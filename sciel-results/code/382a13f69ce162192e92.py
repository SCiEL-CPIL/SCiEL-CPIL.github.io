"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Design family: Projected (Nesterov) Gradient Descent on a LogSumExp smooth
maximum surrogate, followed by an exact coordinate-descent polish and an
L-BFGS refinement.  Core changes vs. the seed:
  * vectorized gradient (removes the O(N^2) Python loop) -> far more iterations
  * Nesterov momentum + backtracking line search
  * coordinate-descent polish on the EXACT nonsmooth objective
  * L-BFGS refinement on the smooth surrogate
  * richer initial candidates and more restarts
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
_DEADLINE = time.time() + 140  # safety margin before the 150s kill


def _linear_overlap_values(h):
    """C_k = (2/N) * (N/2 - r_k), r_k = sum_{i=k}^{N-1} h_i h_{i-k} (autocorr)."""
    n = len(h)
    n_fft = 2 * n
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad, n=n_fft)
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    r = corr[:n]
    return (2.0 / n) * (n / 2.0 - r)


def _score(h):
    """Maximum linear overlap (the quantity to minimize)."""
    h = np.asarray(h, dtype=np.float64)
    return float(np.max(_linear_overlap_values(h)))


def _smooth_max_grad(h, beta):
    """
    Value and gradient of the LogSumExp smooth maximum of the overlap values.
    Gradient computed fully vectorized.
    """
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    cs = _linear_overlap_values(h)
    m = np.max(cs)
    exp_terms = np.exp(beta * (cs - m))
    sum_exp = np.sum(exp_terms)
    w = exp_terms / sum_exp  # softmax weights over the N overlap values
    val = m + np.log(sum_exp) / beta

    # dC_k/dh_j = -(2/n) * [ h_{j-k} (j>=k) + h_{j+k} (j+k<n) ]
    # grad_j = -(2/n) * sum_k w_k [ h_{j-k} + h_{j+k} ]
    # Build the Toeplitz-like matrix M[j,k] = h_{j-k} + h_{j+k} via FFT.
    n_fft = 2 * n
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad, n=n_fft)
    # term A[j,k]=h_{j-k}: rows of the (circular) autocorrelation of h with itself
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    # corr[t] = sum_m h[m] h[m+t]  ->  corr[k] = sum_m h[m] h[m+k]
    # We need per-row vectors.  Use the structure: A[j,k] = h_{j-k} (with wrap).
    # Vectorize by: for each k, A[:,k] = roll(h, k) (circular).  Sum over k weighted.
    # grad_j = -(2/n) * sum_k w_k * ( h_{j-k} + h_{j+k} )
    # sum_k w_k h_{j-k}  =  circular cross-correlation of w with h, evaluated at j.
    # Do both via FFT: circular convolution.
    W = np.fft.rfft(w, n=n_fft)
    # c1[j] = sum_k w_k h_{j-k}  =  (w * h)_j circular convolution
    c1 = np.fft.irfft(W * H, n=n_fft)[:n]
    # c2[j] = sum_k w_k h_{j+k}  =  sum_k w_k h_{j+k}
    # = circular cross-correlation: c2[j] = sum_k w_k h_{j+k}
    # note h_{j+k} = h_{(j+k) mod n};  c2[j] = sum_k w_k h_{j+k}
    # = (w_rev * h)_j where w_rev[k] = w_{-k}.  Equivalent: flip w then convolve.
    w_rev = np.zeros(n_fft)
    w_rev[:n] = w
    w_rev[1:n] = w_rev[1:n][::-1]  # w_rev[k] = w[-k]
    Wrev = np.fft.rfft(w_rev, n=n_fft)
    c2 = np.fft.irfft(Wrev * H, n=n_fft)[:n]

    grad = -(2.0 / n) * (c1 + c2)
    return val, grad


def _project_sum_box(y, target_sum):
    """Project y onto {x: 0<=x_i<=1, sum(x)=target_sum} via bisection on shift."""
    lo = np.min(y) - 1.0
    hi = np.max(y) + 1.0

    def g(tau):
        return np.sum(np.clip(y + tau, 0.0, 1.0)) - target_sum

    if g(lo) > 0:
        return np.clip(y + lo, 0.0, 1.0)
    if g(hi) < 0:
        return np.clip(y + hi, 0.0, 1.0)

    for _ in range(60):
        mid = (lo + hi) / 2.0
        if g(mid) > 0:
            hi = mid
        else:
            lo = mid
    tau = (lo + hi) / 2.0
    return np.clip(y + tau, 0.0, 1.0)


def _nesterov_pgd(h0, beta, steps, lr, deadline, momentum=0.9):
    """Nesterov-accelerated projected gradient descent with backtracking."""
    h = h0.copy()
    target_sum = N / 2.0
    n = len(h)
    y = h.copy()
    current_lr = lr
    prev_val, _ = _smooth_max_grad(h, beta)

    for i in range(steps):
        if time.time() > deadline:
            break
        val, grad = _smooth_max_grad(y, beta)
        step_lr = current_lr
        cand = y - step_lr * grad
        cand_proj = _project_sum_box(cand, target_sum)
        new_val, _ = _smooth_max_grad(cand_proj, beta)
        while new_val > prev_val and step_lr > 1e-9:
            step_lr *= 0.5
            cand = y - step_lr * grad
            cand_proj = _project_sum_box(cand, target_sum)
            new_val, _ = _smooth_max_grad(cand_proj, beta)
        if new_val < prev_val:
            h = cand_proj
            y = h + momentum * (h - y)
            y = _project_sum_box(y, target_sum)
            prev_val = new_val
            if i % 20 == 0:
                current_lr = min(current_lr * 1.05, 0.2)
        else:
            current_lr = max(current_lr * 0.5, 1e-7)
            y = h.copy()
    return h


def _coordinate_polish(h0, deadline, max_sweeps=200):
    """Exact coordinate descent on the nonsmooth max-overlap objective."""
    h = h0.copy()
    target_sum = N / 2.0
    n = len(h)
    cur = _score(h)
    for sweep in range(max_sweeps):
        if time.time() > deadline:
            break
        improved = False
        for j in range(n):
            if time.time() > deadline:
                break
            base = h[j]
            # try moving h[j] up and down by a small amount, fixing sum via a
            # paired adjustment on a "reservoir" coordinate to keep sum exact.
            # Simpler: try a few candidate values for h[j] and re-project.
            for delta in (0.02, -0.02, 0.01, -0.01, 0.05, -0.05):
                h2 = h.copy()
                h2[j] = base + delta
                h2 = _project_sum_box(h2, target_sum)
                s2 = _score(h2)
                if s2 < cur - 1e-12:
                    h = h2
                    cur = s2
                    improved = True
                    break
        if not improved:
            break
    return h, cur


def _lbfgs_refine(h0, beta, deadline):
    """L-BFGS on the smooth surrogate, with projection-based constraints handled
    by optimizing over a free variable and projecting (penalty-free via the
    smooth objective + a final projection)."""
    h = h0.copy()
    target_sum = N / 2.0

    def fun(x):
        return _smooth_max_grad(x, beta)[0]

    def gfun(x):
        return _smooth_max_grad(x, beta)[1]

    try:
        res = minimize(fun, h, jac=gfun, method='L-BFGS-B',
                       options={'maxiter': 400, 'maxfun': 4000,
                                'gtol': 1e-6, 'ftol': 1e-9})
        if res.x is not None:
            hp = _project_sum_box(res.x, target_sum)
            return hp
    except Exception:
        pass
    return h


def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    t0 = time.time()
    np.random.seed(42)

    best_h = None
    best_score = float('inf')

    def _update_best(h):
        nonlocal best_h, best_score
        h = _project_sum_box(h, N / 2.0)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()

    i = np.arange(N)
    candidates = []

    # Multiple-frequency sine/cosine waves
    for freq in (1, 2, 3, 5):
        candidates.append(0.5 + 0.5 * np.cos(2.0 * np.pi * freq * i / N))
        candidates.append(0.5 + 0.5 * np.sin(2.0 * np.pi * freq * i / N))
    # Alternating
    candidates.append(np.where(i % 2 == 0, 0.5 + 0.2, 0.5 - 0.2))
    # Sawtooth-ish
    candidates.append(0.5 + 0.4 * (2.0 * (i / N) - 1.0))
    # Random starts
    for _ in range(8):
        candidates.append(np.random.uniform(0, 1, N))

    for c in candidates:
        cp = _project_sum_box(np.asarray(c, dtype=np.float64), N / 2.0)
        _update_best(cp)

    # Optimization phases (coarse -> fine beta), then exact polish + L-BFGS.
    if best_h is not None and time.time() - t0 < _DEADLINE - 70:
        _update_best(_nesterov_pgd(best_h, beta=10.0, steps=400, lr=0.05,
                                   deadline=_DEADLINE - 70))
    if best_h is not None and time.time() - t0 < _DEADLINE - 45:
        _update_best(_nesterov_pgd(best_h, beta=50.0, steps=600, lr=0.01,
                                   deadline=_DEADLINE - 45))
    if best_h is not None and time.time() - t0 < _DEADLINE - 25:
        _update_best(_nesterov_pgd(best_h, beta=150.0, steps=600, lr=0.004,
                                   deadline=_DEADLINE - 25))
    if best_h is not None and time.time() - t0 < _DEADLINE - 15:
        _update_best(_nesterov_pgd(best_h, beta=300.0, steps=300, lr=0.001,
                                   deadline=_DEADLINE - 15))

    # Exact coordinate-descent polish on the nonsmooth objective.
    if best_h is not None and time.time() - t0 < _DEADLINE - 5:
        hp, _ = _coordinate_polish(best_h, deadline=_DEADLINE - 5, max_sweeps=200)
        _update_best(hp)

    # L-BFGS refinement on a high-beta smooth surrogate, then re-polish.
    if best_h is not None and time.time() - t0 < _DEADLINE - 2:
        _update_best(_lbfgs_refine(best_h, beta=200.0, deadline=_DEADLINE - 2))
        if time.time() - t0 < _DEADLINE - 1:
            hp, _ = _coordinate_polish(best_h, deadline=_DEADLINE - 1, max_sweeps=50)
            _update_best(hp)

    if best_h is None:
        best_h = np.full(N, 0.5)

    # Final projection for exact feasibility.
    best_h = _project_sum_box(best_h, N / 2.0)
    return best_h
