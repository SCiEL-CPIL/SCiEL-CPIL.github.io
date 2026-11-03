"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Uses a frequency-domain convex formulation: the overlap C_k is the inverse
DFT of the PSD S = |H|^2.  Minimizing max_k C_k is equivalent to minimizing
the peak of the inverse DFT of S.  We optimize S directly (convex in S) with
a log-sum-exp smoothed objective, then reconstruct h and polish."""
import time
import numpy as np
from scipy.optimize import minimize, linprog

N = 100
_DEADLINE = time.time() + 140  # Safety margin before 150s kill


def _autocorr(h):
    """Linear autocorrelation r_k = sum_{i=k}^{N-1} h_i * h_{i-k} for k=0..N-1."""
    n = len(h)
    n_fft = 2 * n
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad, n=n_fft)
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    return corr[:n]


def _overlap_values(h):
    """C_k = (2/N) * (N/2 - r_k) for k=0..N-1."""
    n = len(h)
    r = _autocorr(h)
    return (2.0 / n) * (n / 2.0 - r)


def _score(h):
    """Maximum linear overlap."""
    h = np.asarray(h, dtype=np.float64)
    cs = _overlap_values(h)
    return float(np.max(cs))


def _smooth_max_value(h, beta):
    """Smooth maximum of overlap values via log-sum-exp."""
    h = np.asarray(h, dtype=np.float64)
    cs = _overlap_values(h)
    m = np.max(cs)
    return m + np.log(np.sum(np.exp(beta * (cs - m)))) / beta


def _smooth_max_grad(h, beta):
    """Gradient of smooth max objective. Returns (value, gradient)."""
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    cs = _overlap_values(h)
    m = np.max(cs)
    exp_terms = np.exp(beta * (cs - m))
    sum_exp = np.sum(exp_terms)
    w = exp_terms / sum_exp

    val = m + np.log(sum_exp) / beta

    grad = np.zeros(n)
    for k in range(n):
        if w[k] == 0.0:
            continue
        factor = -(2.0 / n) * w[k]
        if k > 0:
            grad[k:] += factor * h[:n-k]
        if k < n:
            grad[:n-k] += factor * h[k:]

    return val, grad


def _project_sum_box(h):
    """Project h onto {x: sum(x)=N/2, 0<=x_i<=1} via bisection."""
    h = np.array(h, dtype=np.float64, copy=True)
    target = N / 2.0
    lo, hi = -10.0, 10.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        projected = np.clip(h - mid, 0.0, 1.0)
        s = np.sum(projected)
        if s > target:
            lo = mid
        else:
            hi = mid
    lambda_star = (lo + hi) / 2.0
    h_proj = np.clip(h - lambda_star, 0.0, 1.0)
    diff = target - np.sum(h_proj)
    if abs(diff) > 1e-15:
        mask = (h_proj > 1e-12) & (h_proj < 1.0 - 1e-12)
        if np.any(mask):
            h_proj[mask] += diff / np.sum(mask)
            h_proj = np.clip(h_proj, 0.0, 1.0)
    return h_proj


def _fix_sum(h):
    """Adjust h to have sum exactly N/2, keeping values in [0,1]."""
    h = np.array(h, dtype=np.float64, copy=True)
    target = N / 2.0
    diff = target - np.sum(h)
    h += diff / N
    h = np.clip(h, 0.0, 1.0)
    diff = target - np.sum(h)
    if abs(diff) > 1e-12:
        mask = (h > 0) & (h < 1)
        if np.any(mask):
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
    diff = target - np.sum(h)
    if abs(diff) > 1e-15:
        if diff > 0:
            idx = np.argmin(h)
        else:
            idx = np.argmax(h)
        h[idx] += diff
        h = np.clip(h, 0.0, 1.0)
    return h


def _psd_to_overlap(S):
    """
    Given PSD S (length N/2+1, real, non-negative), compute overlap values C_k.
    C_k = (2/N)(N/2 - r_k) where r_k is the linear autocorrelation.
    For the circular case, r_k = sum_j h_j h_{j-k} = (1/N) sum_m S_m exp(2*pi*i*m*k/N).
    We use circular autocorrelation as a proxy (close for smooth h).
    """
    # S has length N//2 + 1 (rfft output)
    # Circular autocorrelation: r_k = (1/N) * ifft(fft(S) * conj(fft(S)))... 
    # Actually, for circular: r_k = (1/N) sum_m S_m exp(2*pi*i*m*k/N)
    # Let's compute via IFFT of S (padded to full DFT)
    n = N
    # Build full DFT of S: S_full[m] for m=0..N-1
    S_full = np.zeros(n, dtype=np.complex128)
    S_full[:len(S)] = S
    # For real signal, S_full[N-m] = S_full[m]
    for m in range(1, n // 2 + 1):
        S_full[n - m] = S_full[m]
    # Circular autocorrelation r_k = (1/N) * IFFT(S_full)
    r_circ = np.fft.ifft(S_full).real / 1.0  # ifft includes 1/N factor
    # Actually np.fft.ifft(x) = (1/N) sum x_m exp(2*pi*i*m*k/N), so r_circ = ifft(S_full)
    r_circ = np.fft.ifft(S_full).real
    # C_k = (2/N)(N/2 - r_k)
    C = (2.0 / n) * (n / 2.0 - r_circ)
    return C


def _smooth_max_psd(S, beta):
    """Smooth max of overlap values computed from PSD."""
    C = _psd_to_overlap(S)
    m = np.max(C)
    return m + np.log(np.sum(np.exp(beta * (C - m)))) / beta


def _smooth_max_psd_grad(S, beta):
    """Gradient of smooth max PSD objective w.r.t. S."""
    C = _psd_to_overlap(S)
    m = np.max(C)
    exp_terms = np.exp(beta * (C - m))
    sum_exp = np.sum(exp_terms)
    w = exp_terms / sum_exp
    val = m + np.log(sum_exp) / beta

    # dC_k/dS_m = (2/N) * (1/N) * (-exp(2*pi*i*m*k/N)) ... 
    # C_k = (2/N)(N/2 - r_k), r_k = (1/N) sum_m S_m exp(2*pi*i*m*k/N)
    # dC_k/dS_m = -(2/N^2) exp(2*pi*i*m*k/N)
    # d(val)/dS_m = sum_k w_k * dC_k/dS_m = -(2/N^2) sum_k w_k exp(2*pi*i*m*k/N)
    # = -(2/N^2) * FFT(w)_m  (where FFT is unnormalized DFT)
    
    n = N
    W = np.fft.fft(w)  # W[m] = sum_k w_k exp(-2*pi*i*m*k/N)
    # We need sum_k w_k exp(2*pi*i*m*k/N) = conj(W[m]) for real w
    # Actually W[m] = sum_k w_k exp(-2*pi*i*m*k/N), so sum_k w_k exp(2*pi*i*m*k/N) = W[-m mod N] = conj(W[m])
    grad_S_full = -(2.0 / (n * n)) * np.conj(W)
    # S_full[m] for m=0..N//2, and S_full[N-m] = S_full[m]
    # d/dS[m] = grad_S_full[m] + grad_S_full[N-m] (for m=1..N//2-1)
    # For m=0: just grad_S_full[0]
    # For m=N//2: just grad_S_full[N//2]
    grad_S = np.zeros(len(S))
    grad_S[0] = grad_S_full[0].real
    for m in range(1, len(S)):
        if m < n - m:
            grad_S[m] = (grad_S_full[m] + grad_S_full[n - m]).real
        elif m == n - m:  # m = N/2
            grad_S[m] = grad_S_full[m].real
    return val, grad_S


def _h_to_psd(h):
    """Convert h to PSD S = |H|^2 where H = rfft(h)."""
    H = np.fft.rfft(h)
    S = np.abs(H) ** 2
    return S


def _psd_to_h(S):
    """Reconstruct h from PSD (choosing phases to make h real and in [0,1]).
    Use zero-phase: h_i = (1/N) sum_m S_m^(1/2) exp(0) = mean of sqrt(S)...
    Better: use the phase that minimizes max |h_i - 0.5| or just use h = ifft(sqrt(S_full))
    and then project."""
    n = N
    S_full = np.zeros(n, dtype=np.complex128)
    S_full[:len(S)] = S
    for m in range(1, n // 2 + 1):
        S_full[n - m] = S_full[m]
    # h = ifft(sqrt(S_full)) with zero phase
    amp = np.sqrt(np.maximum(S_full, 0))
    h = np.fft.ifft(amp).real
    return h


def _optimize_psd(S0, beta, maxiter=5000, lr_init=0.1):
    """PGD on PSD space with projection onto non-negative and fixed mean."""
    S = np.array(S0, dtype=np.float64, copy=True)
    # Constraint: sum(S) = N^2/4 (since sum(h)=N/2, by Parseval sum|H|^2 = N*sum(h^2)... 
    # Actually sum(h) = N/2 means H[0] = N/2, so S[0] = N^2/4.
    # The other S[m] are free but non-negative.
    # Let's just optimize S[1:] with S[0] fixed.
    S[0] = (N / 2.0) ** 2  # H[0] = sum(h) = N/2
    lr = lr_init
    best_S = S.copy()
    best_val = _smooth_max_psd(S, beta)
    
    for step in range(maxiter):
        if time.time() > _DEADLINE - 5:
            break
        val, grad = _smooth_max_psd(S, beta)
        # Project gradient: S[0] is fixed
        grad[0] = 0.0
        S_new = S - lr * grad
        S_new = np.maximum(S_new, 0.0)  # Non-negativity
        S_new[0] = (N / 2.0) ** 2  # Fix DC
        val_new = _smooth_max_psd(S_new, beta)
        if val_new < val:
            S = S_new
            val = val_new
            lr = min(lr * 1.03, 1.0)
        else:
            lr *= 0.8
        if val < best_val:
            best_val = val
            best_S = S.copy()
    return best_S


def _run_slsqp(h0, beta, maxiter, ftol, gtol):
    """SLSQP on h-space."""
    cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N/2.0,
             'jac': lambda x: np.ones(N)}]
    bounds = [(0.0, 1.0)] * N
    def obj_func(x):
        val, _ = _smooth_max_grad(x, beta)
        return val
    def jac_func(x):
        _, grad = _smooth_max_grad(x, beta)
        return grad
    try:
        res = minimize(obj_func, h0.copy(), jac=jac_func, method='SLSQP',
                       bounds=bounds, constraints=cons,
                       options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol})
        return res.x
    except Exception:
        return h0


def _pgd_optimize(h0, beta, n_steps=5000, lr_init=0.01):
    """PGD in h-space."""
    h = h0.copy()
    lr = lr_init
    best_h = h.copy()
    best_val = _smooth_max_value(h, beta)
    for step in range(n_steps):
        if time.time() > _DEADLINE - 5:
            break
        val, grad = _smooth_max_grad(h, beta)
        h_new = _project_sum_box(h - lr * grad)
        val_new = _smooth_max_value(h_new, beta)
        if val_new < val:
            h = h_new
            val = val_new
            lr = min(lr * 1.05, 0.05)
        else:
            lr *= 0.7
        if val < best_val:
            best_val = val
            best_h = h.copy()
    return best_h


def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    t0 = time.time()
    np.random.seed(42)
    best_h = None
    best_score = float('inf')

    def _update_best(h):
        nonlocal best_h, best_score
        h = _fix_sum(h)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()

    i = np.arange(N)

    # Initial candidates
    h_cos = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
    _update_best(_fix_sum(h_cos))
    h_sin = 0.5 + 0.5 * np.sin(2.0 * np.pi * i / N)
    _update_best(_fix_sum(h_sin))
    h_alt = np.where(i % 2 == 0, 0.75, 0.25)
    _update_best(_fix_sum(h_alt))
    h_block = np.where(i < N//2, 0.75, 0.25)
    _update_best(_fix_sum(h_block))
    for seed in [1, 2, 3]:
        np.random.seed(seed)
        h_rand = np.random.uniform(0, 1, N)
        _update_best(_fix_sum(h_rand))

    # PSD-space optimization
    if best_h is not None and time.time() - t0 < _DEADLINE - 50:
        S0 = _h_to_psd(best_h)
        for beta in [20.0, 50.0, 100.0]:
            if time.time() - t0 > _DEADLINE - 20:
                break
            S_opt = _optimize_psd(S0, beta, maxiter=3000, lr_init=0.5)
            h_from_psd = _psd_to_h(S_opt)
            _update_best(h_from_psd)
            S0 = S_opt  # Warm-start next beta

    # PGD in h-space
    if time.time() - t0 < _DEADLINE - 40:
        h_pgd = _pgd_optimize(best_h, beta=50.0, n_steps=10000, lr_init=0.01)
        _update_best(h_pgd)

    # SLSQP
    if time.time() - t0 < _DEADLINE - 30:
        h_opt = _run_slsqp(best_h, beta=50.0, maxiter=300, ftol=1e-10, gtol=1e-8)
        _update_best(h_opt)

    # Random restarts
    if time.time() - t0 < _DEADLINE - 25:
        for _ in range(3):
            if time.time() - t0 > _DEADLINE - 15:
                break
            h_rand = np.random.uniform(0, 1, N)
            h_rand = _fix_sum(h_rand)
            h_pgd_rand = _pgd_optimize(h_rand, beta=50.0, n_steps=3000, lr_init=0.01)
            _update_best(h_pgd_rand)

    # High-beta polish
    if best_h is not None and time.time() - t0 < _DEADLINE - 15:
        h_polished = _pgd_optimize(best_h, beta=100.0, n_steps=5000, lr_init=0.005)
        _update_best(h_polished)

    # Final SLSQP polish
    if best_h is not None and time.time() - t0 < _DEADLINE - 10:
        h_final = _run_slsqp(best_h, beta=200.0, maxiter=1000, ftol=1e-14, gtol=1e-12)
        h_final = _fix_sum(h_final)
        _update_best(h_final)

    # Very final PGD
    if best_h is not None and time.time() - t0 < _DEADLINE - 5:
        h_very_final = _pgd_optimize(best_h, beta=500.0, n_steps=3000, lr_init=0.002)
        _update_best(h_very_final)

    if best_h is None:
        best_h = np.full(N, 0.5)
    best_h = _fix_sum(best_h)
    return best_h
