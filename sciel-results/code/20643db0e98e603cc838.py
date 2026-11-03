"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Combines FFT-accelerated linear overlap with Nesterov-accelerated projected gradient
on a smooth-max surrogate, plus a final coordinate-descent polish on the active set.
Structural change: replaces SLSQP with accelerated projected gradient descent,
which is more robust to the non-convex landscape and allows tighter convergence
to the true max via coordinate refinement."""
import time
import numpy as np

N = 100
_DEADLINE = time.time() + 110  # Safety margin before 120s kill


def _linear_overlap_values(h):
    """
    Compute the linear overlap values C_k for k=0..N-1.
    C_k = (2/N) * (N/2 - sum_{i=k}^{N-1} h_i h_{i-k})
    Vectorized using FFT for speed.
    """
    n = len(h)
    n_fft = 1
    while n_fft < 2 * n:
        n_fft *= 2
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad, n=n_fft)
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    r = corr[:n]
    cs = (2.0 / n) * (n / 2.0 - r)
    return cs


def _score(h):
    """Compute the maximum linear overlap."""
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    return float(np.max(cs))


def _smooth_max_val_grad(h, beta):
    """
    Return (smooth_max_value, gradient) of max(C_k) using log-sum-exp.
    C_k = 1 - (2/n) * r_k where r_k = sum_{i=k}^{n-1} h[i]*h[i-k]
    dC_k/dh_j = -(2/n) * (h_{j-k} [if j>=k] + h_{j+k} [if j+k<n])
    """
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    cs = _linear_overlap_values(h)
    m = np.max(cs)
    exp_terms = np.exp(beta * (cs - m))
    sum_exp = np.sum(exp_terms)
    w = exp_terms / sum_exp
    val = m + np.log(sum_exp) / beta

    grad = np.zeros(n)
    for k in range(n):
        wk = w[k]
        if wk < 1e-14:
            continue
        factor = -(2.0 / n) * wk
        # dC_k/dh_j for j >= k: factor * h[j-k]
        if k > 0:
            grad[k:] += factor * h[:n - k]
        # dC_k/dh_j for j+k < n: factor * h[j+k]
        if k < n:
            grad[:n - k] += factor * h[k:]

    return val, grad


def _fix_sum(h):
    """Adjust h to have sum exactly N/2, keeping values in [0, 1]."""
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


def _nesterov_projected_gradient(h0, beta, max_iter, lr_init, deadline):
    """
    Nesterov-accelerated projected gradient descent on smooth max surrogate.
    Projects onto {0<=h<=1, sum(h)=N/2} at each step.
    """
    h = np.array(h0, dtype=np.float64, copy=True)
    h = _fix_sum(h)
    y = h.copy()
    t = 1.0
    lr = lr_init
    best_h = h.copy()
    best_val = _score(h)
    prev_val = _smooth_max_val_grad(h, beta)[0]

    for it in range(max_iter):
        if time.time() > deadline:
            break

        # Nesterov lookahead
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        momentum = (t - 1.0) / t_new
        y = h + momentum * (h - y)
        y = np.clip(y, 0.0, 1.0)

        # Gradient at y
        val_y, grad_y = _smooth_max_val_grad(y, beta)

        # Gradient step
        h_new = y - lr * grad_y
        h_new = _fix_sum(h_new)

        # Line search: accept if smooth max improved
        val_new, _ = _smooth_max_val_grad(h_new, beta)
        if val_new < val_y:
            h = h_new
            t = t_new
            # Adaptive learning rate
            if val_new < best_val:
                best_val = val_new
                best_h = h.copy()
        else:
            lr *= 0.5
            if lr < 1e-10:
                break

    # Final true score check
    true_score = _score(best_h)
    return best_h, true_score


def _coordinate_polish(h, max_iter=200, deadline=None):
    """
    Coordinate descent on the true max(C_k) objective.
    For each coordinate j, try small perturbations and accept if max overlap decreases.
    This handles the non-smooth max better than gradient methods.
    """
    h = np.array(h, dtype=np.float64, copy=True)
    h = _fix_sum(h)
    current_score = _score(h)
    best_h = h.copy()
    best_score = current_score

    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break

        improved = False
        for j in range(N):
            if deadline is not None and time.time() > deadline:
                break

            # Try small steps in + and - directions
            for delta in [0.01, 0.005, 0.001, -0.01, -0.005, -0.001]:
                h_trial = h.copy()
                h_trial[j] += delta
                h_trial = _fix_sum(h_trial)
                s = _score(h_trial)
                if s < best_score - 1e-10:
                    h = h_trial
                    best_h = h.copy()
                    best_score = s
                    improved = True
                    break

        if not improved:
            # Try smaller steps
            for j in range(N):
                if deadline is not None and time.time() > deadline:
                    break
                for delta in [0.0005, 0.0002, -0.0005, -0.0002]:
                    h_trial = h.copy()
                    h_trial[j] += delta
                    h_trial = _fix_sum(h_trial)
                    s = _score(h_trial)
                    if s < best_score - 1e-12:
                        h = h_trial
                        best_h = h.copy()
                        best_score = s
                        improved = True
                        break

        if not improved:
            break

    return best_h, best_score


def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    t0 = time.time()
    deadline = t0 + 105  # 15s safety margin
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

    # Phase 1: Multiple diverse starts with Nesterov accelerated gradient
    starts = []

    # Cosine wave
    i = np.arange(N)
    h_cos = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
    starts.append(_fix_sum(h_cos))

    # Sine wave
    h_sin = 0.5 + 0.5 * np.sin(2.0 * np.pi * i / N)
    starts.append(_fix_sum(h_sin))

    # Alternating
    h_alt = np.where(i % 2 == 0, 0.7, 0.3)
    starts.append(_fix_sum(h_alt))

    # Random starts
    rng = np.random.default_rng(42)
    for _ in range(4):
        h_rand = rng.uniform(0, 1, N)
        starts.append(_fix_sum(h_rand))

    # Triangular wave
    h_tri = 0.5 + 0.5 * np.abs(2.0 * (i / N - 0.5) - 0.5)
    starts.append(_fix_sum(h_tri))

    # Optimize each start with Nesterov
    for idx, h0 in enumerate(starts):
        if time.time() > deadline - 15:
            break

        _update_best(h0)

        # Run Nesterov with moderate beta
        h_opt, s_opt = _nesterov_projected_gradient(
            h0, beta=50.0, max_iter=300, lr_init=0.02, deadline=deadline - 15
        )
        _update_best(h_opt)

        # Polish with higher beta
        if time.time() < deadline - 10:
            h_opt2, s_opt2 = _nesterov_projected_gradient(
                best_h, beta=100.0, max_iter=200, lr_init=0.01, deadline=deadline - 10
            )
            _update_best(h_opt2)

    # Phase 2: Coordinate descent polish on the best solution
    if best_h is not None and time.time() < deadline - 5:
        h_polished, s_polished = _coordinate_polish(
            best_h, max_iter=100, deadline=deadline - 3
        )
        _update_best(h_polished)

    # Phase 3: One more round of Nesterov + coordinate polish if time allows
    if best_h is not None and time.time() < deadline - 5:
        h_opt3, _ = _nesterov_projected_gradient(
            best_h, beta=200.0, max_iter=150, lr_init=0.005, deadline=deadline - 3
        )
        _update_best(h_opt3)

        if time.time() < deadline - 2:
            h_polished2, _ = _coordinate_polish(
                best_h, max_iter=50, deadline=deadline - 1
            )
            _update_best(h_polished2)

    if best_h is None:
        best_h = np.full(N, 0.5)

    best_h = _fix_sum(best_h)
    return best_h
