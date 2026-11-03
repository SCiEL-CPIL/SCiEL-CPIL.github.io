"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Structural change: direct minimization of smooth-max surrogate using L-BFGS-B
with sum constraint handled via parameterization, plus local refinement with
projected gradient. Uses FFT for O(n log n) overlap computation."""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
_DEADLINE = time.time() + 142


def _linear_overlap_values(h):
    """Compute C_k = (2/N)*(N/2 - r_k) where r_k = sum_{i=k}^{N-1} h_i h_{i-k}."""
    n = len(h)
    n_fft = 1
    while n_fft < 2 * n:
        n_fft *= 2
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad)
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    r = corr[:n]
    cs = (2.0 / n) * (n / 2.0 - r)
    return cs


def _score(h):
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    return float(np.max(cs))


def _fix_sum(h):
    """Adjust h to have sum exactly N/2, keeping values in [0, 1]."""
    h = np.array(h, dtype=np.float64, copy=True)
    target = N / 2.0
    diff = target - np.sum(h)
    h += diff / N
    h = np.clip(h, 0.0, 1.0)
    diff = target - np.sum(h)
    if abs(diff) > 1e-12:
        mask = (h > 1e-10) & (h < 1.0 - 1e-10)
        if np.any(mask):
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
    diff = target - np.sum(h)
    if abs(diff) > 1e-14:
        if diff > 0:
            idx = np.argmin(h)
        else:
            idx = np.argmax(h)
        h[idx] += diff
        h = np.clip(h, 0.0, 1.0)
    return h


def _smooth_max_val_grad(h, beta):
    """Return (smooth_max_value, gradient) of max(C_k) using log-sum-exp."""
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
        if wk < 1e-15:
            continue
        factor = -(2.0 / n) * wk
        if k > 0:
            grad[k:] += factor * h[:n - k]
        grad[:n - k] += factor * h[k:]
    return val, grad


def _projected_gradient_step(h, grad, lr):
    """One projected gradient step with box + sum constraint."""
    h_new = h - lr * grad
    h_new = np.clip(h_new, 0.0, 1.0)
    h_new = _fix_sum(h_new)
    return h_new


def _accelerated_gradient(h0, beta, max_iter=300):
    """Nesterov accelerated projected gradient on smooth max."""
    h = np.array(h0, dtype=np.float64, copy=True)
    h = _fix_sum(h)
    h_prev = h.copy()
    t_k = 1.0
    lr = 0.05

    best_h = h.copy()
    best_val = _score(h)

    for it in range(max_iter):
        if time.time() > _DEADLINE - 5:
            break
        val, grad = _smooth_max_val_grad(h, beta)

        h_new = h - lr * grad
        h_new = np.clip(h_new, 0.0, 1.0)
        h_new = _fix_sum(h_new)

        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t_k * t_k)) / 2.0
        h_next = h_new + ((t_k - 1.0) / t_new) * (h_new - h_prev)
        h_next = np.clip(h_next, 0.0, 1.0)
        h_next = _fix_sum(h_next)

        h_prev = h
        h = h_next
        t_k = t_new

        if it % 20 == 19:
            lr = max(0.005, min(0.2, lr * 0.9))

        s = _score(h)
        if s < best_val:
            best_val = s
            best_h = h.copy()

    return best_h, best_val


def _lbfgs_smooth_max(h0, beta, max_iter=200):
    """L-BFGS-B optimization of smooth max with sum constraint via reparameterization."""
    h0 = np.array(h0, dtype=np.float64, copy=True)
    h0 = _fix_sum(h0)

    # Use logit parameterization: h_i = sigmoid(x_i)
    # But we need sum = N/2, so we work in a constrained space.
    # Alternative: use L-BFGS-B with bounds [0,1] and add sum penalty.
    
    def objective(x):
        if time.time() > _DEADLINE - 5:
            return 1e6, np.zeros(len(x))
        h = np.clip(x, 0.0, 1.0)
        # Penalty for sum constraint
        sum_penalty = 100.0 * (np.sum(h) - N / 2.0) ** 2
        val, grad = _smooth_max_val_grad(h, beta)
        # Gradient of penalty
        grad_pen = 200.0 * (np.sum(h) - N / 2.0)
        return val + sum_penalty, grad + grad_pen

    res = minimize(objective, h0, method='L-BFGS-B', 
                   bounds=[(0, 1)] * N,
                   options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8})
    
    h_opt = _fix_sum(res.x)
    s = _score(h_opt)
    return h_opt, s


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

    # Diverse initial guesses
    starts = []

    # Constant
    starts.append(np.full(N, 0.5))

    # Cosine
    h_cos = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
    starts.append(_fix_sum(h_cos))

    # Sine
    h_sin = 0.5 + 0.5 * np.sin(2.0 * np.pi * i / N)
    starts.append(_fix_sum(h_sin))

    # Two-frequency
    h_2f = 0.5 + 0.3 * np.cos(2.0 * np.pi * i / N) + 0.2 * np.cos(4.0 * np.pi * i / N)
    starts.append(_fix_sum(h_2f))

    # Three-frequency
    h_3f = 0.5 + 0.25 * np.cos(2.0 * np.pi * i / N) + 0.15 * np.cos(4.0 * np.pi * i / N) + 0.1 * np.cos(6.0 * np.pi * i / N)
    starts.append(_fix_sum(h_3f))

    # Sawtooth-like
    h_saw = 0.5 + 0.5 * np.sin(2.0 * np.pi * i / N) * np.sign(np.sin(4.0 * np.pi * i / N))
    starts.append(_fix_sum(h_saw))

    # Random starts
    for seed in range(8):
        np.random.seed(seed)
        h_r = np.random.uniform(0, 1, N)
        starts.append(_fix_sum(h_r))

    # Evaluate all starts
    for h0 in starts:
        _update_best(h0)

    # Phase 1: Accelerated gradient from each start with moderate beta
    for h0 in starts[:6]:
        if time.time() > _DEADLINE - 10:
            break
        h_candidate, s_candidate = _accelerated_gradient(h0, beta=50.0, max_iter=200)
        if s_candidate < best_score:
            _update_best(h_candidate)

    # Phase 2: L-BFGS-B from best and a few other starts
    for h0 in [best_h] + starts[:3]:
        if time.time() > _DEADLINE - 10:
            break
        h_candidate, s_candidate = _lbfgs_smooth_max(h0, beta=100.0, max_iter=150)
        if s_candidate < best_score:
            _update_best(h_candidate)

    # Phase 3: Polish with high-beta accelerated gradient
    if best_h is not None and time.time() < _DEADLINE - 8:
        h_final, s_final = _accelerated_gradient(best_h, beta=200.0, max_iter=300)
        if s_final < best_score:
            _update_best(h_final)

    # Phase 4: Local refinement with very high beta
    if best_h is not None and time.time() < _DEADLINE - 5:
        h_final2, s_final2 = _accelerated_gradient(best_h, beta=500.0, max_iter=200)
        if s_final2 < best_score:
            _update_best(h_final2)

    if best_h is None:
        best_h = np.full(N, 0.5)

    best_h = _fix_sum(best_h)
    return best_h
