"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Combines spectral initialization, Nesterov accelerated epigraph optimization,
and SLSQP polishing for high-precision results.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
_T_START = time.time()
# Safety margin: 150s limit. We stop at 135s to allow for overhead.
_DEADLINE = _T_START + 135


def _linear_overlap_values(h):
    """
    Compute C_k = (2/N)*(N/2 - r_k) where r_k = sum_{i=k}^{N-1} h_i h_{i-k}.
    Uses FFT for efficiency.
    """
    n = len(h)
    # Next power of 2 >= 2*n for efficient FFT
    n_fft = 1
    while n_fft < 2 * n:
        n_fft *= 2
    
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    
    H = np.fft.rfft(h_pad)
    # Autocorrelation
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    r = corr[:n]
    
    cs = (2.0 / n) * (n / 2.0 - r)
    return cs


def _score(h):
    """Compute the maximum linear overlap."""
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    return float(np.max(cs))


def _project(h, target_sum):
    """
    Project h onto {x : 0 <= x_i <= 1, sum(x) = target_sum} via bisection
    on the Lagrange multiplier for the sum constraint.
    """
    h = np.asarray(h, dtype=np.float64).copy()
    current_sum = np.sum(h)
    
    # Quick check if already feasible (within tolerance)
    if abs(current_sum - target_sum) < 1e-12 and np.all(h >= -1e-12) and np.all(h <= 1 + 1e-12):
        return np.clip(h, 0.0, 1.0)

    # Bisection bounds for the offset
    # We want sum(clip(h + offset, 0, 1)) = target_sum
    # If offset is very small, sum is small. If offset is large, sum is large.
    low = -np.max(h) - 1.0
    high = 1.0 - np.min(h) + 1.0

    def sum_of_clipped(offset):
        return np.sum(np.clip(h + offset, 0.0, 1.0))

    # Bisection
    for _ in range(50):
        mid = (low + high) / 2.0
        s_mid = sum_of_clipped(mid)
        if s_mid > target_sum:
            high = mid
        else:
            low = mid

    offset = (low + high) / 2.0
    h_proj = np.clip(h + offset, 0.0, 1.0)

    # Correct for any floating point drift in the sum
    diff = target_sum - np.sum(h_proj)
    if abs(diff) > 1e-14:
        if diff > 0:
            # Add to the smallest elements (those least likely to hit 1)
            idx = np.argmin(h_proj)
            h_proj[idx] += diff
        else:
            # Subtract from the largest elements (those least likely to hit 0)
            idx = np.argmax(h_proj)
            h_proj[idx] += diff
        h_proj = np.clip(h_proj, 0.0, 1.0)

    return h_proj


def _smooth_max_obj_grad(h, beta):
    """
    Compute the value and gradient of the smooth max of C_k using FFT.
    F(h) = (1/beta) * log(sum(exp(beta * C_k)))
    """
    cs = _linear_overlap_values(h)
    cs_max = np.max(cs)
    # Numerical stability for exp
    exp_cs = np.exp(beta * (cs - cs_max))
    sum_exp = np.sum(exp_cs)
    w = exp_cs / sum_exp
    val = (1.0 / beta) * (np.log(sum_exp) + beta * cs_max)

    n_fft = 1
    while n_fft < 2 * N:
        n_fft *= 2

    w_pad = np.zeros(n_fft)
    w_pad[:N] = w
    h_pad = np.zeros(n_fft)
    h_pad[:N] = h

    W = np.fft.rfft(w_pad)
    H = np.fft.rfft(h_pad)

    # Convolution for gradient terms
    # grad C_k / dh_i involves h_j and h_{j+k}
    # The gradient of the log-sum-exp is a weighted sum of gradients of C_k.
    # C_k = (2/N) * (N/2 - sum h_i h_{i+k})
    # dC_k/dh_i = -(2/N) * (h_{i+k} + h_{i-k}) (with wrap-around/linear handling)
    
    # Using FFT to compute the weighted sum of shifts
    conv = np.fft.irfft(W * H, n=n_fft)
    A = conv[:N]

    w_rev = w[::-1]
    w_rev_pad = np.zeros(n_fft)
    w_rev_pad[:N] = w_rev
    W_rev = np.fft.rfft(w_rev_pad)

    conv_rev = np.fft.irfft(W_rev * H, n=n_fft)
    B = conv_rev[N-1 : N-1+N]

    grad = -(2.0 / N) * (A + B)
    return val, grad


def _epigraph_obj_grad(h, t, beta):
    """
    Epigraph formulation: minimize t + (1/beta)*log(sum(exp(beta*(C_k - t))))
    Returns (val, grad_h, grad_t).
    """
    cs = _linear_overlap_values(h)
    diff = cs - t
    diff_max = np.max(diff)
    exp_diff = np.exp(beta * (diff - diff_max))
    sum_exp = np.sum(exp_diff)
    w = exp_diff / sum_exp
    val = t + (1.0 / beta) * (np.log(sum_exp) + beta * diff_max)

    n_fft = 1
    while n_fft < 2 * N:
        n_fft *= 2

    w_pad = np.zeros(n_fft)
    w_pad[:N] = w
    h_pad = np.zeros(n_fft)
    h_pad[:N] = h

    W = np.fft.rfft(w_pad)
    H = np.fft.rfft(h_pad)

    conv = np.fft.irfft(W * H, n=n_fft)
    A = conv[:N]

    w_rev = w[::-1]
    w_rev_pad = np.zeros(n_fft)
    w_rev_pad[:N] = w_rev
    W_rev = np.fft.rfft(w_rev_pad)

    conv_rev = np.fft.irfft(W_rev * H, n=n_fft)
    B = conv_rev[N-1 : N-1+N]

    grad_h = -(2.0 / N) * (A + B)
    grad_t = 1.0 - np.sum(w)
    return val, grad_h, grad_t


def _nesterov_epigraph(h0, beta, max_iter=500, lr_init=0.05):
    """
    Nesterov accelerated gradient descent using epigraph formulation.
    This is often more stable for minimax problems.
    """
    h = np.array(h0, dtype=np.float64, copy=True)
    h = _project(h, N / 2.0)
    t_val = _score(h)

    h_prev = h.copy()
    t_prev = t_val
    t_mom = 1.0
    lr_current = lr_init
    best_h = h.copy()
    best_score = _score(h)
    max_backtracks = 10

    for it in range(max_iter):
        if time.time() > _DEADLINE - 5:
            break

        t_mom_new = (1.0 + np.sqrt(1.0 + 4.0 * t_mom * t_mom)) / 2.0
        momentum = (t_mom - 1.0) / t_mom_new

        y_h = h + momentum * (h - h_prev)
        y_t = t_val + momentum * (t_val - t_prev)

        val_y, grad_h, grad_t = _epigraph_obj_grad(y_h, y_t, beta)

        step_lr = lr_current
        h_new = None
        t_new_val = None
        
        # Backtracking line search
        for bt in range(max_backtracks):
            h_candidate = y_h - step_lr * grad_h
            h_candidate = _project(h_candidate, N / 2.0)
            t_candidate = y_t - step_lr * grad_t
            t_candidate = max(0.0, t_candidate)

            # We check if the objective decreased
            # Using the actual score is a good proxy for the smooth objective decrease
            val_candidate = _score(h_candidate) 
            # Note: _score is the exact max, not the smooth max.
            # For Nesterov to work strictly, we should check the smooth objective.
            # However, since beta is high, they are close.
            # To be safe, let's compute the smooth obj value for comparison.
            val_smooth_candidate, _, _ = _epigraph_obj_grad(h_candidate, t_candidate, beta)
            
            if val_smooth_candidate <= val_y + 1e-9:
                h_new = h_candidate
                t_new_val = t_candidate
                lr_current = step_lr
                break
            step_lr *= 0.5

        if h_new is None:
            # Fallback step
            h_new = y_h - lr_current * 0.5 * grad_h
            h_new = _project(h_new, N / 2.0)
            t_new_val = max(0.0, y_t - lr_current * 0.5 * grad_t)
            lr_current *= 0.5

        h_prev = h
        t_prev = t_val
        h = h_new
        t_val = t_new_val
        t_mom = t_mom_new

        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()

    return best_h, best_score


def _spectral_h_from_coeffs(coeffs):
    """
    Reconstruct h from real DFT coefficients (modes 1..K).
    coeffs = [a1, b1, a2, b2, ..., aK, bK]
    h_i = 0.5 + sum_{k=1}^K (a_k cos(2*pi*k*i/N) + b_k sin(2*pi*k*i/N))
    """
    n_modes = len(coeffs) // 2
    h = np.full(N, 0.5)
    i = np.arange(N)
    for k in range(1, n_modes + 1):
        a = coeffs[2 * (k - 1)]
        b = coeffs[2 * (k - 1) + 1]
        h += a * np.cos(2.0 * np.pi * k * i / N)
        h += b * np.sin(2.0 * np.pi * k * i / N)
    return h


def _h_to_spectral_coeffs(h, K):
    """
    Extract the first K real DFT coefficients from h.
    Returns [a1, b1, ..., aK, bK]
    """
    H = np.fft.rfft(h)
    out = []
    for k in range(1, K + 1):
        a = 2.0 / N * H[k].real
        b = -2.0 / N * H[k].imag
        out.append(a)
        out.append(b)
    return out


def _optimize_spectral(h_init, K, beta, maxiter=200, lr_init=0.01):
    """
    Optimize in spectral coefficient space using L-BFGS-B with penalty for bounds.
    This helps navigate the landscape before switching to full-space optimization.
    """
    coeffs0 = _h_to_spectral_coeffs(h_init, K)

    def obj(coeffs):
        h = _spectral_h_from_coeffs(coeffs)
        # Penalty for out-of-bounds
        penalty = 0.0
        if np.any(h < 0) or np.any(h > 1):
            penalty = 100.0 * (np.sum(np.maximum(0, -h) ** 2) + np.sum(np.maximum(0, h - 1) ** 2))
        val, _ = _smooth_max_obj_grad(h, beta)
        return val + penalty

    def grad(coeffs):
        h = _spectral_h_from_coeffs(coeffs)
        _, grad_h = _smooth_max_obj_grad(h, beta)

        penalty_grad = np.zeros(N)
        penalty_grad[h < 0] += -200.0 * h[h < 0]
        penalty_grad[h > 1] += 200.0 * (h[h > 1] - 1.0)

        total_grad_h = grad_h + penalty_grad

        n_modes = len(coeffs) // 2
        i = np.arange(N)
        grad_coeffs = np.zeros(len(coeffs))
        for k in range(1, n_modes + 1):
            cos_k = np.cos(2.0 * np.pi * k * i / N)
            sin_k = np.sin(2.0 * np.pi * k * i / N)
            grad_coeffs[2 * (k - 1)] = np.sum(total_grad_h * cos_k)
            grad_coeffs[2 * (k - 1) + 1] = np.sum(total_grad_h * sin_k)

        return grad_coeffs

    try:
        res = minimize(obj, coeffs0, jac=grad, method='L-BFGS-B',
                       options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8})
        if res.fun < obj(coeffs0):
            h_opt = _spectral_h_from_coeffs(res.x)
            return h_opt
    except Exception:
        pass

    return h_init


def _run_slsqp(h0, beta, maxiter=300, ftol=1e-12, gtol=1e-10):
    """
    Run SLSQP optimization on the smooth max objective.
    """
    cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N / 2.0,
             'jac': lambda x: np.ones(N)}]
    bounds = [(0.0, 1.0)] * N

    def obj_func(x):
        val, _ = _smooth_max_obj_grad(x, beta)
        return val

    def jac_func(x):
        _, grad = _smooth_max_obj_grad(x, beta)
        return grad

    try:
        res = minimize(obj_func, h0.copy(), jac=jac_func, method='SLSQP',
                       bounds=bounds, constraints=cons,
                       options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol})
        if res.x is not None:
            return res.x
    except Exception:
        pass
    return h0


def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    t0 = time.time()
    np.random.seed(42)

    best_h = None
    best_score = float('inf')

    def _update_best(h):
        nonlocal best_h, best_score
        h = _project(h, N / 2.0)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()

    i = np.arange(N)

    # --- Phase 1: Diverse structured starts ---
    starts = []

    # Cosine fundamental
    h_cos = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
    starts.append(_project(h_cos, N / 2.0))

    # Sine fundamental
    h_sin = 0.5 + 0.5 * np.sin(2.0 * np.pi * i / N)
    starts.append(_project(h_sin, N / 2.0))

    # Two-frequency
    h_2f = 0.5 + 0.3 * np.cos(2.0 * np.pi * i / N) + 0.2 * np.cos(4.0 * np.pi * i / N)
    starts.append(_project(h_2f, N / 2.0))

    # Three-frequency
    h_3f = 0.5 + 0.25 * np.cos(2.0 * np.pi * i / N) + 0.2 * np.cos(4.0 * np.pi * i / N) + 0.15 * np.cos(6.0 * np.pi * i / N)
    starts.append(_project(h_3f, N / 2.0))

    # Random sines
    for _ in range(15):
        f = np.random.uniform(1, 6)
        phase = np.random.uniform(0, 2 * np.pi)
        h_s = 0.5 + 0.5 * np.sin(2 * np.pi * f * i / N + phase)
        starts.append(_project(h_s, N / 2.0))

    # Random
    for _ in range(15):
        h_r = np.random.rand(N)
        starts.append(_project(h_r, N / 2.0))

    for h0 in starts:
        _update_best(h0)

    # --- Phase 2: Spectral optimization from top starts ---
    # Sort starts by score and take top 3
    start_scores = [( _score(h), h) for h in starts]
    start_scores.sort(key=lambda x: x[0])
    top_starts = [h for _, h in start_scores[:3]]

    for h0 in top_starts:
        if time.time() > _DEADLINE - 40:
            break
        # Try different K values
        for K in [8, 16, 32]:
            if time.time() > _DEADLINE - 35:
                break
            h_spec = _optimize_spectral(h0, K, beta=50.0, maxiter=100)
            _update_best(h_spec)

    # --- Phase 3: Nesterov Epigraph Refinement ---
    # Run from the current best
    if best_h is not None and time.time() < _DEADLINE - 25:
        h_cand, s_cand = _nesterov_epigraph(best_h, beta=80.0, max_iter=400, lr_init=0.05)
        _update_best(h_cand)

    # Run from a perturbed version of best to escape local minima
    if best_h is not None and time.time() < _DEADLINE - 20:
        h_pert = best_h + 0.02 * np.random.randn(N)
        h_pert = _project(h_pert, N / 2.0)
        h_cand2, s_cand2 = _nesterov_epigraph(h_pert, beta=80.0, max_iter=300, lr_init=0.05)
        _update_best(h_cand2)

    # --- Phase 4: SLSQP Polish ---
    # SLSQP is good for final precision
    if best_h is not None and time.time() < _DEADLINE - 10:
        h_opt = _run_slsqp(best_h, beta=150.0, maxiter=400, ftol=1e-12, gtol=1e-10)
        _update_best(h_opt)

    # --- Phase 5: Final High-Beta Polish ---
    # One more SLSQP with even higher beta to tighten the max approximation
    if best_h is not None and time.time() < _DEADLINE - 3:
        h_final = _run_slsqp(best_h, beta=300.0, maxiter=300, ftol=1e-14, gtol=1e-12)
        _update_best(h_final)

    if best_h is None:
        best_h = np.full(N, 0.5)

    best_h = _project(best_h, N / 2.0)
    return best_h
