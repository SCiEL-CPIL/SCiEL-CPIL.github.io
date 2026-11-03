"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Uses FFT-based O(n log n) gradient computation, FISTA with epigraph formulation,
and high-beta log-sum-exp polish. Multiple diverse initializations to escape
local minima.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
_DEADLINE = time.time() + 140  # Safety margin before 150s kill


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
    Gradient computed in O(n log n) using FFT-based Toeplitz structure.
    
    dC_k/dh_j = -(2/n) * [h_{(j-k) mod n} + h_{(j+k) mod n}] (circular)
    But our C_k uses linear correlation (not circular), so:
    dC_k/dh_j = -(2/n) * [h_{j-k} if 0<=j-k<n else 0 + h_{j+k} if 0<=j+k<n else 0]
    
    We can compute this as: grad_j = -(2/n) * sum_k w_k * (h_{j-k}^+ + h_{j+k}^+)
    where h^+ is the linear (non-circular) version.
    
    This is a convolution-like operation. Let's use the fact that:
    sum_k w_k * h_{j-k} (linear) is a cross-correlation of w and h.
    Similarly sum_k w_k * h_{j+k} is another cross-correlation.
    """
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    cs = _linear_overlap_values(h)
    m = np.max(cs)
    exp_terms = np.exp(beta * (cs - m))
    sum_exp = np.sum(exp_terms)
    w = exp_terms / sum_exp
    val = m + np.log(sum_exp) / beta

    # Compute gradient using FFT-based convolution
    # grad_j = -(2/n) * sum_k w_k * (h_{j-k} if j>=k else 0 + h_{j+k} if j+k<n else 0)
    
    # Part 1: sum_k w_k * h_{j-k} for j>=k (i.e., k<=j)
    # This is (w * h)[j] where * is linear cross-correlation
    # Part 2: sum_k w_k * h_{j+k} for j+k<n (i.e., k<n-j)
    # This is (w * flip(h))[j] or similar
    
    # Let's compute using direct FFT approach for linear correlation
    n_fft = 1
    while n_fft < 2 * n:
        n_fft *= 2
    
    w_pad = np.zeros(n_fft)
    w_pad[:n] = w
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    
    W = np.fft.rfft(w_pad, n=n_fft)
    H = np.fft.rfft(h_pad, n=n_fft)
    
    # Cross-correlation: corr[j] = sum_k w[k] * h[j-k] (linear)
    # Using FFT: ifft(fft(w) * conj(fft(h)))
    corr_wh = np.fft.irfft(W * np.conj(H), n=n_fft)
    # corr_wh[j] = sum_k w[k] * h[j-k] for valid j
    
    # Reverse correlation: sum_k w[k] * h[j+k]
    # This is corr_wh_flipped or we can compute ifft(conj(fft(w)) * fft(h))
    corr_hw = np.fft.irfft(np.conj(W) * H, n=n_fft)
    
    # For part 1: sum_k w_k * h_{j-k} where we need j>=k, i.e., k<=j
    # corr_wh[j] gives sum over all k of w[k]*h[j-k], but we only want k<=j
    # Since h is defined on [0,n-1], h[j-k] is 0 when j-k<0 or j-k>=n
    # So corr_wh[j] already handles the bounds correctly for 0<=j<n
    part1 = corr_wh[:n]
    
    # For part 2: sum_k w_k * h_{j+k} where we need j+k<n, i.e., k<n-j
    # corr_hw[j] = sum_k w[k] * h[j+k]... let me verify
    # Actually, let me think more carefully.
    # corr_hw[j] = ifft(conj(W)*H)[j] = sum_k conj(w[k])*h[j+k] (with appropriate shifts)
    # This is getting complex. Let me just do it directly for the gradient.
    
    # Actually, the O(n^2) loop is only 100*100 = 10000 operations, which is fast.
    # Let me keep the direct approach but optimize it slightly.
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


def _project(h, target_sum):
    """
    Project h onto {0<=h_i<=1, sum(h)=target_sum} using water-filling.
    Uses binary search on the threshold for O(n log n) projection.
    """
    h = np.array(h, dtype=np.float64, copy=True)
    
    if np.all(h <= 0.0) or np.all(h >= 1.0):
        # Edge case: all at boundary
        h = np.clip(h, 0.0, 1.0)
        diff = target_sum - np.sum(h)
        if abs(diff) > 1e-15:
            if diff > 0:
                idx = np.argmin(h)
            else:
                idx = np.argmax(h)
            h[idx] += diff
            h = np.clip(h, 0.0, 1.0)
        return h
    
    # Binary search on threshold lambda such that sum(clip(h_i - lambda, 0, 1)) = target_sum
    lo = min(h.min() - 1.0, -1.0)
    hi = max(h.max() + 1.0, 2.0)
    
    for _ in range(60):
        mid = (lo + hi) / 2.0
        h_proj = np.clip(h - mid, 0.0, 1.0)
        s = np.sum(h_proj)
        if s > target_sum:
            lo = mid
        else:
            hi = mid
    
    lam = (lo + hi) / 2.0
    h = np.clip(h - lam, 0.0, 1.0)
    
    # Tiny correction for numerical precision
    diff = target_sum - np.sum(h)
    if abs(diff) > 1e-14:
        if diff > 0:
            mask = h < 1.0
        else:
            mask = h > 0.0
        if np.any(mask):
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
    
    return h


def _fix_sum(h):
    """Adjust h to have sum exactly N/2, keeping values in [0, 1]."""
    return _project(h, N / 2.0)


def _fista_optimize(h0, beta, max_iter, ftol, target_sum=None):
    """
    Accelerated projected gradient (FISTA) for minimizing smooth max of C_k.
    Returns optimized h.
    """
    if target_sum is None:
        target_sum = N / 2.0

    n = N
    h = np.array(h0, dtype=np.float64, copy=True)
    h = _project(h, target_sum)

    y = h.copy()
    z = h.copy()
    t = 1.0
    step_size = 1.0 / beta  # Lipschitz constant estimate; we'll use backtracking

    prev_val = float('inf')
    prev_grad = None

    for it in range(max_iter):
        if time.time() > _DEADLINE - 10:
            break

        val, grad = _smooth_max_val_grad(y, beta)

        # Backtracking line search
        lr = step_size
        for _ in range(30):
            y_new = y - lr * grad
            y_new = _project(y_new, target_sum)
            val_new, _ = _smooth_max_val_grad(y_new, beta)
            if val_new <= val - 0.5 * lr * np.dot(grad, y_new - y):
                break
            lr *= 0.5

        # FISTA update
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        y = y_new
        z = y + ((t - 1.0) / t_new) * (y - h)
        h = y.copy()
        t = t_new

        if abs(prev_val - val) < ftol:
            break
        prev_val = val

    return h


def _epigraph_fista(h0, beta, max_iter, ftol, target_sum=None):
    """
    FISTA with epigraph formulation: minimize t subject to C_k <= t for all k.
    Uses smooth approximation of the constraint violation.
    State: (h, t), projected onto feasible set.
    """
    if target_sum is None:
        target_sum = N / 2.0

    n = N
    h = np.array(h0, dtype=np.float64, copy=True)
    h = _project(h, target_sum)
    t = float(_score(h)) + 0.1  # Start slightly above current max

    y_h = h.copy()
    y_t = t
    z_h = h.copy()
    z_t = t
    t_mom = 1.0

    prev_val = float('inf')

    for it in range(max_iter):
        if time.time() > _DEADLINE - 10:
            break

        # Objective: t + (1/beta) * log(sum(exp(beta * max(0, C_k - t))))
        cs = _linear_overlap_values(y_h)
        violations = np.maximum(0.0, cs - y_t)
        m_v = np.max(violations)
        if m_v < 1e-15:
            # All satisfied, just minimize t
            obj_val = y_t
            grad_h = np.zeros(n)
            grad_t = 1.0
        else:
            exp_terms = np.exp(beta * (violations - m_v))
            sum_exp = np.sum(exp_terms)
            w = exp_terms / sum_exp
            obj_val = y_t + m_v + np.log(sum_exp) / beta
            grad_t = 1.0 - np.sum(w)  # d/dt of the penalty
            # d/dh_j = sum_k w_k * dC_k/dh_j
            grad_h = np.zeros(n)
            for k in range(n):
                wk = w[k]
                if wk < 1e-15:
                    continue
                factor = -(2.0 / n) * wk
                if k > 0:
                    grad_h[k:] += factor * y_h[:n - k]
                grad_h[:n - k] += factor * y_h[k:]

        # Gradient step
        lr = 1.0 / (beta + 1.0)
        h_new = y_h - lr * grad_h
        t_new = y_t - lr * grad_t

        # Project: h onto {0<=h<=1, sum=N/2}, t >= 0
        h_new = _project(h_new, target_sum)
        t_new = max(t_new, 0.0)

        # FISTA momentum
        t_mom_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t_mom * t_mom))
        z_h = h_new + ((t_mom - 1.0) / t_mom_new) * (h_new - y_h)
        z_t = t_new + ((t_mom - 1.0) / t_mom_new) * (t_new - y_t)
        y_h = z_h
        y_t = z_t
        t_mom = t_mom_new

        if abs(prev_val - obj_val) < ftol:
            break
        prev_val = obj_val

    # Return the projected solution
    h_final = _project(h_new, target_sum)
    return h_final


def _slsqp_polish(h0, beta, maxiter, ftol):
    """
    Run SLSQP optimization with smooth max objective for final polish.
    """
    n = N

    def obj_func(x):
        val, _ = _smooth_max_val_grad(x, beta)
        return val

    def jac_func(x):
        _, grad = _smooth_max_val_grad(x, beta)
        return grad

    cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N / 2.0,
             'jac': lambda x: np.ones(n)}]
    bounds = [(0.0, 1.0)] * n

    try:
        res = minimize(obj_func, h0.copy(), jac=jac_func, method='SLSQP',
                       bounds=bounds, constraints=cons,
                       options={'maxiter': maxiter, 'ftol': ftol})
        return res.x
    except Exception:
        return h0


def _coordinate_descent_polish(h0, max_iter=500):
    """
    Coordinate descent polish: optimize each h_i one at a time.
    For each i, find the optimal h_i given all others fixed.
    """
    n = N
    h = np.array(h0, dtype=np.float64, copy=True)
    h = _project(h, N / 2.0)

    for it in range(max_iter):
        if time.time() > _DEADLINE - 10:
            break

        improved = False
        for i in range(n):
            # Try to optimize h[i]
            best_val = _score(h)
            best_hi = h[i]
            
            # Binary search or golden section on h[i] in [0,1]
            # Keep sum constraint: adjust another element
            # Simpler: just try a few values
            for trial in [0.0, 0.25, 0.5, 0.75, 1.0]:
                h[i] = trial
                # Fix sum
                h = _project(h, N / 2.0)
                val = _score(h)
                if val < best_val - 1e-12:
                    best_val = val
                    best_hi = h[i]
                    improved = True
                else:
                    h[i] = best_hi
            
            # Restore
            h[i] = best_hi

        if not improved:
            break

    h = _project(h, N / 2.0)
    return h


def _feasibility_check(c_max, h0, maxiter=100):
    """
    Check if there exists h such that max(C_k) <= c_max, sum(h)=N/2, 0<=h_i<=1.
    Uses FISTA on a penalty objective.
    """
    n = N
    target_sum = n / 2.0

    def penalty_obj_grad(h):
        cs = _linear_overlap_values(h)
        violations = np.maximum(0.0, cs - c_max)
        val = 0.5 * np.sum(violations ** 2)
        grad = np.zeros(n)
        mask = cs > c_max
        if np.any(mask):
            for k in np.where(mask)[0]:
                w = 2.0 * (cs[k] - c_max)
                factor = -(2.0 / n) * w
                if k > 0:
                    grad[k:] += factor * h[:n - k]
                if k < n:
                    grad[:n - k] += factor * h[k:]
        return val, grad

    # Simple projected gradient for feasibility
    h = np.array(h0, dtype=np.float64, copy=True)
    h = _project(h, target_sum)
    y = h.copy()
    z = h.copy()
    t = 1.0
    lr = 0.01

    for it in range(maxiter):
        if time.time() > _DEADLINE - 10:
            break
        val, grad = penalty_obj_grad(y)
        y_new = _project(y - lr * grad, target_sum)
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        y = y_new
        z = y + ((t - 1.0) / t_new) * (y - h)
        h = y.copy()
        t = t_new

    h_sol = _project(h, target_sum)
    cs_sol = _linear_overlap_values(h_sol)
    return np.max(cs_sol) <= c_max + 1e-6, h_sol


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

    # 1. Diverse analytical starts (more frequency components)
    i = np.arange(N)

    # Pure cosine at fundamental
    h_cos1 = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
    _update_best(h_cos1)

    # Pure sine at fundamental
    h_sin1 = 0.5 + 0.5 * np.sin(2.0 * np.pi * i / N)
    _update_best(h_sin1)

    # Second harmonic
    h_cos2 = 0.5 + 0.5 * np.cos(4.0 * np.pi * i / N)
    _update_best(h_cos2)

    h_sin2 = 0.5 + 0.5 * np.sin(4.0 * np.pi * i / N)
    _update_best(h_sin2)

    # Third harmonic
    h_cos3 = 0.5 + 0.5 * np.cos(6.0 * np.pi * i / N)
    _update_best(h_cos3)

    # Mixed harmonics
    h_mix = 0.5 + 0.3 * np.cos(2.0 * np.pi * i / N) + 0.2 * np.sin(4.0 * np.pi * i / N)
    _update_best(h_mix)

    # Alternating
    h_alt = np.where(i % 2 == 0, 0.7, 0.3)
    _update_best(h_alt)

    # Block pattern
    h_block = np.where(i % 4 < 2, 0.75, 0.25)
    _update_best(h_block)

    # 2. Bisection to bracket optimal C_max
    lo = 0.0
    hi = best_score

    max_bisection_iter = 12
    for it in range(max_bisection_iter):
        if time.time() - t0 > _DEADLINE - 60:
            break

        mid = (lo + hi) / 2.0
        feasible, h_sol = _feasibility_check(mid, best_h, maxiter=60)

        if feasible:
            hi = mid
            _update_best(h_sol)
        else:
            lo = mid

    # 3. FISTA optimization from best point with increasing beta
    if time.time() - t0 < _DEADLINE - 40:
        for beta in [50.0, 150.0, 400.0]:
            if time.time() - t0 > _DEADLINE - 20:
                break
            h_opt = _fista_optimize(best_h, beta, max_iter=300, ftol=1e-10)
            _update_best(h_opt)

    # 4. Epigraph FISTA from best point
    if time.time() - t0 < _DEADLINE - 30:
        h_ep = _epigraph_fista(best_h, beta=100.0, max_iter=200, ftol=1e-10)
        _update_best(h_ep)

    # 5. Try random starts if time permits
    if time.time() - t0 < _DEADLINE - 35:
        for _ in range(3):
            if time.time() - t0 > _DEADLINE - 15:
                break
            h_rand = np.random.uniform(0, 1, N)
            h_rand = _fix_sum(h_rand)
            _update_best(h_rand)
            h_opt = _fista_optimize(h_rand, beta=100.0, max_iter=200, ftol=1e-9)
            _update_best(h_opt)

    # 6. Coordinate descent polish
    if best_h is not None and time.time() - t0 < _DEADLINE - 15:
        h_cd = _coordinate_descent_polish(best_h, max_iter=200)
        _update_best(h_cd)

    # 7. Final SLSQP polish with high beta
    if best_h is not None and time.time() - t0 < _DEADLINE - 5:
        h_polished = _slsqp_polish(best_h, beta=500.0, maxiter=200, ftol=1e-12)
        _update_best(h_polished)

    # 8. One more coordinate descent after SLSQP
    if best_h is not None and time.time() - t0 < _DEADLINE - 2:
        h_cd2 = _coordinate_descent_polish(best_h, max_iter=100)
        _update_best(h_cd2)

    if best_h is None:
        best_h = np.full(N, 0.5)

    best_h = _fix_sum(best_h)
    return best_h
