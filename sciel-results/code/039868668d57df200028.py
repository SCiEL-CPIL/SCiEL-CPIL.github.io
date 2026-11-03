"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Uses spectral parameterization, log-sum-exp smoothing, and multi-stage refinement."""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
_T_START = time.time()
_DEADLINE = _T_START + 138  # Safety margin before 150s kill


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
    """Compute the maximum linear overlap."""
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    return float(np.max(cs))


def _project(h, target_sum):
    """Project h onto {x : 0 <= x_i <= 1, sum(x) = target_sum} via bisection."""
    h = np.asarray(h, dtype=np.float64).copy()
    current_sum = np.sum(h)
    if abs(current_sum - target_sum) < 1e-12 and np.all(h >= -1e-12) and np.all(h <= 1 + 1e-12):
        return np.clip(h, 0.0, 1.0)

    low = -np.max(h) - 1.0
    high = 1.0 - np.min(h) + 1.0

    def sum_of_clipped(offset):
        return np.sum(np.clip(h + offset, 0.0, 1.0))

    for _ in range(60):
        mid = (low + high) / 2.0
        s_mid = sum_of_clipped(mid)
        if s_mid > target_sum:
            high = mid
        else:
            low = mid

    offset = (low + high) / 2.0
    h_proj = np.clip(h + offset, 0.0, 1.0)

    diff = target_sum - np.sum(h_proj)
    if abs(diff) > 1e-14:
        if diff > 0:
            idx = np.argmin(h_proj)
            h_proj[idx] += diff
        else:
            idx = np.argmax(h_proj)
            h_proj[idx] += diff
        h_proj = np.clip(h_proj, 0.0, 1.0)

    return h_proj


def _smooth_max_obj_grad(h, beta):
    """Value and gradient of smooth max of C_k using FFT."""
    cs = _linear_overlap_values(h)
    cs_max = np.max(cs)
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


def _nesterov_smooth_max(h0, beta, max_iter=500, lr_init=0.05):
    """Nesterov accelerated projected gradient on smooth max."""
    h = np.array(h0, dtype=np.float64, copy=True)
    h = _project(h, N / 2.0)

    h_prev = h.copy()
    t_mom = 1.0
    lr_current = lr_init
    best_h = h.copy()
    best_score = _score(h)
    max_backtracks = 12

    for it in range(max_iter):
        if time.time() > _DEADLINE - 8:
            break

        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t_mom * t_mom)) / 2.0
        momentum = (t_mom - 1.0) / t_new
        y = h + momentum * (h - h_prev)

        val_y, grad = _smooth_max_obj_grad(y, beta)

        step_lr = lr_current
        h_new = None
        for bt in range(max_backtracks):
            h_candidate = y - step_lr * grad
            h_candidate = _project(h_candidate, N / 2.0)
            val_candidate = _score(h_candidate)
            if val_candidate <= val_y + 1e-8:
                h_new = h_candidate
                lr_current = step_lr
                break
            step_lr *= 0.5

        if h_new is None:
            h_new = y - lr_current * 0.5 * grad
            h_new = _project(h_new, N / 2.0)
            lr_current *= 0.5

        h_prev = h
        h = h_new
        t_mom = t_new

        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()

    return best_h, best_score


def _spectral_h_from_coeffs(coeffs):
    """Reconstruct h from real DFT coefficients. h_i = 0.5 + sum of modes."""
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
    """Extract first K real DFT coefficients from h."""
    H = np.fft.rfft(h)
    out = []
    for k in range(1, K + 1):
        a = 2.0 / N * H[k].real
        b = -2.0 / N * H[k].imag
        out.append(a)
        out.append(b)
    return out


def _optimize_spectral(h_init, K, beta, maxiter=300, lr_init=0.01):
    """Optimize in spectral coefficient space using L-BFGS-B."""
    coeffs0 = _h_to_spectral_coeffs(h_init, K)

    def obj(coeffs):
        h = _spectral_h_from_coeffs(coeffs)
        penalty = 0.0
        if np.any(h < 0) or np.any(h > 1):
            penalty = np.sum(np.maximum(0, -h) ** 2) + np.sum(np.maximum(0, h - 1) ** 2)
        val, _ = _smooth_max_obj_grad(h, beta)
        return val + 1.0 * penalty

    def grad(coeffs):
        h = _spectral_h_from_coeffs(coeffs)
        _, grad_h = _smooth_max_obj_grad(h, beta)

        penalty_grad = np.zeros(N)
        penalty_grad[h < 0] += -2.0 * h[h < 0]
        penalty_grad[h > 1] += 2.0 * (h[h > 1] - 1.0)

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

    # Fallback: Nesterov in spectral space
    coeffs = np.array(coeffs0, dtype=np.float64, copy=True)
    coeffs_prev = coeffs.copy()
    t_k = 1.0
    lr = lr_init
    best_coeffs = coeffs.copy()
    best_val = obj(coeffs)

    for it in range(max(100, maxiter)):
        if time.time() > _DEADLINE - 8:
            break
        val = obj(coeffs)
        g = grad(coeffs)

        coeffs_new = coeffs - lr * g
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t_k * t_k)) / 2.0
        coeffs_next = coeffs_new + ((t_k - 1.0) / t_new) * (coeffs_new - coeffs_prev)

        coeffs_prev = coeffs
        coeffs = coeffs_next
        t_k = t_new

        if it % 50 == 49:
            lr = max(0.001, lr * 0.95)

        v = obj(coeffs)
        if v < best_val:
            best_val = v
            best_coeffs = coeffs.copy()

    h_opt = _spectral_h_from_coeffs(best_coeffs)
    return h_opt


def _spectral_perturb_and_polish(h0, K, beta, n_perturb=5, perturb_scale=0.1, maxiter=100):
    """
    Perturb the spectral coefficients of h0, re-construct h, and polish.
    """
    coeffs0 = _h_to_spectral_coeffs(h0, K)
    best_h = h0.copy()
    best_score = _score(h0)
    
    for _ in range(n_perturb):
        if time.time() > _DEADLINE - 10:
            break
        # Random perturbation in spectral space
        noise = np.random.normal(0, perturb_scale, len(coeffs0))
        coeffs_pert = coeffs0 + noise
        
        # Reconstruct h
        h_pert = _spectral_h_from_coeffs(coeffs_pert)
        
        # Project to valid box [0,1] and sum constraint
        h_pert = _project(h_pert, N / 2.0)
        
        # Polish from this perturbed start using spectral optimization
        h_polished = _optimize_spectral(h_pert, K, beta, maxiter=maxiter)
        
        s = _score(h_polished)
        if s < best_score:
            best_score = s
            best_h = h_polished.copy()
            
    return best_h

def _run_slsqp(h0, beta, maxiter=300, ftol=1e-12, gtol=1e-10):
    """Run SLSQP optimization on the smooth max objective."""
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
        return res.x
    except Exception:
        return h0


def _high_beta_polish(h0, n_iter=300, beta=500.0, lr=0.002):
    """Final high-precision polish with very high beta and small learning rate."""
    h = np.array(h0, dtype=np.float64, copy=True)
    h = _project(h, N / 2.0)

    best_h = h.copy()
    best_score = _score(h)

    for i in range(n_iter):
        if time.time() > _DEADLINE - 3:
            break

        val, grad = _smooth_max_obj_grad(h, beta)
        h_new = h - lr * grad
        h_new = _project(h_new, N / 2.0)

        s = _score(h_new)
        if s < best_score:
            best_score = s
            best_h = h_new.copy()
        h = h_new

    return best_h, best_score


def _binary_alternating_search():
    """Search over binary alternating sequences h[2i]=1-h[2i+1]."""
    n_half = N // 2
    np.random.seed(999)
    bits = np.random.randint(0, 2, n_half)
    h = np.zeros(N)
    h[0::2] = bits
    h[1::2] = 1 - bits

    best_score = _score(h)
    best_h = h.copy()

    for iteration in range(800):
        if time.time() > _DEADLINE - 15:
            break
        improved = False
        for idx in range(n_half):
            old_bit = bits[idx]
            bits[idx] = 1 - old_bit
            h[0::2] = bits
            h[1::2] = 1 - bits
            s = _score(h)
            if s < best_score:
                best_score = s
                best_h = h.copy()
                improved = True
            else:
                bits[idx] = old_bit
        if not improved:
            break

    return best_h, best_score


def _coordinate_descent_polish(h0, max_iter=2000, beta=200.0):
    """Coordinate descent: optimize one coordinate at a time with exact line search."""
    h = np.array(h0, dtype=np.float64, copy=True)
    h = _project(h, N / 2.0)
    best_h = h.copy()
    best_score = _score(h)

    for it in range(max_iter):
        if time.time() > _DEADLINE - 5:
            break

        improved = False
        for idx in range(N):
            # Try a few candidate values for h[idx]
            current = h[idx]
            candidates = [max(0.0, min(1.0, current + delta))
                         for delta in [-0.05, -0.02, -0.01, 0.01, 0.02, 0.05, -0.1, 0.1]]
            candidates.append(current)

            best_cand = current
            best_s = best_score

            for cand in candidates:
                h[idx] = cand
                # Adjust sum: redistribute the difference
                diff = N / 2.0 - np.sum(h)
                if abs(diff) > 1e-10:
                    # Redistribute to other coordinates proportionally
                    others = np.arange(N)
                    others = others[others != idx]
                    h[others] += diff / (N - 1)
                    h = np.clip(h, 0.0, 1.0)
                    # Re-project for exact sum
                    h = _project(h, N / 2.0)

                s = _score(h)
                if s < best_s:
                    best_s = s
                    best_cand = cand

            h[idx] = best_cand
            h = _project(h, N / 2.0)
            s = _score(h)
            if s < best_score:
                best_score = s
                best_h = h.copy()
                improved = True

        if not improved:
            break

    return best_h, best_score


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

    # --- Phase 0: Binary alternating sequence search ---
    if time.time() < _DEADLINE - 20:
        h_binary, s_binary = _binary_alternating_search()
        _update_best(h_binary)

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

    # --- Phase 2: Spectral optimization from best start ---
    for K in [2, 4, 8, 16, 32, 49]:
        if time.time() - t0 > _DEADLINE - 35:
            break
        h_spec = _optimize_spectral(best_h, K, beta=50.0, maxiter=200)
        _update_best(h_spec)

    # --- Phase 2.5: Spectral Perturbation and Polish ---
    if best_h is not None and time.time() < _DEADLINE - 40:
        K_pert = 32
        h_pert_best = _spectral_perturb_and_polish(best_h, K_pert, beta=50.0, n_perturb=5, perturb_scale=0.1, maxiter=100)
        _update_best(h_pert_best)

    # --- Phase 3: Nesterov smooth max refinement ---
    if best_h is not None and time.time() < _DEADLINE - 25:
        h_cand, s_cand = _nesterov_smooth_max(best_h, beta=50.0, max_iter=400, lr_init=0.05)
        _update_best(h_cand)

    # --- Phase 4: Random restarts with spectral polish ---
    if time.time() < _DEADLINE - 30:
        for _ in range(8):
            if time.time() - t0 > _DEADLINE - 20:
                break
            h_rand = np.random.uniform(0, 1, N)
            h_rand = _project(h_rand, N / 2.0)
            h_spec_rand = _optimize_spectral(h_rand, K=8, beta=50.0, maxiter=100)
            _update_best(h_spec_rand)

    # --- Phase 5: SLSQP polish ---
    if best_h is not None and time.time() < _DEADLINE - 12:
        h_opt = _run_slsqp(best_h, beta=100.0, maxiter=300, ftol=1e-12, gtol=1e-10)
        _update_best(h_opt)

    # --- Phase 6: High-beta polish ---
    if best_h is not None and time.time() < _DEADLINE - 5:
        h_final, _ = _high_beta_polish(best_h, n_iter=200, beta=300.0, lr=0.003)
        _update_best(h_final)

    # --- Phase 7: Coordinate descent polish ---
    if best_h is not None and time.time() < _DEADLINE - 4:
        h_cd, _ = _coordinate_descent_polish(best_h, max_iter=500, beta=200.0)
        _update_best(h_cd)

    # --- Phase 8: Final SLSQP with very high beta ---
    if best_h is not None and time.time() < _DEADLINE - 2:
        h_final2 = _run_slsqp(best_h, beta=500.0, maxiter=400, ftol=1e-14, gtol=1e-12)
        _update_best(h_final2)

    if best_h is None:
        best_h = np.full(N, 0.5)

    best_h = _project(best_h, N / 2.0)
    return best_h
