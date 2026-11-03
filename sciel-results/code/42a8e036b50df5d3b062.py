import numpy as np
from scipy.optimize import minimize
import time
import warnings

warnings.filterwarnings('ignore')

N = 50
_TIME_LIMIT = 145.0
_start_time = time.time()

def _precompute_conv_kernel():
    """
    Precompute the autoconvolution kernel for step functions.
    For step function h(t) = sum_i h_i * 1_{[x_i, x_{i+1}]}(t),
    the autoconvolution (h*h)(t) = integral h(s) h(t-s) ds
    can be computed as sum_{i,j} h_i * h_j * K_{ij}(t)
    where K_{ij}(t) = measure of {s: s in [x_i,x_{i+1}] and t-s in [x_j,x_{j+1}]}
    """
    dw = 1.0 / N
    x = np.linspace(-0.25, 0.25, N + 1)
    
    # Use enough evaluation points for the autoconvolution
    n_pts = 4 * N
    t_grid = np.linspace(-0.5, 0.5, n_pts + 1)
    midpoints = (t_grid[:-1] + t_grid[1:]) / 2.0
    
    # K[i,j,k] = max(0, min(x[i+1], t_k - x[j]) - max(x[i], t_k - x[j+1]))
    # This is the overlap length of [x_i, x_{i+1}] and [t_k - x_{j+1}, t_k - x_j]
    
    i_idx = np.arange(N)[:, None, None]
    j_idx = np.arange(N)[None, :, None]
    k_idx = np.arange(n_pts)[None, None, :]
    
    x_i = x[i_idx]
    x_i1 = x[i_idx + 1]
    x_j = x[j_idx]
    x_j1 = x[j_idx + 1]
    t_k = midpoints[k_idx]
    
    lo = np.maximum(x_i, t_k - x_j1)
    hi = np.minimum(x_i1, t_k - x_j)
    
    K = np.maximum(0.0, hi - lo)  # (N, N, n_pts)
    
    return K, midpoints

_K, _t_mid = _precompute_conv_kernel()
_K_flat = _K.reshape(N * N, -1)  # (N*N, n_pts)
_n_conv_pts = _K_flat.shape[1]

def _score(h):
    """
    Score: (L2^2) / (L1 * sup|autoconv|)
    For step functions on [-1/4, 1/4] with N equal intervals.
    """
    h = np.asarray(h, dtype=float)
    h = np.maximum(h, 0.0)
    
    s1 = np.sum(h)
    if s1 < 1e-15:
        return 0.0
    
    dw = 1.0 / N
    L1 = s1 * dw
    L2_sq = np.dot(h, h) * dw
    
    # Compute autoconvolution
    h_outer = np.outer(h, h).flatten()
    autoconv = h_outer @ _K_flat  # (n_conv_pts,)
    
    sup_conv = np.max(np.abs(autoconv))
    if sup_conv < 1e-15:
        return 0.0
    
    return L2_sq / (L1 * sup_conv)

def _score_smooth(h, beta=100.0):
    """
    Smooth surrogate using log-sum-exp for the supremum.
    """
    h = np.asarray(h, dtype=float)
    h = np.maximum(h, 0.0)
    
    s1 = np.sum(h)
    if s1 < 1e-15:
        return 0.0
    
    dw = 1.0 / N
    L1 = s1 * dw
    L2_sq = np.dot(h, h) * dw
    
    h_outer = np.outer(h, h).flatten()
    autoconv = h_outer @ _K_flat
    
    abs_ac = np.abs(autoconv)
    max_val = np.max(abs_ac)
    if max_val < 1e-15:
        return 0.0
    
    # Log-sum-exp for smooth max
    shifted = beta * (abs_ac - max_val)
    log_sum = max_val + np.log(np.sum(np.exp(shifted))) / beta
    
    if log_sum < 1e-15:
        return 0.0
    
    return L2_sq / (L1 * log_sum)

def _score_neg(h, beta=100.0):
    """Negative smooth score for minimization."""
    return -_score_smooth(h, beta)

def _score_neg_exact(h):
    """Negative exact score for minimization (non-smooth)."""
    return -_score(h)

def _local_polish_exact(h0, max_iter=200):
    """
    Local polish using L-BFGS-B on the exact (non-smooth) objective.
    """
    h = np.array(h0, dtype=float)
    h = np.maximum(h, 0.0)
    best_h = h.copy()
    best_score = _score(h)
    
    try:
        result = minimize(
            _score_neg_exact,
            h,
            method='L-BFGS-B',
            bounds=[(0, 10)] * N,
            options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 20}
        )
        h_opt = np.maximum(result.x, 0.0)
        s_opt = _score(h_opt)
        if s_opt > best_score:
            best_score = s_opt
            best_h = h_opt
    except Exception:
        pass
    
    return best_h, best_score

def _refined_search(h0, max_iter=150):
    """
    Refine using coordinate descent and small perturbations.
    """
    h = np.array(h0, dtype=float)
    h = np.maximum(h, 0.0)
    best_h = h.copy()
    best_score = _score(h)
    
    rng = np.random.default_rng(12345)
    
    for _ in range(max_iter):
        if time.time() - _start_time > _TIME_LIMIT - 10:
            break
        
        # Try random small perturbations
        for _ in range(5):
            h_pert = h.copy()
            idx = rng.integers(0, N)
            delta = rng.uniform(-0.05, 0.05)
            h_pert[idx] = max(0.0, h_pert[idx] + delta)
            s_pert = _score(h_pert)
            if s_pert > best_score:
                best_score = s_pert
                best_h = h_pert.copy()
                h = h_pert.copy()
    
    return best_h, best_score

def construct():
    global _start_time
    _start_time = time.time()
    
    # Known good starting point
    h0 = np.zeros(N)
    h0[0] = 2.0 / 3.0
    h0[1] = 1.0 / 3.0
    
    best_h = h0.copy()
    best_score = _score(best_h)
    
    # Phase 1: L-BFGS-B with smooth surrogate, annealing beta
    # Start with low beta (smooth) and increase
    starts = [
        h0.copy(),
        np.array([1.0/3.0, 1.0/3.0, 1.0/3.0] + [0.0]*(N-3)),
        np.array([0.5, 0.5] + [0.0]*(N-2)),
        np.array([0.6, 0.3, 0.1] + [0.0]*(N-3)),
        np.array([0.7, 0.2, 0.1] + [0.0]*(N-3)),
        np.ones(N) / N,
        np.array([1.0/N]*N),
        np.array([0.4, 0.3, 0.2, 0.1] + [0.0]*(N-4)),
    ]
    
    # Anneal beta: start smooth, get sharper
    beta_schedule = [10.0, 30.0, 100.0, 300.0, 1000.0]
    
    for beta in beta_schedule:
        if time.time() - _start_time > _TIME_LIMIT - 50:
            break
        
        for s in starts:
            if time.time() - _start_time > _TIME_LIMIT - 45:
                break
            
            try:
                result = minimize(
                    _score_neg,
                    s,
                    method='L-BFGS-B',
                    bounds=[(0, 10)] * N,
                    args=(beta,),
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 30}
                )
                h_trial = np.maximum(result.x, 0.0)
                score_trial = _score(h_trial)
                if score_trial > best_score:
                    best_score = score_trial
                    best_h = h_trial.copy()
            except Exception:
                pass
    
    # Phase 2: Continue from best with higher beta
    for beta in [500.0, 2000.0, 10000.0]:
        if time.time() - _start_time > _TIME_LIMIT - 30:
            break
        
        try:
            result = minimize(
                _score_neg,
                best_h,
                method='L-BFGS-B',
                bounds=[(0, 10)] * N,
                args=(beta,),
                options={'maxiter': 300, 'ftol': 1e-13, 'gtol': 1e-10, 'maxls': 30}
            )
            h_trial = np.maximum(result.x, 0.0)
            score_trial = _score(h_trial)
            if score_trial > best_score:
                best_score = score_trial
                best_h = h_trial.copy()
        except Exception:
            pass
    
    # Phase 3: Exact local polish
    if time.time() - _start_time < _TIME_LIMIT - 20:
        h_polished, s_polished = _local_polish_exact(best_h, max_iter=100)
        if s_polished > best_score:
            best_score = s_polished
            best_h = h_polished
    
    # Phase 4: Random perturbation refinement
    if time.time() - _start_time < _TIME_LIMIT - 10:
        h_refined, s_refined = _refined_search(best_h, max_iter=50)
        if s_refined > best_score:
            best_score = s_refined
            best_h = h_refined
    
    # Final exact polish
    if time.time() - _start_time < _TIME_LIMIT - 5:
        h_final, s_final = _local_polish_exact(best_h, max_iter=50)
        if s_final > best_score:
            best_score = s_final
            best_h = h_final
    
    return [float(x) for x in best_h]
