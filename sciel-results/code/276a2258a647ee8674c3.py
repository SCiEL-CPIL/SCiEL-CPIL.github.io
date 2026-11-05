"""Bermudan exercise-policy research with smooth surrogate coordinate ascent optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time
from scipy.stats import norm

_DEADLINE = 115.0  # seconds; safety margin under 120s wall-clock limit (150s kill)

def _bs_put_price(S, K, r, sigma, T):
    if T <= 0:
        return np.maximum(K - S, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    N = norm.cdf
    price = K * np.exp(-r * T) * N(-d2) - S * N(-d1)
    return price

def _find_bs_threshold(K, r, sigma, T_remaining, dt, S_grid):
    T_next = T_remaining - dt
    if T_next < 0: T_next = 0
    intrinsic = K - S_grid
    cont_val = np.exp(-r * dt) * _bs_put_price(S_grid, K, r, sigma, T_next)
    diff = intrinsic - cont_val
    sign_change = np.where(np.diff(np.sign(diff)) != 0)[0]
    if len(sign_change) == 0:
        if np.all(diff > 0):
            return np.max(S_grid) 
        else:
            return np.min(S_grid)
    idx = sign_change[0]
    S1, S2 = S_grid[idx], S_grid[idx+1]
    d1, d2 = diff[idx], diff[idx+1]
    if d2 == d1:
        return S1
    S_star = S1 - d1 * (S2 - S1) / (d2 - d1)
    return S_star

class ExercisePolicy:
    def __init__(self, thresholds, r, dt):
        self.thresholds = thresholds
        self.r = r
        self.dt = dt
        self.n_dates = len(thresholds)

    def exercise(self, current_state, time):
        if isinstance(time, (int, np.integer)):
            idx = int(time)
        else:
            idx = int(np.round(time * 12)) - 1
            if idx < 0:
                idx = 0
            if idx >= self.n_dates:
                idx = self.n_dates - 1
        if idx < 0 or idx >= self.n_dates:
            return np.zeros(len(current_state), dtype=bool)
        return current_state < self.thresholds[idx]

def _evaluate_policy_fast(S, K, r, T, n_dates, thresholds, Intrinsic=None, disc=None):
    dt = T / n_dates
    n_paths = S.shape[1]
    if Intrinsic is None:
        Intrinsic = np.maximum(K - S, 0.0)
    if disc is None:
        disc = np.exp(-r * dt)
    V = np.zeros(n_paths)
    for t in range(n_dates - 1, -1, -1):
        ex_mask = S[t] < thresholds[t]
        V = Intrinsic[t] * ex_mask + disc * V * (~ex_mask)
    return np.mean(V)

def _precompute_prefix_affine(S, K, r, T, n_dates, thresholds, t_end, Intrinsic=None, disc=None):
    dt = T / n_dates
    n_paths = S.shape[1]
    if Intrinsic is None:
        Intrinsic = np.maximum(K - S, 0.0)
    if disc is None:
        disc = np.exp(-r * dt)
    
    C = np.ones(n_paths)
    D = np.zeros(n_paths)
    
    for k in range(t_end - 1, -1, -1):
        ex_mask = S[k] < thresholds[k]
        A_k = (~ex_mask) * disc
        B_k = ex_mask * Intrinsic[k]
        
        C = A_k * C
        D = A_k * D + B_k
        
    return C, D

def _smooth_threshold_value(theta, S_t, C, D, Intrinsic_t, V_next, disc, alpha=100.0):
    """Smooth surrogate for threshold value using sigmoid soft-min."""
    # Soft indicator: 1 if S_t < theta, 0 otherwise
    # Use sigmoid: sigma(alpha * (theta - S_t))
    z = alpha * (theta - S_t)
    # Clip for numerical stability
    z = np.clip(z, -50, 50)
    soft_ex = 1.0 / (1.0 + np.exp(-z))
    
    V_t = Intrinsic_t * soft_ex + disc * V_next * (1.0 - soft_ex)
    V_0 = C * V_t + D
    return -np.mean(V_0)

def fit_exercise_policy(training_paths, contract):
    t_start = _time.time()

    S0 = contract.get('S0', 100.0)
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_dates = contract.get('n_dates', 12)
    
    S = training_paths
    if S.ndim != 2:
        if S.shape[1] == n_dates:
            S = S.T
        else:
            S = S.reshape(n_dates, -1)
            
    n_paths = S.shape[1]
    dt = T / n_dates
    disc = np.exp(-r * dt)
    Intrinsic = np.maximum(K - S, 0.0)
    
    # 1. Initialize thresholds using BS heuristic
    thresholds = np.zeros(n_dates)
    S_min = np.min(S)
    S_max = np.max(S)
    S_grid = np.linspace(S_min * 0.5, S_max * 1.5, 200)
    
    for t in range(n_dates):
        T_remaining = (n_dates - t) * dt
        thresholds[t] = _find_bs_threshold(K, r, sigma, T_remaining, dt, S_grid)
        
    thresholds = np.clip(thresholds, 0.0, K + 10.0)
    
    best_score = _evaluate_policy_fast(S, K, r, T, n_dates, thresholds, Intrinsic, disc)
    
    # Phase 1: Smooth surrogate coordinate ascent
    # Start with coarse alpha, refine over iterations
    alphas = [50.0, 100.0, 200.0, 500.0, 1000.0]
    
    for alpha in alphas:
        if _time.time() - t_start > _DEADLINE - 5.0:
            break
        
        # Multiple sweeps with decreasing step
        for sweep in range(3):
            if _time.time() - t_start > _DEADLINE - 5.0:
                break
            
            improved = False
            for t in range(n_dates):
                if _time.time() - t_start > _DEADLINE - 5.0:
                    break
                
                # Precompute V_{t+1}
                V_next = np.zeros(n_paths)
                for k in range(n_dates - 1, t, -1):
                    ex_mask = S[k] < thresholds[k]
                    V_next = Intrinsic[k] * ex_mask + disc * V_next * (~ex_mask)
                
                # Precompute C, D
                C, D = _precompute_prefix_affine(S, K, r, T, n_dates, thresholds, t, Intrinsic, disc)
                
                # Use golden section search on smooth objective
                # Search range around current threshold
                current = thresholds[t]
                S_t = S[t]
                search_lo = max(0.0, current - 2.0 * sigma * np.sqrt(dt) * 2)
                search_hi = min(2.0 * K, current + 2.0 * sigma * np.sqrt(dt) * 2)
                
                # If current is at boundary, expand search
                if current <= search_lo + 1e-6:
                    search_lo = 0.0
                    search_hi = max(search_hi, current + 5.0)
                if current >= search_hi - 1e-6:
                    search_hi = 2.0 * K
                    search_lo = min(search_lo, current - 5.0)
                
                # Golden section search
                gr = (np.sqrt(5) - 1) / 2
                a, b = search_lo, search_hi
                
                def obj(theta):
                    return _smooth_threshold_value(theta, S_t, C, D, Intrinsic[t], V_next, disc, alpha)
                
                c = b - gr * (b - a)
                d = a + gr * (b - a)
                fc = obj(c)
                fd = obj(d)
                
                for _ in range(30):
                    if _time.time() - t_start > _DEADLINE - 5.0:
                        break
                    if fc < fd:
                        b = d
                        d = c
                        fd = fc
                        c = b - gr * (b - a)
                        fc = obj(c)
                    else:
                        a = c
                        c = d
                        fc = fd
                        d = a + gr * (b - a)
                        fd = obj(d)
                
                optimal_theta = (a + b) / 2.0
                optimal_theta = np.clip(optimal_theta, 0.0, 2.0 * K)
                
                if abs(optimal_theta - thresholds[t]) > 1e-4:
                    old_thresh = thresholds[t]
                    thresholds[t] = optimal_theta
                    new_score = _evaluate_policy_fast(S, K, r, T, n_dates, thresholds, Intrinsic, disc)
                    if new_score > best_score:
                        best_score = new_score
                        improved = True
                    else:
                        thresholds[t] = old_thresh
            
            if not improved:
                break

    # Phase 2: Fine-grained discrete search for final polish
    if _time.time() - t_start < _DEADLINE - 2.0:
        for t in range(n_dates):
            if _time.time() - t_start > _DEADLINE - 2.0:
                break
            
            V_next = np.zeros(n_paths)
            for k in range(n_dates - 1, t, -1):
                ex_mask = S[k] < thresholds[k]
                V_next = Intrinsic[k] * ex_mask + disc * V_next * (~ex_mask)
            
            C, D = _precompute_prefix_affine(S, K, r, T, n_dates, thresholds, t, Intrinsic, disc)
            
            # Fine grid around current threshold
            current = thresholds[t]
            step = 0.01
            candidates = np.arange(max(0, current - 0.5), min(2*K, current + 0.5) + step, step)
            
            best_local = -np.inf
            best_local_thresh = current
            
            for c in candidates:
                ex_mask = S[t] < c
                V_t = Intrinsic[t] * ex_mask + disc * V_next * (~ex_mask)
                V_0 = C * V_t + D
                score = np.mean(V_0)
                
                if score > best_local:
                    best_local = score
                    best_local_thresh = c
            
            if best_local_thresh != thresholds[t]:
                old_thresh = thresholds[t]
                thresholds[t] = best_local_thresh
                new_score = _evaluate_policy_fast(S, K, r, T, n_dates, thresholds, Intrinsic, disc)
                if new_score > best_score:
                    best_score = new_score
                else:
                    thresholds[t] = old_thresh

    return ExercisePolicy(thresholds, r, dt)
# EVOLVE-BLOCK-END
