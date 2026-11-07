import numpy as np
from scipy.optimize import minimize, differential_evolution, minimize_scalar
import time

N = 100
TARGET_SUM = N / 2.0
DEADLINE = 138.0  # safety margin before 150s kill

def _corr_vector(h):
    """Compute all lags of correlation."""
    return np.correlate(h, 1.0 - h, mode='full')

def _correlation_max(h):
    """Compute the raw value: max correlation / N * 2."""
    corr = _corr_vector(h)
    return np.max(corr) / N * 2.0

def _project_feasible(h):
    """Project h to satisfy 0 <= h <= 1 and sum(h) = N/2."""
    h = np.asarray(h, dtype=float).copy()
    h = np.clip(h, 0.0, 1.0)
    for _ in range(200):
        diff = TARGET_SUM - np.sum(h)
        if abs(diff) < 1e-14:
            break
        mask = (h > 1e-12) & (h < 1.0 - 1e-12)
        n_free = np.sum(mask)
        if n_free == 0:
            break
        h[mask] += diff / n_free
        h = np.clip(h, 0.0, 1.0)
    diff = TARGET_SUM - np.sum(h)
    if abs(diff) > 1e-15:
        idx = np.argmin(np.abs(h - 0.5))
        h[idx] += diff
        h = np.clip(h, 0.0, 1.0)
    return h

def _logsumexp(x, beta=1.0):
    """Compute log(sum(exp(x/beta))) * beta, which approximates max(x) as beta->0."""
    x_max = np.max(x)
    if x_max == -np.inf:
        return x_max
    shifted = x - x_max
    return x_max + beta * np.log(np.sum(np.exp(shifted / beta)))

def _smooth_objective(h, beta=0.05):
    """Smooth approximation of the max correlation using log-sum-exp."""
    corr = _corr_vector(h)
    ls = _logsumexp(corr, beta)
    return ls / N * 2.0

def _smooth_objective_grad(h, beta=0.05):
    """Gradient of the smooth objective using finite differences."""
    corr = _corr_vector(h)
    x_max = np.max(corr)
    shifted = corr - x_max
    weights = np.exp(shifted / beta) / np.sum(np.exp(shifted / beta))
    
    dh = np.zeros(N)
    eps = 1e-7
    for i in range(N):
        h_plus = h.copy()
        h_plus[i] += eps
        h_minus = h.copy()
        h_minus[i] -= eps
        corr_plus = _corr_vector(h_plus)
        corr_minus = _corr_vector(h_minus)
        dc = (corr_plus - corr_minus) / (2 * eps)
        dh[i] = np.sum(weights * dc)
    
    return 2.0 * dh / N

def _block_based_h(block_heights, block_lengths):
    """
    Generate h from block structure.
    block_heights: array of K heights
    block_lengths: array of K integer lengths (sum to N)
    """
    h = np.empty(N)
    pos = 0
    for bh, bl in zip(block_heights, block_lengths):
        h[pos:pos+bl] = bh
        pos += bl
    return _project_feasible(h)

def _block_objective(params):
    """
    Objective for block-based search.
    params: [h1, h2, ..., hK, l1, l2, ..., lK-1]
    where lK = N - sum(l1...lK-1)
    """
    K = len(params) // 2
    if K < 2:
        return 1.0
    heights = params[:K]
    lengths = params[K:]
    # Convert lengths to integers
    lengths_int = np.round(lengths).astype(int)
    lengths_int = np.maximum(lengths_int, 1)
    # Ensure sum is N
    current_sum = np.sum(lengths_int)
    if current_sum != N:
        # Adjust last block
        lengths_int[-1] += N - current_sum
        if lengths_int[-1] < 1:
            # Redistribute
            lengths_int = np.ones(K, dtype=int)
            lengths_int[-1] = N - (K - 1)
    
    h = _block_based_h(heights, lengths_int)
    return _correlation_max(h)

def _coordinate_polish(h, max_iters=50, tol=1e-12):
    """
    Coordinate descent polish.
    """
    h = _project_feasible(h).copy()
    current_val = _correlation_max(h)
    
    for it in range(max_iters):
        improved = False
        for i in range(N):
            # Find j to compensate
            j_candidates = [j for j in range(N) if j != i]
            j = min(j_candidates, key=lambda jj: abs(h[jj] - 0.5))
            
            sum_others = np.sum(h) - h[i]
            x_min = max(0.0, sum_others - TARGET_SUM)
            x_max = min(1.0, 1.0 - (TARGET_SUM - sum_others))
            
            if x_min >= x_max:
                continue
            
            def obj_1d(x_i):
                h_j = TARGET_SUM - sum_others + x_i
                if h_j < 0.0 or h_j > 1.0:
                    return 1e10
                h_temp = h.copy()
                h_temp[i] = x_i
                h_temp[j] = h_j
                return _correlation_max(h_temp)
            
            try:
                res = minimize_scalar(obj_1d, bounds=(x_min, x_max), method='bounded',
                                     options={'xatol': tol})
                if res.success:
                    h_j = TARGET_SUM - sum_others + res.x
                    if 0.0 <= h_j <= 1.0:
                        h_temp = h.copy()
                        h_temp[i] = res.x
                        h_temp[j] = h_j
                        val_temp = _correlation_max(h_temp)
                        if val_temp < current_val - tol:
                            h = h_temp
                            current_val = val_temp
                            improved = True
            except Exception:
                pass
        
        if not improved:
            break
    
    return _project_feasible(h)

def construct():
    start_time = time.time()
    
    best_h = np.full(N, 0.5)
    best_val = _correlation_max(best_h)
    
    # Phase 1: Differential Evolution for global search
    if time.time() - start_time < DEADLINE - 80:
        bounds = [(0.0, 1.0) for _ in range(N)]
        
        def de_objective(x):
            h = _project_feasible(x)
            return _smooth_objective(h, beta=0.1)
        
        try:
            res = differential_evolution(
                de_objective,
                bounds,
                seed=42,
                maxiter=80,
                tol=1e-8,
                popsize=12,
                init='latinhypercube',
                update_interval=10,
                polish=True,
                workers=-1
            )
            h_res = _project_feasible(res.x)
            val = _correlation_max(h_res)
            if val < best_val - 1e-10:
                best_val = val
                best_h = h_res.copy()
        except Exception:
            pass
    
    # Phase 2: Block-based structural search
    if time.time() - start_time < DEADLINE - 50:
        # Try different numbers of blocks
        for K in [4, 6, 8, 10, 12]:
            if time.time() - start_time > DEADLINE - 40:
                break
            
            # Initial guess: alternating high/low blocks
            block_heights = np.zeros(K)
            block_lengths = np.zeros(K)
            for k in range(K):
                block_heights[k] = 0.8 if k % 2 == 0 else 0.2
                block_lengths[k] = N // K
            block_lengths[-1] += N - np.sum(block_lengths)
            
            h_init = _block_based_h(block_heights, block_lengths)
            val_init = _correlation_max(h_init)
            if val_init < best_val - 1e-10:
                best_val = val_init
                best_h = h_init.copy()
            
            # Optimize block heights and lengths using SLSQP
            # Parameters: [h1,...,hK, l1,...,lK-1]
            n_params = 2 * K - 1
            bounds_block = [(0.0, 1.0)] * K + [(1.0, N - 1.0)] * (K - 1)
            
            def block_obj(params):
                heights = params[:K]
                lengths = params[K:]
                lengths_int = np.round(lengths).astype(int)
                lengths_int = np.maximum(lengths_int, 1)
                current_sum = np.sum(lengths_int)
                if current_sum != N:
                    lengths_int[-1] += N - current_sum
                    if lengths_int[-1] < 1:
                        lengths_int = np.ones(K, dtype=int)
                        lengths_int[-1] = N - (K - 1)
                h = _block_based_h(heights, lengths_int)
                return _correlation_max(h)
            
            x0 = np.concatenate([block_heights, block_lengths[:-1]])
            
            try:
                res = minimize(
                    block_obj,
                    x0,
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12}
                )
                h_res = _block_based_h(res.x[:K], np.round(res.x[K:]).astype(int))
                # Fix lengths
                lengths_int = np.round(res.x[K:]).astype(int)
                lengths_int = np.maximum(lengths_int, 1)
                current_sum = np.sum(lengths_int)
                if current_sum != N:
                    lengths_int[-1] += N - current_sum
                h_res = _block_based_h(res.x[:K], lengths_int)
                val = _correlation_max(h_res)
                if val < best_val - 1e-10:
                    best_val = val
                    best_h = h_res.copy()
            except Exception:
                pass
    
    # Phase 3: SLSQP with smooth objective from best solution
    if time.time() - start_time < DEADLINE - 30:
        bounds = [(0.0, 1.0) for _ in range(N)]
        constraint = {'type': 'eq', 'fun': lambda x: np.sum(x) - TARGET_SUM, 'jac': lambda x: np.ones(N)}
        
        try:
            res = minimize(
                lambda x: _smooth_objective(x, 0.05),
                best_h,
                method='SLSQP',
                bounds=bounds,
                constraints=[constraint],
                jac=lambda x: _smooth_objective_grad(x, 0.05),
                options={'maxiter': 400, 'ftol': 1e-12, 'disp': False}
            )
            h_res = _project_feasible(res.x)
            val = _correlation_max(h_res)
            if val < best_val - 1e-10:
                best_val = val
                best_h = h_res.copy()
        except Exception:
            pass
        
        # Try from alternating patterns
        for base in [0.15, 0.2, 0.25, 0.3, 0.35]:
            if time.time() - start_time > DEADLINE - 20:
                break
            alt = np.where(np.arange(N) % 2 == 0, base, 1.0 - base)
            h_alt = _project_feasible(alt)
            try:
                res = minimize(
                    lambda x: _smooth_objective(x, 0.05),
                    h_alt,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=[constraint],
                    jac=lambda x: _smooth_objective_grad(x, 0.05),
                    options={'maxiter': 250, 'ftol': 1e-12, 'disp': False}
                )
                h_res = _project_feasible(res.x)
                val = _correlation_max(h_res)
                if val < best_val - 1e-10:
                    best_val = val
                    best_h = h_res.copy()
            except Exception:
                pass
        
        # Try from sinusoidal patterns
        for freq in [1, 2, 3, 5, 7]:
            if time.time() - start_time > DEADLINE - 15:
                break
            t = np.arange(N)
            for amp in [0.25, 0.35, 0.45]:
                for phase in [0, np.pi/3, 2*np.pi/3]:
                    h_sin = 0.5 + amp * np.sin(2 * np.pi * freq * t / N + phase)
                    h_sin = _project_feasible(h_sin)
                    try:
                        res = minimize(
                            lambda x: _smooth_objective(x, 0.05),
                            h_sin,
                            method='SLSQP',
                            bounds=bounds,
                            constraints=[constraint],
                            jac=lambda x: _smooth_objective_grad(x, 0.05),
                            options={'maxiter': 150, 'ftol': 1e-12, 'disp': False}
                        )
                        h_res = _project_feasible(res.x)
                        val = _correlation_max(h_res)
                        if val < best_val - 1e-10:
                            best_val = val
                            best_h = h_res.copy()
                    except Exception:
                        pass
    
    # Phase 4: Coordinate descent polish
    if time.time() - start_time < DEADLINE - 10:
        h_cd = _coordinate_polish(best_h, max_iters=40, tol=1e-12)
        val_cd = _correlation_max(h_cd)
        if val_cd < best_val - 1e-12:
            best_val = val_cd
            best_h = h_cd.copy()
    
    # Phase 5: Final SLSQP with tighter tolerance
    if time.time() - start_time < DEADLINE - 5:
        bounds = [(0.0, 1.0) for _ in range(N)]
        constraint = {'type': 'eq', 'fun': lambda x: np.sum(x) - TARGET_SUM, 'jac': lambda x: np.ones(N)}
        
        try:
            res = minimize(
                lambda x: _smooth_objective(x, 0.02),
                best_h,
                method='SLSQP',
                bounds=bounds,
                constraints=[constraint],
                options={'maxiter': 300, 'ftol': 1e-14, 'disp': False}
            )
            h_polish = _project_feasible(res.x)
            val_polish = _correlation_max(h_polish)
            if val_polish < best_val - 1e-12:
                best_val = val_polish
                best_h = h_polish.copy()
        except Exception:
            pass
    
    # Final coordinate polish
    if time.time() - start_time < DEADLINE - 2:
        h_final = _coordinate_polish(best_h, max_iters=20, tol=1e-14)
        val_final = _correlation_max(h_final)
        if val_final < best_val - 1e-14:
            best_val = val_final
            best_h = h_final.copy()
    
    # Ensure final validity
    best_h = _project_feasible(best_h)
    
    return best_h
