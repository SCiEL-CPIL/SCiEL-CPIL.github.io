import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 50
DELTA = 1.0 / N

def _build_overlap_matrices():
    num_lags = 2 * N - 1
    M = np.zeros((num_lags, N, N))
    for k in range(num_lags):
        for i in range(N):
            for j in range(N):
                lo = max(i, k - j)
                hi = min(i + 1, k - j + 1)
                M[k, i, j] = max(0.0, hi - lo)
    return M

_M_matrices = _build_overlap_matrices()
_NUM_LAGS = 2 * N - 1

def _autoconv_sup(h):
    h = np.maximum(h, 0)
    ff_vals = np.array([DELTA * (h @ _M_matrices[k] @ h) for k in range(_NUM_LAGS)])
    return np.max(ff_vals)

def _ratio(h):
    h = np.maximum(h, 0)
    sum_h2 = np.sum(h**2)
    sum_h = np.sum(h)
    if sum_h < 1e-15:
        return 0.0
    sup_ff = _autoconv_sup(h)
    if sup_ff < 1e-15:
        return 0.0
    return sum_h2 / (sum_h * sup_ff)

def _neg_ratio_log(x):
    """Objective for Differential Evolution in log-space."""
    # Clip to avoid overflow/underflow in exp
    x = np.clip(x, -15, 15)
    h = np.exp(x)
    s = np.sum(h)
    if s < 1e-15:
        return 0.0
    h = h / s
    return -_ratio(h)

def construct():
    start_time = time.time()
    DEADLINE = 145.0
    
    best_h = np.ones(N) / N
    best_score = _ratio(best_h)
    
    # 1. Global Search with Differential Evolution in log-space
    # Bounds for log(h): -10 to 10
    bounds = [(-10, 10)] * N
    
    # Use a smaller population and maxiter to fit in time budget
    # DE is stochastic, so set seed for reproducibility if needed, but usually default is fine
    try:
        result_de = differential_evolution(
            _neg_ratio_log, 
            bounds, 
            maxiter=100, 
            popsize=15, 
            tol=1e-5,
            seed=42,
            init='latinhypercube'
        )
        
        h_de = np.exp(result_de.x)
        s_de = np.sum(h_de)
        if s_de > 1e-15:
            h_de = h_de / s_de
        score_de = _ratio(h_de)
        
        if score_de > best_score:
            best_score = score_de
            best_h = h_de.copy()
            
    except Exception as e:
        pass

    # 2. Local Polish with SLSQP in linear space
    # Start from the best found so far (either initial uniform or DE result)
    h_current = best_h.copy()
    
    # Run SLSQP a few times from slightly perturbed starts to escape local minima
    np.random.seed(42)
    
    for attempt in range(5):
        if time.time() - start_time > DEADLINE - 10:
            break
            
        if attempt == 0:
            h0 = h_current.copy()
        else:
            # Perturb the best solution
            noise = np.random.normal(0, 0.1 * np.std(h_current) + 1e-6, N)
            h0 = np.maximum(h_current + noise, 0)
            s0 = np.sum(h0)
            if s0 < 1e-15:
                continue
            h0 = h0 / s0
        
        def neg_ratio(x):
            x = np.maximum(x, 0)
            s = np.sum(x)
            if s < 1e-15:
                return 0.0
            return -_ratio(x / s)
        
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(np.maximum(x, 0)) - 1.0}]
        bounds_lin = [(0, None)] * N
        
        try:
            result_slsqp = minimize(
                neg_ratio, 
                h0, 
                method='SLSQP',
                bounds=bounds_lin, 
                constraints=constraints,
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            
            h_candidate = np.maximum(result_slsqp.x, 0)
            s_cand = np.sum(h_candidate)
            if s_cand > 1e-15:
                h_candidate = h_candidate / s_cand
                score_cand = _ratio(h_candidate)
                
                if score_cand > best_score:
                    best_score = score_cand
                    best_h = h_candidate.copy()
                    
        except Exception:
            pass

    return best_h.tolist()
