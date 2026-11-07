"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _correlation_value(h):
    """Compute the evaluator's value."""
    corr = np.correlate(h, 1 - h, mode="full")
    return np.max(corr) / N * 2.0

def _smooth_max_corr(h, beta=80.0):
    """Smooth approximation of max correlation using log-sum-exp."""
    corr = np.correlate(h, 1 - h, mode="full")
    # Use log-sum-exp for smooth max
    m = np.max(corr)
    return m + np.log(np.sum(np.exp((corr - m) * beta)) / len(corr)) / beta

def _epigraph_objective(x, beta=80.0):
    """Epigraph formulation: x[0:N]=h, x[N]=t. Minimize t + small regularization."""
    h = x[:N]
    t = x[N]
    corr = np.correlate(h, 1 - h, mode="full")
    # Smooth max: t >= max(corr) approximated smoothly
    m = np.max(corr)
    smooth_max = m + np.log(np.sum(np.exp((corr - m) * beta)) / len(corr)) / beta
    # Penalty for t being below smooth max
    penalty = np.maximum(0, smooth_max - t) ** 2
    return t + 0.01 * penalty

def _project_to_feasible(h):
    """Project h onto feasible set: sum=N/2, 0<=h<=1."""
    h = np.array(h, dtype=float).copy()
    h = np.clip(h, 0.0, 1.0)
    for _ in range(200):
        diff = N / 2.0 - np.sum(h)
        if abs(diff) < 1e-14:
            break
        if diff > 0:
            mask = h < 1.0
        else:
            mask = h > 0.0
        if not np.any(mask):
            break
        h[mask] += diff / np.sum(mask)
        h = np.clip(h, 0.0, 1.0)
    return h

def _local_polish(h, max_iters=300, tol=1e-12):
    """Efficient local polish using gradient-like updates on the smooth surrogate."""
    h = np.array(h, dtype=float).copy()
    h = _project_to_feasible(h)
    best_val = _correlation_value(h)
    beta = 80.0
    
    for it in range(max_iters):
        # Compute gradient of smooth max correlation w.r.t. h
        corr = np.correlate(h, 1 - h, mode="full")
        m = np.max(corr)
        weights = np.exp((corr - m) * beta)
        weights /= np.sum(weights)
        
        # Gradient of correlation[k] w.r.t. h[i]
        # corr[k] = sum_j h[j] * (1-h[j+k-N]) for appropriate shifts
        # Use finite differences for simplicity but vectorized
        grad = np.zeros(N)
        for i in range(N):
            # Perturb h[i] slightly
            eps = 1e-6
            h_plus = h.copy()
            h_plus[i] += eps
            h_plus = np.clip(h_plus, 0, 1)
            # Re-project to maintain sum constraint approximately
            diff = N/2.0 - np.sum(h_plus)
            if abs(diff) > 1e-10:
                mask = h_plus < 1.0 if diff > 0 else h_plus > 0.0
                if np.any(mask):
                    h_plus[mask] += diff / np.sum(mask)
                    h_plus = np.clip(h_plus, 0, 1)
            
            corr_plus = np.correlate(h_plus, 1 - h_plus, mode="full")
            m_plus = np.max(corr_plus)
            w_plus = np.exp((corr_plus - m_plus) * beta)
            w_plus /= np.sum(w_plus)
            smooth_plus = m_plus + np.log(np.sum(w_plus) / len(corr_plus)) / beta
            
            grad[i] = (smooth_plus - _smooth_max_corr(h, beta)) / eps
        
        # Gradient descent step with adaptive learning rate
        lr = 0.01 / (np.max(np.abs(grad)) + 1e-8)
        h_new = h - lr * grad
        h_new = _project_to_feasible(h_new)
        
        val = _correlation_value(h_new)
        if val < best_val - tol:
            h = h_new.copy()
            best_val = val
        else:
            # Try smaller step
            lr *= 0.1
            h_new = h - lr * grad
            h_new = _project_to_feasible(h_new)
            val = _correlation_value(h_new)
            if val < best_val - tol:
                h = h_new.copy()
                best_val = val
            else:
                break
    
    return h, best_val

def _generate_starts(n_starts=6, seed=42):
    """Generate diverse starting points."""
    rng = np.random.RandomState(seed)
    starts = []
    
    # Start 1: uniform 0.5
    starts.append(np.full(N, 0.5))
    
    # Start 2: alternating
    alt = np.ones(N)
    alt[::2] = 0.0
    alt[1::2] = 1.0
    # Adjust to sum = N/2
    alt = _project_to_feasible(alt)
    starts.append(alt)
    
    # Start 3-6: random
    for _ in range(n_starts - 2):
        h = rng.uniform(0.1, 0.9, size=N)
        h = _project_to_feasible(h)
        starts.append(h)
    
    return starts

def _slsqp_optimize(h_init, deadline):
    """Optimize using SLSQP with smooth objective."""
    bounds = [(0.0, 1.0)] * N
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N / 2.0}]
    
    def obj(x):
        return _smooth_max_corr(x, beta=80.0)
    
    try:
        result = minimize(
            obj, h_init, method='SLSQP',
            bounds=bounds, constraints=constraints,
            options={'maxiter': 200, 'ftol': 1e-12}
        )
        if time.time() > deadline:
            return h_init, _correlation_value(h_init)
        
        h_cand = _project_to_feasible(result.x)
        val = _correlation_value(h_cand)
        return h_cand, val
    except Exception:
        return h_init, _correlation_value(h_init)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    starts = _generate_starts(n_starts=6, seed=42)
    
    best_h = starts[0].copy()
    best_val = _correlation_value(best_h)
    
    for i, h_init in enumerate(starts):
        if time.time() > deadline - 15.0:
            break
        
        # SLSQP optimization
        h_opt, val_opt = _slsqp_optimize(h_init, deadline)
        
        # Local polish
        if time.time() < deadline - 5.0:
            h_pol, val_pol = _local_polish(h_opt, max_iters=200)
            if val_pol < val_opt:
                h_opt = h_pol
                val_opt = val_pol
        
        if val_opt < best_val:
            best_val = val_opt
            best_h = h_opt.copy()
    
    # Final high-precision polish on the best solution
    if time.time() < deadline - 2.0:
        h_final, val_final = _local_polish(best_h, max_iters=500, tol=1e-14)
        if val_final < best_val:
            best_h = h_final
            best_val = val_final
    
    # Final validation
    best_h = _project_to_feasible(best_h)
    return best_h
# EVOLVE-BLOCK-END
