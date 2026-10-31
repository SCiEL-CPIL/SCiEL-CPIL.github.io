"""Rewritten program: minimize max overlap of h with shifted 1-h."""
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 100


def _overlap_objective(h):
    """Compute the maximum overlap of h with shifted (1-h).
    
    Overlap at shift k: sum_i h[i] * (1 - h[(i+k) % N])
    We want to minimize the maximum over all shifts k.
    """
    N = len(h)
    h = np.asarray(h)
    one_minus_h = 1.0 - h
    max_overlap = 0.0
    for k in range(N):
        shifted = np.roll(one_minus_h, k)
        overlap = np.sum(h * shifted)
        if overlap > max_overlap:
            max_overlap = overlap
    return max_overlap


def _smoothed_overlap_objective(h, beta=50.0):
    """Soft-max surrogate: approximate max via log-sum-exp."""
    N = len(h)
    h = np.asarray(h)
    one_minus_h = 1.0 - h
    overlaps = np.zeros(N)
    for k in range(N):
        shifted = np.roll(one_minus_h, k)
        overlaps[k] = np.sum(h * shifted)
    # soft max
    max_val = np.max(overlaps)
    if beta == 0:
        return max_val
    log_sum_exp = max_val + np.log(np.sum(np.exp(beta * (overlaps - max_val))))
    return (1.0 / beta) * log_sum_exp


def _constraint_sum(h):
    """Constraint: sum(h) = N/2."""
    return np.sum(h) - N / 2.0


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill
    
    best_h = np.full(N, 0.5)
    best_score = _overlap_objective(best_h)
    
    # Try differential evolution first for global search
    try:
        def de_objective(h):
            # Add penalty for constraint violation
            penalty = 10.0 * abs(np.sum(h) - N / 2.0)
            return _smoothed_overlap_objective(h, beta=30.0) + penalty
        
        bounds = [(0.0, 1.0)] * N
        
        # Use differential evolution with limited population
        elapsed = time.time() - start_time
        if elapsed < deadline - 60.0:
            result = differential_evolution(
                de_objective,
                bounds=bounds,
                maxiter=50,
                popsize=15,
                tol=1e-4,
                seed=42,
                polish=False,
                init='latinhypercube'
            )
            if result.success or result.fun < best_score:
                candidate = result.x
                # Project back to satisfy constraint
                candidate = np.clip(candidate, 0.0, 1.0)
                # Adjust to satisfy sum constraint
                diff = N / 2.0 - np.sum(candidate)
                candidate = candidate + diff / N
                candidate = np.clip(candidate, 0.0, 1.0)
                # Fine-tune to get exact sum
                residual = N / 2.0 - np.sum(candidate)
                if abs(residual) > 1e-12:
                    # Distribute residual
                    idx = np.argmax(candidate < 1.0) if residual > 0 else np.argmin(candidate > 0.0)
                    candidate[idx] += residual
                    candidate = np.clip(candidate, 0.0, 1.0)
                
                score = _overlap_objective(candidate)
                if score < best_score:
                    best_score = score
                    best_h = candidate.copy()
        
        elapsed = time.time() - start_time
        if elapsed < deadline - 30.0:
            # Local polish using SLSQP
            def local_obj(h):
                return _smoothed_overlap_objective(h, beta=100.0)
            
            constraints = [{'type': 'eq', 'fun': _constraint_sum}]
            bounds_local = [(0.0, 1.0)] * N
            
            res = minimize(
                local_obj,
                best_h,
                method='SLSQP',
                bounds=bounds_local,
                constraints=constraints,
                options={'maxiter': 200, 'ftol': 1e-10}
            )
            
            if res.fun is not None:
                candidate = np.clip(res.x, 0.0, 1.0)
                # Ensure sum constraint
                diff = N / 2.0 - np.sum(candidate)
                candidate = candidate + diff / N
                candidate = np.clip(candidate, 0.0, 1.0)
                residual = N / 2.0 - np.sum(candidate)
                if abs(residual) > 1e-12:
                    idx = np.argmax(candidate < 1.0) if residual > 0 else np.argmin(candidate > 0.0)
                    candidate[idx] += residual
                    candidate = np.clip(candidate, 0.0, 1.0)
                
                score = _overlap_objective(candidate)
                if score < best_score:
                    best_score = score
                    best_h = candidate.copy()
        
        # Second round of local polish with higher beta
        elapsed = time.time() - start_time
        if elapsed < deadline - 15.0:
            def local_obj2(h):
                return _smoothed_overlap_objective(h, beta=200.0)
            
            constraints = [{'type': 'eq', 'fun': _constraint_sum}]
            bounds_local = [(0.0, 1.0)] * N
            
            res = minimize(
                local_obj2,
                best_h,
                method='SLSQP',
                bounds=bounds_local,
                constraints=constraints,
                options={'maxiter': 300, 'ftol': 1e-12}
            )
            
            if res.fun is not None:
                candidate = np.clip(res.x, 0.0, 1.0)
                diff = N / 2.0 - np.sum(candidate)
                candidate = candidate + diff / N
                candidate = np.clip(candidate, 0.0, 1.0)
                residual = N / 2.0 - np.sum(candidate)
                if abs(residual) > 1e-12:
                    idx = np.argmax(candidate < 1.0) if residual > 0 else np.argmin(candidate > 0.0)
                    candidate[idx] += residual
                    candidate = np.clip(candidate, 0.0, 1.0)
                
                score = _overlap_objective(candidate)
                if score < best_score:
                    best_score = score
                    best_h = candidate.copy()
    
    except Exception:
        pass
    
    # Final check: ensure validity
    best_h = np.clip(best_h, 0.0, 1.0)
    diff = N / 2.0 - np.sum(best_h)
    best_h = best_h + diff / N
    best_h = np.clip(best_h, 0.0, 1.0)
    residual = N / 2.0 - np.sum(best_h)
    if abs(residual) > 1e-14:
        idx = np.argmax(best_h < 1.0) if residual > 0 else np.argmin(best_h > 0.0)
        best_h[idx] += residual
        best_h = np.clip(best_h, 0.0, 1.0)
    
    return best_h
