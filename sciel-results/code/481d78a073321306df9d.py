"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _max_overlap(h):
    """Compute the maximum overlap of h with shifted (1-h) over all shifts."""
    N = len(h)
    overlaps = np.zeros(N)
    for k in range(N):
        shifted_1mh = 1.0 - np.roll(h, -k)
        overlaps[k] = np.dot(h, shifted_1mh)
    return np.max(overlaps)

def _overlap_vector(h):
    """Compute all overlaps of h with shifted (1-h)."""
    N = len(h)
    overlaps = np.zeros(N)
    for k in range(N):
        shifted_1mh = 1.0 - np.roll(h, -k)
        overlaps[k] = np.dot(h, shifted_1mh)
    return overlaps

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Start with uniform solution
    best_h = np.full(N, 0.5)
    best_val = _max_overlap(best_h)
    
    # Use smooth surrogate: soft-max of overlaps via log-sum-exp
    # to make the objective differentiable for scipy optimizers
    
    def soft_max_objective(h):
        """Soft-max surrogate for the max overlap, plus constraint penalty."""
        overlaps = _overlap_vector(h)
        # Use log-sum-exp for smooth max approximation
        beta = 50.0  # higher beta = sharper max approximation
        m = np.max(overlaps)
        soft_max_val = m + np.log(np.sum(np.exp((overlaps - m) * beta)) / N) / beta
        
        # Penalty for constraint violations
        penalty = 0.0
        s = np.sum(h)
        penalty += 1000.0 * (s - N / 2.0) ** 2
        penalty += 100.0 * np.sum(np.maximum(0, h - 1.0) ** 2)
        penalty += 100.0 * np.sum(np.maximum(0, -h) ** 2)
        
        return soft_max_val + penalty
    
    # Try SLSQP with explicit constraints
    def objective_slsqp(h):
        overlaps = _overlap_vector(h)
        return np.max(overlaps)
    
    def objective_grad(h):
        """Gradient of the max overlap (subgradient)."""
        overlaps = _overlap_vector(h)
        k_max = np.argmax(overlaps)
        # Gradient of overlap_k w.r.t. h[i]:
        # d/dh[i] sum_j h[j]*(1-h[(j+k)%N])
        # = (1-h[(i+k)%N]) - h[(i-k)%N] for the term where j=i
        # but we need to be careful with the double sum
        # Actually: overlap_k = sum_j h[j]*(1-h[(j+k)%N])
        # d/dh[i] = (1-h[(i+k)%N]) + sum_j h[j]*d/dh[i](1-h[(j+k)%N])
        # The second term: d/dh[i](1-h[(j+k)%N]) = -1 if (j+k)%N == i, i.e., j = (i-k)%N
        # So d/dh[i] = (1-h[(i+k)%N]) - h[(i-k)%N]
        grad = np.zeros(N)
        for i in range(N):
            grad[i] = (1.0 - h[(i + k_max) % N]) - h[(i - k_max) % N]
        return grad
    
    # SLSQP optimization
    constraints = [
        {'type': 'eq', 'fun': lambda h: np.sum(h) - N / 2.0, 
         'jac': lambda h: np.ones(N)}
    ]
    bounds = [(0.0, 1.0)] * N
    
    h_slsqp = best_h.copy()
    try:
        result = minimize(
            objective_slsqp,
            h_slsqp,
            method='SLSQP',
            jac=objective_grad,
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 500, 'ftol': 1e-10}
        )
        if time.time() < deadline - 30.0:
            h_candidate = np.clip(result.x, 0, 1)
            # Enforce exact sum
            s = np.sum(h_candidate)
            if s > 0:
                h_candidate = h_candidate * (N / 2.0) / s
                h_candidate = np.clip(h_candidate, 0, 1)
            # Final sum correction
            diff = N / 2.0 - np.sum(h_candidate)
            if abs(diff) > 1e-12:
                if diff > 0:
                    idx = np.where(h_candidate < 1.0)[0]
                else:
                    idx = np.where(h_candidate > 0.0)[0]
                if len(idx) > 0:
                    h_candidate[idx[0]] += diff
                    h_candidate = np.clip(h_candidate, 0, 1)
            
            val = _max_overlap(h_candidate)
            if val < best_val - 1e-10:
                best_val = val
                best_h = h_candidate.copy()
    except Exception:
        pass
    
    # L-BFGS-B with penalty method
    if time.time() < deadline - 20.0:
        h_lbfgs = best_h.copy()
        try:
            result = minimize(
                soft_max_objective,
                h_lbfgs,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-12}
            )
            h_candidate = np.clip(result.x, 0, 1)
            s = np.sum(h_candidate)
            if s > 0:
                h_candidate = h_candidate * (N / 2.0) / s
                h_candidate = np.clip(h_candidate, 0, 1)
            diff = N / 2.0 - np.sum(h_candidate)
            if abs(diff) > 1e-12:
                if diff > 0:
                    idx = np.where(h_candidate < 1.0)[0]
                else:
                    idx = np.where(h_candidate > 0.0)[0]
                if len(idx) > 0:
                    h_candidate[idx[0]] += diff
                    h_candidate = np.clip(h_candidate, 0, 1)
            
            val = _max_overlap(h_candidate)
            if val < best_val - 1e-10:
                best_val = val
                best_h = h_candidate.copy()
        except Exception:
            pass
    
    # Local search polish with multiple random restarts
    if time.time() < deadline - 10.0:
        np.random.seed(42)
        for trial in range(20):
            if time.time() > deadline - 5.0:
                break
            # Random perturbation maintaining sum = N/2
            h = best_h + np.random.randn(N) * 0.02
            h = np.clip(h, 0, 1)
            s = np.sum(h)
            if s > 0:
                h = h * (N / 2.0) / s
                h = np.clip(h, 0, 1)
            diff = N / 2.0 - np.sum(h)
            if diff > 0:
                idx = np.where(h < 1.0)[0]
                if len(idx) > 0:
                    h[idx] += diff / len(idx)
            else:
                idx = np.where(h > 0.0)[0]
                if len(idx) > 0:
                    h[idx] += diff / len(idx)
            
            val = _max_overlap(h)
            if val < best_val - 1e-10:
                best_val = val
                best_h = h.copy()
    
    # Gradient-based polish on the best solution
    if time.time() < deadline - 3.0:
        h = best_h.copy()
        for it in range(200):
            if time.time() > deadline - 2.0:
                break
            overlaps = _overlap_vector(h)
            k_max = np.argmax(overlaps)
            grad = np.zeros(N)
            for i in range(N):
                grad[i] = (1.0 - h[(i + k_max) % N]) - h[(i - k_max) % N]
            
            step = 0.0005
            h_new = h - step * grad
            h_new = np.clip(h_new, 0, 1)
            s = np.sum(h_new)
            if s > 0:
                h_new = h_new * (N / 2.0) / s
                h_new = np.clip(h_new, 0, 1)
            
            val = _max_overlap(h_new)
            if val < best_val - 1e-12:
                best_val = val
                best_h = h_new.copy()
                h = h_new
            else:
                h = h_new
    
    # Ensure exact sum
    best_h = np.clip(best_h, 0, 1)
    s = np.sum(best_h)
    if s > 0:
        best_h = best_h * (N / 2.0) / s
        best_h = np.clip(best_h, 0, 1)
    
    # Final correction for exact sum
    diff = N / 2.0 - np.sum(best_h)
    if abs(diff) > 1e-15:
        if diff > 0:
            idx = np.where(best_h < 1.0)[0]
        else:
            idx = np.where(best_h > 0.0)[0]
        if len(idx) > 0:
            best_h[idx[0]] += diff
            best_h = np.clip(best_h, 0, 1)
    
    return best_h
# EVOLVE-BLOCK-END
