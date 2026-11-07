"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _soft_max(corr, beta=100.0):
    """Smooth approximation of max via log-sum-exp."""
    m = np.max(corr)
    # To avoid overflow/underflow issues, shift by m
    # exp(beta * (corr - m)) is safe since corr - m <= 0
    return m + np.log(np.sum(np.exp(beta * (corr - m)))) / beta

def _objective(h, beta=100.0):
    """Compute the value to minimize using soft-max."""
    corr = np.correlate(h, 1 - h, mode="full")
    return _soft_max(corr, beta) / N * 2

def _constraints_check(h):
    """Check if h is valid."""
    if not np.all(np.isfinite(h)):
        return False
    if np.any(h < -1e-12) or np.any(h > 1 + 1e-12):
        return False
    if abs(np.sum(h) - N / 2) > 1e-9 * N:
        return False
    return True

def _project_sum(h):
    """Project h to satisfy sum(h) = N/2 and h in [0, 1]."""
    h = np.clip(h, 0, 1)
    target = N / 2.0
    current_sum = np.sum(h)
    diff = target - current_sum
    
    # If diff is small, just add it
    if abs(diff) < 1e-9:
        return h
    
    # Distribute diff among non-saturated elements
    # Elements at 0 can only increase, elements at 1 can only decrease
    # We need to find a set of indices where we can adjust.
    
    # Simple iterative approach:
    # 1. Identify free elements (0 < h < 1)
    # 2. If no free elements, we might be stuck, but with sum=50 and N=100,
    #    and h in [0,1], there should be free elements unless all are 0 or 1,
    #    which would imply sum is 0 or 100.
    
    # Let's try a few iterations of redistribution
    for _ in range(10):
        if abs(np.sum(h) - target) < 1e-12:
            break
        free_mask = (h > 1e-8) & (h < 1 - 1e-8)
        if not np.any(free_mask):
            # If no free elements, we need to move some 0s to 1s or vice versa
            # This is a harder combinatorial problem.
            # For now, just clip and hope the optimizer handles it.
            break
        
        num_free = np.sum(free_mask)
        adj = diff / num_free
        h[free_mask] += adj
        h = np.clip(h, 0, 1)
        diff = target - np.sum(h)
        
    return h

def construct():
    deadline = time.time() + 140  # Safety margin before 150s limit
    best_h = np.full(N, 0.5)
    best_val = _objective(best_h)
    
    # Try smooth sinusoidal patterns
    for freq in range(1, 20):
        for phase in np.linspace(0, 2*np.pi, 8, endpoint=False):
            h = 0.5 + 0.3 * np.sin(2*np.pi*freq*np.arange(N)/N + phase)
            h = _project_sum(h)
            if _constraints_check(h):
                val = _objective(h)
                if val < best_val:
                    best_val = val
                    best_h = h.copy()
    
    # Try local optimization from best found
    h0 = best_h.copy()
    
    def obj_func(x):
        # x is the raw variable from L-BFGS-B
        # We don't clip here to allow gradients to flow, but we must ensure
        # the penalty handles the constraints.
        # However, L-BFGS-B handles bounds [0,1] natively.
        # So x is already in [0,1] mostly.
        # The sum constraint is handled by penalty.
        sum_penalty = (np.sum(x) - N/2) ** 2 * 1e8
        return _objective(x) + sum_penalty
    
    # Use L-BFGS-B with bounds
    bounds = [(0, 1)] * N
    result = minimize(obj_func, h0, method='L-BFGS-B', bounds=bounds, 
                      options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-14})
    
    # Project the result to ensure feasibility
    h_opt = _project_sum(result.x)
    if _constraints_check(h_opt):
        val = _objective(h_opt)
        if val < best_val:
            best_val = val
            best_h = h_opt.copy()
    
    # Try more local optimizations from different starting points
    if time.time() < deadline:
        rng = np.random.RandomState(42)
        for _ in range(50):
            if time.time() > deadline:
                break
            h0 = 0.5 + 0.4 * np.sin(2*np.pi*rng.randint(1,20)*np.arange(N)/N + rng.uniform(0, 2*np.pi))
            h0 = _project_sum(h0)
            if not _constraints_check(h0):
                continue
            
            result = minimize(obj_func, h0, method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-14})
            
            h_opt = _project_sum(result.x)
            if _constraints_check(h_opt):
                val = _objective(h_opt)
                if val < best_val:
                    best_val = val
                    best_h = h_opt.copy()
    
    # Final projection to ensure exact sum
    best_h = _project_sum(best_h)
    
    # Verify
    if not _constraints_check(best_h):
        return np.full(N, 0.5)
    
    return best_h
# EVOLVE-BLOCK-END
