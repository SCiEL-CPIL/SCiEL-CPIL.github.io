"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_score(centers):
    """Compute the radius for a given set of centers."""
    if centers.shape != (N, 3):
        return 0.0
    # Check bounds
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    
    # Wall distances
    wall_dists = []
    for i in range(N):
        for k in range(3):
            wall_dists.append(0.5 - abs(centers[i, k]))
    min_wall = min(wall_dists)
    
    # Pairwise distances
    min_pair = float('inf')
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < min_pair:
                min_pair = d
    
    return min(min_pair / 2.0, min_wall)

def _soft_min_obj_and_grad(centers, beta):
    """
    Compute the smooth surrogate objective and its gradient.
    Objective: maximize min(d_k). Equivalent to minimize -soft_min(d_k).
    soft_min = -1/beta * log(sum(exp(-beta * d_k)))
    We want to MINIMIZE -soft_min.
    Let G = -soft_min.
    G = 1/beta * log(sum(exp(-beta * d_k)))
    grad G = - sum( w_k * grad d_k ) where w_k = exp(-beta d_k) / sum(exp(-beta d_j))
    """
    N = centers.shape[0]
    dists = []
    grads = [] # List of (indices, gradient_vector) or just accumulate
    
    # Accumulate gradient of G
    grad_G = np.zeros_like(centers)
    
    # Compute all distances and their contributions
    # We need sum(exp(-beta * d_k))
    # To avoid overflow/underflow, we can shift by max d, but beta is moderate.
    # Let's just compute exp(-beta * d) directly. If beta is large, exp becomes 0 for large d.
    
    exp_terms = []
    
    # Pairwise distances
    for i in range(N):
        for j in range(i + 1, N):
            diff = centers[i] - centers[j]
            d = np.linalg.norm(diff)
            if d < 1e-12:
                d = 1e-12
                # Gradient is undefined, use random or zero? Zero is safer to avoid NaNs.
                g_i = np.zeros(3)
                g_j = np.zeros(3)
            else:
                g_i = diff / d
                g_j = -diff / d
            exp_terms.append(np.exp(-beta * d))
            # Store gradient info for this term
            # We need to accumulate w_k * grad d_k later
            # Let's store (i, j, g_i, g_j, exp_term)
            # Actually, it's better to compute weights first, then accumulate gradients.
            
    # Wall distances
    for i in range(N):
        for k in range(3):
            # Distance to wall: 0.5 - |c_ik|
            # Let's handle abs carefully.
            # If c_ik > 0, d = 0.5 - c_ik, grad = -1
            # If c_ik < 0, d = 0.5 + c_ik, grad = 1
            # If c_ik == 0, undefined, use 0.
            
            c = centers[i, k]
            if c > 0:
                d = 0.5 - c
                g = -1.0
            elif c < 0:
                d = 0.5 + c
                g = 1.0
            else:
                d = 0.5
                g = 0.0
            
            exp_terms.append(np.exp(-beta * d))
            # Store (i, k, g, exp_term)
            
    # To efficiently compute weights and gradients, we should have stored the gradients.
    # Let's restructure: 
    # 1. Compute all d_k and exp(-beta d_k).
    # 2. Compute sum_exp.
    # 3. Compute weights w_k = exp_k / sum_exp.
    # 4. Accumulate grad_G = - sum( w_k * grad d_k ).
    
    # Re-doing the loop to store data is inefficient. Let's do it in one pass if possible?
    # No, we need sum_exp first.
    
    sum_exp = sum(exp_terms)
    if sum_exp == 0:
        # All exp terms are 0, which means all d are large. 
        # This shouldn't happen if beta is not huge or d is not huge.
        return 0.0, np.zeros_like(centers)
    
    # We need to iterate again to accumulate gradients with weights.
    # This is 2x cost, but acceptable for N=50.
    
    # Pairwise
    for i in range(N):
        for j in range(i + 1, N):
            diff = centers[i] - centers[j]
            d = np.linalg.norm(diff)
            if d < 1e-12:
                continue
            w = np.exp(-beta * d) / sum_exp
            g_i = diff / d
            g_j = -diff / d
            # grad d_k wrt x_i is g_i, wrt x_j is g_j
            # grad G = - sum w_k grad d_k
            grad_G[i] -= w * g_i
            grad_G[j] -= w * g_j
            
    # Walls
    for i in range(N):
        for k in range(3):
            c = centers[i, k]
            if c > 0:
                d = 0.5 - c
                g = -1.0
            elif c < 0:
                d = 0.5 + c
                g = 1.0
            else:
                continue # grad 0
            
            w = np.exp(-beta * d) / sum_exp
            # grad d_k wrt x_ik is g
            grad_G[i, k] -= w * g
            
    # Objective value: G = 1/beta * log(sum_exp)
    obj_val = (1.0 / beta) * np.log(sum_exp)
    
    return obj_val, grad_G

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin
    
    # Initial best
    best_score = 0.0
    best_centers = None
    
    # Initial guess: Jittered grid
    # 4x4x4 grid = 64 points. Take 50.
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    # We will run multiple restarts with different random jitters
    # and increasing beta (continuation).
    
    rng = np.random.default_rng(42)
    
    num_restarts = 10
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Create jittered initial guess
        # Use the first 50 points of the grid, add noise
        # Or random points? Grid is better start.
        # Let's pick 50 random points from a fine grid or just jitter the 50 grid points.
        
        # Strategy: Start with a 4x4x4 grid, keep 50, jitter.
        # Or just random uniform? Random uniform often gets stuck.
        # Let's use a "blue noise" like distribution or just jittered grid.
        
        # Jittered grid:
        init_centers = grid[:N].copy()
        # Add small random noise to break symmetry
        noise = rng.normal(0, 0.01, size=init_centers.shape)
        init_centers += noise
        # Ensure inside bounds
        init_centers = np.clip(init_centers, -0.49, 0.49)
        
        # Continuation on beta
        # Start with low beta (smooth), increase to high beta (sharp)
        betas = [5, 10, 20, 40, 80, 160, 320]
        
        current_centers = init_centers.copy()
        
        for beta in betas:
            if time.time() > deadline:
                break
                
            # Define objective and gradient
            def obj_grad(x):
                val, grad = _soft_min_obj_and_grad(x, beta)
                return val, grad
            
            # Use L-BFGS-B
            # Bounds: [-0.5, 0.5] for all coordinates
            bounds = [(-0.5, 0.5)] * (N * 3)
            
            try:
                res = minimize(
                    obj_grad,
                    current_centers.flatten(),
                    jac=True,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12}
                )
                current_centers = res.x.reshape(N, 3)
            except Exception as e:
                # If optimization fails, keep current_centers
                pass
        
        # Evaluate final score
        score = _compute_score(current_centers)
        if score > best_score:
            best_score = score
            best_centers = current_centers.copy()
            
    # Final check
    if best_centers is None:
        # Fallback to simple grid
        return grid[:N]
        
    return best_centers
# EVOLVE-BLOCK-END
