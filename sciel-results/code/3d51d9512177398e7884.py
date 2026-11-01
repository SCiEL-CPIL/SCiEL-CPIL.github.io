"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

def _score(c):
    """Compute the largest common radius for centres c (shape (N,3))."""
    if c.shape != (N, 3):
        return -1.0
    if not np.all(np.isfinite(c)):
        return -1.0
    # distance to walls
    wall = np.min(np.min(0.5 - np.abs(c), axis=1), axis=0)
    # pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)
    return min(wall, min_dist / 2.0)

def _init_grid():
    """Simple cubic grid as starting point."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    # N=100, m=5. 5x5x5=125 points. We take the first 100.
    # To make it more uniform for 100, we can use a 4x5x5 grid or similar?
    # 4*5*5 = 100 exactly.
    # Let's construct a 4x5x5 grid.
    nx, ny, nz = 4, 5, 5
    xs = (np.arange(nx) + 0.5) / nx - 0.5
    ys = (np.arange(ny) + 0.5) / ny - 0.5
    zs = (np.arange(nz) + 0.5) / nz - 0.5
    
    grid = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    return grid[:N].copy()

def _soft_min_obj_grad(x, beta):
    """
    Objective function and gradient for soft-min of distances.
    x: flattened array of shape (N*3,)
    beta: sharpness parameter
    Returns: (obj, grad)
    Minimize this to maximize the minimum distance.
    We consider both wall distances and pairwise distances.
    
    Wall distances: d_wi = 0.5 - |xi| for each coordinate.
    Pairwise distances: d_ij = ||xi - xj||.
    
    Soft min: -1/beta * log(sum(exp(-beta * d_k)))
    Gradient: sum(exp(-beta * d_k) * grad_d_k) / sum(exp(-beta * d_k))
    """
    x = x.reshape(N, 3)
    
    # 1. Compute all distances and their gradients
    # Wall distances
    # d_wi_k = 0.5 - |x_ik|
    # grad d/dx = -sign(x)
    
    wall_dists = np.zeros((N, 3))
    wall_grads = np.zeros((N, 3, 3))
    
    for i in range(N):
        for k in range(3):
            val = x[i, k]
            wall_dists[i, k] = 0.5 - abs(val)
            if val > 0:
                wall_grads[i, k, k] = -1.0
            else:
                wall_grads[i, k, k] = 1.0 # derivative of -|x| is -sign(x). If x<0, -(-1)=1. If x>0, -(1)=-1.
                # Wait. d = 0.5 - |x|. dd/dx = -sign(x).
                # If x > 0, sign=1, grad=-1.
                # If x < 0, sign=-1, grad=1.
                # My code: if val>0 grad=-1. else grad=1. Correct.

    # Pairwise distances
    # d_ij = ||xi - xj||
    # grad d/dxi = (xi - xj) / d_ij
    # grad d/dxj = (xj - xi) / d_ij
    
    # We need to sum contributions from all pairs.
    # To avoid O(N^3) memory for all distances at once if N is large, 
    # but N=100, N^2=10000, which is fine.
    
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :] # (N, N, 3)
    dists = np.sqrt(np.sum(diff**2, axis=2)) # (N, N)
    np.fill_diagonal(dists, np.inf)
    
    # Mask for i < j to avoid double counting in the sum
    # But the soft min is over all constraints.
    # Constraints are:
    # 1. N*3 wall constraints
    # 2. N*(N-1)/2 pairwise constraints (symmetric)
    
    # Let's flatten all distances into a single vector D
    # And corresponding gradients G (Jacobian of D w.r.t x)
    
    # Wall part
    D_w = wall_dists.flatten() # size 3N
    G_w = wall_grads.reshape(N*3, 3*N) # (3N, 3N)
    
    # Pairwise part
    # We only need i < j
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu] # size M
    # Gradients for pair distances
    # For a pair (i, j), vector v = xi - xj. d = ||v||.
    # grad xi = v/d, grad xj = -v/d
    
    G_p = np.zeros((len(pair_dists), 3*N))
    for idx, (i, j) in enumerate(zip(iu[0], iu[1])):
        d = pair_dists[idx]
        if d > 1e-12:
            v = x[i] - x[j]
            grad_v = v / d
            G_p[idx, i*3:(i+1)*3] = grad_v
            G_p[idx, j*3:(j+1)*3] = -grad_v
        else:
            # If distance is 0, gradient is undefined. Add small noise or skip?
            # In optimization, if they are coincident, push them apart randomly?
            # For now, set to 0, relying on the objective value (which is 0) to drive movement via other terms or restarts.
            # Actually, if d=0, exp(-beta*0)=1. The term is active.
            # Gradient of 0 is problematic.
            # We can perturb slightly in the calculation if d is too small.
            pass
            
    D = np.concatenate([D_w, pair_dists])
    G = np.vstack([G_w, G_p])
    
    # Soft Min Calculation
    # To prevent overflow, shift by min
    min_D = np.min(D)
    shifted = -beta * (D - min_D)
    
    # Clip to prevent overflow in exp
    shifted = np.clip(shifted, -700, 700)
    
    exp_terms = np.exp(shifted)
    sum_exp = np.sum(exp_terms)
    
    if sum_exp == 0:
        obj = -min_D
        grad = np.zeros(3*N)
    else:
        # obj = -1/beta * log(sum_exp) - min_D
        obj = -1.0 / beta * np.log(sum_exp) - min_D
        
        # grad = sum( exp_term_k * G_k ) / sum_exp
        # weights = exp_terms / sum_exp
        weights = exp_terms / sum_exp
        grad = G.T @ weights
        
    return obj, grad

def _optimize_lfgs(c_init, time_limit, betas):
    """
    Multi-stage L-BFGS-B optimization.
    """
    c = c_init.copy()
    x0 = c.flatten()
    
    start_time = time.time()
    
    # Bounds: [-0.5, 0.5]
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    for beta in betas:
        if time.time() > time_limit:
            break
            
        # Run L-BFGS-B
        try:
            res = minimize(
                _soft_min_obj_grad,
                x0,
                args=(beta,),
                jac=True,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            x0 = res.x
        except Exception:
            pass
            
    c = x0.reshape(N, 3)
    return c

def _polish(c, time_limit):
    """
    Sequential pairwise repulsion polish.
    Iterates through pairs and pushes them apart if they are too close,
    while respecting boundaries.
    """
    c = c.copy()
    start_time = time.time()
    
    # Estimate a good radius from current config
    r = _score(c)
    
    # We want to maximize min dist.
    # Target distance = 2 * r_target
    # Let's try to increase r slightly.
    
    # Simple approach:
    # For many iterations:
    #   For each pair (i, j):
    #      if dist < 2*r_current:
    #          push apart
    
    # However, r_current is fixed in this loop?
    # Better: maximize the minimum distance directly by pushing the closest pair apart.
    
    # Let's use a local search:
    # Find the closest pair (i, j) and the closest wall (k).
    # Push i, j apart and k away from wall.
    
    for _ in range(5000): # Fixed number of iterations for polish
        if time.time() > time_limit:
            break
            
        # Compute distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Find min pair
        min_idx = np.unravel_index(np.argmin(dists), dists.shape)
        i, j = min_idx
        d_min = dists[i, j]
        
        # Find min wall dist
        wall_dists = np.min(0.5 - np.abs(c), axis=1)
        min_wall_idx = np.argmin(wall_dists)
        k = min_wall_idx
        d_wall = wall_dists[k]
        
        r_min = min(d_min / 2.0, d_wall)
        
        # If we can't improve, break?
        # We will try to increase the minimum constraint.
        
        # Strategy: Move the most constrained entities.
        
        # 1. Handle pair (i, j)
        if d_min < 2 * r_min + 1e-9: # If pair is the bottleneck
            # Push apart
            vec = c[i] - c[j]
            norm = np.linalg.norm(vec)
            if norm > 1e-12:
                direction = vec / norm
                # Move i in direction, j in -direction
                # Check bounds
                # Simple step
                step = 0.01 * (2*r_min - d_min + 1e-6)
                if step > 0.05: step = 0.05
                
                new_i = c[i] + direction * step
                new_j = c[j] - direction * step
                
                # Clip to bounds
                new_i = np.clip(new_i, -0.5, 0.5)
                new_j = np.clip(new_j, -0.5, 0.5)
                
                # Check if this improves the score (or at least doesn't hurt the min)
                # For simplicity in polish, we just accept if it increases d_min or keeps it same and improves wall?
                # Actually, we want to maximize the GLOBAL min.
                # Let's just do a small random jitter if stuck?
                # No, let's try to move i and j apart and see if score improves.
                
                c[i] = new_i
                c[j] = new_j
                
                # If score dropped, revert?
                # Checking score every step is O(N^2). 5000 * 100^2 = 50M ops. OK.
                # But we can just do it blindly with small steps?
                # Blindly moving might reduce other distances.
                # Let's revert if score decreases.
                new_score = _score(c)
                old_score = _score(c) # Wait, I modified c. I need old score.
                
                # This is inefficient. Let's store old c.
                # Re-do logic properly.
                
        # This simple polish is getting complicated with score checks.
        # Let's use a simpler "repulsion" force model for a few iterations.
        
        # Force based approach:
        # F_i = sum_{j != i} (c_i - c_j)/||c_i - c_j||^2  (if dist < 2r)
        # Plus wall repulsion.
        
        # Let's stick to a simple local gradient step on the true min (non-diff) using subgradient?
        # Or just return the L-BFGS result. The L-BFGS with high beta should be good.
        
        break # Skip complex polish for now, rely on L-BFGS stages.
        
    return c

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    # 1. Initialize with grid
    c = _init_grid()
    
    # 2. Multi-stage L-BFGS-B optimization
    # Start with low beta (smooth), end with high beta (sharp)
    betas = [10, 50, 100, 200, 500]
    
    # Allocate time to L-BFGS
    # L-BFGS is fast. We can run multiple stages.
    # Let's give it 100s.
    c_opt = _optimize_lfgs(c, deadline - 20.0, betas)
    
    # 3. Final polish with high beta L-BFGS again to ensure convergence
    if time.time() < deadline - 5.0:
         c_opt = _optimize_lfgs(c_opt, deadline - 5.0, [1000])
         
    return c_opt
# EVOLVE-BLOCK-END
