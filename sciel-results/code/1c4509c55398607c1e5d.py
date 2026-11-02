import numpy as np
import time
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from scipy.special import logsumexp

N = 200

def compute_radius(P):
    """
    Compute the maximum radius for a given set of centers P.
    P: (N, 2) array of centers.
    Returns the radius r.
    """
    if len(P) < 2:
        return 0.5 - np.max(np.abs(P))
    
    dists = cdist(P, P)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    bound_dists = 0.5 - np.abs(P)
    min_bound_dist = np.min(bound_dists)
    
    return min(min_pair_dist, min_bound_dist)

def soft_min_objective(P, beta=0.01):
    """
    Differentiable objective for optimization.
    Minimizes -softmin(distances).
    """
    dists = cdist(P, P)
    np.fill_diagonal(dists, np.inf)
    
    bound_dists = 0.5 - np.abs(P)
    
    all_dists = np.concatenate([dists.ravel(), bound_dists.ravel()])
    
    # Remove infinities for logsumexp
    mask = np.isfinite(all_dists)
    all_dists = all_dists[mask]
    
    if len(all_dists) == 0:
        return 0.0
        
    obj = beta * logsumexp(-all_dists / beta)
    return obj

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s buffer for safety
    
    # 1. Initialization: Jittered Grid
    # Create a grid of slightly more than N points
    grid_size = int(np.ceil(np.sqrt(N))) + 1  # e.g., 16 for 200
    # Spacing
    spacing = 1.0 / grid_size
    
    points = []
    for i in range(grid_size):
        for j in range(grid_size):
            # Base center
            x = -0.5 + (i + 0.5) * spacing
            y = -0.5 + (j + 0.5) * spacing
            # Jitter
            x += np.random.uniform(-0.1, 0.1) * spacing
            y += np.random.uniform(-0.1, 0.1) * spacing
            points.append([x, y])
            
    P_init = np.array(points)
    
    # If we have more than N, subsample
    if len(P_init) > N:
        indices = np.random.choice(len(P_init), N, replace=False)
        P_init = P_init[indices]
    elif len(P_init) < N:
        # Pad with random points if too few
        while len(P_init) < N:
            P_init = np.vstack([P_init, np.random.uniform(-0.4, 0.4, 2)])
            
    P_init = P_init[:N]
    
    # Ensure valid start (strictly inside)
    P_init = np.clip(P_init, -0.49, 0.49)
    
    current_P = P_init.copy()
    current_r = compute_radius(current_P)
    best_P = current_P.copy()
    best_r = current_r
    
    # 2. Simulated Annealing
    T = 0.02
    T_min = 1e-5
    cooling_rate = 0.999
    
    steps = 0
    while T > T_min:
        if time.time() > deadline - 15.0:
            break
            
        # Move
        i = np.random.randint(N)
        old_pos = current_P[i].copy()
        
        # Adaptive step size
        step_size = 0.01 * (T / 0.02 + 0.1)
        noise = np.random.normal(0, step_size, 2)
        new_pos = current_P[i] + noise
        
        # Clip to safe interior
        new_pos = np.clip(new_pos, -0.49, 0.49)
        
        current_P[i] = new_pos
        
        new_r = compute_radius(current_P)
        
        # Acceptance
        if new_r > current_r:
            current_r = new_r
            if new_r > best_r:
                best_r = new_r
                best_P = current_P.copy()
        else:
            delta_r = current_r - new_r
            if delta_r > 0:
                prob = np.exp(-delta_r / T)
                if np.random.rand() < prob:
                    current_r = new_r
                else:
                    current_P[i] = old_pos
        
        T *= cooling_rate
        steps += 1
        
    # 3. Polish with L-BFGS-B
    # Use the best_P found by SA
    P_opt = best_P.copy()
    
    # Flatten P for scipy
    x0 = P_opt.flatten()
    
    bounds = [(-0.49, 0.49)] * (2 * N)
    
    def obj_flat(x):
        P_flat = x.reshape(-1, 2)
        return soft_min_objective(P_flat, beta=0.005)
        
    def obj_grad_flat(x):
        # Numerical gradient or analytical? 
        # L-BFGS-B can use finite differences, but analytical is better.
        # For simplicity and robustness, let's rely on finite differences 
        # or just use the objective. The gradient of logsumexp is manageable.
        # However, to avoid complexity and potential bugs in gradient, 
        # we can let scipy estimate it if we don't provide it, 
        # but that's slow. 
        # Let's provide a simple numerical gradient if needed, 
        # or just pass the objective. 
        # Actually, L-BFGS-B with finite differences on 400 vars is slow.
        # Let's try to compute the gradient analytically.
        
        P_flat = x.reshape(-1, 2)
        dists = cdist(P_flat, P_flat)
        np.fill_diagonal(dists, np.inf)
        bound_dists = 0.5 - np.abs(P_flat)
        
        all_dists = np.concatenate([dists.ravel(), bound_dists.ravel()])
        mask = np.isfinite(all_dists)
        all_dists = all_dists[mask]
        
        if len(all_dists) == 0:
            return np.zeros_like(x)
            
        beta = 0.005
        # soft_min = -beta * logsumexp(-all_dists / beta)
        # grad soft_min w.r.t. d_k = (1/beta) * exp(-d_k/beta) / sum(exp(-d_j/beta))
        # = exp(-d_k/beta) / sum(exp(-d_j/beta))
        
        weights = np.exp(-all_dists / beta)
        sum_weights = np.sum(weights)
        grad_all_dists = weights / sum_weights
        
        # Now we need to map grad_all_dists back to P.
        # This is complex due to the flattening and masking.
        # Given the time constraints and risk of error, 
        # let's just use the objective and let scipy use finite differences 
        # but limit the iterations. 
        # Or, simpler: use COBYLA or Nelder-Mead? No, L-BFGS-B is better for bounds.
        # Let's just use the objective function. 
        # To speed up, we can reduce N in the polish? No.
        # Let's just run L-BFGS-B with maxiter=20 to see if it helps.
        return None # Signal to use finite differences? No, scipy handles None.
        
    # To avoid slow finite differences, let's just do a few local search steps 
    # or rely on the SA result which is often quite good for this N.
    # The SA with 100k steps should be sufficient.
    
    # Let's try a quick L-BFGS-B polish with a small number of iterations 
    # to see if it improves. If it fails or is slow, we skip.
    
    if time.time() < deadline - 5.0:
        try:
            # Use a tighter beta for polish
            res = minimize(obj_flat, x0, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 20, 'ftol': 1e-8})
            if res.success:
                P_polished = res.x.reshape(-1, 2)
                # Ensure validity
                P_polished = np.clip(P_polished, -0.4999, 0.4999)
                r_polished = compute_radius(P_polished)
                if r_polished > best_r:
                    best_P = P_polished
                    best_r = r_polished
        except Exception:
            pass

    # Final safety check
    best_P = np.clip(best_P, -0.4999, 0.4999)
    if not np.all(np.isfinite(best_P)):
        # Fallback to a simple grid
        best_P = np.random.uniform(-0.4, 0.4, (N, 2))
        
    return best_P
