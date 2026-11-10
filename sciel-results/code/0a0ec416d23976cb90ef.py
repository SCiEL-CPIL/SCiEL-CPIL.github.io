import numpy as np
import time
from scipy.optimize import differential_evolution, minimize, basinhopping

def construct():
    start_time = time.time()
    deadline = start_time + 148.0  # Leave 2s for safety

    n = 8
    # Precompute indices for unordered pairs
    pair_indices = []
    for i in range(n):
        for j in range(i + 1, n):
            pair_indices.append((i, j))
    pair_indices = np.array(pair_indices)
    idx_i = pair_indices[:, 0]
    idx_j = pair_indices[:, 1]
    num_pairs = len(pair_indices)
    
    # Constants for the objective function
    # log(Score) = 56*log(2) + 2*sum(log(d_ij)) - 56*log(D) - 8*log(8)
    # We minimize -log(Score)
    LOG_2 = np.log(2.0)
    LOG_8 = np.log(8.0)
    CONST_OFFSET = 56.0 * LOG_2 - 8.0 * LOG_8

    def objective(x):
        # Check time
        if time.time() > deadline:
            return 1e10
            
        points = x.reshape(n, 2)
        
        # Calculate pairwise distances
        diff = points[idx_i, :] - points[idx_j, :]
        dists = np.sqrt(np.sum(diff**2, axis=1))
        
        # Check for zero distances (distinct points)
        if np.any(dists < 1e-12):
            return 1e10
            
        D = np.max(dists)
        if D < 1e-12:
            return 1e10
            
        # Sum of logs of distances
        sum_log_d = np.sum(np.log(dists))
        
        # log(Score) = CONST_OFFSET + 2 * sum_log_d - 56 * log(D)
        log_score = CONST_OFFSET + 2.0 * sum_log_d - 56.0 * np.log(D)
        
        return -log_score

    # Bounds
    # The scale is determined by the diameter, so the absolute scale doesn't matter, 
    # but the optimizer needs a finite box. 
    # Since we can scale any configuration to have diameter 2, 
    # we can restrict the search to points within a box of size, say, [-5, 5].
    # Actually, it's better to allow a larger box to accommodate different aspect ratios 
    # before the internal scaling in the objective function normalizes it.
    bounds = [(-5.0, 5.0)] * (2 * n)
    
    # Initial guesses
    # 1. Regular Octagon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x0_oct = np.column_stack([np.cos(angles), np.sin(angles)]).flatten()
    
    # 2. Two Squares (Concentric, Rotated)
    # This configuration often outperforms the regular polygon for n=8.
    # Let's try to optimize the radii and rotation angle of two squares.
    # We can generate a few such starting points.
    
    starts = [x0_oct]
    
    # Generate diverse starts for local optimization
    np.random.seed(42)
    
    # Random perturbations of octagon
    for _ in range(10):
        noise = np.random.normal(0, 0.3, 2 * n)
        starts.append(x0_oct + noise)
        
    # Two squares configurations
    # 4 points at (r1*cos(a_k), r1*sin(a_k)), 4 points at (r2*cos(b_k), r2*sin(b_k))
    # a_k = k*pi/2, b_k = theta + k*pi/2
    for r1 in [0.5, 0.8, 1.0, 1.2, 1.5]:
        for r2 in [0.5, 0.8, 1.0, 1.2, 1.5]:
            for theta in [0.0, np.pi/8, np.pi/4, np.pi/3, np.pi/2]:
                p1 = []
                p2 = []
                for k in range(4):
                    a1 = k * np.pi / 2
                    p1.append([r1 * np.cos(a1), r1 * np.sin(a1)])
                    a2 = theta + k * np.pi / 2
                    p2.append([r2 * np.cos(a2), r2 * np.sin(a2)])
                pts = np.array(p1 + p2)
                starts.append(pts.flatten())
                
    # Random starts
    for _ in range(20):
        starts.append(np.random.uniform(-2, 2, 2 * n))

    best_val = float('inf')
    best_x = x0_oct
    
    # Phase 1: Local Refinement of Initial Starts
    # Use L-BFGS-B for speed, then Powell for precision
    for s in starts:
        if time.time() > deadline - 10.0:
            break
        try:
            # Quick local search
            res = minimize(
                objective, 
                s, 
                method='L-BFGS-B', 
                options={'maxiter': 100, 'ftol': 1e-15}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass

    # Phase 2: Global Search with Differential Evolution
    # If time remains, run DE to find better basins
    if time.time() < deadline - 30.0:
        try:
            # DE is stochastic, so we run it for a fixed number of iterations
            # or until time runs out.
            # We use a smaller population to fit in time.
            res = differential_evolution(
                objective, 
                bounds, 
                maxiter=100, 
                popsize=20, 
                tol=1e-6, 
                seed=42, 
                init='sobol',
                polish=False # We will polish manually
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass
            
    # Phase 3: Basin Hopping around the best solution found
    if time.time() < deadline - 15.0:
        try:
            def step(x):
                # Add noise
                noise = np.random.normal(0, 0.1, len(x))
                return x + noise
                
            bh_res = basinhopping(
                objective, 
                best_x, 
                niter=20, 
                minimizer_kwargs={'method': 'L-BFGS-B', 'options': {'maxiter': 50}},
                stepsize=0.2,
                seed=42
            )
            if bh_res.fun < best_val:
                best_val = bh_res.fun
                best_x = bh_res.x
        except Exception:
            pass
            
    # Phase 4: Final Polish
    # Use Powell for very high precision
    if time.time() < deadline - 2.0:
        try:
            res_final = minimize(
                objective, 
                best_x, 
                method='Powell', 
                options={'maxiter': 1000, 'xatol': 1e-15, 'fatol': 1e-15}
            )
            if res_final.fun < best_val:
                best_val = res_final.fun
                best_x = res_final.x
        except Exception:
            pass

    best_points = best_x.reshape(n, 2)
    return [[float(p[0]), float(p[1])] for p in best_points]
