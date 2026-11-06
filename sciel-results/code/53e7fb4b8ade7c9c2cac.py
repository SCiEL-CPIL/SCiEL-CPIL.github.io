import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

def construct():
    # Start time for deadline guard
    start_time = time.time()
    deadline = 140.0  # 150s limit, keep 10s margin
    
    n = 8
    # Objective function to minimize: negative sum of log distances
    # We will scale points to diameter 1 during evaluation to ensure constraint
    # But for optimization, it's easier to constrain the diameter directly.
    
    def objective(x):
        # x is flattened array of 16 coordinates
        pts = x.reshape(n, 2)
        # Calculate pairwise distances
        # Using broadcasting
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        # Mask diagonal
        mask = ~np.eye(n, dtype=bool)
        dists = dists[mask]
        
        # Avoid log(0)
        dists = np.maximum(dists, 1e-10)
        
        # Sum of log distances
        log_prod = np.sum(np.log(dists))
        return -log_prod

    # Constraints for L-BFGS-B: max distance <= 1
    # We will use this in the local refinement step.
    # For DE, we can just use the objective and hope it finds points with small diameter,
    # or we can add a penalty.
    # Let's use a penalty for DE to keep diameter small.
    
    def objective_penalized(x):
        pts = x.reshape(n, 2)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        mask = ~np.eye(n, dtype=bool)
        dists = dists[mask]
        
        max_dist = np.max(dists)
        
        # Penalty if diameter > 1
        penalty = 0.0
        if max_dist > 1.0:
            penalty = (max_dist - 1.0) ** 2 * 1000.0
            
        dists_safe = np.maximum(dists, 1e-10)
        log_prod = np.sum(np.log(dists_safe))
        
        return -log_prod + penalty

    # DE Bounds: coordinates in [-1, 1] is reasonable for diameter 1
    bounds = [(-1, 1)] * (2 * n)
    
    # Run Differential Evolution
    # Use a moderate number of iterations and popsize to fit in time
    # popsize=20, maxiter=100
    # This might be too slow if not careful.
    # Let's try a smaller popsize and more iterations, or just a few restarts.
    
    best_x = None
    best_val = np.inf
    
    # Check time
    if time.time() - start_time < deadline - 30:
        try:
            # DE is stochastic, so seed is important for reproducibility in tests, but here we just want to find a good one.
            # However, the evaluator runs this once.
            res = differential_evolution(
                objective_penalized, 
                bounds, 
                seed=42, 
                maxiter=50, 
                popsize=15, 
                tol=1e-6,
                polish=False # We will polish manually
            )
            best_x = res.x
            best_val = res.fun
        except Exception as e:
            # Fallback to regular octagon if DE fails
            pass
            
    # If DE didn't run or failed, use regular octagon as initial guess
    if best_x is None:
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        best_x = np.column_stack([np.cos(angles), np.sin(angles)]).flatten()
        
    # Local Refinement using L-BFGS-B with constraints
    # We want to maximize sum log dist, so minimize negative sum log dist.
    # Constraints: dist(i,j) <= 1 for all i<j.
    
    def constraints(x):
        pts = x.reshape(n, 2)
        cons = []
        for i in range(n):
            for j in range(i+1, n):
                d2 = np.sum((pts[i] - pts[j])**2)
                cons.append(1.0 - d2) # d2 <= 1 => 1 - d2 >= 0
        return cons

    # L-BFGS-B does not support equality/inequality constraints directly in the same way as SLSQP.
    # L-BFGS-B only supports bounds.
    # So we should use SLSQP or trust-constr for constrained optimization.
    # SLSQP is generally robust for this size.
    
    def slsqp_objective(x):
        return objective(x) # Minimize -log_prod

    # Check time
    if time.time() - start_time < deadline - 10:
        try:
            res2 = minimize(
                slsqp_objective, 
                best_x, 
                method='SLSQP', 
                bounds=[(-10, 10)] * (2*n), # Wide bounds, constraints will handle diameter
                constraints=[{'type': 'ineq', 'fun': lambda x, i=i, j=j: 1.0 - np.sum((x.reshape(n,2)[i] - x.reshape(n,2)[j])**2)} for i in range(n) for j in range(i+1, n)],
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            if res2.success:
                best_x = res2.x
                best_val = res2.fun
        except Exception as e:
            pass
            
    # Final scaling to ensure diameter is exactly 2 for output?
    # The evaluator scales it, so we can just return the points.
    # But to be safe and clean, let's scale to diameter 2.
    
    pts = best_x.reshape(n, 2)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    mask = ~np.eye(n, dtype=bool)
    max_d = np.max(dists[mask])
    
    if max_d > 1e-10:
        scale = 2.0 / max_d
        pts = pts * scale
        
    # Check distinctness
    # If points are too close, perturb them slightly
    # For a valid solution, they should be distinct.
    
    return [[float(x), float(y)] for x, y in pts]
