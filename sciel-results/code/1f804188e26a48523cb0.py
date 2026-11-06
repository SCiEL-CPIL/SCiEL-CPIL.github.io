import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin for 150s limit
    
    # Objective function to maximize: product of all pairwise distances
    # We minimize the negative log of the product, which is -sum(log(distances))
    # This is equivalent to maximizing the product and avoids numerical underflow.
    # The score is Delta / n^n, where Delta is the product of distances after scaling to diameter 2.
    # Let P be the product of distances. Let D be the diameter (max distance).
    # Scaled product Delta = P * (2/D)^n
    # Score = Delta / n^n = P * (2/D)^n / n^n
    # We want to maximize this.
    
    def objective(params):
        if time.time() > deadline:
            return 1e100  # Large penalty to stop optimization
        pts = params.reshape(8, 2)
        
        # Calculate all pairwise distances
        # Use broadcasting for efficiency
        dists = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)
        # We need product of distances for all ordered pairs i != j? 
        # The task says "product over ordered pairs". 
        # Usually, for n points, there are n*(n-1) ordered pairs.
        # However, the regular octagon scores 1.
        # Let's check the regular octagon score.
        # If we use n*(n-1) pairs, the product is (product of unordered pairs)^2.
        # Let's assume "ordered pairs" means i != j.
        # Mask out the diagonal (distance 0)
        mask = ~np.eye(8, dtype=bool)
        dists = dists[mask]
        
        if np.any(dists < 1e-12):
            return 1e100
            
        # Diameter is the max distance
        diam = np.max(dists)
        if diam < 1e-12:
            return 1e100
            
        # Delta = product(dists) * (2/diam)^8
        # Score = Delta / 8^8
        # Maximize Score <=> Maximize Delta <=> Maximize product(dists) / diam^8
        # Minimize -log(product(dists) / diam^8) = -sum(log(dists)) + 8*log(diam)
        
        log_prod = np.sum(np.log(dists))
        log_diam = np.log(diam)
        
        # We want to minimize: 8*log(diam) - log_prod
        return 8 * log_diam - log_prod

    # Initial bounds: |coord| < 10^6, but we can scale down.
    # Let's search in a box [-1, 1] x [-1, 1] initially, as the diameter constraint
    # will effectively scale the points. The relative geometry is what matters.
    # Actually, the objective is scale-invariant?
    # Let's check: if we scale pts by k, dists scale by k.
    # diam scales by k.
    # 8*log(k*diam) - log(k^28 * prod(dists)) = 8*log(k) + 8*log(diam) - 28*log(k) - log(prod)
    # = 8*log(diam) - 20*log(k) - log(prod).
    # This is NOT scale invariant.
    # Wait, the score is Delta / n^n.
    # Delta is the product AFTER scaling to diameter 2.
    # So the value we compute should be invariant to the initial scale of the input points,
    # because we normalize by the diameter.
    # My objective: 8*log(diam) - sum(log(dists)).
    # Let's re-verify.
    # Score = (P * (2/D)^8) / 8^8.
    # P = prod(dists). D = max(dists).
    # log(Score) = log(P) + 8*log(2) - 8*log(D) - 8*log(8).
    # log(P) = sum(log(dists)).
    # So we want to maximize: sum(log(dists)) - 8*log(D).
    # Equivalently, minimize: 8*log(D) - sum(log(dists)).
    # This objective IS scale invariant.
    # If pts -> k*pts, dists -> k*dists, D -> k*D.
    # 8*log(k*D) - sum(log(k*dists)) = 8*log(k) + 8*log(D) - (28*log(k) + sum(log(dists)))
    # = 8*log(D) - sum(log(dists)) - 20*log(k).
    # This is NOT scale invariant. Why?
    # Ah, "product over ordered pairs". There are 8*7 = 56 ordered pairs.
    # So P = prod_{i!=j} d_ij.
    # log(P) = sum_{i!=j} log(d_ij).
    # If we scale by k, log(P) increases by 56*log(k).
    # log(D) increases by log(k).
    # Objective: 8*log(D) - log(P).
    # Change: 8*log(k) - 56*log(k) = -48*log(k).
    # So it depends on scale.
    # This implies the "best" configuration depends on the scale?
    # No, the task says "after scaling the set to diameter 2".
    # This implies we should first normalize the set so that diameter is 2, THEN calculate the product.
    # So the procedure is:
    # 1. Take points.
    # 2. Calculate diameter D.
    # 3. Scale points by 2/D.
    # 4. Calculate product of distances P_scaled.
    # 5. Score = P_scaled / 8^8.
    
    # Let's re-implement objective to reflect this exactly.
    
    def objective_scaled(params):
        if time.time() > deadline:
            return 1e100
        pts = params.reshape(8, 2)
        
        dists = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)
        mask = ~np.eye(8, dtype=bool)
        dists = dists[mask]
        
        if np.any(dists < 1e-12):
            return 1e100
            
        diam = np.max(dists)
        if diam < 1e-12:
            return 1e100
            
        # Scale to diameter 2
        scale = 2.0 / diam
        scaled_dists = dists * scale
        
        # Product of scaled distances
        # log(P_scaled) = sum(log(scaled_dists))
        log_p_scaled = np.sum(np.log(scaled_dists))
        
        # Score = P_scaled / 8^8
        # Maximize Score <=> Maximize log_p_scaled
        # Minimize -log_p_scaled
        
        return -log_p_scaled

    # Bounds for the 16 parameters. 
    # Since the objective is scale-invariant (in terms of the final score), 
    # the absolute scale of the input points doesn't matter, only the relative geometry.
    # However, the optimizer might wander to very large or small scales if not constrained.
    # But since the objective value is the same for any scale, the gradient with respect to scale is zero?
    # Let's check:
    # f(k) = -sum(log(k * d_ij)) = -56*log(k) - sum(log(d_ij)).
    # df/dk = -56/k.
    # This is not zero. The objective as defined in `objective_scaled` is NOT scale invariant in the raw parameter space.
    # Wait. `scaled_dists = dists * (2/diam)`.
    # If I scale `pts` by k, `dists` becomes k*dists, `diam` becomes k*diam.
    # `scale` becomes 2/(k*diam) = (1/k) * (2/diam).
    # `scaled_dists` becomes (k*dists) * (1/k) * (2/diam) = dists * (2/diam).
    # So `scaled_dists` is invariant.
    # Therefore `log_p_scaled` is invariant.
    # Therefore `objective_scaled` is scale invariant.
    # My previous derivation was wrong because I didn't substitute the scale factor correctly into the log sum.
    # Yes, `log_p_scaled` is constant w.r.t global scaling.
    # So we can bound the coordinates to a reasonable range, e.g., [-1, 1].
    
    bounds = [(-1.0, 1.0)] * 16
    
    # Global search using Differential Evolution
    # DE is good for global optimization.
    # maxiter should be sufficient to explore the space.
    # popsize is the size of the population.
    
    best_result = None
    best_score = -1e100
    
    try:
        # Use DE to find a good starting point
        # tol and maxiter control the convergence.
        # We will run DE for a short time, then refine.
        
        # To ensure we don't just return the regular octagon (local max),
        # DE should explore other regions.
        
        res = differential_evolution(
            objective_scaled, 
            bounds, 
            maxiter=50, 
            popsize=20, 
            tol=1e-4, 
            seed=42,
            polish=False  # We will do our own polish
        )
        
        if time.time() > deadline - 10:
            # If we are running out of time, just use the DE result
            best_params = res.x
        else:
            # Refine with L-BFGS-B
            # Start from DE result
            res_local = minimize(
                objective_scaled, 
                res.x, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-12}
            )
            best_params = res_local.x
            best_score = -res_local.fun
            
            # Try a few more random restarts if time permits
            # This helps escape local optima
            num_restarts = 5
            for i in range(num_restarts):
                if time.time() > deadline - 15:
                    break
                    
                # Random initial guess
                x0 = np.random.uniform(-1, 1, 16)
                
                # Local refinement
                res_restart = minimize(
                    objective_scaled, 
                    x0, 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={'maxiter': 50, 'ftol': 1e-12}
                )
                
                if -res_restart.fun > best_score:
                    best_score = -res_restart.fun
                    best_params = res_restart.x

    except Exception:
        # Fallback to regular octagon if something goes wrong
        best_params = np.array([
            [np.cos(2 * np.pi * k / 8), np.sin(2 * np.pi * k / 8)] for k in range(8)
        ]).flatten()
        
    # Construct the final points
    pts = best_params.reshape(8, 2).tolist()
    
    # Ensure distinct points (should be guaranteed by optimizer, but just in case)
    # And ensure |coord| < 10^6 (bounds were -1, 1 so this is safe)
    
    return pts
