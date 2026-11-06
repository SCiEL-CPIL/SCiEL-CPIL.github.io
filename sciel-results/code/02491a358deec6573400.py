"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
def construct():
    import numpy as np
    from scipy.optimize import differential_evolution
    
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    def objective(x):
        """
        x is a flat array of 16 floats (8 points in 2D).
        Returns negative log of product of distances (to maximize product, minimize this).
        We also need to enforce diameter <= 2, but the scaling is done externally.
        Actually, the problem says: "score is Delta/n^n, where Delta is the product over ordered pairs of distances after scaling the set to diameter 2".
        So we can optimize the geometry directly, then scale.
        
        To avoid singularities (points coinciding), we can work with a fixed scale during optimization?
        No, the objective is scale-invariant if we normalize.
        Let's define the objective as:
        1. Compute all pairwise distances.
        2. Find max distance D.
        3. If D == 0, return large penalty.
        4. Scale points so max distance is 2. (Divide by D/2)
        5. Compute product of all 8*7=56 ordered pair distances (or just 28 unordered, squared).
           Product P = prod_{i<j} d_{ij}
           Since we scale to diameter 2, the distances are d_{ij} * (2/D).
           So P_scaled = prod_{i<j} (d_{ij} * 2/D) = (2/D)^28 * prod_{i<j} d_{ij}.
           We want to maximize P_scaled.
           This is equivalent to maximizing (prod d_{ij}) / D^28.
           Let's maximize log(prod d_{ij}) - 28 * log(D).
        """
        if time.time() > deadline:
            return 1e100 # Penalty to stop optimization? Or just return current best?
            # Differential evolution might not handle dynamic stopping well.
            # Better to check time in the wrapper.
            
        pts = x.reshape(8, 2)
        # Compute pairwise distances
        # Using broadcasting
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        
        # Upper triangle for unique pairs
        iu = np.triu_indices(8, k=1)
        d = dists[iu]
        
        # Check for zero distances (coincident points)
        if np.min(d) < 1e-12:
            return 1e100
        
        D = np.max(d)
        
        # log objective: sum(log(d_i)) - 28 * log(D)
        # We want to maximize this, so differential_evolution minimizes negative.
        val = np.sum(np.log(d)) - 28.0 * np.log(D)
        return -val

    # Initial guess: Regular Octagon
    # But we can randomize starting points to find better local optima?
    # Differential Evolution is a global optimizer, so bounds are important.
    # Since the problem is scale invariant, we can fix one point or normalize the scale during optimization?
    # If we don't fix scale, the optimizer might drift to infinity or zero.
    # However, the objective is scale invariant: f(k*x) = f(x).
    # This creates a "valley" of constant value. DE might struggle or wander.
    # Better approach: Fix the scale.
    # Let's constrain the points to lie within a box, say [-2, 2]x[-2, 2].
    # Or better: Fix one point at (0,0) and another at (2,0) to fix scale and rotation?
    # No, that removes degrees of freedom (rotation and translation).
    # We have 16 coordinates.
    # Translation invariance: 2 DOF.
    # Rotation invariance: 1 DOF.
    # Scale invariance: 1 DOF.
    # So we can fix 4 DOF.
    # Let's fix P0 = (0,0).
    # Let's fix P1 = (2,0) ? No, that forces diameter to be exactly 2 and aligned.
    # But the diameter might not be between P0 and P1 in the optimal solution.
    # However, we can just optimize the 16 coordinates and let the objective handle scaling.
    # To prevent drift, we can add a small penalty for large coordinates, or use bounds.
    # Bounds: [-10, 10] for each coordinate.
    
    bounds = [(-10, 10)] * 16
    
    # Regular octagon initial guess
    angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
    init_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    # Scale to diameter 2? Regular octagon on unit circle has diameter 2.
    # So init_pts is already diameter 2.
    init_x = init_pts.flatten()
    
    # Use Differential Evolution
    # We need to be careful with the time limit.
    # DE can be slow.
    # Let's try a simpler gradient-based method from multiple starts?
    # Or use scipy.optimize.minimize with L-BFGS-B from multiple random starts?
    # L-BFGS-B is fast but local.
    
    # Strategy:
    # 1. Try DE with low population size and few generations to see if it finds something.
    # 2. If time permits, refine with L-BFGS-B.
    
    best_score = 1.0 # Regular octagon score
    best_pts = init_pts.copy()
    
    def score_pts(pts):
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        iu = np.triu_indices(8, k=1)
        d = dists[iu]
        if np.min(d) < 1e-12:
            return 0.0
        D = np.max(d)
        # Product of scaled distances
        # P_scaled = prod(d_i * 2/D) = (2/D)^28 * prod(d_i)
        log_P = 28*np.log(2) - 28*np.log(D) + np.sum(np.log(d))
        P = np.exp(log_P)
        return P / (8**8) # Score = Delta / n^n. Delta is product of ordered pairs?
        # "product over ordered pairs"
        # Ordered pairs (i,j) with i!=j. There are 8*7=56 pairs.
        # Unordered pairs i<j. There are 28 pairs.
        # Product_ordered = prod_{i!=j} d_{ij} = (prod_{i<j} d_{ij})^2
        # So Delta = (prod_{i<j} d_{ij})^2
        # Score = (prod_{i<j} d_{ij})^2 / 8^8
        # My previous calculation:
        # P_scaled_unordered = (2/D)^28 * prod_{i<j} d_{ij}
        # Delta_scaled = (P_scaled_unordered)^2
        # Score = Delta_scaled / 8^8
        # Let's re-calculate in score_pts to be sure.
        # The regular octagon scores exactly 1.
        # Let's test regular octagon.
        # d_ij for regular octagon on unit circle.
        # Distances are 2*sin(k*pi/8) for k=1..4.
        # Counts:
        # k=1: 8 pairs. d=2*sin(pi/8)
        # k=2: 8 pairs. d=2*sin(2pi/8)
        # k=3: 8 pairs. d=2*sin(3pi/8)
        # k=4: 4 pairs. d=2*sin(4pi/8)=2
        # Product_unordered = (2*sin(pi/8))^8 * (2*sin(2pi/8))^8 * (2*sin(3pi/8))^8 * 2^4
        # Let's just compute numerically in the function.
        
    def actual_score(pts):
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        iu = np.triu_indices(8, k=1)
        d = dists[iu]
        if np.min(d) < 1e-12:
            return 0.0
        D = np.max(d)
        # Scale to diameter 2
        scaled_d = d * (2.0 / D)
        prod_unordered = np.prod(scaled_d)
        # Delta is product over ordered pairs
        # Ordered pairs include (i,j) and (j,i).
        # So product_ordered = prod_unordered^2
        Delta = prod_unordered ** 2
        n = 8
        score = Delta / (n ** n)
        return score

    # Verify regular octagon score
    reg_score = actual_score(init_pts)
    # Should be 1.0
    # print(reg_score)
    
    # Objective for DE: minimize -actual_score?
    # No, DE works better on smooth functions. -log(score) is better.
    # score = exp(2 * (sum(log(d_scaled)) - 8*log(8)))
    # log(score) = 2 * (sum(log(d) - log(D) + log(2)) - 8*log(8))
    #             = 2 * (sum(log(d)) - 28*log(D) + 28*log(2) - 8*log(8))
    # To maximize score, maximize log(score).
    # Minimize -log(score).
    
    def de_objective(x):
        pts = x.reshape(8, 2)
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        iu = np.triu_indices(8, k=1)
        d = dists[iu]
        if np.min(d) < 1e-12:
            return 1e100
        D = np.max(d)
        log_score = 2.0 * (np.sum(np.log(d)) - 28.0*np.log(D) + 28.0*np.log(2.0) - 8.0*np.log(8.0))
        return -log_score

    # Run DE
    # popsize=15, maxiter=100 might be too slow.
    # Let's try popsize=10, maxiter=50.
    # And check time.
    
    try:
        # We will run DE in a way that we can interrupt it.
        # scipy DE doesn't have a built-in callback to stop based on time easily,
        # but we can wrap the objective to check time and return a penalty,
        # or just run a small number of iterations.
        
        # Let's use a smaller budget for DE.
        # If DE is too slow, we fall back to L-BFGS-B.
        
        # Actually, L-BFGS-B from multiple random starts is often faster and effective for this type of problem.
        # Let's do that.
        
        from scipy.optimize import minimize
        
        best_val = -np.inf
        best_x = init_x.copy()
        
        # 1. Regular Octagon
        val = de_objective(init_x)
        if val < best_val:
            best_val = val
            best_x = init_x.copy()
        
        # 2. Random starts
        # Generate random points in a box, scale them, and optimize.
        n_starts = 20
        for i in range(n_starts):
            if time.time() > deadline - 5.0:
                break
            
            # Random points
            # To avoid coincident points, ensure some spread.
            x_rand = np.random.uniform(-2, 2, 16)
            
            # Optimize
            res = minimize(de_objective, x_rand, method='L-BFGS-B', bounds=bounds, options={'maxiter': 100, 'ftol': 1e-12})
            
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
            
            # Check time
            if time.time() > deadline - 10.0:
                break
        
        # 3. Try some structured perturbations of the regular octagon
        # e.g. move vertices radially
        for i in range(10):
            if time.time() > deadline - 5.0:
                break
            
            # Perturb radii
            angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
            radii = np.ones(8) * (1.0 + 0.2 * np.random.randn(8))
            # Ensure positive
            radii = np.abs(radii) + 0.1
            pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
            x_start = pts.flatten()
            
            res = minimize(de_objective, x_start, method='L-BFGS-B', bounds=bounds, options={'maxiter': 100, 'ftol': 1e-12})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
                
        # 4. Try perturbing angles
        for i in range(10):
            if time.time() > deadline - 5.0:
                break
            
            angles = np.linspace(0, 2*np.pi, 8, endpoint=False) + 0.1 * np.random.randn(8)
            radii = np.ones(8)
            pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
            x_start = pts.flatten()
            
            res = minimize(de_objective, x_start, method='L-BFGS-B', bounds=bounds, options={'maxiter': 100, 'ftol': 1e-12})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()

        # 5. Final refinement with tighter tolerance
        if time.time() < deadline - 2.0:
            res = minimize(de_objective, best_x, method='L-BFGS-B', bounds=bounds, options={'maxiter': 500, 'ftol': 1e-15})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()

        best_pts = best_x.reshape(8, 2)
        
        # Ensure distinct points
        # If any points are too close, it's invalid.
        # The objective penalizes this, so it should be fine.
        
        return best_pts.tolist()
        
    except Exception as e:
        # Fallback to regular octagon
        return init_pts.tolist()

# EVOLVE-BLOCK-END
