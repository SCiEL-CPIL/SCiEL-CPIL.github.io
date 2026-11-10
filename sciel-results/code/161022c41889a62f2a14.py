# EVOLVE-BLOCK-START
import numpy as np
import time

def moving_sofa_2d() -> np.ndarray:
    # Grid definition for final scoring
    x_grid = np.arange(-3.0, 1.5, 0.01)
    y_grid = np.arange(-1.5, 1.5, 0.01)
    
    X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    n_points = len(X_flat)
    
    # Precompute for fast evaluation
    def check_inside(x, y, a_deg):
        theta = -np.deg2rad(a_deg)
        c = np.cos(theta)
        s = np.sin(theta)
        
        wx = X_flat - x
        wy = Y_flat - y
        
        u = wx * c - wy * s
        v = wx * s + wy * c
        
        in_r1 = (u >= 0.0) & (u <= 1.0) & (v >= -3.0) & (v <= 1.0)
        in_r2 = (u >= -3.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0)
        
        return in_r1 | in_r2
    
    def evaluate_path_fast(path):
        """
        Fast path evaluation using vectorized numpy operations.
        """
        full_path = np.vstack([
            [[0.0, 0.0, 0.0]],
            path,
            [[-2.0, 0.0, 90.0]]
        ])
        
        # Collect all (x, y, a) placements
        placements = []
        for i in range(len(full_path) - 1):
            p1 = full_path[i]
            p2 = full_path[i+1]
            da = p2[2] - p1[2]
            m = max(1, int(np.ceil(abs(da))))
            ts = np.linspace(0, 1, m+1)
            for t in ts:
                x = p1[0] + t * (p2[0] - p1[0])
                y = p1[1] + t * (p2[1] - p1[1])
                a = p1[2] + t * (p2[2] - p1[2])
                placements.append((x, y, a))
        
        if len(placements) == 0:
            return float(n_points) * 0.0001
        
        mask = np.ones(n_points, dtype=bool)
        
        # Process placements in batches for memory efficiency
        batch_size = 50
        for start in range(0, len(placements), batch_size):
            end = min(start + batch_size, len(placements))
            batch = placements[start:end]
            
            for x, y, a in batch:
                theta = -np.deg2rad(a)
                c = np.cos(theta)
                s = np.sin(theta)
                
                wx = X_flat - x
                wy = Y_flat - y
                
                u = wx * c - wy * s
                v = wx * s + wy * c
                
                in_r1 = (u >= 0.0) & (u <= 1.0) & (v >= -3.0) & (v <= 1.0)
                in_r2 = (u >= -3.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0)
                
                inside = in_r1 | in_r2
                mask &= inside
                
                if not mask.any():
                    return 0.0
        
        return np.sum(mask) * 0.0001
    
    # Smooth surrogate using soft-min
    def soft_min_surrogate(path, beta=20.0):
        """
        Smooth approximation of the min over all placements of the indicator.
        Uses log-sum-exp trick for numerical stability.
        """
        full_path = np.vstack([
            [[0.0, 0.0, 0.0]],
            path,
            [[-2.0, 0.0, 90.0]]
        ])
        
        # Collect all placements
        placements = []
        for i in range(len(full_path) - 1):
            p1 = full_path[i]
            p2 = full_path[i+1]
            da = p2[2] - p1[2]
            m = max(1, int(np.ceil(abs(da))))
            ts = np.linspace(0, 1, m+1)
            for t in ts:
                x = p1[0] + t * (p2[0] - p1[0])
                y = p1[1] + t * (p2[1] - p1[1])
                a = p1[2] + t * (p2[2] - p1[2])
                placements.append((x, y, a))
        
        if len(placements) == 0:
            return float(n_points)
        
        # For each point, compute the minimum "distance" to being outside
        # across all placements. Use a smooth approximation.
        
        # For each placement, compute a "score" for each point:
        # 1 if inside, 0 if outside. We want the min over placements.
        # Soft min: -log(sum(exp(-beta * score_i))) / beta
        # But score is binary, so let's use a different approach.
        
        # Alternative: for each point, compute the minimum "margin" of being inside.
        # For a point to be inside placement (x,y,a), it must satisfy:
        # (0<=u<=1, -3<=v<=1) or (-3<=u<=1, 0<=v<=1)
        # The margin is the minimum distance to the boundary of the L-shape.
        
        # This is complex. Let's use a simpler approach:
        # Compute the product (or sum of logs) of indicators, which approximates AND.
        
        # For numerical stability, we'll compute the sum of "violations"
        # and use a soft penalty.
        
        total_violation = np.zeros(n_points)
        
        batch_size = 20
        for start in range(0, len(placements), batch_size):
            end = min(start + batch_size, len(placements))
            batch = placements[start:end]
            
            for x, y, a in batch:
                theta = -np.deg2rad(a)
                c = np.cos(theta)
                s = np.sin(theta)
                
                wx = X_flat - x
                wy = Y_flat - y
                
                u = wx * c - wy * s
                v = wx * s + wy * c
                
                # Compute violation: how far outside the L-shape
                # R1: 0<=u<=1, -3<=v<=1
                # R2: -3<=u<=1, 0<=v<=1
                # A point is inside if it's in R1 or R2.
                # Violation for R1: max(0, -u, u-1, v+3, v-1)
                # Violation for R2: max(0, u+3, u-1, -v, v-1)
                # Total violation: min(violation_R1, violation_R2)
                
                viol_r1 = np.maximum.reduce([
                    np.zeros_like(u),
                    -u,
                    u - 1.0,
                    v + 3.0,
                    v - 1.0
                ])
                
                viol_r2 = np.maximum.reduce([
                    np.zeros_like(u),
                    u + 3.0,
                    u - 1.0,
                    -v,
                    v - 1.0
                ])
                
                viol = np.minimum(viol_r1, viol_r2)
                
                # Soft penalty: exp(-beta * viol) approximates indicator
                # Sum of violations (soft)
                total_violation += viol
        
        # The sofa points are those with total_violation close to 0
        # Score approximation: count points with small violation
        # Use a smooth count: sum(exp(-beta * total_violation))
        beta_local = 50.0
        scores = np.exp(-beta_local * total_violation)
        
        return np.sum(scores)
    
    # Build a path analytically based on the sofa geometry
    # The sofa is 4.5 long and 3 wide
    # We need to move it from corridor 1 (horizontal, extending right from origin)
    # to corridor 2 (vertical, extending up from (-2,0))
    
    # Key insight: The sofa needs to pivot around the inner corner
    # At angle 0: sofa is in horizontal corridor, inner corner at (x,y)
    # At angle 90: sofa is in vertical corridor, inner corner at (-2,0)
    
    # The critical constraint is that the sofa must fit in the L-shaped corridor
    # at all times. The corridor has width 1.
    
    # Let's build a smooth path with key waypoints
    # The sofa's long axis rotates from 0 to 90 degrees
    
    # Analytical path: the inner corner traces a curve that keeps the sofa
    # tangent to the outer walls of the L-corridor
    
    # For a sofa of length L=4.5 and width W=3, moving through an L-corridor
    # of width 1, the inner corner path is complex but we can approximate
    
    # Key positions:
    # a=0: sofa in horizontal arm, inner corner at (0,0) [fixed by evaluator]
    # a=90: sofa in vertical arm, inner corner at (-2,0) [fixed by evaluator]
    
    # The sofa has to "turn the corner" - this is the classic sofa problem
    
    # Let's use a path where the sofa slides along the outer wall then pivots
    # The critical region is when the sofa is at ~45 degrees
    
    # Build path with multiple segments
    # Segment 1: slide right while starting to rotate (a: 0 -> 30)
    # Segment 2: pivot through the corner (a: 30 -> 60)  
    # Segment 3: slide down while finishing rotation (a: 60 -> 90)
    
    # Based on geometric analysis, the optimal path should keep the sofa
    # centered in the overlap region. The overlap between start and end
    # is x in [-3,1], y in [0,1].
    
    # Let's try a path that moves the corner along a curve that keeps the
    # L-region covering this overlap region.
    
    def make_path_from_params(params):
        """
        params: array of shape (n, 2) where each row is (x, y) for a fixed angle
        Angles are fixed at specific values.
        """
        # Fixed angles for control points
        angles = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 85.0]
        path = []
        for i, a in enumerate(angles):
            path.append([params[2*i], params[2*i+1], a])
        return np.array(path, dtype=np.float64)
    
    # Initial parameters: based on geometric analysis
    # The corner should move from (0,0) to (-2,0) while rotating 0 to 90
    # A good initial guess: linear interpolation with some curvature
    
    # For each angle, estimate the optimal (x,y) position
    # The L-region must cover the overlap region x in [-3,1], y in [0,1]
    
    # At angle a, the L-region is rotated. To cover the overlap region,
    # the corner position should be chosen so that the L covers the region.
    
    # Simple heuristic: move x from 0 to -2, keep y near 0.5 (center of overlap)
    # But adjust based on angle to keep the L covering the region
    
    def initial_params():
        params = np.zeros((9, 2))
        for i, a in enumerate(angles):
            t = a / 90.0
            # Linear interpolation with slight curve
            x = -2.0 * t
            # Keep y near the center of the overlap region
            y = 0.3  # Slightly above 0 to help cover the overlap
            params[i] = [x, y]
        return params.flatten()
    
    current_params = initial_params()
    
    # Evaluate initial path
    initial_path = make_path_from_params(current_params)
    initial_score = evaluate_path_fast(initial_path)
    
    # Also try the seed path
    seed_path = np.array(
        [
            [-0.1, 0.1, 15.0],
            [-0.2, 0.2, 20.0],
            [-0.3, 0.3, 34.0],
            [-0.3, 0.3, 40.0],
            [-0.4, 0.3, 50.0],
            [-0.5, 0.2, 55.0],
            [-0.6, 0.2, 75.0],
            [-0.7, 0.3, 86.0],
        ],
        dtype=np.float64,
    )
    seed_score = evaluate_path_fast(seed_path)
    
    best_score = max(initial_score, seed_score)
    best_path = initial_path if initial_score > seed_score else seed_path
    
    start_time = time.time()
    time_limit = 580.0
    deadline = start_time + time_limit
    
    # Nelder-Mead optimization
    def nelder_mead(f, x0, max_iter=200, tol=1e-6, time_limit_check=None):
        n = len(x0)
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        simplex = [x0.copy()]
        fvals = [f(x0)]
        
        for i in range(n):
            xi = x0.copy()
            xi[i] += 0.1 if x0[i] != 0 else 0.1
            simplex.append(xi)
            fvals.append(f(xi))
        
        best_params = x0.copy()
        best_val = min(fvals)
        
        for iteration in range(max_iter):
            if time_limit_check is not None and time.time() > time_limit_check:
                break
            
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]
            
            if abs(fvals[0] - fvals[-1]) < tol:
                break
            
            centroid = np.mean(simplex[:-1], axis=0)
            
            xr = centroid + alpha * (centroid - simplex[-1])
            fr = f(xr)
            
            if fr < fvals[0]:
                xe = centroid + gamma * (centroid - simplex[-1])
                fe = f(xe)
                if fe < fr:
                    simplex[-1] = xe
                    fvals[-1] = fe
                else:
                    simplex[-1] = xr
                    fvals[-1] = fr
            elif fr < fvals[-2]:
                simplex[-1] = xr
                fvals[-1] = fr
            else:
                xc = centroid + rho * (simplex[-1] - centroid)
                fc = f(xc)
                if fc < fvals[-1]:
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    for i in range(1, n+1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        fvals[i] = f(simplex[i])
            
            if fvals[0] < best_val:
                best_val = fvals[0]
                best_params = simplex[0].copy()
        
        return best_params, best_val
    
    # Objective using exact evaluation
    def objective_exact(params):
        path = make_path_from_params(params)
        return -evaluate_path_fast(path)
    
    # Run Nelder-Mead with exact objective
    np_params, np_val = nelder_mead(
        objective_exact, current_params,
        max_iter=150, tol=1e-7,
        time_limit_check=deadline
    )
    
    if -np_val > best_score:
        best_score = -np_val
        best_path = make_path_from_params(np_params)
    
    # Try random restarts
    np.random.seed(42)
    for restart in range(8):
        if time.time() > deadline - 30:
            break
        
        perturbed = current_params + np.random.normal(0, 0.15, size=len(current_params))
        bp, bv = nelder_mead(
            objective_exact, perturbed,
            max_iter=60, tol=1e-6,
            time_limit_check=deadline
        )
        
        if -bv > best_score:
            best_score = -bv
            best_path = make_path_from_params(bp)
    
    # Final polish: use tighter tolerance
    if time.time() < deadline - 15:
        # Use best path as starting point
        polish_params = np.array([
            best_path[0, 0], best_path[0, 1],
            best_path[1, 0], best_path[1, 1],
            best_path[2, 0], best_path[2, 1],
            best_path[3, 0], best_path[3, 1],
            best_path[4, 0], best_path[4, 1],
            best_path[5, 0], best_path[5, 1],
            best_path[6, 0], best_path[6, 1],
            best_path[7, 0], best_path[7, 1],
            best_path[8, 0], best_path[8, 1],
        ]) if len(best_path) >= 9 else current_params
        
        # Polish with tighter tolerance
        pp, pv = nelder_mead(
            objective_exact, polish_params,
            max_iter=100, tol=1e-8,
            time_limit_check=deadline
        )
        
        if -pv > best_score:
            best_score = -pv
            best_path = make_path_from_params(pp)
    
    # Safety: ensure we return a valid path
    if len(best_path) < 1:
        best_path = np.array([[-1.0, 0.5, 45.0]])
    
    # Check all values are finite
    if not np.all(np.isfinite(best_path)):
        best_path = np.array([[-1.0, 0.5, 45.0]])
    
    return best_path

# EVOLVE-BLOCK-END
