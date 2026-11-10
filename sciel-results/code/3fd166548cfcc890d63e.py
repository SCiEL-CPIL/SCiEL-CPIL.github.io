# EVOLVE-BLOCK-START
import numpy as np
import time

def moving_sofa_2d() -> np.ndarray:
    # Grid definition
    x_grid = np.arange(-3.0, 1.5, 0.01)
    y_grid = np.arange(-1.5, 1.5, 0.01)
    
    X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    n_points = len(X_flat)
    
    def check_inside(x, y, a_deg):
        theta = -np.deg2rad(a_deg)
        c = np.cos(theta)
        s = np.sin(theta)
        
        wx = X_flat - x
        wy = Y_flat - y
        
        u = c * wx - s * wy
        v = s * wx + c * wy
        
        in_r1 = (u >= 0.0) & (u <= 1.0) & (v >= -3.0) & (v <= 1.0)
        in_r2 = (u >= -3.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0)
        
        return in_r1 | in_r2
    
    def evaluate_path(path):
        full_path = np.vstack([
            [[0.0, 0.0, 0.0]],
            path,
            [[-2.0, 0.0, 90.0]]
        ])
        
        mask = np.ones(n_points, dtype=bool)
        
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
                
                inside = check_inside(x, y, a)
                mask &= inside
                
                if not mask.any():
                    return 0.0
        
        return np.sum(mask) * 0.0001
    
    # Start with a good seed path
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
    
    best_path = seed_path.copy()
    best_score = evaluate_path(best_path)
    
    start_time = time.time()
    time_limit = 580.0
    
    # Use Nelder-Mead with more control points
    # Parameters: x,y for each of 6 control points at angles 10,25,40,55,70,85
    def make_path_from_params(params):
        angles = [10.0, 25.0, 40.0, 55.0, 70.0, 85.0]
        path = []
        for i, a in enumerate(angles):
            path.append([params[2*i], params[2*i+1], a])
        return np.array(path, dtype=np.float64)
    
    # Initial parameters from seed path (interpolated)
    # a=10: approx (-0.05, 0.05)
    # a=25: approx (-0.25, 0.25)
    # a=40: (-0.3, 0.3)
    # a=55: (-0.5, 0.2)
    # a=70: approx (-0.55, 0.2)
    # a=85: approx (-0.68, 0.28)
    current_params = np.array([
        -0.05, 0.05,
        -0.25, 0.25,
        -0.3, 0.3,
        -0.5, 0.2,
        -0.55, 0.2,
        -0.68, 0.28
    ])
    
    def nelder_mead(f, x0, max_iter=200, tol=1e-6, time_limit_check=None):
        n = len(x0)
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        # Initial simplex
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
            
            # Sort by function value
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]
            
            # Check convergence
            if abs(fvals[0] - fvals[-1]) < tol:
                break
            
            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            fr = f(xr)
            
            if fr < fvals[0]:
                # Expansion
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
                # Contraction
                xc = centroid + rho * (simplex[-1] - centroid)
                fc = f(xc)
                if fc < fvals[-1]:
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    # Shrink
                    for i in range(1, n+1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        fvals[i] = f(simplex[i])
            
            # Track best
            if fvals[0] < best_val:
                best_val = fvals[0]
                best_params = simplex[0].copy()
        
        return best_params, best_val
    
    # Objective function (negative for minimization)
    def objective(params, soft_beta=0.0):
        path = make_path_from_params(params)
        if soft_beta > 0:
            return -evaluate_path_soft(path, soft_beta)
        return -evaluate_path(path)
    
    def evaluate_path_soft(path, beta=50.0):
        full_path = np.vstack([
            [[0.0, 0.0, 0.0]],
            path,
            [[-2.0, 0.0, 90.0]]
        ])
        
        # Compute log-sum-exp of negative distances for soft min
        # For each point, compute the "violation" at each placement
        # soft_min = -log(sum(exp(-violation_i / beta))) * beta
        
        n_placements = 0
        # First count total placements
        for i in range(len(full_path) - 1):
            p1 = full_path[i]
            p2 = full_path[i+1]
            da = p2[2] - p1[2]
            m = max(1, int(np.ceil(abs(da))))
            n_placements += m + 1
        
        # Accumulate log-sum-exp for each grid point
        # log_sum_exp_i = log(sum_j exp(-d_ij / beta))
        # where d_ij is the minimum distance to boundary for placement j at point i
        # Actually, we want: point is "inside" if it's inside ALL placements
        # soft version: product of sigmoid(distances) or similar
        
        # Simpler: compute the minimum "margin" across all placements, softened
        # For each point, track the min margin (positive = inside)
        # soft_min = -beta * log(sum(exp(-margin_j / beta)))
        
        # But this is expensive. Let's use a different approach:
        # Count how many placements each point fails, then soft-penalize
        fail_count = np.zeros(n_points, dtype=np.float64)
        
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
                
                theta = -np.deg2rad(a)
                c = np.cos(theta)
                s = np.sin(theta)
                
                wx = X_flat - x
                wy = Y_flat - y
                
                u = c * wx - s * wy
                v = s * wx + c * wy
                
                # Distance to boundary of each rectangle (positive = inside)
                # For r1: [0,1] x [-3,1]
                d1_u = np.minimum(u, 1.0 - u)
                d1_v = np.minimum(v + 3.0, 1.0 - v)
                dist_r1 = np.minimum(d1_u, d1_v)
                
                # For r2: [-3,1] x [0,1]
                d2_u = np.minimum(u + 3.0, 1.0 - u)
                d2_v = np.minimum(v, 1.0 - v)
                dist_r2 = np.minimum(d2_u, d2_v)
                
                dist = np.maximum(dist_r1, dist_r2)
                
                # Soft penalty: sigmoid-like
                fail_count += np.exp(-dist * beta)
        
        # Score: points with low fail_count are "safe"
        # Use soft threshold: safe = 1 / (1 + fail_count / n_placements * 2)
        safe = 1.0 / (1.0 + fail_count / (n_placements * 0.5))
        return np.sum(safe) * 0.0001
    
    # Run Nelder-Mead with soft objective first (smoother landscape)
    deadline = start_time + time_limit
    def objective_soft(params):
        return objective(params, soft_beta=50.0)
    
    best_params_soft, best_val_soft = nelder_mead(
        objective_soft, current_params, 
        max_iter=200, tol=1e-6,
        time_limit_check=deadline
    )
    
    # Convert soft result to hard score
    path_soft = make_path_from_params(best_params_soft)
    score_soft = evaluate_path(path_soft)
    
    if score_soft > best_score:
        best_score = score_soft
        best_path = path_soft
        best_params = best_params_soft
    else:
        best_params = current_params
    
    # Now refine with hard objective
    def objective_hard(params):
        return objective(params, soft_beta=0.0)
    
    best_params, best_val = nelder_mead(
        objective_hard, best_params, 
        max_iter=300, tol=1e-7,
        time_limit_check=deadline
    )
    
    # Convert back to score
    if -best_val > best_score:
        best_score = -best_val
        best_path = make_path_from_params(best_params)
    
    # Try a few more random restarts if time allows
    np.random.seed(42)
    for restart in range(10):
        if time.time() > deadline - 30:
            break
        
        # Perturb initial params
        perturbed = best_params + np.random.normal(0, 0.1, size=len(best_params))
        
        bp, bv = nelder_mead(
            objective_hard, perturbed,
            max_iter=100, tol=1e-6,
            time_limit_check=deadline
        )
        
        if -bv > best_score:
            best_score = -bv
            best_path = make_path_from_params(bp)
            best_params = bp
    
    # Final high-precision polish pass
    if time.time() < deadline - 60:
        def objective_polish(params):
            return objective(params, soft_beta=0.0)
        
        bp_polish, bv_polish = nelder_mead(
            objective_polish, best_params,
            max_iter=500, tol=1e-9,
            time_limit_check=deadline - 30
        )
        
        if -bv_polish > best_score:
            best_score = -bv_polish
            best_path = make_path_from_params(bp_polish)
    
    return best_path

# EVOLVE-BLOCK-END
