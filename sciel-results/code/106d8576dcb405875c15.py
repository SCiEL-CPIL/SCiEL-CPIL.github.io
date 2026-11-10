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
    
    def evaluate_path(path):
        """Exact evaluation: count grid points inside all placements."""
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
    
    def soft_surrogate(path, beta=30.0):
        """
        Smooth surrogate using log-sum-exp for soft-min over placements.
        For each grid point, compute violation distance to L-shape boundary,
        then use soft-min across all placements.
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
        
        # For each point, compute the soft-min of "inside-ness" across placements.
        # Use violation-based approach: for each placement, compute distance to outside.
        # soft_min of indicators = -log(sum(exp(-beta*indicator_i)))/beta
        # But indicators are binary. Instead, use violation margins.
        
        # Accumulate log-sum-exp of exp(-beta * viol_i) for each point
        # This gives a smooth count of points that are "inside" all placements.
        
        # For memory efficiency, process in chunks
        chunk_size = 1000
        total_score = 0.0
        
        for start in range(0, n_points, chunk_size):
            end = min(start + chunk_size, n_points)
            Xc = X_flat[start:end]
            Yc = Y_flat[start:end]
            
            # log_sum_exp over all placements of -beta * viol
            # = -beta * soft_min(viol_i)
            # We want sum over points of exp(-beta * soft_min(viol_i))
            # which approximates count of points with small min-violation
            
            # For numerical stability, track max and sum
            max_log = np.full(end - start, -np.inf)
            sum_exp = np.zeros(end - start)
            
            for x, y, a in placements:
                theta = -np.deg2rad(a)
                c = np.cos(theta)
                s = np.sin(theta)
                
                wx = Xc - x
                wy = Yc - y
                
                u = wx * c - wy * s
                v = wx * s + wy * c
                
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
                
                log_val = -beta * viol
                
                # Online log-sum-exp update
                new_max = np.maximum(max_log, log_val)
                # Avoid log(0)
                new_max_safe = np.where(new_max > -1e10, new_max, -1e10)
                sum_exp = sum_exp * np.exp(max_log - new_max_safe) + np.exp(log_val - new_max_safe)
                max_log = new_max
            
            # exp(max_log + log(sum_exp)) = sum of exp(log_val_i)
            # This is sum of exp(-beta * viol_i) over placements
            # We want the soft-min: -log(sum(exp(-beta*viol_i)))/beta
            # But for counting, we want: 1 if soft_min is small, 0 otherwise
            # Use: exp(-beta * (-log(sum_exp + exp(max_log))/beta)) = sum_exp * exp(max_log)
            # Actually, let's just use the sum directly as a score.
            # Points with all violations ~0 will have large sum_exp*exp(max_log)
            
            scores_chunk = sum_exp * np.exp(max_log - 50.0)  # scale down for stability
            total_score += np.sum(np.clip(scores_chunk, 0.0, 1.0))
        
        return total_score
    
    # Path construction
    angles = [10.0, 25.0, 40.0, 55.0, 70.0, 85.0]
    
    def make_path_from_params(params):
        path = []
        for i, a in enumerate(angles):
            path.append([params[2*i], params[2*i+1], a])
        return np.array(path, dtype=np.float64)
    
    def make_simple_path(n_pts=6):
        t = np.linspace(0.1, 0.9, n_pts)
        path = []
        for ti in t:
            path.append([-2.0 * ti, 0.0, 90.0 * ti])
        return np.array(path, dtype=np.float64)
    
    # Initial parameters
    current_params = np.array([
        -2.0*10.0/90.0, 0.0,
        -2.0*25.0/90.0, 0.0,
        -2.0*40.0/90.0, 0.0,
        -2.0*55.0/90.0, 0.0,
        -2.0*70.0/90.0, 0.0,
        -2.0*85.0/90.0, 0.0,
    ])
    
    # Try seed path
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
    
    simple_path = make_simple_path(6)
    simple_score = evaluate_path(simple_path)
    seed_score = evaluate_path(seed_path)
    
    best_path = seed_path if seed_score >= simple_score else simple_path
    best_score = max(seed_score, simple_score)
    
    start_time = time.time()
    deadline = start_time + 570.0  # 60s safety margin
    
    def nelder_mead(f, x0, max_iter=200, tol=1e-6, deadline_check=None):
        n = len(x0)
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        simplex = [x0.copy()]
        fvals = [f(x0)]
        
        for i in range(n):
            xi = x0.copy()
            xi[i] += 0.1
            simplex.append(xi)
            fvals.append(f(xi))
        
        best_params = x0.copy()
        best_val = min(fvals)
        
        for iteration in range(max_iter):
            if deadline_check is not None and time.time() > deadline_check:
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
    
    # Phase 1: Coarse optimization with soft surrogate (beta=15)
    if time.time() < deadline - 120:
        def obj_soft_coarse(params):
            path = make_path_from_params(params)
            return -soft_surrogate(path, beta=15.0)
        
        p1, v1 = nelder_mead(obj_soft_coarse, current_params, max_iter=100, tol=1e-5, deadline_check=deadline - 120)
        # Verify with exact
        exact_s = evaluate_path(make_path_from_params(p1))
        if exact_s > best_score:
            best_score = exact_s
            best_path = make_path_from_params(p1)
            current_params = p1
    
    # Phase 2: Medium optimization with soft surrogate (beta=30)
    if time.time() < deadline - 80:
        def obj_soft_med(params):
            path = make_path_from_params(params)
            return -soft_surrogate(path, beta=30.0)
        
        p2, v2 = nelder_mead(obj_soft_med, current_params, max_iter=80, tol=1e-6, deadline_check=deadline - 80)
        exact_s = evaluate_path(make_path_from_params(p2))
        if exact_s > best_score:
            best_score = exact_s
            best_path = make_path_from_params(p2)
            current_params = p2
    
    # Phase 3: Exact optimization with Nelder-Mead
    if time.time() < deadline - 40:
        def obj_exact(params):
            path = make_path_from_params(params)
            return -evaluate_path(path)
        
        p3, v3 = nelder_mead(obj_exact, current_params, max_iter=100, tol=1e-7, deadline_check=deadline - 40)
        if -v3 > best_score:
            best_score = -v3
            best_path = make_path_from_params(p3)
            current_params = p3
    
    # Phase 4: Random restarts with exact evaluation
    np.random.seed(42)
    for restart in range(10):
        if time.time() > deadline - 20:
            break
        
        perturbed = current_params + np.random.normal(0, 0.1, size=len(current_params))
        
        def obj_exact_r(params):
            path = make_path_from_params(params)
            return -evaluate_path(path)
        
        bp, bv = nelder_mead(obj_exact_r, perturbed, max_iter=40, tol=1e-6, deadline_check=deadline - 20)
        
        if -bv > best_score:
            best_score = -bv
            best_path = make_path_from_params(bp)
            current_params = bp
    
    # Phase 5: Final polish with very tight tolerance
    if time.time() < deadline - 5:
        def obj_exact_polish(params):
            path = make_path_from_params(params)
            return -evaluate_path(path)
        
        pp, pv = nelder_mead(obj_exact_polish, current_params, max_iter=50, tol=1e-9, deadline_check=deadline - 5)
        
        if -pv > best_score:
            best_score = -pv
            best_path = make_path_from_params(pp)
    
    # Safety checks
    if len(best_path) < 1:
        best_path = make_simple_path(5)
    
    if not np.all(np.isfinite(best_path)):
        best_path = make_simple_path(5)
    
    return best_path

# EVOLVE-BLOCK-END
