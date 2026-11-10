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
    
    # Precompute grid point coordinates for fast rotation
    # Store as complex for efficient rotation: z = x + iy
    Z = X_flat + 1j * Y_flat
    
    def get_placements(path):
        """Expand path with interpolation into full placement list."""
        full = np.vstack([[[0.0, 0.0, 0.0]], path, [[-2.0, 0.0, 90.0]]])
        placements = []
        for i in range(len(full) - 1):
            p1, p2 = full[i], full[i + 1]
            da = p2[2] - p1[2]
            m = max(1, int(np.ceil(abs(da))))
            ts = np.linspace(0, 1, m + 1)
            for t in ts:
                placements.append((p1[0] + t * (p2[0] - p1[0]),
                                   p1[1] + t * (p2[1] - p1[1]),
                                   p1[2] + t * (p2[2] - p1[2])))
        return placements
    
    def evaluate_path_exact(path):
        """Count grid points inside ALL placements (exact)."""
        placements = get_placements(path)
        if not placements:
            return float(n_points) * 0.0001
        
        mask = np.ones(n_points, dtype=bool)
        for (x, y, a) in placements:
            theta = -np.deg2rad(a)
            c, s = np.cos(theta), np.sin(theta)
            # Rotate: u = x*c - y*s, v = x*s + y*c  (where x,y are relative)
            wx = X_flat - x
            wy = Y_flat - y
            u = wx * c - wy * s
            v = wx * s + wy * c
            in_r1 = (u >= 0.0) & (u <= 1.0) & (v >= -3.0) & (v <= 1.0)
            in_r2 = (u >= -3.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0)
            mask &= (in_r1 | in_r2)
            if not mask.any():
                return 0.0
        return np.count_nonzero(mask) * 0.0001
    
    def evaluate_path_vectorized(path):
        """Vectorized evaluation: process all placements at once per chunk."""
        placements = get_placements(path)
        n_pl = len(placements)
        if n_pl == 0:
            return float(n_points) * 0.0001
        
        # Process in chunks of grid points for memory
        chunk = 500
        total_inside = 0
        
        for start in range(0, n_points, chunk):
            end = min(start + chunk, n_points)
            Xc = X_flat[start:end]
            Yc = Y_flat[start:end]
            
            # For each placement, compute inside mask for this chunk
            # Then AND across all placements
            inside_all = np.ones(end - start, dtype=bool)
            
            for (x, y, a) in placements:
                theta = -np.deg2rad(a)
                c, s = np.cos(theta), np.sin(theta)
                wx = Xc - x
                wy = Yc - y
                u = wx * c - wy * s
                v = wx * s + wy * c
                in_r1 = (u >= 0.0) & (u <= 1.0) & (v >= -3.0) & (v <= 1.0)
                in_r2 = (u >= -3.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0)
                inside_all &= (in_r1 | in_r2)
                if not inside_all.any():
                    break
            
            total_inside += np.count_nonzero(inside_all)
        
        return total_inside * 0.0001
    
    def soft_surrogate(path, beta=25.0):
        """
        Smooth surrogate: for each grid point, compute soft-min of violation
        across all placements. Score = sum of exp(-beta * soft_min_viol).
        """
        placements = get_placements(path)
        n_pl = len(placements)
        if n_pl == 0:
            return float(n_points)
        
        chunk = 800
        total = 0.0
        
        for start in range(0, n_points, chunk):
            end = min(start + chunk, n_points)
            Xc = X_flat[start:end]
            Yc = Y_flat[start:end]
            
            # log-sum-exp accumulator
            max_log = np.full(end - start, -1e18)
            sum_exp = np.zeros(end - start)
            
            for (x, y, a) in placements:
                theta = -np.deg2rad(a)
                c, s = np.cos(theta), np.sin(theta)
                wx = Xc - x
                wy = Yc - y
                u = wx * c - wy * s
                v = wx * s + wy * c
                
                # Violation distance to L-shape (0 if inside, >0 if outside)
                viol_r1 = np.maximum.reduce([
                    np.zeros_like(u), -u, u - 1.0, v + 3.0, v - 1.0
                ])
                viol_r2 = np.maximum.reduce([
                    np.zeros_like(u), u + 3.0, u - 1.0, -v, v - 1.0
                ])
                viol = np.minimum(viol_r1, viol_r2)
                
                log_val = -beta * viol
                # Online log-sum-exp
                new_max = np.maximum(max_log, log_val)
                # Avoid issues with -inf
                safe_max = np.where(new_max > -1e17, new_max, -1e17)
                sum_exp = sum_exp * np.exp(max_log - safe_max) + np.exp(log_val - safe_max)
                max_log = new_max
            
            # sum of exp(-beta*viol_i) over placements
            # Points deep inside all placements: viol~0, sum ~ n_pl
            # Points outside at least one: that placement has large viol, contributes ~0
            scores = sum_exp * np.exp(max_log - 50.0)
            total += np.sum(np.clip(scores, 0.0, 1.0))
        
        return total
    
    # Path parameterization: fixed angles, optimize (x, y) per angle
    angles = np.array([8.0, 20.0, 33.0, 47.0, 61.0, 75.0, 86.0])
    n_angles = len(angles)
    
    def params_to_path(params):
        path = np.empty((n_angles, 3))
        path[:, 0] = params[0:n_angles]
        path[:, 1] = params[n_angles:2*n_angles]
        path[:, 2] = angles
        return path
    
    def path_to_params(path):
        return np.concatenate([path[:, 0], path[:, 1]])
    
    # Initial guess: linear interpolation from (0,0,0) to (-2,0,90)
    t_init = angles / 90.0
    init_params = np.concatenate([-2.0 * t_init, np.zeros(n_angles)])
    
    # A few hand-crafted starting paths
    def make_arc_path(n_pts=7):
        t = np.linspace(0.05, 0.95, n_pts)
        x = -2.0 * t
        y = 0.3 * np.sin(np.pi * t)
        a = 90.0 * t
        path = np.column_stack([x, y, a])
        # Adjust angles to match our fixed angles
        for i in range(n_angles):
            path[i, 2] = angles[i]
        # Rescale x,y to match
        t_match = angles / 90.0
        path[:, 0] = -2.0 * t_match
        path[:, 1] = 0.3 * np.sin(np.pi * t_match)
        return path
    
    def make_s_curve(n_pts=7):
        t_match = angles / 90.0
        x = -2.0 * t_match
        y = 0.4 * np.sin(2 * np.pi * t_match)
        path = np.column_stack([x, y, angles])
        return path
    
    # Evaluate initial candidates
    candidates = [
        params_to_path(init_params),
        make_arc_path(),
        make_s_curve(),
    ]
    
    start_time = time.time()
    deadline = start_time + 580.0  # 50s safety margin
    
    best_score = -1.0
    best_path = candidates[0]
    for c in candidates:
        s = evaluate_path_exact(c)
        if s > best_score:
            best_score = s
            best_path = c
    
    current_params = path_to_params(best_path)
    
    def nelder_mead(f, x0, max_iter=150, tol=1e-7, deadline_check=None):
        n = len(x0)
        alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
        
        simplex = [x0.copy()]
        fvals = [f(x0)]
        for i in range(n):
            xi = x0.copy()
            xi[i] += 0.05
            simplex.append(xi)
            fvals.append(f(xi))
        
        best_val = min(fvals)
        best_x = simplex[np.argmin(fvals)].copy()
        
        for it in range(max_iter):
            if deadline_check is not None and time.time() > deadline_check:
                break
            
            order = np.argsort(fvals)
            simplex = [simplex[i].copy() for i in order]
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
                    simplex[-1] = xe; fvals[-1] = fe
                else:
                    simplex[-1] = xr; fvals[-1] = fr
            elif fr < fvals[-2]:
                simplex[-1] = xr; fvals[-1] = fr
            else:
                xc = centroid + rho * (simplex[-1] - centroid)
                fc = f(xc)
                if fc < fvals[-1]:
                    simplex[-1] = xc; fvals[-1] = fc
                else:
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        fvals[i] = f(simplex[i])
            
            if fvals[0] < best_val:
                best_val = fvals[0]
                best_x = simplex[0].copy()
        
        return best_x, best_val
    
    # Phase 1: Coarse soft surrogate optimization
    if time.time() < deadline - 150:
        def obj_soft_coarse(params):
            return -soft_surrogate(params_to_path(params), beta=15.0)
        
        p1, v1 = nelder_mead(obj_soft_coarse, current_params, max_iter=80, tol=1e-5, deadline_check=deadline - 150)
        s1 = evaluate_path_exact(params_to_path(p1))
        if s1 > best_score:
            best_score = s1
            best_path = params_to_path(p1)
            current_params = p1
    
    # Phase 2: Medium soft surrogate
    if time.time() < deadline - 100:
        def obj_soft_med(params):
            return -soft_surrogate(params_to_path(params), beta=30.0)
        
        p2, v2 = nelder_mead(obj_soft_med, current_params, max_iter=60, tol=1e-6, deadline_check=deadline - 100)
        s2 = evaluate_path_exact(params_to_path(p2))
        if s2 > best_score:
            best_score = s2
            best_path = params_to_path(p2)
            current_params = p2
    
    # Phase 3: Exact Nelder-Mead
    if time.time() < deadline - 50:
        def obj_exact(params):
            return -evaluate_path_exact(params_to_path(params))
        
        p3, v3 = nelder_mead(obj_exact, current_params, max_iter=80, tol=1e-8, deadline_check=deadline - 50)
        s3 = -v3
        if s3 > best_score:
            best_score = s3
            best_path = params_to_path(p3)
            current_params = p3
    
    # Phase 4: Random restarts
    rng = np.random.default_rng(42)
    for restart in range(8):
        if time.time() > deadline - 20:
            break
        noise_scale = 0.15 / (restart + 1)
        perturbed = current_params + rng.normal(0, noise_scale, size=len(current_params))
        
        def obj_exact_r(params):
            return -evaluate_path_exact(params_to_path(params))
        
        bp, bv = nelder_mead(obj_exact_r, perturbed, max_iter=30, tol=1e-7, deadline_check=deadline - 20)
        if -bv > best_score:
            best_score = -bv
            best_path = params_to_path(bp)
            current_params = bp
    
    # Phase 5: Final polish
    if time.time() < deadline - 5:
        def obj_polish(params):
            return -evaluate_path_exact(params_to_path(params))
        
        pp, pv = nelder_mead(obj_polish, current_params, max_iter=40, tol=1e-10, deadline_check=deadline - 5)
        if -pv > best_score:
            best_score = -pv
            best_path = params_to_path(pp)
    
    # Safety
    if len(best_path) < 1 or not np.all(np.isfinite(best_path)):
        best_path = params_to_path(init_params)
    
    return best_path

# EVOLVE-BLOCK-END
