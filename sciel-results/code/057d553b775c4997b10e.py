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
    
    # Number of control points
    n_ctrl = 12
    # Fixed angles for control points (evenly spaced from ~5 to ~85 degrees)
    ctrl_angles = np.linspace(5.0, 85.0, n_ctrl)
    
    def make_path_from_params(params):
        """params: array of shape (2*n_ctrl,) with [x0, y0, x1, y1, ...]"""
        path = []
        for i in range(n_ctrl):
            path.append([params[2*i], params[2*i+1], ctrl_angles[i]])
        return np.array(path, dtype=np.float64)
    
    def make_path_from_params_and_angles(params, angles):
        """params: array of shape (2*n_ctrl,) with [x0, y0, x1, y1, ...]"""
        path = []
        for i in range(len(angles)):
            path.append([params[2*i], params[2*i+1], angles[i]])
        return np.array(path, dtype=np.float64)
    
    # Initial parameters: linear interpolation from (0,0) to (-2,0)
    current_params = np.zeros(2 * n_ctrl)
    for i in range(n_ctrl):
        t = ctrl_angles[i] / 90.0
        current_params[2*i] = -2.0 * t
        current_params[2*i+1] = 0.0
    
    # Evaluate initial path
    initial_path = make_path_from_params(current_params)
    initial_score = evaluate_path(initial_path)
    
    # Try a few hand-crafted paths
    hand_paths = [
        # Linear path
        current_params.copy(),
        # Slightly curved path (move up in middle)
        np.array([
            -2.0*5/90, 0.0,
            -2.0*10/90, 0.05,
            -2.0*15/90, 0.1,
            -2.0*20/90, 0.15,
            -2.0*25/90, 0.18,
            -2.0*30/90, 0.2,
            -2.0*35/90, 0.2,
            -2.0*40/90, 0.18,
            -2.0*45/90, 0.15,
            -2.0*50/90, 0.1,
            -2.0*55/90, 0.05,
            -2.0*60/90, 0.0,
        ]),
    ]
    
    best_params = current_params.copy()
    best_score = initial_score
    
    for hp in hand_paths[1:]:
        hp_path = make_path_from_params(hp)
        hp_score = evaluate_path(hp_path)
        if hp_score > best_score:
            best_score = hp_score
            best_params = hp.copy()
    
    start_time = time.time()
    time_limit = 580.0
    deadline = start_time + time_limit
    
    # Smooth surrogate objective using soft-min
    def smooth_objective(params, beta=50.0):
        """
        Smooth approximation of the objective.
        Instead of exact intersection, we use a soft version.
        """
        path = make_path_from_params(params)
        full_path = np.vstack([
            [[0.0, 0.0, 0.0]],
            path,
            [[-2.0, 0.0, 90.0]]
        ])
        
        # For each grid point, compute the minimum "margin" across all placements
        # margin > 0 means inside, margin < 0 means outside
        # Use soft-min to get a smooth objective
        
        # This is expensive, so use a subsample of points
        n_sub = min(n_points, 2000)
        indices = np.random.choice(n_points, n_sub, replace=False)
        X_sub = X_flat[indices]
        Y_sub = Y_flat[indices]
        
        margins = np.zeros(n_sub)
        first = True
        
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
                
                wx = X_sub - x
                wy = Y_sub - y
                
                u = wx * c - wy * s
                v = wx * s + wy * c
                
                # Margin for each rectangle (positive = inside)
                # R1: 0 <= u <= 1, -3 <= v <= 1
                margin_r1 = np.minimum(
                    np.minimum(u, 1.0 - u),
                    np.minimum(v + 3.0, 1.0 - v)
                )
                # R2: -3 <= u <= 1, 0 <= v <= 1
                margin_r2 = np.minimum(
                    np.minimum(u + 3.0, 1.0 - u),
                    np.minimum(v, 1.0 - v)
                )
                
                margin = np.maximum(margin_r1, margin_r2)
                
                if first:
                    margins = margin.copy()
                    first = False
                else:
                    # Soft-min: log-sum-exp approximation
                    # min(a,b) ≈ -log(exp(-beta*a) + exp(-beta*b))/beta for large beta
                    # Use stable version
                    m_max = np.maximum(margins, margin)
                    m_min = np.minimum(margins, margin)
                    # soft_min = m_min - (1/beta) * log(1 + exp(-beta*(m_max - m_min)))
                    diff = m_max - m_min
                    soft_min = m_min - (1.0/beta) * np.log1p(np.exp(-beta * diff))
                    margins = soft_min
        
        # Objective: maximize the fraction of points with margin > 0
        # Use soft version: sum of sigmoid(beta * margin)
        score = np.sum(1.0 / (1.0 + np.exp(-beta * margins)))
        return -score / n_sub  # Negate for minimization
    
    # Exact objective for final evaluation
    def exact_objective(params):
        path = make_path_from_params(params)
        return -evaluate_path(path)
    
    # Nelder-Mead with adaptive parameters
    def nelder_mead(f, x0, max_iter=150, tol=1e-6, time_limit_check=None, 
                     alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5):
        n = len(x0)
        
        simplex = [x0.copy()]
        fvals = [f(x0)]
        
        # Initialize simplex with small perturbations
        step_size = 0.2
        for i in range(n):
            xi = x0.copy()
            xi[i] += step_size
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
                # Accept reflection
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
            
            if fvals[0] < best_val:
                best_val = fvals[0]
                best_params = simplex[0].copy()
        
        return best_params, best_val
    
    # Phase 1: Optimize with smooth objective
    np_params, np_val = nelder_mead(
        lambda p: smooth_objective(p, beta=30.0),
        best_params,
        max_iter=100, tol=1e-5,
        time_limit_check=deadline - 60
    )
    
    # Evaluate with exact objective
    np_exact = evaluate_path(make_path_from_params(np_params))
    if np_exact > best_score:
        best_score = np_exact
        best_params = np_params.copy()
    
    # Phase 2: Optimize with exact objective (more precise)
    if time.time() < deadline - 30:
        np_params2, np_val2 = nelder_mead(
            exact_objective,
            best_params,
            max_iter=80, tol=1e-7,
            time_limit_check=deadline - 15
        )
        
        np_exact2 = -np_val2
        if np_exact2 > best_score:
            best_score = np_exact2
            best_params = np_params2.copy()
    
    # Phase 3: Random restarts with exact objective
    np.random.seed(42)
    for restart in range(8):
        if time.time() > deadline - 20:
            break
        
        perturbed = best_params + np.random.normal(0, 0.1, size=len(best_params))
        bp, bv = nelder_mead(
            exact_objective,
            perturbed,
            max_iter=50, tol=1e-6,
            time_limit_check=deadline - 10
        )
        
        if -bv > best_score:
            best_score = -bv
            best_params = bp.copy()
    
    # Phase 4: Final polish with very tight tolerance
    if time.time() < deadline - 5:
        bp_final, bv_final = nelder_mead(
            exact_objective,
            best_params,
            max_iter=30, tol=1e-8,
            time_limit_check=deadline - 2,
            alpha=0.5, gamma=1.5, rho=0.3, sigma=0.3
        )
        
        if -bv_final > best_score:
            best_score = -bv_final
            best_params = bp_final.copy()
    
    # Construct final path
    best_path = make_path_from_params(best_params)
    
    # Safety checks
    if len(best_path) < 1:
        best_path = make_path_from_params(np.zeros(2*n_ctrl))
    
    if not np.all(np.isfinite(best_path)):
        best_path = make_path_from_params(np.zeros(2*n_ctrl))
    
    return best_path

# EVOLVE-BLOCK-END
