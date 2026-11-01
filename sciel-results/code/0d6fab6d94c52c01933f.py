import numpy as np
import scipy.optimize as so
import time
import itertools

N = 14
D = 3
SEED = 42

def _pairwise_dist2(X):
    """Compute squared pairwise distances matrix for X (N x D)."""
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    return np.sum(diff**2, axis=2)

def _raw_score(X):
    """
    Calculate the actual score: (max_dist / min_dist)^2.
    Returns 1e10 if degenerate.
    """
    if X.shape != (N, D):
        return 1e10
    dist2 = _pairwise_dist2(X)
    mask = ~np.eye(N, dtype=bool)
    d2 = dist2[mask]
    if len(d2) == 0:
        return 1e10
    d_min = np.min(d2)
    d_max = np.max(d2)
    if d_min < 1e-12:
        return 1e10
    return (d_max / d_min)

def _objective_smooth(X_flat, tau):
    """
    Smooth objective: minimizes log((max_d/min_d)^2) = 2*(soft_max(log d2) - soft_min(log d2)).
    Uses numerically stable log-sum-exp.
    """
    X = X_flat.reshape(N, D)
    dist2 = _pairwise_dist2(X)
    mask = ~np.eye(N, dtype=bool)
    d2 = dist2[mask]
    
    # Ensure positive values for log
    d2_safe = np.maximum(d2, 1e-15)
    log_d2 = np.log(d2_safe)
    
    if tau <= 0:
        tau = 1e-10
    
    # Soft max
    max_log = np.max(log_d2)
    term_max = tau * (log_d2 - max_log)
    term_max = np.clip(term_max, -50, 50)
    sum_exp_max = np.sum(np.exp(term_max))
    soft_max_val = max_log + (1.0 / tau) * np.log(sum_exp_max)
    
    # Soft min
    min_log = np.min(log_d2)
    term_min = -tau * (log_d2 - min_log)
    term_min = np.clip(term_min, -50, 50)
    sum_exp_min = np.sum(np.exp(term_min))
    soft_min_val = min_log - (1.0 / tau) * np.log(sum_exp_min)
    
    log_ratio = soft_max_val - soft_min_val
    return 2.0 * log_ratio

def _objective_nm(X_flat):
    """Raw objective for Nelder-Mead polishing."""
    X = X_flat.reshape(N, D)
    return _raw_score(X)

def _sqp_objective(X_flat):
    """
    Objective for SQP: minimize -min_pairwise_dist2.
    This is equivalent to maximizing the minimum pairwise distance.
    """
    X = X_flat.reshape(N, D)
    dist2 = _pairwise_dist2(X)
    mask = ~np.eye(N, dtype=bool)
    d2 = dist2[mask]
    return -np.min(d2)

def _sqp_constraints(X_flat):
    """
    Constraints for SQP: all pairwise distances <= 1.
    Returns list of constraint dicts.
    """
    X = X_flat.reshape(N, D)
    dist2 = _pairwise_dist2(X)
    mask = ~np.eye(N, dtype=bool)
    d2 = dist2[mask]
    # Each constraint: d2_i <= 1, i.e., 1 - d2_i >= 0
    return 1.0 - d2

def _sqp_grad(X_flat):
    """Gradient of -min(d2) w.r.t. X."""
    X = X_flat.reshape(N, D)
    dist2 = _pairwise_dist2(X)
    mask = ~np.eye(N, dtype=bool)
    d2 = dist2[mask]
    min_idx = np.argmin(d2)
    
    # Get the pair indices
    pairs = []
    idx = 0
    for i in range(N):
        for j in range(i+1, N):
            if idx == min_idx:
                pairs.append((i, j))
                break
            idx += 1
        if pairs:
            break
    
    i, j = pairs[0]
    grad = np.zeros_like(X_flat)
    # d/dX_i (||X_i - X_j||^2) = 2*(X_i - X_j)
    # d/dX_j (||X_i - X_j||^2) = 2*(X_j - X_i)
    diff = X[i] - X[j]
    grad[i*D:(i+1)*D] = 2.0 * diff
    grad[j*D:(j+1)*D] = -2.0 * diff
    
    # We're minimizing -min(d2), so gradient is -gradient of min(d2)
    return -grad

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_score = 1e10
    best_X = None
    
    np.random.seed(SEED)
    
    # Generate diverse initial points
    inits = []
    
    # 1. Fibonacci sphere
    golden_angle = np.pi * (3. - np.sqrt(5.))
    X_fib = np.zeros((N, 3))
    for i in range(N):
        z = 1. - (i / (N - 1.)) * 2.
        radius = np.sqrt(1. - z**2)
        theta = golden_angle * i
        x = np.cos(theta) * radius
        y = np.sin(theta) * radius
        X_fib[i] = [x, y, z]
    inits.append(X_fib)
    
    # 2. Symmetric ring structure: two rings of 7 points each, offset
    X_rings = np.zeros((N, 3))
    ring1_z = 0.5
    ring1_r = np.sqrt(1 - ring1_z**2)
    for i in range(7):
        theta = 2 * np.pi * i / 7
        X_rings[i] = [ring1_r * np.cos(theta), ring1_r * np.sin(theta), ring1_z]
    ring2_z = -0.5
    ring2_r = np.sqrt(1 - ring2_z**2)
    for i in range(7):
        theta = 2 * np.pi * (i + 0.5) / 7
        X_rings[i + 7] = [ring2_r * np.cos(theta), ring2_r * np.sin(theta), ring2_z]
    inits.append(X_rings)
    
    # 3. Three-layer structure: 5-4-5
    X_layers = np.zeros((N, 3))
    z1 = -0.6
    r1 = np.sqrt(1 - z1**2)
    for i in range(5):
        theta = 2 * np.pi * i / 5
        X_layers[i] = [r1 * np.cos(theta), r1 * np.sin(theta), z1]
    z2 = 0.0
    r2 = 1.0
    for i in range(4):
        theta = 2 * np.pi * (i + 0.25) / 4
        X_layers[i + 5] = [r2 * np.cos(theta), r2 * np.sin(theta), z2]
    z3 = 0.6
    r3 = np.sqrt(1 - z3**2)
    for i in range(5):
        theta = 2 * np.pi * (i + 0.2) / 5
        X_layers[i + 9] = [r3 * np.cos(theta), r3 * np.sin(theta), z3]
    inits.append(X_layers)
    
    # 4. Random points on sphere
    for _ in range(4):
        pts = np.random.randn(N, 3)
        norms = np.linalg.norm(pts, axis=1)
        pts /= norms[:, np.newaxis]
        inits.append(pts)
        
    # 5. Random points in ball
    for _ in range(3):
        pts = np.random.randn(N, 3)
        max_norm = np.max(np.linalg.norm(pts, axis=1))
        if max_norm > 0:
            pts /= max_norm
        inits.append(pts)
        
    # 6. Lattice-like
    side = int(np.ceil(N ** (1/3))) + 1
    grid = list(itertools.product(range(side), repeat=3))[:N]
    X_lat = np.array(grid, dtype=float)
    X_lat -= np.mean(X_lat, axis=0)
    lat_norm = np.max(np.linalg.norm(X_lat, axis=1))
    if lat_norm > 0:
        X_lat /= lat_norm
    inits.append(X_lat)
    
    # SQP constraints setup
    # We need to scale the problem so that max distance = 1
    # Then we maximize min distance, and ratio^2 = 1 / min_dist2
    
    for init_idx, X_init in enumerate(inits):
        if time.time() > deadline - 5:
            break
            
        x_current = X_init.flatten()
        
        # Normalize: scale so max pairwise distance = 1
        dist2_init = _pairwise_dist2(X_init)
        mask_init = ~np.eye(N, dtype=bool)
        d2_init = dist2_init[mask_init]
        max_d2 = np.max(d2_init)
        if max_d2 > 0:
            x_current = x_current / np.sqrt(max_d2)
        
        # Phase 1: SQP - maximize min distance subject to all distances <= 1
        if time.time() < deadline - 15:
            try:
                # Build constraint list for SLSQP
                constraints = []
                for i in range(N):
                    for j in range(i+1, N):
                        def make_constraint(i, j):
                            def constraint(x):
                                Xi = x[i*D:(i+1)*D]
                                Xj = x[j*D:(j+1)*D]
                                return 1.0 - np.sum((Xi - Xj)**2)
                            return constraint
                        constraints.append({'type': 'ineq', 'fun': make_constraint(i, j)})
                
                res_sqp = so.minimize(
                    _sqp_objective,
                    x_current,
                    method='SLSQP',
                    constraints=constraints,
                    options={'maxiter': 200, 'ftol': 1e-12}
                )
                if res_sqp.x is not None and res_sqp.success:
                    x_sqp = res_sqp.x
                    pts_sqp = x_sqp.reshape(N, D)
                    score_sqp = _raw_score(pts_sqp)
                    if score_sqp < best_score:
                        best_score = score_sqp
                        best_X = pts_sqp.copy()
                        x_current = x_sqp
            except Exception:
                pass
        
        # Phase 2: Smooth objective with tau annealing
        tau_schedule = [20, 100, 300]
        for tau in tau_schedule:
            if time.time() > deadline - 10:
                break
            try:
                res = so.minimize(
                    _objective_smooth, 
                    x_current, 
                    args=(tau,),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                if res.x is not None:
                    x_current = res.x
            except Exception:
                pass
        
        # Evaluate after Phase 2
        pts = x_current.reshape(N, D)
        score = _raw_score(pts)
        if score < best_score:
            best_score = score
            best_X = pts.copy()
            
        # Phase 3: High-tau polish
        if time.time() < deadline - 5:
            try:
                res_polish = so.minimize(
                    _objective_smooth, 
                    x_current, 
                    args=(2000.0,),
                    method='L-BFGS-B',
                    options={'maxiter': 150, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                if res_polish.x is not None:
                    x_polish = res_polish.x
                    pts_p = x_polish.reshape(N, D)
                    score_p = _raw_score(pts_p)
                    if score_p < best_score:
                        best_score = score_p
                        best_X = pts_p.copy()
                        x_current = x_polish
            except Exception:
                pass
        
        # Phase 4: Nelder-Mead on raw objective
        if time.time() < deadline - 3:
            try:
                res_nm = so.minimize(
                    _objective_nm,
                    x_current,
                    method='Nelder-Mead',
                    options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
                )
                if res_nm.x is not None:
                    x_nm = res_nm.x
                    pts_nm = x_nm.reshape(N, D)
                    score_nm = _raw_score(pts_nm)
                    if score_nm < best_score:
                        best_score = score_nm
                        best_X = pts_nm.copy()
            except Exception:
                pass

    if best_X is None:
        side = int(np.ceil(N ** (1 / D)))
        grid = list(itertools.product(range(side), repeat=D))[:N]
        return np.array(grid, dtype=float)
        
    return best_X
