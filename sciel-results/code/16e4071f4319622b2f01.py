import numpy as np
import scipy.optimize as so
import time

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

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
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
        radius = np.sqrt(max(0, 1 - z**2))
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
    
    # 3. Three-layer structure
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
    import itertools
    grid = list(itertools.product(range(side), repeat=3))[:N]
    X_lat = np.array(grid, dtype=float)
    X_lat -= np.mean(X_lat, axis=0)
    lat_norm = np.max(np.linalg.norm(X_lat, axis=1))
    if lat_norm > 0:
        X_lat /= lat_norm
    inits.append(X_lat)
    
    # Optimization Strategy:
    # Phase 1: Multi-stage tau annealing with L-BFGS-B
    # Phase 2: High-tau L-BFGS-B polish
    # Phase 3: Nelder-Mead on raw objective for final local refinement
    
    tau_schedule = [20, 100, 300]
    
    for init_idx, X_init in enumerate(inits):
        if time.time() > deadline - 10:
            break
            
        x_current = X_init.flatten()
        
        # Center to improve conditioning
        x_c = x_current.reshape(N, D) - np.mean(X_init, axis=0)
        x_current = x_c.flatten()
        
        # Phase 1: Tau annealing
        for tau in tau_schedule:
            if time.time() > deadline - 15:
                break
                
            try:
                res = so.minimize(
                    _objective_smooth, 
                    x_current, 
                    args=(tau,),
                    method='L-BFGS-B',
                    options={'maxiter': 400, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                if res.x is not None:
                    x_current = res.x
            except Exception:
                pass
        
        # Evaluate after Phase 1
        pts = x_current.reshape(N, D)
        score = _raw_score(pts)
        if score < best_score:
            best_score = score
            best_X = pts.copy()
            
        # Phase 2: High-tau L-BFGS-B polish
        if time.time() < deadline - 8:
            try:
                res_polish = so.minimize(
                    _objective_smooth, 
                    x_current, 
                    args=(5000.0,),
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
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
        
        # Phase 3: Nelder-Mead on raw objective
        if time.time() < deadline - 5:
            try:
                res_nm = so.minimize(
                    _objective_nm,
                    x_current,
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
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
        # Fallback
        import itertools
        side = int(np.ceil(N ** (1 / D)))
        grid = list(itertools.product(range(side), repeat=D))[:N]
        return np.array(grid, dtype=float)
        
    return best_X
