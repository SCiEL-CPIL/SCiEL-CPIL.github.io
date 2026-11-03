import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 100
HALF = 0.5
TIME_LIMIT = 150.0
SAFETY = 15.0
START = time.time()

def _radius(pts):
    """Exact feasible common radius (vectorized, double precision)."""
    pts = np.asarray(pts, dtype=np.float64)
    if not (pts.shape == (N, 3) and np.all(np.isfinite(pts))):
        return 0.0
    
    # Pairwise distances
    D = cdist(pts, pts)
    np.fill_diagonal(D, np.inf)
    min_pair = np.min(D)
    
    # Wall distances
    min_wall = np.min(HALF - np.max(np.abs(pts), axis=1))
    
    r = 0.5 * min(min_pair, 2.0 * min_wall)
    if not np.isfinite(r) or r < 0:
        return 0.0
    return float(r)

def _bcc_points(n_max=4):
    """BCC lattice points centred at origin, nearest N kept."""
    pts = []
    for i in range(-n_max, n_max + 1):
        for j in range(-n_max, n_max + 1):
            for k in range(-n_max, n_max + 1):
                pts.append([i, j, k])
    for i in range(-n_max, n_max + 1):
        for j in range(-n_max, n_max + 1):
            for k in range(-n_max, n_max + 1):
                pts.append([i + 0.5, j + 0.5, k + 0.5])
    pts = np.array(pts, dtype=np.float64)
    pts -= pts.mean(axis=0)
    d = np.linalg.norm(pts, axis=1)
    order = np.argsort(d)
    return pts[order[:N]]

def _init_grid():
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].astype(np.float64)

def _init_bcc_scaled():
    """Initialize BCC lattice scaled to fit roughly in the cube."""
    V = _bcc_points(4)
    ext = np.max(V.max(axis=0) - V.min(axis=0))
    s = 0.8 / ext
    C = s * V
    C -= C.mean(axis=0)
    return C

def _soft_surrogate(pts, beta):
    """
    Smooth objective: -r + penalty.
    Uses log-sum-exp to approximate min distance smoothly.
    """
    diff = pts[:, None, :] - pts[None, :, :]
    d2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(d2, 0.0)
    ii, jj = np.triu_indices(N, 1)
    d2u = d2[ii, jj]
    d2u = np.clip(d2u, 1e-12, None)
    
    d = np.sqrt(d2u)
    beta_val = 1.0 / beta
    
    # Prevent overflow in exp
    d_min = np.min(d)
    exp_terms = np.exp(-beta_val * (d - d_min))
    log_sum = np.log(np.sum(exp_terms))
    soft_min_d = d_min - log_sum / beta_val
    
    # Wall distances
    w = HALF - np.max(np.abs(pts), axis=1)
    w_min = np.min(w)
    exp_terms_w = np.exp(-beta_val * (w - w_min))
    log_sum_w = np.log(np.sum(exp_terms_w))
    soft_min_w = w_min - log_sum_w / beta_val
    
    val1 = soft_min_d
    val2 = 2.0 * soft_min_w
    
    # Smooth min of val1, val2
    m = np.minimum(val1, val2)
    exp1 = np.exp(-beta_val * (val1 - m))
    exp2 = np.exp(-beta_val * (val2 - m))
    soft_r = m - np.log(exp1 + exp2) / beta_val
    
    r_smooth = 0.5 * soft_r
    
    # Penalty for wall breach (hard constraint violation)
    breach = np.clip(-w, 0.0, None)
    pen = 100.0 * np.sum(breach) + 1000.0 * np.max(breach)
    
    return -(r_smooth) + pen

def _polish_smooth(pts, deadline):
    """
    Polish using smooth surrogate and L-BFGS-B.
    """
    best_pts = pts.copy()
    best_r = _radius(pts)
    
    # Try a few iterations with decreasing beta (increasing sharpness)
    betas = [0.05, 0.02, 0.01, 0.005]
    
    for beta in betas:
        if time.time() > deadline:
            break
            
        def obj(x):
            return _soft_surrogate(x.reshape(N, 3), beta)
        
        try:
            bounds = [(-HALF, HALF)] * (N * 3)
            res = minimize(obj, best_pts.ravel(), method='L-BFGS-B',
                           bounds=bounds,
                           options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
            
            cand = res.x.reshape(N, 3)
            r = _radius(cand)
            if np.all(np.isfinite(cand)) and r > best_r:
                best_r = r
                best_pts = cand.copy()
                
        except Exception:
            pass
            
    return best_pts, best_r

def _slda(pts, deadline):
    """
    Stochastic Gradient Langevin Dynamics to escape local minima.
    Uses a repulsive potential with thermal noise.
    """
    best_pts = pts.copy()
    best_r = _radius(pts)
    
    # Parameters
    T0 = 0.01  # Initial temperature
    T_final = 0.001  # Final temperature
    n_steps = 500
    dt = 0.01  # Time step
    
    current = best_pts.copy()
    
    for step in range(n_steps):
        if time.time() > deadline:
            break
        
        # Anneal temperature
        T = T0 * (T_final / T0) ** (step / n_steps)
        
        # Compute forces
        diff = current[:, None, :] - current[None, :, :]
        d2 = np.einsum('ijk,ijk->ij', diff, diff)
        np.fill_diagonal(d2, 0.0)
        d2 = np.clip(d2, 1e-12, None)
        d = np.sqrt(d2)
        
        # Repulsive force: F_ij = k * (1/d^2) * (r_i - r_j) for d < threshold
        # Use a soft repulsion to keep it smooth
        # Force on i due to j: k * (r_i - r_j) / d^3 (like inverse square law)
        # But we want to maximize min distance, so we push apart when close.
        # Gradient of -min(d) is complex, so use a surrogate: sum of repulsive terms.
        # Potential: U = sum_{i<j} k / d_ij^2 (repulsive)
        # Force: F_i = -grad U_i = sum_{j!=i} 2k (r_i - r_j) / d_ij^4
        
        # For efficiency, only consider pairs within a cutoff
        # But for N=100, all pairs is fine
        forces = np.zeros_like(current)
        for i in range(N):
            for j in range(i + 1, N):
                rij = current[i] - current[j]
                dij = np.linalg.norm(rij)
                if dij < 0.2:  # Cutoff
                    # Force magnitude ~ 1/d^3, direction rij/d
                    f_mag = 1.0 / (dij ** 3 + 1e-12)
                    f_vec = rij / dij * f_mag
                    forces[i] += f_vec
                    forces[j] -= f_vec
        
        # Wall repulsion
        for i in range(N):
            for k in range(3):
                if current[i, k] > HALF - 0.05:
                    forces[i, k] -= 10.0 * (current[i, k] - (HALF - 0.05))
                elif current[i, k] < -HALF + 0.05:
                    forces[i, k] += 10.0 * ((-HALF + 0.05) - current[i, k])
        
        # Langevin update: x_{t+1} = x_t + dt * F + sqrt(2*T*dt) * noise
        noise = np.random.randn(*current.shape) * np.sqrt(2 * T * dt)
        current = current + dt * forces + noise
        
        # Project back into cube (hard constraint)
        current = np.clip(current, -HALF + 1e-6, HALF - 1e-6)
        
        # Check if improved
        r = _radius(current)
        if r > best_r:
            best_r = r
            best_pts = current.copy()
    
    return best_pts, best_r

def _local_refine(pts, deadline):
    """
    Final high-precision polish: move each point slightly to increase min distance.
    """
    best_pts = pts.copy()
    best_r = _radius(pts)
    
    # Try small perturbations
    for trial in range(50):
        if time.time() > deadline:
            break
        
        # Randomly pick a few points to nudge
        n_move = 5
        idx = np.random.choice(N, n_move, replace=False)
        
        cand = best_pts.copy()
        for i in idx:
            # Try moving in random direction
            for _ in range(10):
                direction = np.random.randn(3)
                direction /= np.linalg.norm(direction)
                step = 0.001 * np.random.uniform(0.5, 2.0)
                
                cand[i] += direction * step
                cand[i] = np.clip(cand[i], -HALF + 1e-6, HALF - 1e-6)
                
                r = _radius(cand)
                if r > best_r:
                    best_r = r
                    best_pts = cand.copy()
                else:
                    cand[i] -= direction * step  # revert
    
    return best_pts, best_r

def construct():
    rng = np.random.default_rng(42)
    deadline = START + (TIME_LIMIT - SAFETY)
    
    # 1. Generate BCC lattice
    V = _bcc_points(4)
    
    # 2. Global Optimization of Scale and Translation
    def objective_global(params):
        s = params[0]
        t = params[1:4]
        
        if s <= 0:
            return 1e6
            
        C = s * V + t
        
        min_wall = np.min(HALF - np.max(np.abs(C), axis=1))
        if min_wall < 0:
            return 100.0 * (-min_wall) + 10.0
        
        D = cdist(C, C)
        np.fill_diagonal(D, np.inf)
        min_pair = np.min(D)
        
        r = 0.5 * min(min_pair, 2.0 * min_wall)
        return -r

    V_extent = np.max(V.max(axis=0) - V.min(axis=0))
    s_init = 0.8 / V_extent
    t_init = [0.0, 0.0, 0.0]
    x0 = np.concatenate(([s_init], t_init))
    
    best_pts = None
    best_r = -1.0
    
    # Run global optimization
    try:
        res = minimize(objective_global, x0, method='L-BFGS-B',
                       options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
        s_opt = res.x[0]
        t_opt = res.x[1:4]
        C_opt = s_opt * V + t_opt
        r_opt = _radius(C_opt)
        
        if r_opt > best_r:
            best_r = r_opt
            best_pts = C_opt.copy()
            
    except Exception:
        pass
        
    # Fallback to grid if global opt failed or is bad
    if best_pts is None or best_r < 0.1:
        grid_pts = _init_grid()
        r_grid = _radius(grid_pts)
        if r_grid > best_r:
            best_r = r_grid
            best_pts = grid_pts.copy()
    else:
        grid_pts = _init_grid()
        r_grid = _radius(grid_pts)
        if r_grid > best_r:
            best_r = r_grid
            best_pts = grid_pts.copy()

    # 3. Local Polish with Smooth Surrogate
    if time.time() < deadline:
        pol_pts, pol_r = _polish_smooth(best_pts, deadline)
        if pol_r > best_r:
            best_r = pol_r
            best_pts = pol_pts

    # 4. Stochastic Langevin Dynamics to escape local minima
    if time.time() < deadline:
        slda_pts, slda_r = _slda(best_pts, deadline)
        if slda_r > best_r:
            best_r = slda_r
            best_pts = slda_pts

    # 5. Final local refinement
    if time.time() < deadline:
        ref_pts, ref_r = _local_refine(best_pts, deadline)
        if ref_r > best_r:
            best_r = ref_r
            best_pts = ref_pts

    # 6. Final Validity Guard
    pts = np.asarray(best_pts, dtype=np.float64)
    if not (pts.shape == (N, 3) and np.all(np.isfinite(pts))):
        pts = _init_grid()
        
    if np.any(np.abs(pts) > HALF + 1e-9):
        pts = np.clip(pts, -HALF, HALF)
        
    return pts
