"""Rewritten program for packing 200 equal circles in a unit square."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
from scipy.spatial import cKDTree


def _compute_radius(c):
    """Compute the largest common radius for given centres."""
    n = c.shape[0]
    boundary_dists = 0.5 - np.abs(c)
    min_boundary = np.min(boundary_dists)
    
    if n < 2:
        return min_boundary
    
    dists = pdist(c, metric='euclidean')
    min_pair_dist = np.min(dists)
    
    return min(min_pair_dist / 2.0, min_boundary)


def _soft_min(x, k=200.0):
    """Smooth approximation of min(x) via log-sum-exp."""
    x = np.asarray(x)
    if x.size == 0:
        return 0.0
    m = np.min(x)
    return m - np.log(np.sum(np.exp(k * (x - m)))) / k


def _smooth_objective(params, k=200.0):
    """Minimize negative smooth radius using soft-min surrogate."""
    c = np.clip(params.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
    n = c.shape[0]
    
    # Boundary distances
    boundary_dists = 0.5 - np.abs(c)
    
    # Pairwise distances / 2
    tree = cKDTree(c)
    pairs = tree.query_pairs(0.2)  # conservative upper bound
    if len(pairs) > 0:
        pair_dists = np.array([np.linalg.norm(c[i] - c[j]) / 2.0 for i, j in pairs])
    else:
        pair_dists = np.array([0.5])
    
    # Combine all constraints
    all_dists = np.concatenate([boundary_dists.ravel(), pair_dists])
    r_soft = _soft_min(all_dists, k=k)
    
    return -r_soft


def _get_active_pairs(c, r, margin=1e-6):
    """Get pairs and boundary indices that are near-active (within margin of r)."""
    n = c.shape[0]
    tree = cKDTree(c)
    pairs = tree.query_pairs(2 * (r + margin))
    
    active_pairs = []
    for i, j in pairs:
        d = np.linalg.norm(c[i] - c[j]) / 2.0
        if abs(d - r) < margin or d < r + margin:
            active_pairs.append((i, j, d))
    
    # Boundary constraints
    boundary_dists = 0.5 - np.abs(c)
    min_boundary = np.min(boundary_dists)
    active_boundaries = []
    for i in range(n):
        for k in range(2):
            d = boundary_dists[i, k]
            if abs(d - r) < margin or d < r + margin:
                active_boundaries.append((i, k, d))
    
    return active_pairs, active_boundaries, min_boundary


def _slsqp_polish(c0, r0, max_time=20.0):
    """SLSQP polish on active constraints only."""
    start_time = time.time()
    c = c0.copy()
    r = r0
    n = c.shape[0]
    
    def objective(x):
        c_local = np.clip(x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
        return -_compute_radius(c_local)
    
    # Build constraints from active set
    active_pairs, active_boundaries, min_boundary = _get_active_pairs(c, r, margin=1e-4)
    
    constraints = []
    # Pairwise constraints: |c_i - c_j| / 2 >= r
    for i, j, d in active_pairs[:50]:  # limit number of constraints
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, i=i, j=j: np.linalg.norm(x[2*i:2*i+2] - x[2*j:2*j+2]) / 2.0 - r,
            'jac': lambda x, i=i, j=j: _pair_jacobian(x, i, j, r)
        })
    
    # Boundary constraints
    for i, k, d in active_boundaries[:30]:
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, i=i, k=k: 0.5 - np.abs(x[2*i+k]) - r,
            'jac': lambda x, i=i, k=k: _boundary_jacobian(x, i, k, r)
        })
    
    bounds = [(-0.5 + 1e-12, 0.5 - 1e-12)] * (2 * n)
    
    try:
        result = minimize(
            objective,
            c.flatten(),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 100, 'ftol': 1e-12}
        )
        
        if result.fun < -r:
            c_new = np.clip(result.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
            r_new = _compute_radius(c_new)
            if r_new > r:
                c = c_new
                r = r_new
    except Exception:
        pass
    
    return c, r


def _pair_jacobian(x, i, j, r):
    """Jacobian for pairwise constraint |c_i - c_j|/2 - r >= 0."""
    n = len(x) // 2
    jac = np.zeros(2 * n)
    ci = x[2*i:2*i+2]
    cj = x[2*j:2*j+2]
    diff = ci - cj
    d = np.linalg.norm(diff)
    if d > 1e-12:
        grad = diff / (2 * d)
        jac[2*i:2*i+2] = grad
        jac[2*j:2*j+2] = -grad
    return jac


def _boundary_jacobian(x, i, k, r):
    """Jacobian for boundary constraint 0.5 - |c_ik| - r >= 0."""
    n = len(x) // 2
    jac = np.zeros(2 * n)
    val = x[2*i+k]
    if val >= 0:
        jac[2*i+k] = -1.0
    else:
        jac[2*i+k] = 1.0
    return jac


def _initialize_hexagonal():
    """Initialize with a hexagonal pattern optimized for the square."""
    best_c = None
    best_r = 0
    
    for d in np.linspace(0.06, 0.08, 15):
        points = []
        row_spacing = d * np.sqrt(3) / 2
        x_offset = d / 2
        
        y = -0.5 + d / 2
        row = 0
        while y <= 0.5 - d / 2:
            x = -0.5 + d / 2
            if row % 2 == 1:
                x += x_offset
            while x <= 0.5 - d / 2:
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                    points.append([x, y])
                x += d
            y += row_spacing
            row += 1
        
        points = np.array(points)
        if len(points) >= N:
            c = points[:N].copy()
            r = _compute_radius(c)
            if r > best_r:
                best_r = r
                best_c = c.copy()
    
    if best_c is not None:
        return best_c
    
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()


def _local_optimize_smooth(c0, max_time=30.0):
    """Local optimization using smooth surrogate with L-BFGS-B."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    # Phase 1: Coarse optimization with lower k
    k_coarse = 100.0
    try:
        result = minimize(
            lambda x: _smooth_objective(x, k=k_coarse),
            c0.flatten(),
            method='L-BFGS-B',
            bounds=[(-0.5 + 1e-12, 0.5 - 1e-12)] * (2 * N),
            options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-6}
        )
        
        if result.fun < -best_r:
            c_new = np.clip(result.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
            r_new = _compute_radius(c_new)
            if r_new > best_r:
                best_c = c_new
                best_r = r_new
    except Exception:
        pass
    
    # Phase 2: Fine optimization with higher k
    if time.time() - start_time < max_time - 10:
        k_fine = 300.0
        try:
            result2 = minimize(
                lambda x: _smooth_objective(x, k=k_fine),
                best_c.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.5 + 1e-12, 0.5 - 1e-12)] * (2 * N),
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result2.fun < -best_r:
                c_new = np.clip(result2.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
                r_new = _compute_radius(c_new)
                if r_new > best_r:
                    best_c = c_new
                    best_r = r_new
        except Exception:
            pass
    
    # Phase 3: SLSQP polish on active constraints
    if time.time() - start_time < max_time - 5:
        try:
            c_polish, r_polish = _slsqp_polish(best_c, best_r, max_time=min(10.0, max_time - (time.time() - start_time) - 2))
            if r_polish > best_r:
                best_c = c_polish
                best_r = r_polish
        except Exception:
            pass
    
    return best_c, best_r


def _cutting_plane_optimize(c0, max_time=25.0):
    """Cutting-plane style optimization using growing set of active constraints."""
    start_time = time.time()
    c = c0.copy()
    r = _compute_radius(c0)
    
    # Maintain a set of "cutting planes" (active constraints)
    active_constraints = []
    
    for iteration in range(5):
        if time.time() > max_time:
            break
        
        # Get current active set
        active_pairs, active_boundaries, min_boundary = _get_active_pairs(c, r, margin=5e-5)
        
        # Build objective that penalizes violation of active constraints
        def cp_objective(x):
            c_local = np.clip(x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
            r_local = _compute_radius(c_local)
            
            # Penalty for violating active constraints
            penalty = 0.0
            for i, j, d in active_pairs[:20]:
                d_new = np.linalg.norm(c_local[i] - c_local[j]) / 2.0
                if d_new < r_local:
                    penalty += (r_local - d_new) ** 2
            for i, k, d in active_boundaries[:10]:
                d_new = 0.5 - np.abs(c_local[i, k])
                if d_new < r_local:
                    penalty += (r_local - d_new) ** 2
            
            return -r_local + 0.1 * penalty
        
        try:
            result = minimize(
                cp_objective,
                c.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.5 + 1e-12, 0.5 - 1e-12)] * (2 * N),
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result.fun < -r:
                c_new = np.clip(result.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
                r_new = _compute_radius(c_new)
                if r_new > r:
                    c = c_new
                    r = r_new
        except Exception:
            pass
    
    return c, r


def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    best_radius = 0.0
    best_c = None
    
    # Initialize with hexagonal pattern
    c_hex = _initialize_hexagonal()
    r_hex = _compute_radius(c_hex)
    if r_hex > best_radius:
        best_radius = r_hex
        best_c = c_hex.copy()
    
    # Main optimization: smooth surrogate + SLSQP polish
    if time.time() < deadline - 30:
        c_opt, r_opt = _local_optimize_smooth(c_hex, max_time=min(40.0, deadline - time.time() - 10))
        if r_opt > best_radius:
            best_radius = r_opt
            best_c = c_opt.copy()
    
    # Additional trials with random initializations
    np.random.seed(42)
    for trial in range(2):
        if time.time() > deadline - 25:
            break
        
        c_rand = np.random.uniform(-0.45, 0.45, size=(N, 2))
        r_rand = _compute_radius(c_rand)
        
        if r_rand > best_radius:
            best_radius = r_rand
            best_c = c_rand.copy()
        
        if time.time() < deadline - 20:
            c_opt, r_opt = _local_optimize_smooth(c_rand, max_time=min(15.0, deadline - time.time() - 5))
            if r_opt > best_radius:
                best_radius = r_opt
                best_c = c_opt.copy()
    
    # Final cutting-plane polish if time allows
    if time.time() < deadline - 10 and best_c is not None:
        try:
            c_cp, r_cp = _cutting_plane_optimize(best_c, max_time=min(8.0, deadline - time.time() - 3))
            if r_cp > best_radius:
                best_radius = r_cp
                best_c = c_cp.copy()
        except Exception:
            pass
    
    if best_c is None:
        best_c = _initialize_hexagonal()
    
    best_c = np.clip(best_c, -0.5, 0.5)
    
    return best_c
# EVOLVE-BLOCK-END
