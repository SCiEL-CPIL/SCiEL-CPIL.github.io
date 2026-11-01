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
    if n == 0:
        return 0.0
    boundary_dists = 0.5 - np.abs(c)
    min_boundary = np.min(boundary_dists)
    
    if n < 2:
        return min_boundary
    
    dists = pdist(c, metric='euclidean')
    min_pair_dist = np.min(dists)
    
    return min(min_pair_dist / 2.0, min_boundary)


def _soft_min(values, beta):
    """Smooth approximation of min using log-sum-exp."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return 0.0
    v_min = np.min(values)
    shifted = values - v_min
    # Clip to avoid overflow
    max_shift = 50.0 / beta
    shifted = np.clip(shifted, 0, max_shift)
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    return v_min - log_sum / beta


def _get_active_pairs(c, k=10):
    """
    Get indices of active pairs (nearest neighbors) using KDTree.
    Returns a list of (i, j) tuples where i < j.
    """
    n = c.shape[0]
    tree = cKDTree(c)
    # Query k+1 to include self, so we get k neighbors
    dists, indices = tree.query(c, k=k + 1)
    
    pairs = set()
    for i in range(n):
        for j in indices[i, 1:]:  # Skip self
            if i < j:
                pairs.add((i, j))
            else:
                pairs.add((j, i))
    
    return list(pairs)


def _dense_smooth_objective(params, beta):
    """
    Dense smooth objective using all pair distances.
    For N=200, ~20k pairs is manageable.
    """
    c = np.clip(params.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
    n = c.shape[0]
    
    boundary_dists = 0.5 - np.abs(c)
    
    # All pair distances
    dists = pdist(c, metric='euclidean')
    
    all_dists = np.concatenate([dists / 2.0, boundary_dists.ravel()])
    
    r_soft = _soft_min(all_dists, beta)
    
    if r_soft <= 0:
        return 1.0
    return -r_soft


def _local_optimize_dense(c0, max_time=30.0):
    """
    Local optimization using dense smooth objective with annealing.
    """
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    # Anneal beta from low to high
    betas = [5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0]
    
    for beta in betas:
        if time.time() > start_time + max_time - 5:
            break
        
        try:
            def obj(params):
                return _dense_smooth_objective(params, beta)
            
            result = minimize(
                obj,
                best_c.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.5 + 1e-12, 0.5 - 1e-12)] * (2 * N),
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-11, 'maxls': 50}
            )
            
            c_new = np.clip(result.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
            r_new = _compute_radius(c_new)
            
            if r_new > best_r:
                best_c = c_new
                best_r = r_new
                
        except Exception:
            pass
    
    return best_c, best_r


def _slsqp_active_polish(c0, max_time=20.0):
    """
    High-precision polish using SLSQP with active constraints and Jacobian.
    """
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    r_target = best_r
    
    for iteration in range(6):
        if time.time() > start_time + max_time:
            break
        
        # Get active pairs for current config
        pairs = _get_active_pairs(best_c, k=14)
        
        if not pairs:
            break
        
        n = N
        pairs_arr = np.array(pairs, dtype=int)
        
        def constraint_func(x):
            c = x.reshape(-1, 2)
            bd = 0.5 - np.abs(c)
            min_bd = np.min(bd)
            
            min_pair = np.inf
            for (i, j) in pairs:
                diff = c[i] - c[j]
                d = np.linalg.norm(diff)
                if d < min_pair:
                    min_pair = d
            
            r_min = min(min_bd, min_pair / 2.0)
            return r_min - r_target
        
        def constraint_jac(x):
            c = x.reshape(-1, 2)
            bd = 0.5 - np.abs(c)
            min_bd_idx = np.argmin(bd)
            min_bd_val = bd[min_bd_idx]
            
            # Find which constraint is most active
            min_pair = np.inf
            active_pair = None
            for (i, j) in pairs:
                diff = c[i] - c[j]
                d = np.linalg.norm(diff)
                if d < min_pair:
                    min_pair = d
                    active_pair = (i, j)
            
            r_pair = min_pair / 2.0
            r_min = min(min_bd_val, r_pair)
            
            jac = np.zeros(2 * n)
            
            if min_bd_val <= r_pair:
                # Boundary constraint is most active
                i = min_bd_idx // 2
                k = min_bd_idx % 2
                # d/dx_k of (0.5 - |c[i,k]|) = -sign(c[i,k])
                jac[2*i + k] = -np.sign(c[i, k])
            else:
                # Pair constraint is most active
                i, j = active_pair
                diff = c[i] - c[j]
                d = np.linalg.norm(diff)
                if d > 1e-15:
                    # d/dx_i of (||c_i - c_j|| / 2) = (c_i - c_j) / (2*d)
                    jac[2*i:2*i+2] = diff / (2.0 * d)
                    jac[2*j:2*j+2] = -diff / (2.0 * d)
            
            return jac
        
        try:
            result = minimize(
                lambda x: 0.0,
                best_c.flatten(),
                method='SLSQP',
                bounds=[(-0.5 + 1e-12, 0.5 - 1e-12)] * (2 * n),
                constraints=[
                    {'type': 'ineq', 'fun': constraint_func, 'jac': constraint_jac}
                ],
                options={'maxiter': 100, 'ftol': 1e-13, 'disp': False}
            )
            
            c_new = np.clip(result.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
            r_new = _compute_radius(c_new)
            
            if r_new > best_r + 1e-10:
                best_c = c_new
                best_r = r_new
                r_target = best_r * 1.0003
            else:
                break
                
        except Exception:
            break
            
    return best_c, best_r


def _initialize_hexagonal():
    """Initialize with a hexagonal pattern, trying multiple spacings and offsets."""
    best_c = None
    best_r = 0.0
    
    for d in np.linspace(0.052, 0.078, 30):
        row_spacing = d * np.sqrt(3) / 2
        x_offset = d / 2
        
        # Try a few x-offsets to better fill the square
        for x_shift in [0.0, d * 0.1, d * 0.2]:
            points = []
            y = -0.5 + d / 2
            row = 0
            while y <= 0.5 - d / 2 + 1e-9:
                x = -0.5 + d / 2 + x_shift
                if row % 2 == 1:
                    x += x_offset
                while x <= 0.5 - d / 2 + 1e-9:
                    if -0.5 + 1e-9 <= x <= 0.5 - 1e-9 and -0.5 + 1e-9 <= y <= 0.5 - 1e-9:
                        points.append([x, y])
                    x += d
                y += row_spacing
                row += 1
            
            if len(points) >= N:
                # If we have more than N, take the first N (top-left bias)
                # But also try taking from the center
                c = np.array(points[:N])
                r = _compute_radius(c)
                if r > best_r:
                    best_r = r
                    best_c = c.copy()
                
                # Also try centering: take N points most central
                if len(points) > N:
                    pts = np.array(points)
                    # Distance from center
                    dists_to_center = np.sum(pts**2, axis=1)
                    idx = np.argsort(dists_to_center)[:N]
                    c2 = pts[idx]
                    r2 = _compute_radius(c2)
                    if r2 > best_r:
                        best_r = r2
                        best_c = c2.copy()
    
    if best_c is not None:
        return best_c
    
    # Fallback: grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()


def _coordinate_descent_polish(c0, max_time=10.0):
    """
    Final coordinate descent polish: nudge each point along its most constraining direction.
    """
    start_time = time.time()
    c = c0.copy()
    best_r = _compute_radius(c)
    
    for sweep in range(50):
        if time.time() > start_time + max_time:
            break
        
        improved = False
        for i in range(N):
            # Find the most constraining neighbor or boundary
            # Try small nudges in several directions
            best_move = None
            best_r_candidate = best_r
            
            # Get nearest neighbors
            tree = cKDTree(c)
            dists, indices = tree.query(c[i], k=8)
            
            for j in indices[1:]:
                if j == i:
                    continue
                diff = c[i] - c[j]
                d = np.linalg.norm(diff)
                if d < 1e-12:
                    continue
                # Direction to move away from j
                direction = diff / d
                # Try small step
                step = 0.0001
                c_trial = c.copy()
                c_trial[i] = c[i] + direction * step
                c_trial[i] = np.clip(c_trial[i], -0.5 + 1e-12, 0.5 - 1e-12)
                r_trial = _compute_radius(c_trial)
                if r_trial > best_r_candidate:
                    best_r_candidate = r_trial
                    best_move = (direction, step)
            
            # Also try boundary directions
            for axis in range(2):
                for sign in [-1, 1]:
                    # Move away from boundary
                    c_trial = c.copy()
                    c_trial[i, axis] += sign * 0.0001
                    c_trial[i, axis] = np.clip(c_trial[i, axis], -0.5 + 1e-12, 0.5 - 1e-12)
                    r_trial = _compute_radius(c_trial)
                    if r_trial > best_r_candidate:
                        best_r_candidate = r_trial
                        best_move = (np.array([0.0, 0.0]) + sign * np.eye(2)[axis], 0.0001)
            
            if best_move is not None:
                direction, step = best_move
                c[i] = c[i] + direction * step
                c[i] = np.clip(c[i], -0.5 + 1e-12, 0.5 - 1e-12)
                improved = True
                if best_r_candidate > best_r:
                    best_r = best_r_candidate
        
        if not improved:
            break
    
    return c, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin (evaluator kills at 150s)
    
    best_radius = 0.0
    best_c = None
    
    # Start with hexagonal initialization
    c_hex = _initialize_hexagonal()
    r_hex = _compute_radius(c_hex)
    if r_hex > best_radius:
        best_radius = r_hex
        best_c = c_hex.copy()
    
    # Primary optimization with dense smooth objective
    if time.time() < deadline - 40:
        remaining = deadline - time.time() - 40
        c_opt, r_opt = _local_optimize_dense(c_hex, max_time=min(45.0, remaining))
        if r_opt > best_radius:
            best_radius = r_opt
            best_c = c_opt.copy()
    
    # Try perturbed starts with shorter optimization
    np.random.seed(42)
    n_trials = 4
    for trial in range(n_trials):
        if time.time() > deadline - 40:
            break
        
        # Perturb the best solution found so far
        if best_c is not None:
            # Smaller perturbation for later trials
            noise_scale = 0.005 / (trial + 1)
            c_pert = best_c + np.random.normal(0, noise_scale, size=(N, 2))
            c_pert = np.clip(c_pert, -0.5 + 1e-12, 0.5 - 1e-12)
        else:
            c_pert = np.random.uniform(-0.45, 0.45, size=(N, 2))
        
        r_pert = _compute_radius(c_pert)
        if r_pert > best_radius:
            best_radius = r_pert
            best_c = c_pert.copy()
        
        if time.time() < deadline - 35:
            remaining = deadline - time.time() - 35
            c_opt, r_opt = _local_optimize_dense(c_pert, max_time=min(20.0, remaining))
            if r_opt > best_radius:
                best_radius = r_opt
                best_c = c_opt.copy()
    
    # Final high-precision polish on the best solution found
    if best_c is not None and time.time() < deadline - 8:
        remaining = max(8.0, deadline - time.time() - 8)
        c_final, r_final = _slsqp_active_polish(best_c, max_time=remaining)
        if r_final > best_radius:
            best_radius = r_final
            best_c = c_final
    
    # Coordinate descent polish
    if best_c is not None and time.time() < deadline - 3:
        remaining = max(3.0, deadline - time.time() - 3)
        c_cd, r_cd = _coordinate_descent_polish(best_c, max_time=remaining)
        if r_cd > best_radius:
            best_radius = r_cd
            best_c = c_cd
    
    # One more SLSQP polish if time permits
    if best_c is not None and time.time() < deadline - 2:
        remaining = max(2.0, deadline - time.time() - 2)
        c_final2, r_final2 = _slsqp_active_polish(best_c, max_time=remaining)
        if r_final2 > best_radius:
            best_radius = r_final2
            best_c = c_final2
    
    if best_c is None:
        best_c = _initialize_hexagonal()
    
    best_c = np.clip(best_c, -0.5, 0.5)
    
    return best_c
# EVOLVE-BLOCK-END
