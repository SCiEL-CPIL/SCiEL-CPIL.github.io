"""Rewritten program for packing 200 equal circles in a unit square."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
from scipy.optimize import LinearConstraint


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
    shifted = np.clip(shifted, 0, 100.0 / beta)
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    return v_min - log_sum / beta


def _smooth_objective(params, beta):
    """Smooth objective: minimize -soft_min(radii_components)."""
    c = np.clip(params.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
    n = c.shape[0]
    
    boundary_dists = 0.5 - np.abs(c)
    min_boundary = np.min(boundary_dists)
    
    if n < 2:
        return -min_boundary
    
    dists = pdist(c, metric='euclidean')
    pair_dists_half = dists / 2.0
    
    # Combine all distance components
    all_dists = np.concatenate([pair_dists_half, boundary_dists.ravel()])
    r_soft = _soft_min(all_dists, beta)
    
    if r_soft <= 0:
        return 1.0
    return -r_soft


def _initialize_hexagonal():
    """Initialize with a hexagonal pattern."""
    best_c = None
    best_r = 0
    
    for d in np.linspace(0.055, 0.075, 20):
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
    
    # Fallback: grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()


def _slsqp_polish(c0, target_r, max_time=20.0):
    """SLSQP polish: maximize min distance with explicit constraints."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    if best_r <= 0:
        return best_c, best_r
    
    # Use SLSQP with explicit inequality constraints
    # Maximize r by minimizing -r where r is bounded by all constraints
    # We'll use an auxiliary variable approach: minimize -t
    # subject to all pair distances >= 2*t and boundary distances >= t
    
    n = N
    
    def objective(params):
        """Objective: -t (last variable)"""
        return -params[-1]
    
    def make_constraints(t_idx):
        """Create constraint functions for SLSQP."""
        constraints = []
        
        # Boundary constraints: for each point i, each coordinate k:
        # 0.5 - |c_ik| >= t
        for i in range(n):
            for k in range(2):
                # 0.5 - |x_ik| >= t  =>  0.5 - |x_ik| - t >= 0
                def boundary_con(params, i=i, k=k):
                    return 0.5 - abs(params[2*i + k]) - params[t_idx]
                constraints.append({'type': 'ineq', 'fun': boundary_con})
        
        # Pairwise distance constraints: |c_i - c_j| >= 2*t for all i<j
        # |c_i - c_j|^2 >= (2*t)^2
        for i in range(n):
            for j in range(i+1, n):
                def pair_con(params, i=i, j=j):
                    dx = params[2*i] - params[2*j]
                    dy = params[2*i+1] - params[2*j+1]
                    return dx*dx + dy*dy - 4.0 * params[t_idx] * params[t_idx]
                constraints.append({'type': 'ineq', 'fun': pair_con})
        
        return constraints
    
    t_idx = 2 * n  # index of auxiliary variable t
    
    # Initial guess for t
    t_init = min(best_r * 0.99, 0.035)
    
    x0 = np.concatenate([c0.flatten(), [t_init]])
    
    bounds = [(-0.5, 0.5)] * (2 * n) + [(0.0, 0.05)]
    
    try:
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=make_constraints(t_idx),
            options={'maxiter': 50, 'ftol': 1e-10, 'disp': False}
        )
        
        if result.success or result.fun < 0:
            c_new = np.clip(result.x[:2*n].reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
            t_new = result.x[t_idx]
            r_new = _compute_radius(c_new)
            if r_new > best_r:
                best_c = c_new
                best_r = r_new
    except Exception:
        pass
    
    return best_c, best_r


def _high_precision_polish(c0, max_time=15.0):
    """High-precision polish: use very high beta and tight tolerances."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    # Very high beta values for near-exact min
    betas = [1000.0, 2000.0, 5000.0]
    
    for beta in betas:
        if time.time() > start_time + max_time - 1:
            break
        
        try:
            def obj(params):
                return _smooth_objective(params, beta)
            
            result = minimize(
                obj,
                best_c.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.5 + 1e-12, 0.5 - 1e-12)] * (2 * N),
                options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
            )
            
            c_new = np.clip(result.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
            r_new = _compute_radius(c_new)
            if r_new > best_r:
                best_c = c_new
                best_r = r_new
        except Exception:
            pass
    
    # SLSQP polish with explicit constraints if time permits
    if time.time() < start_time + max_time - 3:
        remaining = max(3.0, start_time + max_time - time.time() - 3)
        c_slsqp, r_slsqp = _slsqp_polish(best_c, best_r, max_time=remaining)
        if r_slsqp > best_r:
            best_c = c_slsqp
            best_r = r_slsqp
    
    return best_c, best_r


def _local_optimize_smooth(c0, max_time=30.0):
    """Local optimization using smooth objective with annealing."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    # Anneal beta from low to high with more intermediate steps
    betas = [15.0, 30.0, 60.0, 120.0, 250.0, 500.0]
    
    for beta in betas:
        if time.time() > start_time + max_time - 5:
            break
        
        try:
            def obj(params):
                return _smooth_objective(params, beta)
            
            result = minimize(
                obj,
                best_c.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.5 + 1e-12, 0.5 - 1e-12)] * (2 * N),
                options={'maxiter': 120, 'ftol': 1e-13, 'gtol': 1e-10}
            )
            
            c_new = np.clip(result.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
            r_new = _compute_radius(c_new)
            if r_new > best_r:
                best_c = c_new
                best_r = r_new
        except Exception:
            pass
    
    # High-precision polish if time permits
    if time.time() < start_time + max_time - 2:
        c_polished, r_polished = _high_precision_polish(
            best_c, max_time=max(2.0, max_time - (time.time() - start_time) - 2)
        )
        if r_polished > best_r:
            best_c = c_polished
            best_r = r_polished
    
    return best_c, best_r


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
    
    # Smooth optimization from hexagonal start (primary optimization)
    if time.time() < deadline - 40:
        remaining = deadline - time.time() - 40
        c_opt, r_opt = _local_optimize_smooth(c_hex, max_time=min(45.0, remaining))
        if r_opt > best_radius:
            best_radius = r_opt
            best_c = c_opt.copy()
    
    # Try a few perturbed starts with shorter optimization
    np.random.seed(42)
    n_trials = 2
    for trial in range(n_trials):
        if time.time() > deadline - 45:
            break
        
        # Perturb the best solution found so far
        if best_c is not None:
            c_pert = best_c + np.random.normal(0, 0.004, size=(N, 2))
            c_pert = np.clip(c_pert, -0.5 + 1e-12, 0.5 - 1e-12)
        else:
            c_pert = np.random.uniform(-0.45, 0.45, size=(N, 2))
        
        r_pert = _compute_radius(c_pert)
        if r_pert > best_radius:
            best_radius = r_pert
            best_c = c_pert.copy()
        
        if time.time() < deadline - 40:
            remaining = deadline - time.time() - 40
            c_opt, r_opt = _local_optimize_smooth(c_pert, max_time=min(15.0, remaining))
            if r_opt > best_radius:
                best_radius = r_opt
                best_c = c_opt.copy()
    
    # Final high-precision polish on the best solution found
    if best_c is not None and time.time() < deadline - 5:
        remaining = max(4.0, deadline - time.time() - 5)
        c_final, r_final = _high_precision_polish(best_c, max_time=remaining)
        if r_final > best_radius:
            best_radius = r_final
            best_c = c_final
    
    if best_c is None:
        best_c = _initialize_hexagonal()
    
    best_c = np.clip(best_c, -0.5, 0.5)
    
    return best_c
# EVOLVE-BLOCK-END
