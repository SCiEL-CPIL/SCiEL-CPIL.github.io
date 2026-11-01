"""Rewritten program for packing 200 equal circles in a unit square."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, cdist
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


def _soft_min_radius(c, beta):
    """
    Compute a soft approximation of the minimum radius using log-sum-exp.
    Efficiently computes soft-min over all boundary and pairwise constraints.
    """
    # Boundary distances: for each point, distance to each of 4 walls
    b_left = 0.5 - c[:, 0]
    b_right = 0.5 + c[:, 0]
    b_bottom = 0.5 - c[:, 1]
    b_top = 0.5 + c[:, 1]
    
    # Pairwise half-distances
    d = pdist(c, metric='euclidean')
    p_terms = 0.5 * d
    
    # Combine all constraint values
    all_terms = np.concatenate([b_left, b_right, b_bottom, b_top, p_terms])
    
    # Log-sum-exp trick for soft min:
    # soft_min = m - (1/beta) * log(sum(exp(-beta * (all_terms - m))))
    m = np.min(all_terms)
    shifted = all_terms - m  # all >= 0, min is 0
    exp_terms = np.exp(-beta * shifted)
    log_sum = np.log(np.sum(exp_terms))
    
    soft_min_val = m - (1.0 / beta) * log_sum
    return soft_min_val


def _objective(c_flat, beta):
    """Objective function for minimization: -soft_min_radius."""
    c = c_flat.reshape(-1, 2)
    r = _soft_min_radius(c, beta)
    return -r


def _initialize_hexagonal():
    """Initialize with a hexagonal pattern, trying multiple spacings."""
    best_c = None
    best_r = 0.0
    
    for d in np.linspace(0.054, 0.076, 15):
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


def _get_active_pairs(c, r, tol=1e-6):
    """Get indices of pairs that are active (within tol of minimum distance)."""
    d = pdist(c, metric='euclidean')
    active_mask = d <= 2 * r + tol
    # Convert condensed indices to (i, j) pairs
    n = c.shape[0]
    pairs = []
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if active_mask[idx]:
                pairs.append((i, j))
            idx += 1
    return pairs


def _get_active_boundaries(c, r, tol=1e-6):
    """Get indices of points with active boundary constraints."""
    active = []
    for i in range(c.shape[0]):
        if 0.5 - c[i, 0] <= r + tol:
            active.append((i, 'left'))
        if 0.5 + c[i, 0] <= r + tol:
            active.append((i, 'right'))
        if 0.5 - c[i, 1] <= r + tol:
            active.append((i, 'bottom'))
        if 0.5 + c[i, 1] <= r + tol:
            active.append((i, 'top'))
    return active


def _optimize_smooth(c0, max_time=60.0):
    """Optimize using L-BFGS-B on the soft-min objective with gradual beta annealing."""
    start_time = time.time()
    
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    # Gradual beta schedule for better convergence
    beta_schedule = [5.0, 10.0, 20.0, 40.0, 80.0, 150.0, 300.0, 600.0]
    
    for b_val in beta_schedule:
        elapsed = time.time() - start_time
        if elapsed > max_time - 8.0:
            break
            
        def objective_wrapper(x, b=b_val):
            return _objective(x, b)
            
        try:
            res = minimize(
                objective_wrapper,
                best_c.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 800}
            )
            
            c_opt = res.x.reshape(-1, 2)
            # Clip to ensure validity
            c_opt = np.clip(c_opt, -0.5, 0.5)
            r_opt = _compute_radius(c_opt)
            
            if r_opt > best_r:
                best_r = r_opt
                best_c = c_opt.copy()
        except Exception:
            continue

    return best_c, best_r


def _slsqp_polish(c0, max_time=20.0):
    """
    High-precision polish using SLSQP with all constraints vectorized.
    Variables: [c_flat (2N), r (1)] -> minimize -r subject to constraints >= 0.
    """
    start_time = time.time()
    c = c0.copy()
    r_init = _compute_radius(c)
    
    # SLSQP with vectorized inequality constraints
    def obj(x):
        return -x[2 * N]
    
    def ineq_constraints(x):
        c_flat = x[:2 * N]
        r = x[2 * N]
        c = c_flat.reshape(-1, 2)
        
        # Boundary constraints: 0.5 - |c_i| - r >= 0
        b_left = 0.5 - c[:, 0] - r
        b_right = 0.5 + c[:, 0] - r
        b_bottom = 0.5 - c[:, 1] - r
        b_top = 0.5 + c[:, 1] - r
        
        # Pairwise constraints: dist(i,j) - 2r >= 0
        d = pdist(c, metric='euclidean')
        p_terms = d - 2.0 * r
        
        return np.concatenate([b_left, b_right, b_bottom, b_top, p_terms])
    
    x0 = np.concatenate([c.flatten(), [r_init]])
    
    try:
        res = minimize(
            obj,
            x0,
            method='SLSQP',
            constraints=[{'type': 'ineq', 'fun': ineq_constraints}],
            bounds=[(-0.5, 0.5)] * (2 * N) + [(0.0, 0.5)],
            options={'maxiter': 200, 'ftol': 1e-14}
        )
        
        if res.success or res.fun < -r_init:
            c_opt = res.x[:2 * N].reshape(-1, 2)
            c_opt = np.clip(c_opt, -0.5, 0.5)
            r_opt = _compute_radius(c_opt)
            if r_opt > r_init:
                return c_opt, r_opt
    except Exception:
        pass
    
    return c0, r_init


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    best_radius = 0.0
    best_c = None
    
    # Start with hexagonal initialization
    c_hex = _initialize_hexagonal()
    r_hex = _compute_radius(c_hex)
    if r_hex > best_radius:
        best_radius = r_hex
        best_c = c_hex.copy()
    
    # Phase 1: Smooth optimization with L-BFGS-B (gradual beta annealing)
    if time.time() < deadline - 30:
        max_time_p1 = min(50.0, deadline - time.time() - 25)
        c_opt, r_opt = _optimize_smooth(c_hex, max_time=max_time_p1)
        if r_opt > best_radius:
            best_radius = r_opt
            best_c = c_opt.copy()
    
    # Phase 2: SLSQP polish on the best solution
    if time.time() < deadline - 10:
        max_time_p2 = min(20.0, deadline - time.time() - 8)
        c_pol, r_pol = _slsqp_polish(best_c, max_time=max_time_p2)
        if r_pol > best_radius:
            best_radius = r_pol
            best_c = c_pol.copy()
    
    # Phase 3: Try a perturbed restart if time remains
    if time.time() < deadline - 25:
        np.random.seed(42)
        c_pert = best_c + np.random.normal(0, 0.003, size=(N, 2))
        c_pert = np.clip(c_pert, -0.5, 0.5)
        
        max_time_p3 = min(30.0, deadline - time.time() - 15)
        c_opt3, r_opt3 = _optimize_smooth(c_pert, max_time=max_time_p3)
        if r_opt3 > best_radius:
            best_radius = r_opt3
            best_c = c_opt3.copy()
        
        # Quick SLSQP polish on the new best
        if time.time() < deadline - 8:
            c_pol3, r_pol3 = _slsqp_polish(best_c, max_time=min(10.0, deadline - time.time() - 6))
            if r_pol3 > best_radius:
                best_radius = r_pol3
                best_c = c_pol3.copy()
    
    # Phase 4: Another restart with different perturbation
    if time.time() < deadline - 20:
        np.random.seed(123)
        c_pert4 = best_c + np.random.normal(0, 0.008, size=(N, 2))
        c_pert4 = np.clip(c_pert4, -0.5, 0.5)
        
        max_time_p4 = min(25.0, deadline - time.time() - 12)
        c_opt4, r_opt4 = _optimize_smooth(c_pert4, max_time=max_time_p4)
        if r_opt4 > best_radius:
            best_radius = r_opt4
            best_c = c_opt4.copy()
        
        if time.time() < deadline - 6:
            c_pol4, r_pol4 = _slsqp_polish(best_c, max_time=min(8.0, deadline - time.time() - 5))
            if r_pol4 > best_radius:
                best_radius = r_pol4
                best_c = c_pol4.copy()
    
    if best_c is None:
        best_c = _initialize_hexagonal()
    
    # Final projection to ensure strict validity
    best_c = np.clip(best_c, -0.5, 0.5)
    
    return best_c
# EVOLVE-BLOCK-END
