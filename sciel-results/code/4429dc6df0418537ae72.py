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


def _soft_min_radius(c, beta):
    """
    Compute a soft approximation of the minimum radius using log-sum-exp.
    """
    b_left = 0.5 - c[:, 0]
    b_right = 0.5 + c[:, 0]
    b_bottom = 0.5 - c[:, 1]
    b_top = 0.5 + c[:, 1]
    
    d = pdist(c, metric='euclidean')
    p_terms = 0.5 * d
    
    all_terms = np.concatenate([b_left, b_right, b_bottom, b_top, p_terms])
    
    m = np.min(all_terms)
    shifted = all_terms - m
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
    
    for d in np.linspace(0.054, 0.076, 20):
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
                options={'maxiter': 8000, 'ftol': 1e-19, 'gtol': 1e-17, 'maxfun': 15000}
            )
            
            c_opt = res.x.reshape(-1, 2)
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
    """
    start_time = time.time()
    c = c0.copy()
    r_init = _compute_radius(c)
    
    def obj(x):
        return -x[2 * N]
    
    def ineq_constraints(x):
        c_flat = x[:2 * N]
        r = x[2 * N]
        c = c_flat.reshape(-1, 2)
        
        b_left = 0.5 - c[:, 0] - r
        b_right = 0.5 + c[:, 0] - r
        b_bottom = 0.5 - c[:, 1] - r
        b_top = 0.5 + c[:, 1] - r
        
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
            options={'maxiter': 500, 'ftol': 1e-16}
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


def _active_set_polish(c0, max_time=30.0):
    """
    Iterative active-set refinement: identify active constraints, solve reduced problem.
    """
    start_time = time.time()
    c = c0.copy()
    r = _compute_radius(c)
    
    # Identify active constraints with tight tolerance
    tol = 1e-8
    
    def get_active_pairs(c, r, tol):
        d = pdist(c, metric='euclidean')
        active_mask = d <= 2 * r + tol
        n = c.shape[0]
        pairs = []
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                if active_mask[idx]:
                    pairs.append((i, j))
                idx += 1
        return pairs
    
    def get_active_boundaries(c, r, tol):
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
    
    # Iterate: refine active set, solve constrained subproblem
    for iteration in range(10):
        if time.time() - start_time > max_time - 5:
            break
        
        active_pairs = get_active_pairs(c, r, tol)
        active_bounds = get_active_boundaries(c, r, tol)
        
        # Build constraints from active set only
        def obj(x):
            return -x[2 * N]
        
        constraints = []
        
        # Pairwise constraints for active pairs
        if len(active_pairs) > 0:
            def pair_constraint(x):
                c_flat = x[:2 * N]
                r_val = x[2 * N]
                c = c_flat.reshape(-1, 2)
                vals = []
                for (i, j) in active_pairs:
                    dist = np.sqrt((c[i, 0] - c[j, 0])**2 + (c[i, 1] - c[j, 1])**2)
                    vals.append(dist - 2.0 * r_val)
                return np.array(vals)
            constraints.append({'type': 'ineq', 'fun': pair_constraint})
        
        # Boundary constraints for active boundaries
        if len(active_bounds) > 0:
            def bound_constraint(x):
                c_flat = x[:2 * N]
                r_val = x[2 * N]
                c = c_flat.reshape(-1, 2)
                vals = []
                for (i, side) in active_bounds:
                    if side == 'left':
                        vals.append(0.5 - c[i, 0] - r_val)
                    elif side == 'right':
                        vals.append(0.5 + c[i, 0] - r_val)
                    elif side == 'bottom':
                        vals.append(0.5 - c[i, 1] - r_val)
                    else:
                        vals.append(0.5 + c[i, 1] - r_val)
                return np.array(vals)
            constraints.append({'type': 'ineq', 'fun': bound_constraint})
        
        if len(constraints) == 0:
            break
        
        x0 = np.concatenate([c.flatten(), [r]])
        
        try:
            res = minimize(
                obj,
                x0,
                method='SLSQP',
                constraints=constraints,
                bounds=[(-0.5, 0.5)] * (2 * N) + [(0.0, 0.5)],
                options={'maxiter': 200, 'ftol': 1e-16}
            )
            
            c_opt = res.x[:2 * N].reshape(-1, 2)
            c_opt = np.clip(c_opt, -0.5, 0.5)
            r_opt = _compute_radius(c_opt)
            
            if r_opt > r + 1e-12:
                c = c_opt
                r = r_opt
                # Tighten tolerance for next iteration
                tol *= 0.5
            else:
                break
        except Exception:
            break
    
    return c, r


def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    best_radius = 0.0
    best_c = None
    
    # Start with hexagonal initialization
    c_hex = _initialize_hexagonal()
    r_hex = _compute_radius(c_hex)
    if r_hex > best_radius:
        best_radius = r_hex
        best_c = c_hex.copy()
    
    # Phase 1: Smooth optimization with L-BFGS-B (gradual beta annealing)
    if time.time() < deadline - 40:
        max_time_p1 = min(55.0, deadline - time.time() - 35)
        c_opt, r_opt = _optimize_smooth(c_hex, max_time=max_time_p1)
        if r_opt > best_radius:
            best_radius = r_opt
            best_c = c_opt.copy()
    
    # Phase 2: SLSQP polish on the best solution
    if time.time() < deadline - 15:
        max_time_p2 = min(20.0, deadline - time.time() - 12)
        c_pol, r_pol = _slsqp_polish(best_c, max_time=max_time_p2)
        if r_pol > best_radius:
            best_radius = r_pol
            best_c = c_pol.copy()
    
    # Phase 3: Active-set refinement
    if time.time() < deadline - 12:
        max_time_p3 = min(25.0, deadline - time.time() - 10)
        c_active, r_active = _active_set_polish(best_c, max_time=max_time_p3)
        if r_active > best_radius:
            best_radius = r_active
            best_c = c_active.copy()
    
    # Phase 4: Try a perturbed restart if time remains
    if time.time() < deadline - 30:
        np.random.seed(42)
        c_pert = best_c + np.random.normal(0, 0.003, size=(N, 2))
        c_pert = np.clip(c_pert, -0.5, 0.5)
        
        max_time_p4 = min(25.0, deadline - time.time() - 20)
        c_opt4, r_opt4 = _optimize_smooth(c_pert, max_time=max_time_p4)
        if r_opt4 > best_radius:
            best_radius = r_opt4
            best_c = c_opt4.copy()
        
        if time.time() < deadline - 10:
            c_pol4, r_pol4 = _slsqp_polish(best_c, max_time=min(10.0, deadline - time.time() - 8))
            if r_pol4 > best_radius:
                best_radius = r_pol4
                best_c = c_pol4.copy()
        
        if time.time() < deadline - 8:
            c_act4, r_act4 = _active_set_polish(best_c, max_time=min(10.0, deadline - time.time() - 6))
            if r_act4 > best_radius:
                best_radius = r_act4
                best_c = c_act4.copy()
    
    if best_c is None:
        best_c = _initialize_hexagonal()
    
    # Final projection to ensure strict validity
    best_c = np.clip(best_c, -0.5, 0.5)
    
    return best_c
# EVOLVE-BLOCK-END
