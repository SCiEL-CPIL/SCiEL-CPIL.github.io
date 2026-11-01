"""Packing 200 equal circles in a unit square using hybrid soft-min and hard-constraint approach."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.optimize import NonlinearConstraint


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


def _energy(params, temperature=0.02):
    """Smooth energy function to maximize minimum distance."""
    c = params.reshape(-1, 2)
    c = np.clip(c, -0.4999, 0.4999)
    
    # Distance to boundaries
    boundary_dists = 0.5 - np.abs(c)
    min_boundary = np.min(boundary_dists)
    
    if N < 2:
        return -min_boundary
    
    # Use pdist for all pairwise distances
    dists = pdist(c, metric='euclidean')
    min_dist = np.min(dists)
    
    T = max(temperature, 1e-8)
    
    # Soft min of pairwise distances - only consider close pairs for speed
    threshold = min_dist * 1.15 
    mask = dists < threshold
    if np.any(mask):
        d_sub = dists[mask]
        d_min_sub = np.min(d_sub)
        # Numerically stable soft min
        soft_min_pair = d_min_sub - T * np.log(np.sum(np.exp(-(d_sub - d_min_sub) / T)))
    else:
        soft_min_pair = min_dist
    
    # For boundary distances: soft min
    b_min = np.min(boundary_dists)
    b_threshold = b_min * 1.15
    b_mask = boundary_dists < b_threshold
    if np.any(b_mask):
        b_sub = boundary_dists[b_mask]
        b_min_sub = np.min(b_sub)
        soft_min_bound = b_min_sub - T * np.log(np.sum(np.exp(-(b_sub - b_min_sub) / T)))
    else:
        soft_min_bound = b_min
    
    # Combined soft min of r_pair and r_bound
    r_pair = soft_min_pair / 2.0
    r_bound = soft_min_bound
    
    r_vals = np.array([r_pair, r_bound])
    m = np.min(r_vals)
    soft_min_combined = m - T * np.log(np.sum(np.exp(-(r_vals - m) / T)))
    
    return -soft_min_combined


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
        while y <= 0.5 - d / 2 + 1e-9:
            x = -0.5 + d / 2
            if row % 2 == 1:
                x += x_offset
            while x <= 0.5 - d / 2 + 1e-9:
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                    points.append([x, y])
                x += d
            y += row_spacing
            row += 1
        
        if len(points) >= N:
            c = np.array(points[:N])
            r = _compute_radius(c)
            if r > best_r:
                best_r = r
                best_c = c.copy()
        elif len(points) > 0 and best_c is None:
            c = np.array(points)
            if len(c) < N:
                extra = np.random.uniform(-0.45, 0.45, size=(N - len(c), 2))
                c = np.vstack([c, extra])
            r = _compute_radius(c)
            if r > best_r:
                best_r = r
                best_c = c.copy()
    
    if best_c is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_c = grid[:N].copy()
    
    return best_c


def _local_optimize_soft(c0, max_time=10.0, temperature_schedule=None):
    """Local optimization using L-BFGS-B with soft-min energy function."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    if temperature_schedule is None:
        configs = [0.02, 0.01, 0.005, 0.002]
    else:
        configs = list(temperature_schedule)
    
    current_c = c0.copy()
    
    for T in configs:
        if time.time() > start_time + max_time - 1:
            break
        
        def obj(params, _T=T):
            return _energy(params, temperature=_T)
        
        try:
            result = minimize(
                obj,
                current_c.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.4999, 0.4999)] * (2 * N),
                options={'maxiter': 300, 'ftol': 1e-13, 'gtol': 1e-9, 'maxls': 100}
            )
            
            c_new = np.clip(result.x.reshape(-1, 2), -0.4999, 0.4999)
            r_new = _compute_radius(c_new)
            
            if r_new > best_r:
                best_c = c_new.copy()
                best_r = r_new
                current_c = c_new
            else:
                current_c = c_new
        except Exception:
            continue
    
    return best_c, best_r


def _slsqp_polish_hard(c0, r_target, max_time=10.0):
    """
    Hard-constrained SLSQP polish using active set of close pairs.
    Maximizes feasibility for r_target.
    """
    start_time = time.time()
    n = c0.shape[0]
    m = n * 2
    
    # Identify active pairs: those close to the minimum distance
    dists = pdist(c0, metric='euclidean')
    min_dist = np.min(dists)
    
    # Use a threshold to select active pairs
    # We want pairs that might be violating or close to violating 2*r_target
    # If r_target is close to min_dist/2, we need pairs close to min_dist
    threshold = max(min_dist * 1.2, 2.0 * r_target * 1.1)
    
    # Get indices of pairs below threshold
    # pdist returns condensed form. We need to map back to indices or use cKDTree
    # Using cKDTree for efficiency in finding close pairs
    tree = cKDTree(c0)
    # Query pairs within threshold
    # Note: cKDTree.query_pairs returns a list of tuples (i, j)
    pairs = tree.query_pairs(r=threshold, output_type='ndarray')
    
    if len(pairs) == 0:
        # No close pairs? Just check boundaries
        pairs = np.array([], dtype=int).reshape(0, 2)
        
    i_idx = pairs[:, 0] if len(pairs) > 0 else np.array([], dtype=int)
    j_idx = pairs[:, 1] if len(pairs) > 0 else np.array([], dtype=int)
    n_pairs = len(i_idx)
    
    def objective(params):
        return 0.0
    
    def grad_objective(params):
        return np.zeros(m)
    
    def pairwise_constraints(params):
        if n_pairs == 0:
            return np.array([])
        c = params.reshape(n, 2)
        ci = c[i_idx]
        cj = c[j_idx]
        diff = ci - cj
        dists = np.sqrt(np.sum(diff**2, axis=1))
        return dists - 2.0 * r_target
    
    def grad_pairwise_constraints(params):
        if n_pairs == 0:
            return np.zeros((0, m))
        c = params.reshape(n, 2)
        ci = c[i_idx]
        cj = c[j_idx]
        diff = ci - cj
        dists = np.sqrt(np.sum(diff**2, axis=1))
        dists_safe = np.where(dists > 1e-12, dists, 1e-12)
        inv_d = 1.0 / dists_safe
        
        J = np.zeros((n_pairs, m))
        J[:, i_idx * 2] += diff[:, 0] * inv_d
        J[:, i_idx * 2 + 1] += diff[:, 1] * inv_d
        J[:, j_idx * 2] -= diff[:, 0] * inv_d
        J[:, j_idx * 2 + 1] -= diff[:, 1] * inv_d
        return J
    
    def boundary_constraints(params):
        c = params.reshape(n, 2)
        x = c[:, 0]
        y = c[:, 1]
        
        left = x + 0.5 - r_target
        right = 0.5 - r_target - x
        bottom = y + 0.5 - r_target
        top = 0.5 - r_target - y
        
        return np.concatenate([left, right, bottom, top])
    
    def grad_boundary_constraints(params):
        J = np.zeros((4 * n, m))
        J[:n, np.arange(n) * 2] = 1.0
        J[n:2*n, np.arange(n) * 2] = -1.0
        J[2*n:3*n, np.arange(n) * 2 + 1] = 1.0
        J[3*n:4*n, np.arange(n) * 2 + 1] = -1.0
        return J
    
    constraints = []
    if n_pairs > 0:
        constraints.append(NonlinearConstraint(pairwise_constraints, 0.0, np.inf, 
                           jac=grad_pairwise_constraints))
    constraints.append(NonlinearConstraint(boundary_constraints, 0.0, np.inf,
                           jac=grad_boundary_constraints))
    
    # Bounds can be slightly wider to allow movement
    bounds = [(-0.5 + r_target - 0.01, 0.5 - r_target + 0.01)] * m
    
    try:
        result = minimize(
            objective,
            c0.flatten(),
            method='SLSQP',
            jac=grad_objective,
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 50, 'ftol': 1e-12, 'disp': False}
        )
        
        c_new = np.clip(result.x.reshape(n, 2), -0.5 + r_target, 0.5 - r_target)
        r_new = _compute_radius(c_new)
        
        # Check if we actually achieved at least r_target
        if r_new >= r_target - 1e-9:
            return c_new, r_new
        else:
            # Return the improved configuration even if not fully feasible for r_target,
            # but only if it's better than the input
            if r_new > _compute_radius(c0):
                return c_new, r_new
            return c0, _compute_radius(c0)
    except Exception:
        return c0, _compute_radius(c0)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_radius = 0.0
    best_c = None
    
    # Step 1: Initialization
    c_hex = _initialize_hexagonal()
    r_hex = _compute_radius(c_hex)
    best_radius = r_hex
    best_c = c_hex.copy()
    
    # Step 2: Local optimization (Soft-min) from best initial
    if time.time() < deadline - 30:
        max_opt_time = min(20.0, deadline - time.time() - 10)
        c_opt, r_opt = _local_optimize_soft(best_c, max_time=max_opt_time)
        if r_opt > best_radius:
            best_radius = r_opt
            best_c = c_opt.copy()
    
    # Step 3: A few random restarts with soft-min
    np.random.seed(42)
    n_restarts = 4
    
    for trial in range(n_restarts):
        if time.time() > deadline - 30:
            break
        
        # Jittered grid initialization
        m = int(np.ceil(np.sqrt(N)))
        grid_x, grid_y = np.meshgrid(
            np.linspace(-0.45, 0.45, m),
            np.linspace(-0.45, 0.45, m)
        )
        grid = np.column_stack([grid_x.ravel(), grid_y.ravel()])
        np.random.shuffle(grid)
        c_init = grid[:N].copy()
        
        # Add random perturbation
        c_init += np.random.normal(0, 0.01, size=(N, 2))
        c_init = np.clip(c_init, -0.4999, 0.4999)
        
        r_init = _compute_radius(c_init)
        if r_init > best_radius:
            best_radius = r_init
            best_c = c_init.copy()
        
        # Local optimize
        remaining = deadline - time.time() - 30
        if remaining > 5:
            max_opt_time = min(10.0, remaining)
            c_opt, r_opt = _local_optimize_soft(c_init, max_time=max_opt_time)
            if r_opt > best_radius:
                best_radius = r_opt
                best_c = c_opt.copy()
    
    # Step 4: Hard-constrained SLSQP Polish
    # This phase uses the active-set SLSQP to push the radius higher with exact constraints
    if time.time() < deadline - 10:
        remaining = deadline - time.time() - 5
        current_c = best_c.copy()
        current_r = best_radius
        
        # Try to push the radius up using SLSQP
        # We start with a target slightly above the current best
        # and iterate if time permits
        
        # First, just polish the current best to ensure it's as good as possible
        if remaining > 5:
            c_pol, r_pol = _slsqp_polish_hard(current_c, current_r, max_time=min(10.0, remaining/2))
            if r_pol > best_radius:
                best_radius = r_pol
                best_c = c_pol.copy()
                current_c = c_pol
                current_r = r_pol
        
        # Now, try to increase the radius target
        # We can attempt to bump the radius by a small epsilon
        # and see if SLSQP can find a feasible solution
        if time.time() < deadline - 5:
            # Estimate a safe bump size
            # The gap between current radius and theoretical max is unknown, 
            # but we can try small increments
            bump = 1e-5
            attempts = 0
            max_attempts = 10
            
            while time.time() < deadline - 5 and attempts < max_attempts:
                target_r = current_r + bump
                c_new, r_new = _slsqp_polish_hard(current_c, target_r, max_time=5.0)
                
                if r_new > current_r:
                    current_r = r_new
                    current_c = c_new
                    if r_new > best_radius:
                        best_radius = r_new
                        best_c = c_new.copy()
                    # If successful, try again with same bump
                    attempts += 1
                else:
                    # If failed, reduce bump or break
                    # Try a smaller bump
                    bump *= 0.5
                    if bump < 1e-7:
                        break
                    attempts += 1
    
    # Final clip to ensure validity
    best_c = np.clip(best_c, -0.5 + 1e-12, 0.5 - 1e-12)
    
    # Verify the result is valid
    if best_c.shape != (N, 2) or not np.all(np.isfinite(best_c)):
        best_c = _initialize_hexagonal()
        best_c = np.clip(best_c, -0.5 + 1e-12, 0.5 - 1e-12)
    
    return best_c
# EVOLVE-BLOCK-END
