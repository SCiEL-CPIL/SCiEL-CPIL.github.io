"""Packing 200 equal circles in a unit square using model-and-optimize approach."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist, cdist
from scipy.optimize import minimize


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


def _energy(params, temperature=0.01, focus_on_min=True):
    """Smooth energy function to maximize minimum distance using log-sum-exp.
    
    When focus_on_min=True, we weight the minimum distances more heavily
    to guide optimization toward improving the bottleneck constraints.
    """
    c = params.reshape(-1, 2)
    c = np.clip(c, -0.4999, 0.4999)
    
    # Distance to boundaries
    boundary_dists = 0.5 - np.abs(c)
    
    if N < 2:
        T = temperature
        soft_boundary = -T * np.log(np.sum(np.exp(-boundary_dists / T)))
        return -soft_boundary
    
    # Compute all pairwise distances
    dists = pdist(c, metric='euclidean')
    
    T = temperature
    
    # For pairwise distances: use soft-min with focus on the minimum
    min_d = np.min(dists)
    shifted = dists - min_d
    
    # Apply exponential weighting to focus on small distances
    if focus_on_min:
        # Use a sharper soft-min that focuses more on the minimum
        weight_scale = 1.0 / (T * 2.0)
        weights = np.exp(-shifted * weight_scale)
        soft_min_dist = min_d - T * np.log(np.sum(weights))
    else:
        soft_min_dist = min_d - T * np.log(np.sum(np.exp(-shifted / T)))
    
    # For boundary distances
    min_b = np.min(boundary_dists)
    shifted_b = boundary_dists - min_b
    if focus_on_min:
        weight_scale_b = 1.0 / (T * 2.0)
        weights_b = np.exp(-shifted_b * weight_scale_b)
        soft_min_bound = min_b - T * np.log(np.sum(weights_b))
    else:
        soft_min_bound = min_b - T * np.log(np.sum(np.exp(-shifted_b / T)))
    
    # Combine: we want to maximize min(soft_min_dist/2, soft_min_bound)
    r_pair = soft_min_dist / 2.0
    r_bound = soft_min_bound
    
    # Soft minimum of r_pair and r_bound
    T2 = temperature * 0.5
    soft_min_combined = -T2 * np.log(np.exp(-r_pair / T2) + np.exp(-r_bound / T2))
    
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
            # Pad with random points if we don't have enough
            c = np.array(points)
            if len(c) < N:
                extra = np.random.uniform(-0.45, 0.45, size=(N - len(c), 2))
                c = np.vstack([c, extra])
            r = _compute_radius(c)
            if r > best_r:
                best_r = r
                best_c = c.copy()
    
    if best_c is None:
        # Fallback: grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_c = grid[:N].copy()
    
    return best_c


def _initialize_jittered_grid():
    """Initialize with a jittered grid pattern."""
    m = int(np.ceil(np.sqrt(N)))
    grid_x, grid_y = np.meshgrid(
        np.linspace(-0.45, 0.45, m),
        np.linspace(-0.45, 0.45, m)
    )
    grid = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    
    # Shuffle and take N points
    np.random.shuffle(grid)
    c_init = grid[:N].copy()
    
    # Add small random perturbation
    c_init += np.random.normal(0, 0.01, size=(N, 2))
    c_init = np.clip(c_init, -0.4999, 0.4999)
    
    return c_init


def _local_optimize(c0, max_time=30.0):
    """Local optimization using L-BFGS-B with energy function."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    # Try multiple temperature settings with decreasing temperature for finer control
    # Start with coarser focus, then refine
    configs = [
        (0.03, True),   # Coarse: higher T, focus on min
        (0.02, True),
        (0.015, True),
        (0.01, True),   # Fine: lower T
        (0.008, True),  # Very fine
        (0.005, True),  # Extra fine
        (0.003, False), # Ultra fine: no focus, just smooth
    ]
    
    for T, focus in configs:
        if time.time() > start_time + max_time - 2:
            break
        
        def obj(params, _T=T, _focus=focus):
            return _energy(params, temperature=_T, focus_on_min=_focus)
        
        try:
            result = minimize(
                obj,
                c0.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.4999, 0.4999)] * (2 * N),
                options={'maxiter': 600, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
            )
            
            c_new = np.clip(result.x.reshape(-1, 2), -0.4999, 0.4999)
            r_new = _compute_radius(c_new)
            
            if r_new > best_r:
                best_c = c_new
                best_r = r_new
                c0 = c_new.copy()  # Continue from improved point
        except Exception:
            continue
    
    return best_c, best_r


def _anneal_optimize(c0, max_time=40.0):
    """Simulated-annealing-like optimization with rigid-body cluster moves."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    T = 0.05
    T_min = 0.003
    cooling = 0.97
    
    c_current = c0.copy()
    r_current = best_r
    
    step_size = 0.015
    n_accepted = 0
    stagnation_count = 0
    
    def _find_cluster(c, center_idx, radius):
        """Find circles within 'radius' of the center circle (rigid body)."""
        diffs = c - c[center_idx]
        dists = np.sqrt(np.sum(diffs**2, axis=1))
        return np.where(dists < radius)[0]
    
    while time.time() < start_time + max_time - 2:
        # Choose perturbation type: single point or rigid body cluster
        use_cluster = np.random.random() < 0.4
        
        c_trial = c_current.copy()
        
        if use_cluster:
            # Rigid body move: pick a random circle, move it and its neighbors together
            center_idx = np.random.randint(N)
            cluster_radius = 0.08 + np.random.random() * 0.04
            cluster = _find_cluster(c_current, center_idx, cluster_radius)
            
            # Apply same displacement to all circles in the cluster
            displacement = np.random.normal(0, step_size, size=2)
            c_trial[cluster] += displacement
        else:
            # Standard single-point perturbation
            mask = np.random.random((N, 2)) < 0.1
            c_trial[mask] += np.random.normal(0, step_size, size=(N, 2))[mask]
        
        c_trial = np.clip(c_trial, -0.4999, 0.4999)
        
        r_trial = _compute_radius(c_trial)
        
        # Accept if better or with probability exp((r_trial - r_current)/T)
        if r_trial >= r_current:
            accept = True
        else:
            delta = r_trial - r_current
            prob = np.exp(delta / T)
            accept = np.random.random() < prob
        
        if accept:
            c_current = c_trial
            r_current = r_trial
            n_accepted += 1
            stagnation_count = 0
            if r_current > best_r:
                best_r = r_current
                best_c = c_current.copy()
        else:
            stagnation_count += 1
        
        # Cool down
        T *= cooling
        if T < T_min:
            T = T_min
            step_size *= 0.8
            if step_size < 0.002:
                step_size = 0.002
    
    return best_c, best_r


def _slsqp_polish(c0, max_time=10.0):
    """SLSQP polish with hard constraints for final precision."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    # Use SLSQP with hard constraints: circles don't overlap and stay in bounds
    # Objective: maximize minimum clearance (equivalent to minimizing negative of it)
    # We'll use a smooth surrogate that focuses on the minimum distances
    
    def _slsqp_obj(params, current_r):
        c = params.reshape(-1, 2)
        boundary_dists = 0.5 - np.abs(c)
        min_boundary = np.min(boundary_dists)
        dists = pdist(c, metric='euclidean')
        min_pair = np.min(dists)
        r = min(min_pair / 2.0, min_boundary)
        # Penalize if we go below current best
        penalty = max(0, current_r - r) * 100
        return -r + penalty
    
    def _slsqp_constraints(c):
        """Generate constraint functions for SLSQP."""
        n = c.shape[0]
        constraints = []
        
        # Boundary constraints: each circle center must be within [-0.5+r, 0.5-r]
        # We'll use the current best radius as the constraint threshold
        r_constraint = best_r - 0.0005  # Small margin
        
        # For efficiency, only add constraints for the most critical pairs
        # Get all pairwise distances
        dists = pdist(c, metric='euclidean')
        min_dist = np.min(dists)
        
        # Find pairs that are close to the minimum
        close_pairs = np.where(dists < min_dist + 0.005)[0]
        
        # Add distance constraints for close pairs
        for idx in close_pairs[:20]:  # Limit to 20 most critical pairs
            i = int((idx - np.sqrt(2 * idx + 1) + 1) // 2)
            j = int(idx - i * (n - i - 1) - i - 1)
            # Actually, let's use a simpler approach: just use the objective
            pass
        
        return constraints
    
    # Simpler approach: use SLSQP with inequality constraints on each circle's boundary
    # and a few key pairwise distances
    
    # For practical purposes, use a multi-start SLSQP on the soft-min energy
    # with very tight tolerances
    
    configs = [
        (0.003, True),
        (0.002, True),
        (0.001, True),
    ]
    
    for T, focus in configs:
        if time.time() > start_time + max_time - 1:
            break
        
        def obj(params, _T=T, _focus=focus):
            return _energy(params, temperature=_T, focus_on_min=_focus)
        
        try:
            result = minimize(
                obj,
                best_c.flatten(),
                method='SLSQP',
                bounds=[(-0.4999, 0.4999)] * (2 * N),
                options={'maxiter': 200, 'ftol': 1e-14}
            )
            
            c_new = np.clip(result.x.reshape(-1, 2), -0.4999, 0.4999)
            r_new = _compute_radius(c_new)
            
            if r_new > best_r:
                best_c = c_new
                best_r = r_new
        except Exception:
            continue
    
    # Final L-BFGS-B polish
    for T in [0.002, 0.001]:
        if time.time() > start_time + max_time - 0.5:
            break
        
        def obj(params, _T=T):
            return _energy(params, temperature=_T, focus_on_min=True)
        
        try:
            result = minimize(
                obj,
                best_c.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.4999, 0.4999)] * (2 * N),
                options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-12, 'maxls': 80}
            )
            
            c_new = np.clip(result.x.reshape(-1, 2), -0.4999, 0.4999)
            r_new = _compute_radius(c_new)
            
            if r_new > best_r:
                best_c = c_new
                best_r = r_new
        except Exception:
            continue
    
    return best_c, best_r


def _polish(c0, max_time=15.0):
    """High-precision polish pass with very tight tolerances."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    # Phase 1: SLSQP-based polish
    if max_time > 5:
        c_sls, r_sls = _slsqp_polish(best_c, max_time=min(8.0, max_time - 3))
        if r_sls > best_r:
            best_c = c_sls
            best_r = r_sls
    
    # Phase 2: L-BFGS-B with very low temperature
    if time.time() < start_time + max_time - 1:
        configs = [
            (0.003, True),
            (0.002, True),
            (0.001, True),
            (0.0005, False),
        ]
        
        for T, focus in configs:
            if time.time() > start_time + max_time - 0.5:
                break
            
            def obj(params, _T=T, _focus=focus):
                return _energy(params, temperature=_T, focus_on_min=_focus)
            
            try:
                result = minimize(
                    obj,
                    best_c.flatten(),
                    method='L-BFGS-B',
                    bounds=[(-0.4999, 0.4999)] * (2 * N),
                    options={'maxiter': 600, 'ftol': 1e-16, 'gtol': 1e-12, 'maxls': 100}
                )
                
                c_new = np.clip(result.x.reshape(-1, 2), -0.4999, 0.4999)
                r_new = _compute_radius(c_new)
                
                if r_new > best_r:
                    best_c = c_new
                    best_r = r_new
            except Exception:
                continue
    
    return best_c, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin: 5s before 150s limit
    
    best_radius = 0.0
    best_c = None
    
    # Step 1: Hexagonal initialization
    c_hex = _initialize_hexagonal()
    r_hex = _compute_radius(c_hex)
    best_radius = r_hex
    best_c = c_hex.copy()
    
    # Step 2: Local optimization from hex (primary optimization)
    if time.time() < deadline - 30:
        max_opt_time = min(50.0, deadline - time.time() - 10)
        c_opt, r_opt = _local_optimize(c_hex, max_time=max_opt_time)
        if r_opt > best_radius:
            best_radius = r_opt
            best_c = c_opt.copy()
    
    # Step 3: One random restart for diversity
    np.random.seed(42)
    if time.time() < deadline - 25:
        c_init = _initialize_jittered_grid()
        r_init = _compute_radius(c_init)
        if r_init > best_radius:
            best_radius = r_init
            best_c = c_init.copy()
        
        # Local optimize
        remaining = deadline - time.time() - 10
        if remaining > 5:
            max_opt_time = min(25.0, remaining)
            c_opt, r_opt = _local_optimize(c_init, max_time=max_opt_time)
            if r_opt > best_radius:
                best_radius = r_opt
                best_c = c_opt.copy()
    
    # Step 4: Final refinement with annealing (with rigid-body cluster moves)
    if time.time() < deadline - 20:
        remaining = deadline - time.time() - 12
        if remaining > 10:
            c_ann, r_ann = _anneal_optimize(best_c, max_time=min(20.0, remaining))
            if r_ann > best_radius:
                best_radius = r_ann
                best_c = c_ann.copy()
    
    # Step 5: High-precision polish pass (most important for final score)
    if time.time() < deadline - 3:
        remaining = deadline - time.time() - 2
        if remaining > 3:
            c_pol, r_pol = _polish(best_c, max_time=min(18.0, remaining))
            if r_pol > best_radius:
                best_radius = r_pol
                best_c = c_pol.copy()
    
    # Final clip to ensure validity
    best_c = np.clip(best_c, -0.5 + 1e-12, 0.5 - 1e-12)
    
    # Verify the result is valid
    if best_c.shape != (N, 2) or not np.all(np.isfinite(best_c)):
        # Fallback to hex
        best_c = _initialize_hexagonal()
        best_c = np.clip(best_c, -0.5 + 1e-12, 0.5 - 1e-12)
    
    return best_c
# EVOLVE-BLOCK-END
