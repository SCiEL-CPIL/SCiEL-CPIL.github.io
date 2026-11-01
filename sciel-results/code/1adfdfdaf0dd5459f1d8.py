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


def _energy(params, temperature=0.02):
    """Smooth energy function to maximize minimum distance."""
    c = params.reshape(-1, 2)
    c = np.clip(c, -0.4999, 0.4999)
    
    # Distance to boundaries
    boundary_dists = 0.5 - np.abs(c)
    min_boundary = np.min(boundary_dists)
    
    if N < 2:
        return -min_boundary
    
    # Compute all pairwise distances
    dists = pdist(c, metric='euclidean')
    min_dist = np.min(dists)
    
    # Use a smooth approximation to the min distance
    # Soft minimum: -T * log(sum(exp(-d/T)))
    # For numerical stability, use a different approach
    # We want to maximize min_dist, so minimize -min_dist
    # But min is not differentiable, so use a soft version
    
    # Use power-sum approximation: (sum(d^p))^(-1/p) for large p approximates min
    p = 20
    # Only use distances below some threshold to focus on close pairs
    mask = dists < 0.15
    if np.any(mask):
        soft_min = np.sum(dists[mask]**p)**(-1.0/p)
    else:
        soft_min = min_dist
    
    # Combine with boundary constraint
    # We want both min_dist and min_boundary to be large
    # Use soft min for boundary too
    p_b = 20
    soft_boundary = np.sum(boundary_dists**p_b)**(-1.0/p_b)
    
    # Objective: maximize the minimum of soft_min/2 and soft_boundary
    # Minimize: -(min(soft_min/2, soft_boundary))
    # Use soft min to make it differentiable
    r_pair = soft_min / 2.0
    r_bound = soft_boundary
    
    # Soft minimum of r_pair and r_bound
    T = temperature
    soft_min_combined = -T * np.logaddexp(-r_pair/T, -r_bound/T)
    
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


def _local_optimize(c0, max_time=30.0, temperature_schedule=None):
    """Local optimization using L-BFGS-B with energy function."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    # Try multiple temperature settings
    temperatures = [0.01, 0.015, 0.02, 0.03]
    if temperature_schedule is not None:
        temperatures = temperature_schedule
    
    for T in temperatures:
        if time.time() > start_time + max_time - 2:
            break
        
        def obj(params):
            return _energy(params, temperature=T)
        
        try:
            result = minimize(
                obj,
                c0.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.4999, 0.4999)] * (2 * N),
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
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


def _find_clusters(c, radius=0.12):
    """Find clusters of nearby circles for rigid-body perturbation."""
    from scipy.spatial import cKDTree
    
    tree = cKDTree(c)
    visited = np.zeros(N, dtype=bool)
    clusters = []
    
    for i in range(N):
        if not visited[i]:
            # BFS to find all points within radius
            cluster = [i]
            visited[i] = True
            queue = [i]
            while queue:
                idx = queue.pop(0)
                neighbors = tree.query_ball_point(c[idx], radius)
                for nb in neighbors:
                    if not visited[nb]:
                        visited[nb] = True
                        cluster.append(nb)
                        queue.append(nb)
            if len(cluster) > 1:
                clusters.append(cluster)
    
    return clusters


def _force_directed_energy(c, r_target, beta=50.0):
    """Compute force-directed energy with distance-dependent repulsion."""
    n = c.shape[0]
    
    # Boundary repulsion: strong penalty for being outside the safe region
    boundary_dists = 0.5 - np.abs(c)
    boundary_viol = np.maximum(0, r_target - boundary_dists)
    boundary_energy = beta * np.sum(boundary_viol**2)
    
    if n < 2:
        return boundary_energy
    
    # Pairwise repulsion: use a kernel that sharply penalizes overlaps
    dists = pdist(c, metric='euclidean')
    min_dist = 2 * r_target
    
    # Repulsion kernel: for d < min_dist, force ~ 1/(d - min_dist + eps)^2
    # For d >= min_dist, small attractive force to pack tightly
    eps = 1e-6
    overlaps = np.maximum(0, min_dist - dists)
    repulsion_energy = beta * np.sum(overlaps**2)
    
    # Add a small term to encourage packing (minimize average distance)
    # This helps fill the square more uniformly
    mean_dist = np.mean(dists)
    packing_energy = 0.1 * mean_dist
    
    return boundary_energy + repulsion_energy + packing_energy


def _anneal_optimize(c0, max_time=40.0):
    """Multi-scale force-directed annealing with dynamic repulsion strength."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    # Multi-scale annealing: start with large steps, cool down
    T = 0.1
    T_min = 0.001
    cooling = 0.97
    
    c_current = c0.copy()
    r_current = best_r
    
    # Dynamic repulsion strength: increase as we cool to focus on fine details
    beta = 10.0
    
    step_size = 0.03
    n_accepted = 0
    iteration = 0
    
    # Track best radius at different temperature scales
    scale_best_r = best_r
    scale_change_counter = 0
    
    while time.time() < start_time + max_time - 2:
        iteration += 1
        
        # Dynamic beta: increase repulsion strength as temperature decreases
        # This focuses the search on resolving close pairs
        beta = 10.0 + 100.0 * (1.0 - T / T_min)
        
        # Choose move type based on temperature
        # At high T: more single-point moves for exploration
        # At low T: more cluster moves for exploitation
        cluster_prob = 0.3 + 0.4 * (1.0 - T / T_min)
        
        if np.random.random() < cluster_prob and iteration > 100:
            # Cluster move: move a group of nearby points together
            # Find the point with the most violations
            dists = pdist(c_current, metric='euclidean')
            r_est = _compute_radius(c_current)
            violations = np.maximum(0, 2 * r_est - dists)
            
            # Find the most violated pair
            if np.max(violations) > 0:
                max_viol_idx = np.argmax(violations)
                # Convert flat index to pair
                # For N points, flat index k corresponds to pair (i,j) where
                # k = i*(N-i-1)/2 + (j-i-1) for i < j
                # We'll just pick a random point near the violation
                idx = np.random.randint(N)
            else:
                idx = np.random.randint(N)
            
            # Find neighbors within a small radius
            from scipy.spatial import cKDTree
            tree = cKDTree(c_current)
            neighbors = tree.query_ball_point(c_current[idx], 0.08)
            
            c_trial = c_current.copy()
            displacement = np.random.normal(0, step_size * 0.5, size=2)
            for nb in neighbors:
                c_trial[nb] += displacement
        else:
            # Single point move
            c_trial = c_current.copy()
            idx = np.random.randint(N)
            c_trial[idx] += np.random.normal(0, step_size, size=2)
        
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
            if r_current > best_r:
                best_r = r_current
                best_c = c_current.copy()
                scale_best_r = r_current
        
        # Cool down
        T *= cooling
        if T < T_min:
            T = T_min
            # Reduce step size
            step_size *= 0.85
    
    return best_c, best_r


def _slsqp_polish(c0, max_time=30.0):
    """Force-directed polish with iterative radius increase."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    # Iteratively increase the target radius and solve the force-directed problem
    r_target = best_r
    max_iter = 10
    
    for polish_iter in range(max_iter):
        if time.time() > start_time + max_time - 2:
            break
        
        def objective_fd(params):
            c = params.reshape(-1, 2)
            return _force_directed_energy(c, r_target, beta=50.0)
        
        try:
            result = minimize(
                objective_fd,
                best_c.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.4999, 0.4999)] * (2 * N),
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
            )
            
            c_new = np.clip(result.x.reshape(-1, 2), -0.4999, 0.4999)
            r_new = _compute_radius(c_new)
            
            if r_new > best_r:
                best_r = r_new
                best_c = c_new.copy()
                # Increase target radius slightly
                r_target = best_r * 1.001
            else:
                # Reduce target radius
                r_target = best_r * 0.999
        except Exception:
            break
    
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
    
    # Step 2: Local optimization from hex
    if time.time() < deadline - 40:
        max_opt_time = min(40.0, deadline - time.time() - 10)
        c_opt, r_opt = _local_optimize(c_hex, max_time=max_opt_time)
        if r_opt > best_radius:
            best_radius = r_opt
            best_c = c_opt.copy()
    
    # Step 3: Multiple random restarts
    np.random.seed(42)
    n_restarts = 5
    
    for trial in range(n_restarts):
        if time.time() > deadline - 15:
            break
        
        # Generate random initial points with some structure
        # Use a jittered grid as starting point
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
        
        r_init = _compute_radius(c_init)
        if r_init > best_radius:
            best_radius = r_init
            best_c = c_init.copy()
        
        # Local optimize
        remaining = deadline - time.time() - 8
        if remaining > 5:
            max_opt_time = min(20.0, remaining)
            c_opt, r_opt = _local_optimize(c_init, max_time=max_opt_time)
            if r_opt > best_radius:
                best_radius = r_opt
                best_c = c_opt.copy()
    
    # Step 4: Force-directed annealing for refinement
    if time.time() < deadline - 15:
        remaining = deadline - time.time() - 8
        if remaining > 8:
            c_ann, r_ann = _anneal_optimize(best_c, max_time=min(25.0, remaining))
            if r_ann > best_radius:
                best_radius = r_ann
                best_c = c_ann.copy()
    
    # Step 5: Force-directed polish for final refinement
    if time.time() < deadline - 5:
        remaining = deadline - time.time() - 2
        if remaining > 5:
            c_pol, r_pol = _slsqp_polish(best_c, max_time=min(20.0, remaining))
            if r_pol > best_radius:
                best_radius = r_pol
                best_c = c_pol.copy()
    
    # Final clip to ensure validity
    best_c = np.clip(best_c, -0.5, 0.5)
    
    # Verify the result is valid
    if best_c.shape != (N, 2) or not np.all(np.isfinite(best_c)):
        # Fallback to hex
        best_c = _initialize_hexagonal()
        best_c = np.clip(best_c, -0.5, 0.5)
    
    return best_c
# EVOLVE-BLOCK-END
