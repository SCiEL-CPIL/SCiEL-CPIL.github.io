"""Packing 200 equal circles in a unit square using hybrid soft-min, hard-constraint, and cluster-annealing approach."""
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
    threshold = max(min_dist * 1.2, 2.0 * r_target * 1.1)
    
    tree = cKDTree(c0)
    pairs = tree.query_pairs(r=threshold, output_type='ndarray')
    
    if len(pairs) == 0:
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
        
        if r_new >= r_target - 1e-9:
            return c_new, r_new
        else:
            if r_new > _compute_radius(c0):
                return c_new, r_new
            return c0, _compute_radius(c0)
    except Exception:
        return c0, _compute_radius(c0)


def _cluster_anneal(c0, max_time=15.0, n_iter=5000):
    """
    Simulated annealing with rigid-body cluster moves using fixed-topology constellations.
    Circles are partitioned into small groups (3-4 circles) that move and rotate
    as single rigid bodies, preserving local lattice geometry while exploring global configs.
    """
    start_time = time.time()
    current_c = c0.copy()
    current_r = _compute_radius(current_c)
    best_c = current_c.copy()
    best_r = current_r
    
    T_start = 0.008
    T_end = 0.00005
    n_steps = min(n_iter, int(max_time / 0.003))
    
    if n_steps < 200:
        n_steps = 200
    
    # Pre-compute cluster assignments using k-means-like grouping
    # Groups of 3-4 nearby circles form rigid constellations
    def _build_clusters(c, group_sizes=(3, 4)):
        """Assign circles to rigid-body clusters of 3-4 members."""
        tree = cKDTree(c)
        min_dist = 2.0 * _compute_radius(c)
        
        # Find all close pairs
        close_pairs = tree.query_pairs(r=min_dist * 1.8, output_type='ndarray')
        
        if len(close_pairs) == 0:
            # No structure; assign sequentially
            clusters = []
            for i in range(0, N, 4):
                clusters.append(list(range(i, min(i + 4, N))))
            return clusters
        
        # Build adjacency from close pairs
        adj = {i: set() for i in range(N)}
        for i, j in close_pairs:
            adj[i].add(j)
            adj[j].add(i)
        
        # Greedy cluster formation
        used = set()
        clusters = []
        
        # Sort nodes by degree (prefer well-connected nodes as seeds)
        nodes_sorted = sorted(range(N), key=lambda x: -len(adj[x]))
        
        for seed in nodes_sorted:
            if seed in used:
                continue
            
            # BFS to find a connected component of size 3-4
            cluster = [seed]
            used.add(seed)
            frontier = [seed]
            
            while len(cluster) < 3 and frontier:
                next_frontier = []
                for node in frontier:
                    for neighbor in adj[node]:
                        if neighbor not in used and len(cluster) < 4:
                            cluster.append(neighbor)
                            used.add(neighbor)
                            next_frontier.append(neighbor)
                            if len(cluster) >= 4:
                                break
                    if len(cluster) >= 4:
                        break
                frontier = next_frontier
            
            clusters.append(cluster)
        
        # Add any remaining unassigned circles as singletons
        for i in range(N):
            if i not in used:
                clusters.append([i])
                used.add(i)
        
        return clusters
    
    # Build initial cluster structure
    clusters = _build_clusters(current_c)
    
    # Store cluster reference frames for rigid-body transforms
    # For each cluster, store the centroid and relative positions
    def _get_cluster_frame(c, cluster):
        """Get centroid and relative positions for a cluster."""
        pts = c[cluster]
        centroid = np.mean(pts, axis=0)
        rel = pts - centroid
        return centroid, rel
    
    for step in range(n_steps):
        if time.time() > start_time + max_time - 1:
            break
        
        # Cooling schedule
        T = T_start * (T_end / T_start) ** (step / n_steps)
        
        # Rebuild clusters periodically to adapt to new geometry
        if step % 50 == 0:
            clusters = _build_clusters(current_c)
        
        # Pick a random cluster
        cluster_idx = np.random.randint(0, len(clusters))
        cluster = clusters[cluster_idx]
        
        if len(cluster) <= 1:
            # Single point: simple translation
            idx = cluster[0]
            delta = np.random.normal(0, T * 0.5, size=2)
            new_c = current_c.copy()
            new_c[idx] += delta
        elif len(cluster) == 2:
            # Two points: translation + rotation
            i, j = cluster[0], cluster[1]
            centroid, rel = _get_cluster_frame(current_c, cluster)
            
            # Random translation
            d_trans = np.random.normal(0, T * 0.4, size=2)
            # Random rotation angle
            d_angle = np.random.normal(0, T * 0.3)
            
            # Apply rigid transform
            cos_a = np.cos(d_angle)
            sin_a = np.sin(d_angle)
            rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            
            new_c = current_c.copy()
            new_c[i] = centroid + d_trans + rot @ rel[0]
            new_c[j] = centroid + d_trans + rot @ rel[1]
        else:
            # 3+ points: full rigid body (translation + rotation)
            centroid, rel = _get_cluster_frame(current_c, cluster)
            
            # Random translation
            d_trans = np.random.normal(0, T * 0.35, size=2)
            # Random rotation angle
            d_angle = np.random.normal(0, T * 0.25)
            
            # Apply rigid transform
            cos_a = np.cos(d_angle)
            sin_a = np.sin(d_angle)
            rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            
            new_c = current_c.copy()
            for idx, rel_pt in zip(cluster, rel):
                new_c[idx] = centroid + d_trans + rot @ rel_pt
        
        # Clip to bounds
        new_c = np.clip(new_c, -0.4999, 0.4999)
        
        # Evaluate
        new_r = _compute_radius(new_c)
        
        # Accept if better, or with probability exp((new_r - current_r) / T)
        if new_r > current_r:
            current_c = new_c
            current_r = new_r
            if new_r > best_r:
                best_r = new_r
                best_c = new_c.copy()
        else:
            delta_r = new_r - current_r
            if delta_r > -1e-10:
                current_c = new_c
                current_r = new_r
            else:
                prob = np.exp(delta_r / T)
                if np.random.random() < prob:
                    current_c = new_c
                    current_r = new_r
    
    return best_c, best_r


def _force_directed_dynamics(c0, max_time=20.0):
    """
    Force-directed dynamics using symplectic Euler with adaptive time-stepping.
    Circles interact via a high-order polynomial repulsive potential for overlaps,
    and are attracted toward a target spacing. Boundary forces push circles inward.
    """
    start_time = time.time()
    c = c0.copy().astype(np.float64)
    v = np.zeros_like(c)
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    # Parameters
    k_rep = 500.0       # Repulsion strength
    k_att = 50.0        # Attraction to target spacing
    k_bound = 200.0     # Boundary repulsion
    damping = 0.98      # Velocity damping
    target_r = _compute_radius(c)
    target_dist = 2.0 * target_r
    
    # Adaptive time step
    dt = 0.001
    max_dt = 0.005
    min_dt = 0.0001
    
    n_iter = 0
    while time.time() < start_time + max_time - 1:
        n_iter += 1
        
        # Compute forces
        forces = np.zeros_like(c)
        
        # Pairwise repulsion (using KDTree for efficiency)
        tree = cKDTree(c)
        query_r = target_dist * 1.5
        close_pairs = tree.query_pairs(r=query_r, output_type='ndarray')
        
        if len(close_pairs) > 0:
            ci = c[close_pairs[:, 0]]
            cj = c[close_pairs[:, 1]]
            diff = ci - cj
            dists = np.linalg.norm(diff, axis=1)
            
            # Repulsive force: high-order polynomial for overlaps
            # F = k_rep * (target_dist - d)^3 for d < target_dist
            overlaps = np.maximum(0, target_dist - dists)
            force_mag = k_rep * overlaps ** 3
            # Normalize direction
            dists_safe = np.where(dists > 1e-10, dists, 1e-10)
            dir_vecs = diff / dists_safe[:, np.newaxis]
            
            # Apply forces
            for idx in range(len(close_pairs)):
                i, j = close_pairs[idx]
                forces[i] += force_mag[idx] * dir_vecs[idx]
                forces[j] -= force_mag[idx] * dir_vecs[idx]
        
        # Boundary forces: push circles away from walls
        margin = 0.5 - 1e-6
        for axis in range(2):
            # Left wall
            x = c[:, axis]
            left_dist = x + margin
            left_force = k_bound * np.maximum(0, target_r - left_dist) ** 2
            forces[:, axis] += left_force
            
            # Right wall
            right_dist = margin - x
            right_force = k_bound * np.maximum(0, target_r - right_dist) ** 2
            forces[:, axis] -= right_force
        
        # Symplectic Euler integration
        v = v * damping + forces * dt
        c = c + v * dt
        
        # Clip to valid region
        c = np.clip(c, -margin + 1e-9, margin - 1e-9)
        
        # Evaluate radius
        r = _compute_radius(c)
        if r > best_r:
            best_r = r
            best_c = c.copy()
        
        # Adaptive time step: reduce if radius decreased significantly
        if r < best_r - 0.001:
            dt = max(min_dt, dt * 0.7)
        elif r > best_r + 0.0001:
            dt = min(max_dt, dt * 1.1)
        
        # Check for convergence
        if n_iter % 500 == 0 and best_r - r < 1e-8:
            break
    
    return best_c, best_r


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
    if time.time() < deadline - 40:
        max_opt_time = min(15.0, deadline - time.time() - 15)
        c_opt, r_opt = _local_optimize_soft(best_c, max_time=max_opt_time)
        if r_opt > best_radius:
            best_radius = r_opt
            best_c = c_opt.copy()
    
    # Step 3: Rigid cluster annealing to escape local minima
    if time.time() < deadline - 40:
        remaining = deadline - time.time() - 15
        max_anneal_time = min(20.0, remaining)
        c_anneal, r_anneal = _cluster_anneal(best_c, max_time=max_anneal_time)
        if r_anneal > best_radius:
            best_radius = r_anneal
            best_c = c_anneal.copy()
        
        # Local optimize after cluster annealing
        if time.time() < deadline - 25:
            remaining = deadline - time.time() - 10
            max_opt_time = min(12.0, remaining)
            c_opt, r_opt = _local_optimize_soft(best_c, max_time=max_opt_time)
            if r_opt > best_radius:
                best_radius = r_opt
                best_c = c_opt.copy()
        
        # Second round of cluster annealing with different structure
        if time.time() < deadline - 20:
            remaining = deadline - time.time() - 10
            max_anneal_time = min(12.0, remaining)
            c_anneal2, r_anneal2 = _cluster_anneal(best_c, max_time=max_anneal_time)
            if r_anneal2 > best_radius:
                best_radius = r_anneal2
                best_c = c_anneal2.copy()
            
            # Local optimize after second annealing
            if time.time() < deadline - 15:
                remaining = deadline - time.time() - 8
                max_opt_time = min(8.0, remaining)
                c_opt, r_opt = _local_optimize_soft(best_c, max_time=max_opt_time)
                if r_opt > best_radius:
                    best_radius = r_opt
                    best_c = c_opt.copy()
    
    # Step 4: A few random restarts with soft-min
    np.random.seed(42)
    n_restarts = 2
    
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
        remaining = deadline - time.time() - 15
        if remaining > 5:
            max_opt_time = min(8.0, remaining)
            c_opt, r_opt = _local_optimize_soft(c_init, max_time=max_opt_time)
            if r_opt > best_radius:
                best_radius = r_opt
                best_c = c_opt.copy()
    
    # Step 5: Hard-constrained SLSQP Polish
    if time.time() < deadline - 15:
        remaining = deadline - time.time() - 5
        current_c = best_c.copy()
        current_r = best_radius
        
        # Polish the current best
        if remaining > 5:
            c_pol, r_pol = _slsqp_polish_hard(current_c, current_r, max_time=min(10.0, remaining/2))
            if r_pol > best_radius:
                best_radius = r_pol
                best_c = c_pol.copy()
                current_c = c_pol
                current_r = r_pol
        
        # Try to increase the radius target
        if time.time() < deadline - 10:
            bump = 1e-5
            attempts = 0
            max_attempts = 8
            
            while time.time() < deadline - 5 and attempts < max_attempts:
                target_r = current_r + bump
                c_new, r_new = _slsqp_polish_hard(current_c, target_r, max_time=4.0)
                
                if r_new > current_r:
                    current_r = r_new
                    current_c = c_new
                    if r_new > best_radius:
                        best_radius = r_new
                        best_c = c_new.copy()
                    attempts += 1
                else:
                    bump *= 0.5
                    if bump < 1e-7:
                        break
                    attempts += 1
    
    # Step 6: Final ultra-precise polish with very low temperature
    if time.time() < deadline - 5:
        remaining = deadline - time.time() - 3
        if remaining > 3:
            c_pol, r_pol = _local_optimize_soft(best_c, max_time=min(8.0, remaining),
                                                temperature_schedule=[0.0003, 0.0001, 0.00005, 0.00002])
            if r_pol > best_radius:
                best_radius = r_pol
                best_c = c_pol.copy()
    
    # Step 7: Final micro-annealing with very small steps
    if time.time() < deadline - 2:
        remaining = deadline - time.time() - 1
        if remaining > 2:
            c_anneal_final, r_anneal_final = _cluster_anneal(best_c, max_time=min(5.0, remaining), n_iter=2000)
            if r_anneal_final > best_radius:
                best_radius = r_anneal_final
                best_c = c_anneal_final.copy()
    
    # Final clip to ensure validity
    best_c = np.clip(best_c, -0.5 + 1e-12, 0.5 - 1e-12)
    
    # Verify the result is valid
    if best_c.shape != (N, 2) or not np.all(np.isfinite(best_c)):
        best_c = _initialize_hexagonal()
        best_c = np.clip(best_c, -0.5 + 1e-12, 0.5 - 1e-12)
    
    return best_c
# EVOLVE-BLOCK-END
