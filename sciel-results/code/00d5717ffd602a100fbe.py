"""Rewritten program for packing 200 equal circles in a unit square."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
from scipy.spatial import cKDTree


def _compute_radius(c):
    """Compute the largest common radius for given centres (vectorized via KDTree)."""
    n = c.shape[0]
    if n == 0:
        return 0.0
    boundary_dists = 0.5 - np.abs(c)
    min_boundary = np.min(boundary_dists)
    
    if n < 2:
        return min_boundary
    
    tree = cKDTree(c)
    dists, _ = tree.query(c, k=2)
    min_pair_dist = np.min(dists[:, 1])
    
    return min(min_pair_dist / 2.0, min_boundary)


def _soft_min(values, beta):
    """Smooth approximation of min using log-sum-exp (vectorized, clipped)."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return 0.0
    v_min = np.min(values)
    shifted = values - v_min
    max_shift = 40.0 / beta
    shifted = np.clip(shifted, 0.0, max_shift)
    exp_vals = np.exp(-beta * shifted)
    log_sum = np.log(np.sum(exp_vals))
    return v_min - log_sum / beta


def _smooth_objective(params, beta):
    """Smooth objective: minimize -soft_min(radii_components)."""
    c = np.clip(params.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
    n = c.shape[0]
    
    boundary_dists = 0.5 - np.abs(c)
    
    if n < 2:
        return -np.min(boundary_dists)
    
    dists = pdist(c, metric='euclidean')
    pair_dists_half = dists / 2.0
    
    all_dists = np.concatenate([pair_dists_half, boundary_dists.ravel()])
    r_soft = _soft_min(all_dists, beta)
    
    if r_soft <= 0:
        return 1.0
    return -r_soft


def _get_active_pairs(c, r_target, margin=0.005):
    """Get pairs of circles that are close to being at distance 2*r_target."""
    n = c.shape[0]
    tree = cKDTree(c)
    dist_limit = 2 * r_target + margin
    pairs = []
    for i in range(n):
        neighbors = tree.query_ball_point(c[i], dist_limit)
        for j in neighbors:
            if i < j:
                pairs.append((i, j))
    if not pairs:
        return np.empty((0, 2), dtype=np.int32)
    return np.array(pairs, dtype=np.int32)


def _initialize_hexagonal():
    """Initialize with a hexagonal pattern."""
    best_c = None
    best_r = 0
    
    for d in np.linspace(0.055, 0.075, 30):
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
    """Local optimization using smooth objective with annealing."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    betas = [12.0, 25.0, 50.0, 100.0, 200.0, 400.0, 800.0, 1600.0, 3200.0]
    
    for beta in betas:
        if time.time() > start_time + max_time - 2:
            break
        
        try:
            def obj(params):
                return _smooth_objective(params, beta)
            
            maxiter = 60 if beta < 50 else 120 if beta < 400 else 200
            
            result = minimize(
                obj,
                best_c.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.5 + 1e-12, 0.5 - 1e-12)] * (2 * N),
                options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 40}
            )
            
            c_new = np.clip(result.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
            r_new = _compute_radius(c_new)
            if r_new > best_r:
                best_c = c_new
                best_r = r_new
        except Exception:
            pass
    
    return best_c, best_r


def _velocity_verlet_packing(c0, r0, max_time=20.0):
    """
    Physics-based packing using velocity-Verlet integration with
    repulsive potentials and collision damping.
    """
    start_time = time.time()
    best_c = c0.copy()
    best_r = r0
    n = N
    
    # Initial state
    pos = c0.copy().astype(np.float64)
    vel = np.zeros_like(pos)
    
    # Parameters
    dt = 0.0001
    damping = 0.98
    repulsion_strength = 1.0
    target_r = r0 + 1e-6
    
    # Build neighbor list using KDTree
    tree = cKDTree(pos)
    
    max_steps = 5000
    for step in range(max_steps):
        if time.time() > start_time + max_time:
            break
        
        # Compute forces
        forces = np.zeros_like(pos)
        
        # Pair repulsions
        for i in range(n):
            neighbors = tree.query_ball_point(pos[i], 2.5 * target_r)
            for j in neighbors:
                if i >= j:
                    continue
                diff = pos[i] - pos[j]
                dist = np.linalg.norm(diff)
                if dist < 2 * target_r and dist > 1e-12:
                    # Repulsive force: stronger when closer
                    overlap = 2 * target_r - dist
                    force_mag = repulsion_strength * overlap
                    forces[i] += force_mag * diff / dist
                    forces[j] -= force_mag * diff / dist
        
        # Boundary forces
        for i in range(n):
            for k in range(2):
                if pos[i, k] < -0.5 + target_r:
                    forces[i, k] += repulsion_strength * (-0.5 + target_r - pos[i, k])
                elif pos[i, k] > 0.5 - target_r:
                    forces[i, k] -= repulsion_strength * (pos[i, k] - (0.5 - target_r))
        
        # Velocity-Verlet integration
        pos += vel * dt + 0.5 * forces * dt * dt
        vel_new = vel + forces * dt
        vel = vel_new * damping
        
        # Project to square
        pos = np.clip(pos, -0.5 + 1e-12, 0.5 - 1e-12)
        
        # Update best
        r_curr = _compute_radius(pos)
        if r_curr > best_r:
            best_r = r_curr
            best_c = pos.copy()
        
        # Rebuild tree periodically
        if step % 100 == 0:
            tree = cKDTree(pos)
        
        # Adaptive dt: reduce if energy is high
        max_force = np.max(np.abs(forces))
        if max_force > 1.0:
            dt *= 0.9
        elif max_force < 0.1:
            dt = min(dt * 1.1, 0.0002)
    
    return best_c, best_r


def _simulated_annealing_rigid(c0, r0, max_time=40.0):
    """
    Simulated annealing with rigid-body cluster moves to escape local minima.
    """
    start_time = time.time()
    best_c = c0.copy()
    best_r = r0
    current_c = c0.copy()
    current_r = r0
    
    T_start = 0.002
    T_end = 1e-9
    n_steps = 8000
    
    def _build_clusters(c, r_target):
        """Group circles into clusters based on proximity."""
        n = c.shape[0]
        tree = cKDTree(c)
        dist_limit = 2.5 * r_target
        
        parent = list(range(n))
        
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        for i in range(n):
            neighbors = tree.query_ball_point(c[i], dist_limit)
            for j in neighbors:
                if i < j:
                    union(i, j)
        
        clusters = {}
        for i in range(n):
            root = find(i)
            if root not in clusters:
                clusters[root] = []
            clusters[root].append(i)
        
        return list(clusters.values())
    
    clusters = _build_clusters(current_c, current_r)
    cluster_rebuild_interval = 800
    
    for step in range(n_steps):
        if time.time() > start_time + max_time:
            break
        
        T = T_start * (T_end / T_start) ** (step / n_steps)
        
        if step % cluster_rebuild_interval == 0 and step > 0:
            clusters = _build_clusters(current_c, current_r)
        
        if np.random.random() < 0.55 and len(clusters) > 1:
            cluster = clusters[np.random.randint(len(clusters))]
            n_cluster = len(cluster)
            scale = 1.0 / np.sqrt(max(n_cluster, 1))
            angle = np.random.random() * 2 * np.pi
            magnitude = np.random.exponential(0.004 * scale)
            dx = magnitude * np.cos(angle)
            dy = magnitude * np.sin(angle)
            
            new_c = current_c.copy()
            for idx in cluster:
                new_c[idx, 0] += dx
                new_c[idx, 1] += dy
        else:
            idx = np.random.randint(N)
            angle = np.random.random() * 2 * np.pi
            magnitude = np.random.exponential(0.003)
            dx = magnitude * np.cos(angle)
            dy = magnitude * np.sin(angle)
            
            new_c = current_c.copy()
            new_c[idx, 0] += dx
            new_c[idx, 1] += dy
        
        new_c = np.clip(new_c, -0.5 + 1e-12, 0.5 - 1e-12)
        
        new_r = _compute_radius(new_c)
        delta_r = new_r - current_r
        
        if delta_r > 0 or (T > 1e-12 and np.random.random() < np.exp(delta_r / T)):
            current_c = new_c
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_c = current_c.copy()
    
    return best_c, best_r


def _slsqp_polish(c0, r0, max_time=30.0):
    """
    High-precision polish using SLSQP with active constraints.
    """
    start_time = time.time()
    best_c = c0.copy()
    best_r = r0
    n = N
    
    step = 5e-5
    max_iter = 80
    
    for _ in range(max_iter):
        if time.time() > start_time + max_time:
            break
        
        r_target = best_r + step
        
        pairs = _get_active_pairs(best_c, r_target, margin=0.005)
        npairs = len(pairs)
        
        bd = 0.5 - np.abs(best_c)
        
        def constraints_func(x):
            c = x.reshape(-1, 2)
            bd_all = 0.5 - np.abs(c) - r_target
            bd_flat = bd_all.ravel()
            
            if npairs == 0:
                return bd_flat
            
            ci = c[pairs[:, 0]]
            cj = c[pairs[:, 1]]
            diffs = ci - cj
            dists = np.sqrt(np.sum(diffs**2, axis=1))
            pd_vals = dists - 2 * r_target
            
            return np.concatenate([bd_flat, pd_vals])
        
        def jac_func(x):
            c = x.reshape(-1, 2)
            total_constraints = 2 * n + npairs
            jac = np.zeros((total_constraints, 2 * n))
            
            signs = -np.sign(c)
            idx = np.arange(n)
            jac[2*idx, 2*idx] = signs[:, 0]
            jac[2*idx+1, 2*idx+1] = signs[:, 1]
            
            if npairs == 0:
                return jac
            
            ci = c[pairs[:, 0]]
            cj = c[pairs[:, 1]]
            diffs = ci - cj
            dists = np.sqrt(np.sum(diffs**2, axis=1))
            dists_safe = np.where(dists < 1e-12, 1e-12, dists)
            unit_vecs = diffs / dists_safe[:, np.newaxis]
            
            row_indices = 2 * n + np.arange(npairs)
            col_i = pairs[:, 0] * 2
            col_j = pairs[:, 1] * 2
            
            np.add.at(jac, (row_indices, col_i), unit_vecs[:, 0])
            np.add.at(jac, (row_indices, col_i + 1), unit_vecs[:, 1])
            np.add.at(jac, (row_indices, col_j), -unit_vecs[:, 0])
            np.add.at(jac, (row_indices, col_j + 1), -unit_vecs[:, 1])
            
            return jac
        
        try:
            result = minimize(
                lambda x: 0.0,
                best_c.flatten(),
                method='SLSQP',
                bounds=[(-0.5 + 1e-12, 0.5 - 1e-12)] * (2 * n),
                constraints=[
                    {'type': 'ineq', 'fun': constraints_func, 'jac': jac_func}
                ],
                options={'maxiter': 150, 'ftol': 1e-15, 'disp': False}
            )
            
            c_new = np.clip(result.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
            r_new = _compute_radius(c_new)
            
            if r_new > best_r + 1e-14:
                best_c = c_new
                best_r = r_new
                step = min(step * 1.15, 1e-4)
            else:
                step *= 0.5
                if step < 1e-9:
                    break
                    
        except Exception:
            break
            
    return best_c, best_r


def _exact_local_search(c0, r0, max_time=15.0):
    """
    Exact local search: try small perturbations of each circle to improve radius.
    """
    start_time = time.time()
    best_c = c0.copy()
    best_r = r0
    n = N
    
    for round_num in range(8):
        if time.time() > start_time + max_time:
            break
        
        improved = False
        for i in range(n):
            if time.time() > start_time + max_time:
                break
            
            for trial in range(12):
                angle = trial * 2 * np.pi / 12 + np.random.random() * np.pi / 6
                magnitude = 5e-7 * (1 + round_num * 0.3)
                dx = magnitude * np.cos(angle)
                dy = magnitude * np.sin(angle)
                
                new_c = best_c.copy()
                new_c[i, 0] += dx
                new_c[i, 1] += dy
                new_c = np.clip(new_c, -0.5 + 1e-12, 0.5 - 1e-12)
                
                new_r = _compute_radius(new_c)
                if new_r > best_r + 1e-15:
                    best_c = new_c
                    best_r = new_r
                    improved = True
        
        if not improved:
            break
    
    return best_c, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 147  # Safety margin (evaluator kills at 150s)
    
    best_radius = 0.0
    best_c = None
    
    # 1. Start with hexagonal initialization
    c_hex = _initialize_hexagonal()
    r_hex = _compute_radius(c_hex)
    if r_hex > best_radius:
        best_radius = r_hex
        best_c = c_hex.copy()
    
    # 2. Primary optimization with smooth objective
    if time.time() < deadline - 65:
        remaining = deadline - time.time() - 65
        c_opt, r_opt = _local_optimize_smooth(c_hex, max_time=min(30.0, remaining))
        if r_opt > best_radius:
            best_radius = r_opt
            best_c = c_opt.copy()
    
    # 3. Physics-based packing (velocity-Verlet) to escape local minima
    if best_c is not None and time.time() < deadline - 55:
        remaining = max(8.0, deadline - time.time() - 55)
        c_phys, r_phys = _velocity_verlet_packing(best_c, best_radius, max_time=min(18.0, remaining))
        if r_phys > best_radius:
            best_radius = r_phys
            best_c = c_phys.copy()
    
    # 4. Simulated annealing with rigid-body moves
    if best_c is not None and time.time() < deadline - 40:
        remaining = max(8.0, deadline - time.time() - 40)
        c_sa, r_sa = _simulated_annealing_rigid(best_c, best_radius, max_time=min(22.0, remaining))
        if r_sa > best_radius:
            best_radius = r_sa
            best_c = c_sa.copy()
    
    # 5. Another smooth optimization pass
    if best_c is not None and time.time() < deadline - 25:
        remaining = max(5.0, deadline - time.time() - 25)
        c_opt2, r_opt2 = _local_optimize_smooth(best_c, max_time=min(15.0, remaining))
        if r_opt2 > best_radius:
            best_radius = r_opt2
            best_c = c_opt2.copy()
    
    # 6. SLSQP high-precision polish (more time allocated)
    if best_c is not None and time.time() < deadline - 10:
        remaining = max(5.0, deadline - time.time() - 10)
        c_final, r_final = _slsqp_polish(best_c, best_radius, max_time=min(20.0, remaining))
        if r_final > best_radius:
            best_radius = r_final
            best_c = c_final
    
    # 7. Exact local search for final precision
    if best_c is not None and time.time() < deadline - 1:
        remaining = max(1.0, deadline - time.time() - 1)
        c_exact, r_exact = _exact_local_search(best_c, best_radius, max_time=min(8.0, remaining))
        if r_exact > best_radius:
            best_radius = r_exact
            best_c = c_exact
    
    if best_c is None:
        best_c = _initialize_hexagonal()
    
    best_c = np.clip(best_c, -0.5, 0.5)
    
    return best_c
# EVOLVE-BLOCK-END
