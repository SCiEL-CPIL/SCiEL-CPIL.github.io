"""Packing 50 equal circles in a unit circle using force-directed layout and smooth-min optimization."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
DEADLINE = 145.0  # safety margin before 150s kill


def _pairwise_indices(n):
    """Return array of (i, j) pairs for i < j."""
    iu = np.triu_indices(n, k=1)
    return np.column_stack((iu[0], iu[1]))


def _score_vec(points):
    """Vectorized score computation."""
    points = points.reshape(-1, 2)
    n = len(points)
    radii = 1.0 - np.linalg.norm(points, axis=1)
    r = np.min(radii)
    # Compute all pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    # Only consider i < j
    iu = np.triu_indices(n, 1)
    pair_dists = dists[iu] / 2.0
    r = min(r, np.min(pair_dists))
    return r


def _build_contact_graph(points, r, n_contacts=6):
    """Build a contact graph based on nearest neighbors."""
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    
    # For each point, find its nearest neighbors
    edges = set()
    for i in range(n):
        # Sort by distance and take nearest n_contacts
        nearest = np.argsort(dists[i])[:n_contacts]
        for j in nearest:
            if i < j:
                edges.add((i, j))
            else:
                edges.add((j, i))
    
    return np.array(list(edges)) if edges else np.array([], dtype=int).reshape(0, 2)


def _force_directed_energy(points, r, contact_edges, k_contact=50.0, k_boundary=100.0, k_repel=10.0):
    """
    Force-directed layout energy:
    - Attractive forces along contact edges to maintain distance 2*r
    - Repulsive forces between all non-contact pairs
    - Boundary constraint: points should be at distance (1-r) from origin
    """
    n = len(points)
    energy = 0.0
    
    # Contact edge springs: want distance = 2*r
    if len(contact_edges) > 0:
        ci = points[contact_edges[:, 0]]
        cj = points[contact_edges[:, 1]]
        d = np.linalg.norm(ci - cj, axis=1)
        energy += k_contact * np.sum((d - 2 * r) ** 2)
    
    # Boundary: want |points| = 1 - r
    radii = np.linalg.norm(points, axis=1)
    energy += k_boundary * np.sum((radii - (1.0 - r)) ** 2)
    
    # Repulsive forces for all pairs (prevents overlap)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    # Only consider i < j
    iu = np.triu_indices(n, 1)
    pair_dists = dists[iu]
    # Soft repulsion: penalize when distance < 2*r
    violations = np.maximum(0, 2 * r - pair_dists)
    energy += k_repel * np.sum(violations ** 2)
    
    return energy


def _force_directed_optimize(points, r, time_limit=10.0, max_iters=500):
    """Optimize positions using force-directed layout with adaptive cooling."""
    n = len(points)
    pts = points.copy()
    best_pts = pts.copy()
    best_energy = _force_directed_energy(pts, r, _build_contact_graph(pts, r))
    
    t_start = time.time()
    T = 0.01  # temperature for random perturbations
    step_size = 0.005
    
    for it in range(max_iters):
        if time.time() - t_start >= time_limit:
            break
        
        # Build contact graph from current configuration
        contact_edges = _build_contact_graph(pts, r)
        
        # Compute gradient-like forces
        # Spring forces along contact edges
        forces = np.zeros_like(pts)
        if len(contact_edges) > 0:
            ci = pts[contact_edges[:, 0]]
            cj = pts[contact_edges[:, 1]]
            d = np.linalg.norm(ci - cj, axis=1)
            diff_vec = ci - cj
            # Normalize
            diff_vec_norm = diff_vec / np.maximum(d[:, np.newaxis], 1e-10)
            # Spring force: proportional to (d - 2*r)
            spring_force = 50.0 * (d - 2 * r)[:, np.newaxis] * diff_vec_norm
            # Apply to both endpoints
            for idx in range(len(contact_edges)):
                i, j = contact_edges[idx]
                forces[i] += spring_force[idx]
                forces[j] -= spring_force[idx]
        
        # Boundary forces: pull toward radius (1-r)
        radii = np.linalg.norm(pts, axis=1)
        radii_safe = np.maximum(radii, 1e-10)
        boundary_force = 100.0 * (radii - (1.0 - r))[:, np.newaxis] * pts / radii_safe[:, np.newaxis]
        forces -= boundary_force
        
        # Repulsive forces from overlapping pairs
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        iu = np.triu_indices(n, 1)
        pair_dists = dists[iu]
        violations = np.maximum(0, 2 * r - pair_dists)
        
        for idx in range(len(pair_dists)):
            i, j = iu[0][idx], iu[1][idx]
            if violations[idx] > 0:
                d = dists[i, j]
                if d > 1e-10:
                    force_mag = 10.0 * violations[idx] / d
                    forces[i] += force_mag * (pts[i] - pts[j])
                    forces[j] -= force_mag * (pts[i] - pts[j])
        
        # Update positions with cooling
        alpha = step_size * (0.5 + 0.5 * (1 - it / max_iters))
        pts_new = pts + alpha * forces
        # Add small random perturbation
        pts_new += T * np.random.randn(n, 2)
        
        # Check if improvement
        new_energy = _force_directed_energy(pts_new, r, _build_contact_graph(pts_new, r))
        if new_energy < best_energy or np.random.rand() < np.exp(-(new_energy - best_energy) / max(T, 1e-10)):
            pts = pts_new
            if new_energy < best_energy:
                best_energy = new_energy
                best_pts = pts_new.copy()
        
        T *= 0.995  # cool down
    
    return best_pts


def _hexagonal_init(n):
    """Generate a hexagonal lattice initial configuration."""
    best_points = None
    best_r = 0.0
    for spacing in np.linspace(0.05, 0.25, 50):
        s = spacing
        points = []
        max_coord = int(1.0 / s) + 2
        for j in range(-max_coord, max_coord + 1):
            for i in range(-max_coord, max_coord + 1):
                x = (i + 0.5 * (j % 2)) * s
                y = j * s * np.sqrt(3) / 2
                if np.hypot(x, y) <= 1.0 - s / 2:
                    points.append([x, y])
        points = np.array(points)
        if len(points) >= n:
            dists = np.linalg.norm(points, axis=1)
            idx = np.argsort(dists)[:n]
            cand = points[idx]
            r = _score_vec(cand)
            if r > best_r:
                best_r = r
                best_points = cand.copy()
    if best_points is None:
        best_points = np.random.randn(n, 2) * 0.5
    return best_points


def _ring_init(n, n_rings=5):
    """
    Initialize with ring structure matching known optimal packing topology.
    Ring 0: center (1 circle)
    Ring 1: 6 circles
    Ring 2: 12 circles
    Ring 3: 18 circles
    Ring 4: 13 circles (remaining)
    """
    points = []
    
    # Center
    points.append([0.0, 0.0])
    
    # Ring 1: 6 circles at radius ~0.25
    r1 = 0.25
    for k in range(6):
        angle = 2 * np.pi * k / 6
        points.append([r1 * np.cos(angle), r1 * np.sin(angle)])
    
    # Ring 2: 12 circles at radius ~0.5
    r2 = 0.5
    for k in range(12):
        angle = 2 * np.pi * k / 12 + np.pi / 6  # offset
        points.append([r2 * np.cos(angle), r2 * np.sin(angle)])
    
    # Ring 3: 18 circles at radius ~0.75
    r3 = 0.75
    for k in range(18):
        angle = 2 * np.pi * k / 18
        points.append([r3 * np.cos(angle), r3 * np.sin(angle)])
    
    # Ring 4: remaining circles at radius ~0.9
    remaining = n - len(points)
    r4 = 0.9
    for k in range(remaining):
        angle = 2 * np.pi * k / remaining + np.pi / remaining
        points.append([r4 * np.cos(angle), r4 * np.sin(angle)])
    
    return np.array(points[:n])


def _local_optimize(c, time_limit=10.0, beta_schedule=None):
    """Local optimization using L-BFGS-B with smooth-min objective."""
    pair_idx = _pairwise_indices(N)
    n_pairs = len(pair_idx)
    
    if beta_schedule is None:
        beta_schedule = [200.0, 300.0, 500.0]
    
    best_c = c.copy()
    best_r = _score_vec(best_c)
    
    t_start = time.time()
    remaining = time_limit - (time.time() - t_start)
    
    for beta in beta_schedule:
        if time.time() - t_start >= time_limit - 2:
            break
        
        def neg_soft_min(x):
            c_local = x.reshape(N, 2)
            ci = c_local[pair_idx[:, 0]]
            cj = c_local[pair_idx[:, 1]]
            d = np.sqrt(np.sum((ci - cj) ** 2, axis=1)) / 2
            b = 1 - np.hypot(c_local[:, 0], c_local[:, 1])
            all_vals = np.concatenate([d, b])
            return np.log(np.sum(np.exp(-beta * all_vals))) / beta
        
        def callback(x):
            if time.time() - t_start >= time_limit - 1:
                raise Exception("timeout")
        
        try:
            res = minimize(
                neg_soft_min,
                best_c.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 150, 'ftol': 1e-15, 'gtol': 1e-15},
                callback=callback
            )
            if res.success and np.all(np.isfinite(res.x)):
                cand = res.x.reshape(N, 2)
                r = _score_vec(cand)
                if r > best_r:
                    best_r = r
                    best_c = cand.copy()
        except Exception:
            pass
    
    return best_c, best_r


def _anneal(points, t_start, deadline, seed=None):
    """Simulated annealing to escape local minima."""
    if seed is not None:
        np.random.seed(seed)
    
    n = len(points)
    pts = points.copy()
    best_pts = pts.copy()
    best_score = _score_vec(best_pts)
    
    # Energy function: penalize overlaps and boundary violations
    def cost(p):
        p = p.reshape(-1, 2)
        n = len(p)
        r_target = best_score
        overlap_energy = 0.0
        # Pairwise overlaps
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(p[i] - p[j])
                if d < 2 * r_target:
                    overlap_energy += (2 * r_target - d) ** 2
        # Boundary violations
        for i in range(n):
            d = 1.0 - np.linalg.norm(p[i])
            if d < r_target:
                overlap_energy += (r_target - d) ** 2
        return overlap_energy
    
    # SA parameters
    T = 0.05  # initial temperature
    T_min = 0.0001
    cooling = 0.9995
    step_size = 0.02
    
    current_cost = cost(pts)
    
    while time.time() < deadline:
        if T < T_min:
            break
        
        # Propose a move: random perturbation of one or two points
        n_moves = np.random.randint(1, 4)
        new_pts = pts.copy()
        for _ in range(n_moves):
            idx = np.random.randint(n)
            delta = np.random.randn(2) * step_size * (1 + T)
            new_pts[idx] += delta
        
        new_cost = cost(new_pts)
        delta_cost = new_cost - current_cost
        
        # Accept with probability exp(-delta_cost / T) if delta_cost > 0
        if delta_cost < 0 or np.random.rand() < np.exp(-delta_cost / T):
            pts = new_pts
            current_cost = new_cost
            # Update best
            s = _score_vec(pts)
            if s > best_score:
                best_score = s
                best_pts = pts.copy()
        
        T *= cooling
    
    return best_pts, best_score


def _compute_r(c):
    """Compute the max common radius for given centers."""
    return _score_vec(c)


def construct():
    t_start = time.time()
    
    # Start with multiple initializations and pick the best
    inits = [
        _hexagonal_init(N),
        _ring_init(N),
    ]
    
    best_points = None
    best_score = 0.0
    
    for init in inits:
        r = _score_vec(init)
        if r > best_score:
            best_score = r
            best_points = init.copy()
    
    # Force-directed layout refinement
    if time.time() < DEADLINE - 40:
        fd_points = _force_directed_optimize(best_points, best_score, time_limit=15.0)
        fd_score = _score_vec(fd_points)
        if fd_score > best_score:
            best_score = fd_score
            best_points = fd_points.copy()
    
    # Local optimization from best init with multiple betas
    if time.time() < DEADLINE - 25:
        c_opt, r_opt = _local_optimize(best_points, time_limit=min(15.0, DEADLINE - (time.time() - t_start)))
        if r_opt > best_score:
            best_score = r_opt
            best_points = c_opt.copy()
    
    # Second round of force-directed + local optimization
    if time.time() < DEADLINE - 20:
        fd_points2 = _force_directed_optimize(best_points, best_score, time_limit=10.0)
        fd_score2 = _score_vec(fd_points2)
        if fd_score2 > best_score:
            best_score = fd_score2
            best_points = fd_points2.copy()
        
        c_opt2, r_opt2 = _local_optimize(best_points, time_limit=min(10.0, DEADLINE - (time.time() - t_start)))
        if r_opt2 > best_score:
            best_score = r_opt2
            best_points = c_opt2.copy()
    
    # Simulated annealing to escape local minima
    remaining = DEADLINE - (time.time() - t_start)
    if remaining > 25:
        sa_points, sa_score = _anneal(best_points, time.time(), time.time() + min(remaining - 15, 30), seed=42)
        if sa_score > best_score:
            best_score = sa_score
            best_points = sa_points.copy()
        
        # Local optimize after SA
        if time.time() < DEADLINE - 15:
            c_opt, r_opt = _local_optimize(best_points, time_limit=min(15.0, DEADLINE - (time.time() - t_start)))
            if r_opt > best_score:
                best_score = r_opt
                best_points = c_opt.copy()
    
    # Multi-start: perturb best and try again
    n_restarts = 0
    while time.time() < DEADLINE - 15:
        perturbation_scale = 0.02 / (1.0 + 0.1 * n_restarts)
        x0 = best_points.flatten() + np.random.randn(2 * N) * perturbation_scale
        
        n_restarts += 1
        
        try:
            c_opt, r_opt = _local_optimize(x0.reshape(N, 2), time_limit=min(10.0, DEADLINE - (time.time() - t_start)))
            if r_opt > best_score:
                best_score = r_opt
                best_points = c_opt.copy()
        except Exception:
            pass
        
        if n_restarts >= 5:
            break
    
    # Final polish with high beta
    if time.time() < DEADLINE - 5:
        c_final, r_final = _local_optimize(best_points, time_limit=min(8.0, DEADLINE - (time.time() - t_start) - 2), beta_schedule=[500.0, 800.0])
        if r_final > best_score:
            best_points = c_final
            best_score = r_final
    
    # Ensure validity: clamp points that might be slightly outside
    c_final = best_points.reshape(N, 2)
    r2 = np.hypot(c_final[:, 0], c_final[:, 1])
    for i in range(N):
        if r2[i] > 1 - best_score + 1e-10:
            scale = (1 - best_score) / r2[i]
            c_final[i] *= scale
    
    return c_final
