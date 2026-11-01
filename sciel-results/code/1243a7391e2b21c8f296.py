"""Improved program for packing 300 circles in a unit circle using force-directed optimization with topological awareness."""
import numpy as np
import time

N = 300

# EVOLVE-BLOCK-START

def _compute_score(points):
    """Compute the minimum radius for a given set of points."""
    n = len(points)
    if n < 2:
        return min(1.0 - np.linalg.norm(points[0])) if n == 1 else 0.0
    
    # Compute all pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists) / 2.0
    
    # Compute distance to boundary
    boundary_dists = 1.0 - np.linalg.norm(points, axis=1)
    min_boundary = np.min(boundary_dists)
    
    return min(min_pair_dist, min_boundary)

def _hex_lattice_init(n):
    """
    Initialize points in a hexagonal lattice clipped to the unit circle.
    """
    points = []
    s = np.sqrt(2.0 * np.pi / (n * np.sqrt(3.0)))
    
    max_k = int(1.0 / (s * 0.5)) + 2
    
    candidates = []
    for j in range(-max_k, max_k + 1):
        for i in range(-max_k, max_k + 1):
            x = s * (i + 0.5 * (j % 2))
            y = s * j * np.sqrt(3.0) / 2.0
            if x*x + y*y <= 1.0:
                candidates.append([x, y])
    
    candidates = np.array(candidates)
    
    if len(candidates) < n:
        s *= 0.9
        max_k = int(1.0 / (s * 0.5)) + 3
        candidates = []
        for j in range(-max_k, max_k + 1):
            for i in range(-max_k, max_k + 1):
                x = s * (i + 0.5 * (j % 2))
                y = s * j * np.sqrt(3.0) / 2.0
                if x*x + y*y <= 1.0:
                    candidates.append([x, y])
        candidates = np.array(candidates)
        
    if len(candidates) >= n:
        radii = np.linalg.norm(candidates, axis=1)
        indices = np.argsort(radii)[:n]
        points = candidates[indices]
    else:
        points = np.random.rand(n, 2) * 2 - 1
        norms = np.linalg.norm(points, axis=1)
        mask = norms > 1.0
        points[mask] = points[mask] / norms[mask, np.newaxis] * 0.9

    points += np.random.normal(0, 1e-4, points.shape)
    
    return points

def _build_adjacency(points, threshold_factor=1.5):
    """
    Build adjacency graph based on proximity.
    Two points are adjacent if their distance is less than threshold_factor * min_pair_dist.
    """
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    
    min_dist = np.min(dists)
    threshold = threshold_factor * min_dist
    
    # Build adjacency matrix
    adj = np.zeros((n, n), dtype=bool)
    adj = dists < threshold
    np.fill_diagonal(adj, False)
    
    return adj

def _force_directed_step(points, adj, dt=0.01, iterations=500, temp=1.0):
    """
    Force-directed optimization step.
    - Repulsive force between non-adjacent pairs
    - Attractive force between adjacent pairs (to maintain connectivity)
    - Boundary repulsion
    """
    n = len(points)
    points = points.copy()
    
    for it in range(iterations):
        forces = np.zeros_like(points)
        
        # Compute all pairwise distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        
        # Repulsive forces between all pairs (stronger for closer pairs)
        # Use inverse square law
        safe_dists = np.maximum(dists, 1e-10)
        repulsion_strength = 1.0 / (safe_dists ** 2)
        np.fill_diagonal(repulsion_strength, 0.0)
        
        # Attractive forces between adjacent pairs (spring-like)
        # Target distance is the current minimum distance
        min_dist = np.min(dists)
        target_dist = min_dist
        
        # Spring force: k * (d - target) for adjacent pairs
        spring_k = 0.5
        attraction = np.zeros_like(dists)
        for i in range(n):
            for j in range(i+1, n):
                if adj[i, j]:
                    d = dists[i, j]
                    force = spring_k * (d - target_dist)
                    attraction[i, j] = force
                    attraction[j, i] = force
        
        # Combine forces
        # For each pair, force on i from j is:
        # repulsion: repulsion_strength[i,j] * (points[i] - points[j]) / dist
        # attraction: -attraction[i,j] * (points[i] - points[j]) / dist
        
        for i in range(n):
            for j in range(i+1, n):
                if dists[i, j] < 1e-10:
                    continue
                d = dists[i, j]
                unit_vec = diff[i, j] / d
                
                # Net force magnitude
                net_force = repulsion_strength[i, j] - attraction[i, j]
                
                forces[i] += net_force * unit_vec
                forces[j] -= net_force * unit_vec
        
        # Boundary repulsion
        norms = np.linalg.norm(points, axis=1)
        for i in range(n):
            if norms[i] > 0.9:
                # Push inward
                inward_force = (norms[i] - 0.9) * 2.0
                forces[i] -= inward_force * points[i] / norms[i]
        
        # Update positions with cooling
        cooling = 1.0 - (it / iterations) * 0.5
        effective_dt = dt * cooling * temp
        
        points += effective_dt * forces
        
        # Keep points inside the unit circle
        norms = np.linalg.norm(points, axis=1)
        outside = norms > 1.0
        if np.any(outside):
            points[outside] /= norms[outside, np.newaxis]
            points[outside] *= 0.999
    
    return points

def _topological_penalty(points, adj):
    """
    Compute a penalty based on the number of connected components.
    We want a single connected component (or as few as possible).
    """
    n = len(points)
    
    # Count connected components using BFS
    visited = np.zeros(n, dtype=bool)
    num_components = 0
    
    for i in range(n):
        if not visited[i]:
            num_components += 1
            # BFS
            queue = [i]
            visited[i] = True
            while queue:
                node = queue.pop(0)
                for neighbor in np.where(adj[node])[0]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)
    
    # Penalty: we want num_components = 1
    return num_components - 1

def construct():
    start_time = time.time()
    deadline = start_time + 148.0  # 2s safety margin
    
    # Initialize with hex lattice
    points = _hex_lattice_init(N)
    best_points = points.copy()
    best_score = _compute_score(points)
    
    # Precompute upper triangle indices for efficient distance computation
    iu = np.triu_indices(N, k=1)
    n_pairs = len(iu[0])
    
    # Main optimization loop
    # Phase 1: Force-directed with connectivity awareness
    phase1_deadline = deadline - 60.0
    
    while time.time() < phase1_deadline:
        # Build adjacency
        adj = _build_adjacency(points, threshold_factor=1.3)
        
        # Force-directed step
        points = _force_directed_step(points, adj, dt=0.005, iterations=200, temp=1.0)
        
        # Compute score
        score = _compute_score(points)
        
        if score > best_score:
            best_score = score
            best_points = points.copy()
        
        # Reduce temperature over time
        elapsed = time.time() - start_time
        temp = max(0.1, 1.0 - elapsed / 120.0)
    
    # Phase 2: Local optimization with soft-min (similar to original but refined)
    phase2_deadline = deadline - 5.0
    
    current_points = best_points.copy()
    
    def objective_and_grad(x, beta):
        p = x.reshape(-1, 2)
        
        i_idx, j_idx = iu
        vecs = p[i_idx] - p[j_idx]
        pair_dists = np.sqrt(np.sum(vecs**2, axis=1)) / 2.0
        
        norms = np.linalg.norm(p, axis=1)
        bound_dists = 1.0 - norms
        
        z_pairs = -beta * pair_dists
        z_bounds = -beta * bound_dists
        
        max_z_pair = np.max(z_pairs)
        max_z_bound = np.max(z_bounds)
        max_z = max(max_z_pair, max_z_bound)
        
        exp_pairs = np.exp(z_pairs - max_z)
        exp_bounds = np.exp(z_bounds - max_z)
        
        sum_exp = np.sum(exp_pairs) + np.sum(exp_bounds)
        
        log_sum_exp = max_z + np.log(sum_exp)
        
        val = log_sum_exp / beta
        
        w_pairs = exp_pairs / sum_exp
        w_bounds = exp_bounds / sum_exp
        
        grad = np.zeros_like(p)
        
        safe_pair_dists = np.where(pair_dists > 1e-12, pair_dists, 1.0)
        coeff = w_pairs / (2.0 * safe_pair_dists)
        
        np.add.at(grad, i_idx, -coeff[:, np.newaxis] * vecs)
        np.add.at(grad, j_idx, coeff[:, np.newaxis] * vecs)
        
        safe_norms = np.where(norms > 1e-12, norms, 1.0)
        bound_grad_vecs = p / safe_norms[:, np.newaxis]
        
        grad += w_bounds[:, np.newaxis] * bound_grad_vecs
        
        return val, grad
    
    from scipy.optimize import minimize
    
    # Use a few beta values for local refinement
    betas = [100.0, 1000.0, 10000.0]
    
    for beta in betas:
        if time.time() > phase2_deadline:
            break
        
        try:
            res = minimize(
                lambda x: objective_and_grad(x, beta)[0],
                current_points.flatten(),
                jac=lambda x: objective_and_grad(x, beta)[1],
                method='L-BFGS-B',
                options={
                    'maxiter': 300,
                    'gtol': 1e-10,
                    'ftol': 1e-12,
                    'maxls': 40,
                    'disp': False
                }
            )
            
            candidate = res.x.reshape(-1, 2)
            candidate_score = _compute_score(candidate)
            
            if candidate_score > best_score:
                best_score = candidate_score
                best_points = candidate
                current_points = candidate
            else:
                current_points = best_points
                
        except Exception:
            pass
    
    # Final polish
    if time.time() < deadline - 2:
        try:
            final_beta = 50000.0
            res = minimize(
                lambda x: objective_and_grad(x, final_beta)[0],
                best_points.flatten(),
                jac=lambda x: objective_and_grad(x, final_beta)[1],
                method='L-BFGS-B',
                options={
                    'maxiter': 300,
                    'gtol': 1e-12,
                    'ftol': 1e-14,
                    'maxls': 50,
                    'disp': False
                }
            )
            candidate = res.x.reshape(-1, 2)
            candidate_score = _compute_score(candidate)
            if candidate_score > best_score:
                best_score = candidate_score
                best_points = candidate
        except Exception:
            pass
    
    # Ensure points are inside
    norms = np.linalg.norm(best_points, axis=1)
    outside = norms > 1.0
    if np.any(outside):
        best_points[outside] /= norms[outside, np.newaxis]
        best_score = _compute_score(best_points)

    return best_points

# EVOLVE-BLOCK-END
