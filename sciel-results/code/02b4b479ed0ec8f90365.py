"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

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

def _hex_lattice_init(n, spacing_scale=0.95, rotation=0.0):
    """
    Initialize points on a hexagonal lattice clipped to the unit disk.
    """
    # Estimate spacing for n points
    spacing = 2.0 / (np.sqrt(n) + 1.0) * spacing_scale
    
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        
        # Apply rotation
        if rotation != 0.0:
            cos_r = np.cos(rotation)
            sin_r = np.sin(rotation)
            points[:, 0], points[:, 1] = points[:, 0] * cos_r - points[:, 1] * sin_r, points[:, 0] * sin_r + points[:, 1] * cos_r
        
        radius = np.hypot(points[:, 0], points[:, 1])
        # Keep points strictly inside
        mask = radius <= 1.0 - spacing / 2.0
        points, radius = points[mask], radius[mask]
        if len(points) >= n:
            # Sort by radius to pick the most central ones
            points = points[np.argsort(radius, kind="stable")[:n]]
            break
        spacing *= 0.98
        
    return points

def _ring_init(n, num_rings=None):
    """
    Initialize points in concentric rings.
    """
    if num_rings is None:
        num_rings = int(np.sqrt(n) / 2) + 2
    
    points = [[0.0, 0.0]]
    remaining = n - 1
    
    # Distribute remaining points into rings
    ring_counts = []
    for i in range(1, num_rings + 1):
        ring_counts.append(int(remaining * i / (num_rings * (num_rings + 1) / 2.0)))
    
    while sum(ring_counts) < remaining:
        ring_counts[-1] += 1
    while sum(ring_counts) > remaining:
        ring_counts[-1] -= 1
    
    current_count = 0
    for i, count in enumerate(ring_counts):
        if count == 0:
            continue
        ring_radius = (i + 1) / (num_rings + 1) * 0.95
        angles = np.linspace(0, 2 * np.pi, count, endpoint=False) + np.pi / count
        for angle in angles:
            points.append([ring_radius * np.cos(angle), ring_radius * np.sin(angle)])
    
    return np.array(points[:n])

def _soft_min(values, temperature=0.01):
    """
    Compute a smooth approximation of the minimum using the log-sum-exp trick.
    """
    values = np.array(values)
    min_val = np.min(values)
    exp_vals = np.exp((values - min_val) / temperature)
    return min_val - temperature * np.log(np.sum(exp_vals))

def _soft_min_grad(values, temperature=0.01):
    """
    Compute gradient of the soft-min with respect to the input values.
    """
    values = np.array(values)
    min_val = np.min(values)
    exp_vals = np.exp((values - min_val) / temperature)
    weights = exp_vals / np.sum(exp_vals)
    return weights

def _objective_and_grad(x, temperature=0.01):
    """
    Compute the negative soft-min objective and its gradient.
    """
    p = x.reshape(-1, 2)
    n = len(p)
    
    # Calculate pairwise distances
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=-1)
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(dists_sq)
    
    # Boundary distances
    norms = np.linalg.norm(p, axis=1)
    boundary_dists = 1.0 - norms
    
    # Collect all constraint values
    pair_constr = dists.copy()
    np.fill_diagonal(pair_constr, np.inf)
    pair_constr = pair_constr / 2.0
    
    # Flatten pair constraints (upper triangle only to avoid duplicates)
    i_idx, j_idx = np.triu_indices(n, k=1)
    pair_vals = pair_constr[i_idx, j_idx]
    
    # All constraint values
    all_vals = np.concatenate([pair_vals, boundary_dists])
    
    # Soft-min of all constraints
    r_soft = _soft_min(all_vals, temperature)
    
    # We want to maximize r_soft, so minimize -r_soft
    objective = -r_soft
    
    # Compute gradient
    weights = _soft_min_grad(all_vals, temperature)
    
    grad = np.zeros_like(p)
    
    # Pair constraints
    for k, (i, j) in enumerate(zip(i_idx, j_idx)):
        d = dists[i, j]
        if d < 1e-12:
            continue
        w = weights[k]
        u = (p[i] - p[j]) / d
        grad[i] += w * u / 2.0
        grad[j] += w * (-u) / 2.0
    
    # Boundary constraints
    offset = len(pair_vals)
    for k in range(n):
        w = weights[offset + k]
        norm_i = norms[k]
        if norm_i < 1e-12:
            continue
        grad[k] += w * (-p[k] / norm_i)
    
    # Gradient of objective = -gradient of r_soft
    grad_obj = -grad
    
    return objective, grad_obj

def _force_directed_relax(points, steps=50, force_strength=0.01):
    """
    Simple force-directed relaxation to handle overlaps and boundary violations.
    """
    p = points.copy()
    n = len(p)
    
    for step in range(steps):
        forces = np.zeros_like(p)
        
        # Pairwise repulsion
        for i in range(n):
            for j in range(i + 1, n):
                diff = p[i] - p[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    diff = np.array([0.01, 0.01])
                    dist = np.linalg.norm(diff)
                # Repel if too close
                if dist < 0.05:
                    force = force_strength * (0.05 - dist) / dist
                    forces[i] += force * diff
                    forces[j] -= force * diff
        
        # Boundary attraction
        for i in range(n):
            norm = np.linalg.norm(p[i])
            if norm > 0.95:
                force = force_strength * (norm - 0.95) / norm
                forces[i] -= force * p[i]
        
        # Update positions
        p += forces
        # Clamp to boundary
        norms = np.linalg.norm(p, axis=1)
        for i in range(n):
            if norms[i] > 1.0:
                p[i] = p[i] / norms[i] * (1.0 - 1e-10)
    
    return p

def _anneal_optimize(points, time_limit=60.0, temp_schedule=None):
    """
    Optimization with annealed temperature schedule.
    """
    start_time = time.time()
    
    if temp_schedule is None:
        # Default schedule: start high, decrease to low
        temp_schedule = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002]
    
    current_points = points.copy()
    current_score = _compute_score(current_points)
    
    for temp in temp_schedule:
        if time.time() - start_time > time_limit:
            break
        
        try:
            res = minimize(
                lambda x: _objective_and_grad(x, temp)[0],
                current_points.flatten(),
                jac=lambda x: _objective_and_grad(x, temp)[1],
                method='L-BFGS-B',
                options={
                    'maxiter': 200,
                    'gtol': 1e-6,
                    'ftol': 1e-10,
                    'maxls': 30,
                    'disp': False
                }
            )
            
            candidate = res.x.reshape(-1, 2)
            candidate_score = _compute_score(candidate)
            
            if candidate_score > current_score:
                current_points = candidate
                current_score = candidate_score
        
        except Exception:
            pass
    
    return current_points, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Try multiple initializations and pick the best
    initializations = [
        ("hex", _hex_lattice_init(N, 0.95, rotation=0.0)),
        ("hex_rot", _hex_lattice_init(N, 0.95, rotation=np.pi/12)),
        ("hex_tight", _hex_lattice_init(N, 0.90, rotation=0.0)),
        ("hex_loose", _hex_lattice_init(N, 1.00, rotation=np.pi/12)),
        ("ring", _ring_init(N, 8)),
        ("ring_more", _ring_init(N, 12)),
    ]
    
    best_points = None
    best_score = -np.inf
    
    for name, points in initializations:
        if time.time() > deadline - 30:
            break
        
        # Apply force-directed relaxation
        points_relaxed = _force_directed_relax(points, steps=20)
        
        score = _compute_score(points_relaxed)
        if score > best_score:
            best_points = points_relaxed.copy()
            best_score = score
    
    if best_points is None:
        best_points = _hex_lattice_init(N, 0.95)
        best_score = _compute_score(best_points)
    
    # Run annealed optimization from the best initial point
    if time.time() < deadline - 20:
        best_points, best_score = _anneal_optimize(best_points, time_limit=25.0)
    
    # If time remains, try a high-precision polish
    if time.time() < deadline - 15:
        # Use very low temperature for precise optimization
        try:
            res = minimize(
                lambda x: _objective_and_grad(x, 0.0001)[0],
                best_points.flatten(),
                jac=lambda x: _objective_and_grad(x, 0.0001)[1],
                method='L-BFGS-B',
                options={
                    'maxiter': 300,
                    'gtol': 1e-8,
                    'ftol': 1e-12,
                    'maxls': 50,
                    'disp': False
                }
            )
            
            candidate = res.x.reshape(-1, 2)
            candidate_score = _compute_score(candidate)
            
            if candidate_score > best_score:
                best_points = candidate
                best_score = candidate_score
        except Exception:
            pass
    
    # Final validation and cleanup
    norms = np.linalg.norm(best_points, axis=1)
    if np.any(norms > 1.0):
        for i in range(len(best_points)):
            if norms[i] > 1.0:
                best_points[i] = best_points[i] / norms[i] * (1.0 - 1e-10)
        best_score = _compute_score(best_points)
    
    return best_points

# EVOLVE-BLOCK-END
