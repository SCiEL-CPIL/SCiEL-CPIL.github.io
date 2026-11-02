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
    Compute the negative soft-min objective and its gradient (vectorized).
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
    
    # Flatten pair constraints (upper triangle only to avoid duplicates)
    i_idx, j_idx = np.triu_indices(n, k=1)
    pair_vals = dists[i_idx, j_idx] / 2.0
    
    # All constraint values
    all_vals = np.concatenate([pair_vals, boundary_dists])
    
    # Soft-min of all constraints
    min_val = np.min(all_vals)
    exp_vals = np.exp((all_vals - min_val) / temperature)
    sum_exp = np.sum(exp_vals)
    r_soft = min_val - temperature * np.log(sum_exp)
    
    # We want to maximize r_soft, so minimize -r_soft
    objective = -r_soft
    
    # Compute gradient weights
    weights = exp_vals / sum_exp
    
    grad = np.zeros_like(p)
    
    # Pair constraints
    w_pair = weights[:len(i_idx)]
    d_pair = dists[i_idx, j_idx]
    
    # Avoid division by zero
    mask = d_pair > 1e-12
    w_pair_safe = np.where(mask, w_pair, 0.0)
    d_pair_safe = np.where(mask, d_pair, 1.0)
    
    # diff_pair = p[i_idx] - p[j_idx]
    diff_pair = p[i_idx] - p[j_idx]
    
    # coeff = w_pair_safe / (2 * d_pair_safe)
    coeff = w_pair_safe / (2.0 * d_pair_safe)
    
    # contrib_i = coeff[:, np.newaxis] * diff_pair
    contrib_i = coeff[:, np.newaxis] * diff_pair
    
    # Add to grad
    np.add.at(grad, i_idx, contrib_i)
    np.add.at(grad, j_idx, -contrib_i)
    
    # Boundary constraints
    w_bound = weights[len(i_idx):]
    norms_safe = np.where(norms > 1e-12, norms, 1.0)
    w_bound_safe = np.where(norms > 1e-12, w_bound, 0.0)
    coeff_bound = w_bound_safe / norms_safe
    grad -= coeff_bound[:, np.newaxis] * p
    
    # Gradient of objective = -gradient of r_soft
    grad_obj = -grad
    
    return objective, grad_obj

def _force_directed_relax(points, steps=50, force_strength=0.01):
    """
    Vectorized force-directed relaxation to handle overlaps and boundary violations.
    """
    p = points.copy()
    n = len(p)
    
    for step in range(steps):
        forces = np.zeros_like(p)
        
        # Pairwise repulsion
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        
        # Avoid division by zero
        mask = dists > 1e-10
        dists_safe = np.where(mask, dists, 1.0)
        
        # Force magnitude: strength * (target - dist) / dist if dist < target
        target = 0.05
        force_mag = np.where((dists < target) & mask, force_strength * (target - dists) / dists_safe, 0.0)
        
        # Accumulate forces
        # forces[i] += sum_j force_mag[i,j] * diff[i,j]
        contrib = force_mag[:, :, np.newaxis] * diff
        forces += np.sum(contrib, axis=1)
        
        # Boundary attraction
        norms = np.linalg.norm(p, axis=1)
        boundary_mask = norms > 0.95
        norms_safe = np.where(boundary_mask, norms, 1.0)
        b_force_mag = np.where(boundary_mask, force_strength * (norms - 0.95) / norms_safe, 0.0)
        forces -= b_force_mag[:, np.newaxis] * p
        
        # Update positions
        p += forces
        # Clamp to boundary
        norms = np.linalg.norm(p, axis=1)
        out_mask = norms > 1.0
        if np.any(out_mask):
            p[out_mask] = p[out_mask] / norms[out_mask, np.newaxis] * (1.0 - 1e-10)
    
    return p

def _anneal_optimize(points, time_limit=60.0, temp_schedule=None):
    """
    Optimization with annealed temperature schedule.
    """
    start_time = time.time()
    
    if temp_schedule is None:
        # Default schedule: start high, decrease to low
        # Slightly longer at high temps to escape local minima
        temp_schedule = [0.05, 0.03, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
    
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
                    'maxiter': 300,
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
