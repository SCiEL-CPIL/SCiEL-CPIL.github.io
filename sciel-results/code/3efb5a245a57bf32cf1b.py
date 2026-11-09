import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

N = 300

# EVOLVE-BLOCK-START

def _get_score(points):
    """
    Computes the score (common radius r) for a given set of points.
    r = min( min_{i<j} ||c_i - c_j|| / 2, min_i (1 - ||c_i||) )
    """
    if points is None or len(points) != N:
        return 0.0
    
    # Check for non-finite values
    if not np.all(np.isfinite(points)):
        return 0.0

    # Boundary constraint: points must be inside unit circle
    norms = np.linalg.norm(points, axis=1)
    boundary_r = 1.0 - np.max(norms)
    
    if boundary_r < 0:
        return 0.0

    # Pairwise distances
    d = pdist(points)
    if len(d) == 0:
        pair_r = float('inf')
    else:
        pair_r = np.min(d) / 2.0

    r = min(pair_r, boundary_r)
    return max(0.0, r)

def _energy_and_grad(x, r_target):
    """
    Compute energy and gradient for a fixed target radius r_target.
    Energy = Sum of squared overlaps + Sum of squared boundary violations.
    """
    points = x.reshape(-1, 2)
    n = len(points)
    
    # Compute pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Avoid division by zero in gradient calculation
    dists_safe = np.where(dists < 1e-8, 1e-8, dists)
    
    # Overlap penalty: max(0, 2*r - d)^2
    overlap_terms = np.maximum(0.0, 2.0 * r_target - dists)
    
    E_overlap = 0.5 * np.sum(overlap_terms ** 2)
    
    # Gradient of overlap energy w.r.t points
    factor = np.zeros_like(dists)
    mask = dists < 2.0 * r_target
    # Avoid div by zero
    factor[mask] = 2.0 * (2.0 * r_target - dists[mask]) / dists_safe[mask]
    
    # grad_overlap[i] = sum_j factor[i,j] * diff[i,j]
    grad_overlap = np.sum(factor[:, :, np.newaxis] * diff, axis=1)
    
    # Boundary penalty: max(0, ||x_i|| - (1-r))^2
    norms = np.linalg.norm(points, axis=1)
    boundary_violation = np.maximum(0.0, norms - (1.0 - r_target))
    E_boundary = np.sum(boundary_violation ** 2)
    
    # Gradient of boundary energy
    grad_boundary = np.zeros_like(points)
    b_mask = norms > (1.0 - r_target)
    norms_safe = np.where(norms < 1e-8, 1e-8, norms)
    grad_boundary[b_mask] = 2.0 * (norms[b_mask] - (1.0 - r_target)) * (points[b_mask] / norms_safe[b_mask, np.newaxis])
    
    total_energy = E_overlap + E_boundary
    total_grad = (grad_overlap + grad_boundary).flatten()
    
    return total_energy, total_grad

def _hexagonal_lattice(spacing, rotation=0.0):
    """
    Generate a hexagonal lattice of points inside the unit disk.
    """
    k = int(1.0 / spacing) + 5
    pts = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            # Apply rotation
            x_rot = x * np.cos(rotation) - y * np.sin(rotation)
            y_rot = x * np.sin(rotation) + y * np.cos(rotation)
            if np.hypot(x_rot, y_rot) <= 1.0:
                pts.append((x_rot, y_rot))
    
    if len(pts) < N:
        return None
    
    pts = np.array(pts)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists, kind="stable")[:N]
    return pts[idx]

def _concentric_rings(n_rings):
    """
    Generate a concentric rings packing.
    """
    if n_rings <= 0:
        return None
    
    if n_rings == 1:
        angles = 2 * np.pi * np.arange(N) / N
        return np.column_stack([0.5 * np.cos(angles), 0.5 * np.sin(angles)])

    pts = [[0.0, 0.0]]
    remaining = N - 1
    
    ring_radii = np.linspace(0.1, 0.95, n_rings - 1)
    weights = ring_radii
    weights /= np.sum(weights)
    
    counts = np.round(weights * remaining).astype(int)
    diff = remaining - np.sum(counts)
    if diff != 0:
        idx = np.argmax(counts) if diff > 0 else np.argmax(-counts)
        counts[idx] += diff
    
    for i, count in enumerate(counts):
        if count <= 0:
            continue
        r = ring_radii[i]
        angles = 2 * np.pi * np.arange(count) / count
        offset = 0 if i % 2 == 0 else np.pi / count
        x = r * np.cos(angles + offset)
        y = r * np.sin(angles + offset)
        for j in range(count):
            pts.append([x[j], y[j]])
            
    if len(pts) != N:
        return None
        
    return np.array(pts)

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    candidates = []
    
    # 1. Generate initial candidates from various geometric structures
    
    # Hexagonal lattice with various spacings and rotations
    # Rotations help break symmetries and find better local minima
    for spacing in [0.05, 0.055, 0.06, 0.065, 0.07]:
        for rot in [0.0, np.pi/6, np.pi/12]:
            pts = _hexagonal_lattice(spacing, rotation=rot)
            if pts is not None:
                r = _get_score(pts)
                if r > 0:
                    candidates.append((pts, r))
                
    # Concentric rings with various ring counts
    for n_rings in [10, 15, 20]:
        pts = _concentric_rings(n_rings)
        if pts is not None:
            r = _get_score(pts)
            if r > 0:
                candidates.append((pts, r))
                
    if not candidates:
        # Fallback
        angles = 2 * np.pi * np.arange(N) / N
        return np.column_stack([0.5 * np.cos(angles), 0.5 * np.sin(angles)])
        
    # Sort candidates by initial score
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    best_points = candidates[0][0].copy()
    best_score = candidates[0][1]
    
    # 2. Optimize using Gradient Descent (L-BFGS-B) with Penalty Method
    # We will try to improve the top distinct candidates.
    
    # Identify unique candidates to avoid redundant optimization
    unique_candidates = []
    seen_keys = set()
    for pts, r in candidates:
        # Create a coarse hash to check uniqueness
        key = tuple(np.round(pts, 2).flatten())
        if key not in seen_keys:
            seen_keys.add(key)
            unique_candidates.append((pts, r))
            
    # Limit the number of candidates we optimize to manage time
    max_candidates_to_optimize = 5
    candidates_to_optimize = unique_candidates[:max_candidates_to_optimize]
    
    for init_pts, init_r in candidates_to_optimize:
        if time.time() > deadline - 15.0:
            break
            
        # Current radius estimate
        current_r = init_r
        
        # We will try to increase the radius.
        step = 0.0005
        max_steps = 30 # Max radius increase steps
        
        # Optimization settings
        options = {
            'maxiter': 150,
            'ftol': 1e-8,
            'gtol': 1e-5
        }
        
        x_current = init_pts.flatten().copy()
        
        for step_idx in range(max_steps):
            if time.time() > deadline - 10.0:
                break
                
            # Try a slightly larger radius
            target_r = current_r + step
            
            # Check if this is even possible theoretically (area packing)
            # 300 * pi * r^2 < pi * 1^2 -> r < 1/sqrt(300) ~ 0.0577
            if target_r > 0.057:
                break
            
            # Define objective and gradient for this target_r
            def obj_grad(x):
                e, g = _energy_and_grad(x, target_r)
                return e, g
                
            try:
                # Use L-BFGS-B
                result = minimize(obj_grad, x_current, method='L-BFGS-B', jac=True, options=options)
                
                # Check if the optimization succeeded in reducing energy to near zero
                final_energy = result.fun
                # If energy is small, the configuration is valid for target_r
                if final_energy < 1e-5:
                    x_current = result.x
                    current_r = target_r
                    
                    # Update best
                    opt_points = x_current.reshape(-1, 2)
                    opt_score = _get_score(opt_points)
                    if opt_score > best_score:
                        best_score = opt_score
                        best_points = opt_points.copy()
                        
                    # If we are close to the theoretical limit, decrease step
                    if current_r > 0.05:
                        step *= 0.9
                else:
                    # If energy is high, we couldn't fit target_r.
                    # Try a smaller step or break?
                    # For now, break to save time for other candidates
                    break
                    
            except Exception as e:
                # If optimization fails, break
                break
                
    # 3. Final check and return
    return best_points

# EVOLVE-BLOCK-END
