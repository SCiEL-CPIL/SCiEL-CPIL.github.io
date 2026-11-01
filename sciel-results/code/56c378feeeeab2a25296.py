"""Radial-ring decomposition for packing 300 equal circles in a unit circle."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import cdist
import time


def _build_points(radii, phases, ring_counts):
    """
    Build point array from ring radii, phases, and counts.
    Each ring i has count_i points at radius radii[i] with angular offset phases[i].
    """
    points = []
    for i, (r, phase, count) in enumerate(zip(radii, phases, ring_counts)):
        if count == 0:
            continue
        if count == 1:
            # Single point: place at angle phase
            x = r * np.cos(phase)
            y = r * np.sin(phase)
            points.append(np.array([[x, y]]))
        else:
            angles = np.linspace(0, 2 * np.pi, count, endpoint=False) + phase
            xs = r * np.cos(angles)
            ys = r * np.sin(angles)
            points.append(np.column_stack([xs, ys]))
    
    if not points:
        return np.zeros((0, 2))
    
    return np.vstack(points)


def _compute_radius(points):
    """Compute the largest common radius for the given points."""
    if len(points) == 0:
        return 0.0
    if len(points) == 1:
        return 1.0 - np.hypot(*points[0])
    
    # Distance to boundary
    dists_boundary = 1.0 - np.hypot(points[:, 0], points[:, 1])
    min_boundary = np.min(dists_boundary)
    
    # Pairwise distances
    dist_matrix = cdist(points, points)
    np.fill_diagonal(dist_matrix, np.inf)
    min_pair = np.min(dist_matrix) / 2.0
    
    return min(min_boundary, min_pair)


def _objective_constrained(params, ring_counts):
    """
    Objective function with explicit inter-ring constraints.
    params = [r0, r1, ..., r_{k-1}, phi0, phi1, ..., phi_{k-1}]
    where r_i are the radial distances and phi_i are the phase offsets.
    """
    k = len(ring_counts)
    radii = params[:k]
    phases = params[k:]
    
    # Check radii are non-negative and increasing
    if np.any(radii < 0):
        return 1e6
    for i in range(1, k):
        if radii[i] <= radii[i-1]:
            return 1e6
    
    # Check outermost ring fits in unit circle
    if radii[-1] > 1.0:
        return 1e6
    
    points = _build_points(radii, phases, ring_counts)
    
    if len(points) != sum(ring_counts):
        return 1e6
    
    r = _compute_radius(points)
    
    # Add penalty for violating inter-ring minimum distance
    # For adjacent rings i and i+1, the minimum distance between points should be >= 2*r
    # We use a soft penalty to guide optimization
    penalty = 0.0
    for i in range(k - 1):
        if ring_counts[i] == 0 or ring_counts[i+1] == 0:
            continue
        # Extract points from ring i and i+1
        start_i = sum(ring_counts[:i])
        end_i = start_i + ring_counts[i]
        start_ip1 = end_i
        end_ip1 = start_ip1 + ring_counts[i+1]
        
        pts_i = points[start_i:end_i]
        pts_ip1 = points[start_ip1:end_ip1]
        
        # Compute minimum distance between these two sets
        dist_mat = cdist(pts_i, pts_ip1)
        min_dist = np.min(dist_mat)
        
        # The theoretical minimum distance between adjacent rings
        # depends on the radial separation and angular alignment
        # We penalize if min_dist is too small
        target_min = 2 * r * 0.95  # Allow slight violation for optimization
        if min_dist < target_min:
            penalty += (target_min - min_dist) ** 2 * 100
    
    return -(r - penalty)


def _heuristic_radii(ring_counts, r_target=0.053):
    """
    Compute heuristic radial positions based on ring counts.
    For a ring with n points, the minimum radius to fit them without overlap is:
    r_min = r_target / sin(pi/n)
    """
    k = len(ring_counts)
    radii = np.zeros(k)
    
    for i, n in enumerate(ring_counts):
        if n == 1:
            radii[i] = 0.0
        elif n >= 2:
            radii[i] = r_target / np.sin(np.pi / n)
    
    # Ensure radii are strictly increasing with sufficient separation
    for i in range(1, k):
        # Minimum separation between adjacent rings
        min_sep = 2 * r_target
        if radii[i] <= radii[i-1] + min_sep:
            radii[i] = radii[i-1] + min_sep
    
    # Scale to fit in unit circle
    if radii[-1] > 1.0 - r_target:
        scale = (1.0 - r_target) / radii[-1]
        radii *= scale
    
    return radii


def _optimize_ring_structure(ring_counts, max_time=10):
    """
    Optimize radial positions and phases for a given ring structure.
    """
    k = len(ring_counts)
    start_time = time.time()
    
    # Get heuristic initial radii
    r_target = 0.053
    init_radii = _heuristic_radii(ring_counts, r_target)
    
    # Initial phases: stagger rings to maximize inter-ring distances
    init_phases = np.zeros(k)
    for i in range(1, k):
        if ring_counts[i] > 0 and ring_counts[i-1] > 0:
            # Offset by half the angular spacing of the previous ring
            init_phases[i] = np.pi / ring_counts[i-1]
    
    x0 = np.concatenate([init_radii, init_phases])
    
    # Define bounds
    bounds = []
    for i in range(k):
        if i == 0:
            bounds.append((0.0, 1.0))
        else:
            bounds.append((0.0, 1.0))
    for i in range(k):
        bounds.append((0.0, 2 * np.pi))
    
    # Use differential evolution for global search
    best_result = None
    best_fun = np.inf
    
    # Run DE with time limit
    try:
        result = minimize(
            _objective_constrained, x0, args=(ring_counts,),
            method='L-BFGS-B', bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        if result.fun < best_fun:
            best_fun = result.fun
            best_result = result
    except:
        pass
    
    # Try a few local refinements from different starting points
    rng = np.random.default_rng(42)
    for seed in range(3):
        if time.time() - start_time > max_time * 0.7:
            break
        
        # Perturb initial solution
        perturbation = rng.uniform(-0.02, 0.02, size=len(x0))
        x0_perturbed = x0 + perturbation
        # Wrap phases to [0, 2pi]
        x0_perturbed[k:] = np.mod(x0_perturbed[k:], 2 * np.pi)
        
        try:
            result = minimize(
                _objective_constrained, x0_perturbed, args=(ring_counts,),
                method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result.fun < best_fun:
                best_fun = result.fun
                best_result = result
        except:
            pass
    
    if best_result is None:
        return None, 0.0
    
    # Extract optimal parameters
    opt_radii = best_result.x[:k]
    opt_phases = best_result.x[k:]
    
    # Compute actual radius
    points = _build_points(opt_radii, opt_phases, ring_counts)
    r = _compute_radius(points)
    
    return points, r


def _generate_ring_structures():
    """
    Generate candidate ring structures that sum to 300.
    """
    structures = []
    
    # Classic hexagonal layers
    # 1, 6, 12, 18, 24, 30, 36, 42, 48, 54 = 271, need 29 more
    base = [1, 6, 12, 18, 24, 30, 36, 42, 48, 54]
    
    # Add remaining to last ring
    structures.append(base + [29])
    
    # Distribute remaining across last few rings
    for last1 in range(20, 35):
        for last2 in range(20, 40):
            remaining = 29 - last1 - last2
            if remaining < 0 or remaining > 30:
                continue
            new_struct = base.copy()
            new_struct[-2] = last2
            new_struct[-1] = last1
            new_struct.append(remaining)
            if sum(new_struct) == 300 and all(x >= 1 for x in new_struct):
                structures.append(new_struct)
    
    # Try different inner structures
    # Reduce some inner rings and add to outer
    for inner_reduction in range(0, 10):
        if inner_reduction == 0:
            continue
        new_struct = base.copy()
        # Reduce ring 3 (18) by inner_reduction
        new_struct[3] = 18 - inner_reduction
        if new_struct[3] < 1:
            continue
        # Add to last ring
        new_struct[-1] = 54 + inner_reduction
        new_struct.append(29)
        if sum(new_struct) == 300:
            structures.append(new_struct)
    
    # More aggressive restructuring
    # Try: 1, 6, 12, 18, 24, 30, 36, 42, 45, 48, 54, 24
    structures.append([1, 6, 12, 18, 24, 30, 36, 42, 45, 48, 54, 24])
    structures.append([1, 6, 12, 18, 24, 30, 36, 42, 46, 47, 54, 24])
    structures.append([1, 6, 12, 18, 24, 30, 36, 42, 47, 46, 54, 24])
    structures.append([1, 6, 12, 18, 24, 30, 36, 42, 48, 45, 54, 24])
    
    # Try 11 rings with different distribution
    for r9 in range(40, 50):
        for r10 in range(40, 50):
            remaining = 300 - (1 + 6 + 12 + 18 + 24 + 30 + 36 + 42 + r9 + r10)
            if 1 <= remaining <= 40:
                structures.append([1, 6, 12, 18, 24, 30, 36, 42, r9, r10, remaining])
    
    # Remove duplicates and invalid structures
    unique_structures = []
    seen = set()
    for s in structures:
        if sum(s) != 300:
            continue
        if any(x < 1 for x in s):
            continue
        key = tuple(s)
        if key not in seen:
            seen.add(key)
            unique_structures.append(s)
    
    return unique_structures


def construct():
    """Construct the packing configuration."""
    
    best_points = None
    best_r = 0.0
    
    # Time budget
    total_time_budget = 110  # Leave 10s margin
    start_time = time.time()
    
    # Generate ring structures
    structures = _generate_ring_structures()
    
    # Sort by number of rings (fewer rings first, then by total count variance)
    structures.sort(key=lambda s: (len(s), np.std(s)))
    
    # Optimize each structure
    for struct in structures:
        if time.time() - start_time > total_time_budget:
            break
        
        # Estimate potential based on heuristic
        r_target = 0.053
        heuristic_radii = _heuristic_radii(struct, r_target)
        
        # Quick feasibility check: can the outermost ring fit?
        if heuristic_radii[-1] > 1.0:
            continue
        
        # Allocate time based on structure complexity
        time_per_struct = min(5.0, (total_time_budget - (time.time() - start_time)) / max(1, len(structures)))
        
        points, r = _optimize_ring_structure(struct, max_time=time_per_struct)
        
        if points is not None and r > best_r:
            best_r = r
            best_points = points
    
    # If no valid solution found, use fallback
    if best_points is None:
        # Simple hexagonal grid fallback
        spacing = 2.0 / (np.sqrt(N) + 1.0)
        while True:
            k = int(1.0 / spacing) + 2
            rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            pts = np.array(rows)
            radius = np.hypot(pts[:, 0], pts[:, 1])
            mask = radius <= 1.0 - spacing / 2.0
            pts, radius = pts[mask], radius[mask]
            if len(pts) >= N:
                return pts[np.argsort(radius, kind="stable")[:N]]
            spacing *= 0.98
    
    return best_points

# EVOLVE-BLOCK-END
