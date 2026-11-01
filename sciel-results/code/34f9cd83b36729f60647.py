"""Radial-ring decomposition for packing 300 equal circles in a unit circle."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import cdist
import time


def _build_points(radii, phases, ring_counts):
    """Build point array from ring radii, phases, and counts."""
    points = []
    for i, (r, phase, count) in enumerate(zip(radii, phases, ring_counts)):
        if count == 0:
            continue
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
    
    dists_boundary = 1.0 - np.hypot(points[:, 0], points[:, 1])
    min_boundary = np.min(dists_boundary)
    
    dist_matrix = cdist(points, points)
    np.fill_diagonal(dist_matrix, np.inf)
    min_pair = np.min(dist_matrix) / 2.0
    
    return min(min_boundary, min_pair)


def _objective(params, ring_counts):
    """Negative packing radius objective."""
    k = len(ring_counts)
    radii = params[:k]
    phases = params[k:]
    radii = np.maximum(radii, 0.0)
    
    points = _build_points(radii, phases, ring_counts)
    r = _compute_radius(points)
    return -r


def _optimize_ring_configuration(ring_counts, x0, maxiter=3000):
    """Optimize a single ring configuration."""
    k = len(ring_counts)
    bounds = [(0.0, 1.0)] * k + [(0.0, 2 * np.pi)] * k
    
    result = minimize(
        _objective, x0, args=(ring_counts,),
        method='Nelder-Mead',
        options={'maxiter': maxiter, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True}
    )
    
    radii = np.maximum(result.x[:k], 0.0)
    phases = result.x[k:]
    points = _build_points(radii, phases, ring_counts)
    
    if len(points) != N:
        return None, 0.0
    
    r = _compute_radius(points)
    return points, r


def _initial_guess_v2(ring_counts, seed=0):
    """Generate initial guess with better constraints."""
    rng = np.random.default_rng(seed)
    k = len(ring_counts)
    
    # Estimate optimal radius using constraint analysis
    # For ring i with n_i circles at radius R_i:
    #   chord distance >= 2r => R_i >= r / sin(pi/n_i)
    # Adjacent rings: radial gap >= 2r => R_{i+1} >= R_i + 2r
    # Boundary: R_{k-1} + r <= 1
    
    # Try to find r such that all constraints are satisfied
    r_est = 0.05  # initial estimate
    
    radii = np.zeros(k)
    for i, n in enumerate(ring_counts):
        if n > 1:
            r_min_ring = r_est / np.sin(np.pi / n)
        else:
            r_min_ring = 0.0
        r_min_adj = 2.0 * i * r_est
        radii[i] = max(r_min_ring, r_min_adj)
    
    # Scale to fit
    if radii[-1] > 0:
        scale = (1.0 - r_est) / radii[-1]
        if scale < 1.0:
            radii *= scale
    
    # Add jitter
    radii = radii + rng.uniform(-0.005, 0.005, size=k)
    radii = np.maximum(radii, 0.0)
    
    # Phases: stagger for better packing
    phases = np.zeros(k)
    for i in range(1, k):
        if ring_counts[i] > 0 and ring_counts[i-1] > 0:
            phases[i] = np.pi / ring_counts[i-1] + rng.uniform(-0.05, 0.05)
    
    return np.concatenate([radii, phases])


def _try_ring_structure(ring_counts, num_starts=3, maxiter=2000):
    """Try optimizing a ring structure with multiple starts."""
    best_points = None
    best_r = 0.0
    
    for seed in range(num_starts):
        x0 = _initial_guess_v2(ring_counts, seed=seed)
        points, r = _optimize_ring_configuration(ring_counts, x0, maxiter=maxiter)
        
        if points is not None and r > best_r:
            best_r = r
            best_points = points
    
    return best_points, best_r


def _generate_ring_structures():
    """Generate diverse ring structures that sum to 300."""
    structures = []
    
    # Structure 1: Hexagonal-like with increasing counts
    # 1, 6, 12, 18, 24, 30, 36, 42, 48, 54, 29
    structures.append([1, 6, 12, 18, 24, 30, 36, 42, 48, 54, 29])
    
    # Structure 2: More balanced
    structures.append([1, 6, 12, 18, 24, 30, 36, 42, 48, 54, 29])
    
    # Structure 3: Fewer rings, larger counts
    structures.append([1, 12, 24, 36, 48, 60, 72, 47])
    
    # Structure 4: More rings, smaller counts
    structures.append([1, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 33])
    
    # Structure 5: Optimized for hexagonal packing
    # Hexagonal numbers: 1, 7, 19, 37, 61, 91, ...
    # Adjust to sum to 300
    structures.append([1, 7, 19, 37, 61, 91, 84])
    
    # Structure 6: Gradual increase
    structures.append([1, 6, 11, 16, 21, 26, 31, 36, 41, 46, 51, 54])
    
    # Structure 7: Concentric with hexagonal spacing
    structures.append([1, 6, 12, 18, 24, 30, 36, 42, 48, 54, 29])
    
    # Structure 8: Optimized distribution
    structures.append([1, 6, 12, 18, 24, 30, 36, 42, 48, 54, 29])
    
    # Structure 9: More rings for finer control
    structures.append([1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 55, 20])
    
    # Structure 10: Balanced
    structures.append([1, 6, 12, 18, 24, 30, 36, 42, 48, 54, 29])
    
    # Filter valid structures
    valid_structures = []
    for s in structures:
        if sum(s) == N and len(s) > 0:
            valid_structures.append(s)
    
    return valid_structures


def construct():
    """Construct the packing configuration."""
    start_time = time.time()
    best_points = None
    best_r = 0.0
    
    # Generate and try multiple ring structures
    structures = _generate_ring_structures()
    
    for struct in structures:
        if time.time() - start_time > 100:
            break
        
        points, r = _try_ring_structure(struct, num_starts=4, maxiter=2500)
        
        if points is not None and r > best_r:
            best_r = r
            best_points = points
    
    # If we still have time, do a global optimization on the best structure
    if best_points is not None and time.time() - start_time < 90:
        # Find the structure that gave best_r
        for struct in structures:
            if sum(struct) == N:
                # Try differential evolution for this structure
                k = len(struct)
                bounds = [(0.0, 1.0)] * k + [(0.0, 2 * np.pi)] * k
                
                def de_objective(params):
                    return _objective(params, struct)
                
                try:
                    de_result = differential_evolution(
                        de_objective, bounds,
                        seed=42, maxiter=100, tol=1e-8,
                        popsize=15, mutation=(0.3, 1.0)
                    )
                    
                    radii = np.maximum(de_result.x[:k], 0.0)
                    phases = de_result.x[k:]
                    points = _build_points(radii, phases, struct)
                    
                    if len(points) == N:
                        r = _compute_radius(points)
                        if r > best_r:
                            best_r = r
                            best_points = points
                except Exception:
                    pass
                break
    
    # Fallback if no valid configuration found
    if best_points is None:
        # Simple hexagonal lattice fallback
        spacing = 0.1
        while True:
            k = int(1.0 / spacing) + 2
            rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            pts = np.array(rows)
            radius = np.hypot(pts[:, 0], pts[:, 1])
            mask = radius <= 1.0 - spacing / 2.0
            pts = pts[mask]
            radius = radius[mask]
            if len(pts) >= N:
                return pts[np.argsort(radius, kind="stable")[:N]]
            spacing *= 0.95
            if spacing < 0.01:
                break
    
    return best_points

# EVOLVE-BLOCK-END
