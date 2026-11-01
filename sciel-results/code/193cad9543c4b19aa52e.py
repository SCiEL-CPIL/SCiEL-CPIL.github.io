"""Radial-ring decomposition for packing 300 equal circles in a unit circle."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist


def _build_points(radii, phases, ring_counts):
    """Build point array from ring radii, phases, and counts."""
    points = []
    for r, phase, count in zip(radii, phases, ring_counts):
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
    """Objective: negative packing radius."""
    k = len(ring_counts)
    radii = np.maximum(params[:k], 0.0)
    phases = params[k:]
    
    points = _build_points(radii, phases, ring_counts)
    if len(points) != sum(ring_counts):
        return 1e10
    
    r = _compute_radius(points)
    return -r


def _make_initial_guess(ring_counts, r_target=0.053):
    """Generate initial guess for radii and phases."""
    k = len(ring_counts)
    radii = np.zeros(k)
    for i, n in enumerate(ring_counts):
        if n > 1:
            r_min_ring = r_target / np.sin(np.pi / n)
        else:
            r_min_ring = 0.0
        r_min_adj = 2.0 * i * r_target
        radii[i] = max(r_min_ring, r_min_adj)
    
    if radii[-1] > 0:
        scale = (1.0 - r_target) / radii[-1]
        if scale < 1.0:
            radii *= scale
    
    phases = np.zeros(k)
    for i in range(1, k):
        if ring_counts[i] > 0 and ring_counts[i-1] > 0:
            phases[i] = np.pi / ring_counts[i-1]
    
    return np.concatenate([radii, phases])


def construct():
    """Construct the packing configuration using radial-ring decomposition."""
    
    # Ring count patterns summing to 300
    ring_patterns = [
        [1, 6, 12, 18, 24, 30, 36, 42, 48, 54, 29],
        [1, 8, 16, 24, 32, 40, 48, 56, 64, 11],
        [1, 7, 14, 21, 28, 35, 42, 49, 56, 47],
        [1, 6, 12, 18, 24, 30, 36, 42, 48, 54, 29],
        [1, 6, 12, 18, 24, 30, 36, 42, 48, 54, 29],
    ]
    
    # Remove duplicates
    seen = set()
    unique_patterns = []
    for p in ring_patterns:
        key = tuple(p)
        if key not in seen:
            seen.add(key)
            unique_patterns.append(p)
    
    best_points = None
    best_r = 0.0
    
    for ring_counts in unique_patterns:
        ring_counts = np.array(ring_counts)
        if sum(ring_counts) != N:
            continue
        
        num_rings = len(ring_counts)
        x0 = _make_initial_guess(ring_counts)
        
        bounds = [(0.0, 1.0)] * num_rings + [(0.0, 2 * np.pi)] * num_rings
        
        # Try multiple random restarts
        for restart in range(3):
            rng = np.random.default_rng(restart * 100)
            x0_perturbed = x0 + rng.normal(0, 0.01, size=x0.shape)
            x0_perturbed[:num_rings] = np.maximum(x0_perturbed[:num_rings], 0.0)
            x0_perturbed[num_rings:] = x0_perturbed[num_rings:] % (2 * np.pi)
            
            result = minimize(
                _objective, x0_perturbed, args=(ring_counts,),
                method='Nelder-Mead',
                options={'maxiter': 15000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            
            radii = np.maximum(result.x[:num_rings], 0.0)
            phases = result.x[num_rings:]
            
            points = _build_points(radii, phases, ring_counts)
            if len(points) != N:
                continue
            
            r = _compute_radius(points)
            if r > best_r:
                best_r = r
                best_points = points
    
    # Fallback: hexagonal lattice
    if best_points is None:
        spacing = 2.0 / (np.sqrt(N) + 1.0)
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
            spacing *= 0.98
    
    return best_points

# EVOLVE-BLOCK-END
