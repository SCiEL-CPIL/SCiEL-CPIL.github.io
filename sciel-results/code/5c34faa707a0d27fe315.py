import numpy as np
import time
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import cKDTree

N = 300

# EVOLVE-BLOCK-START

def _get_score(points):
    """
    Computes the score (common radius r) for a given set of points.
    r = min( min_{i<j} ||c_i - c_j|| / 2, min_i (1 - ||c_i||) )
    """
    if points is None or len(points) != N:
        return 0.0
    
    if not np.all(np.isfinite(points)):
        return 0.0

    norms = np.linalg.norm(points, axis=1)
    boundary_r = 1.0 - np.max(norms)
    
    if boundary_r < 0:
        return 0.0

    d = pdist(points)
    if len(d) == 0:
        pair_r = float('inf')
    else:
        pair_r = np.min(d) / 2.0

    r = min(pair_r, boundary_r)
    return max(0.0, r)

def _hexagonal_lattice(spacing):
    """
    Generate a hexagonal lattice of points inside the unit disk.
    """
    k = int(1.5 / spacing) + 5
    pts = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if np.hypot(x, y) <= 1.0:
                pts.append((x, y))
    
    if len(pts) < N:
        return None
    
    pts = np.array(pts)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists, kind="stable")[:N]
    return pts[idx]

def _relax(points, time_limit):
    """
    Relax the configuration using Simulated Annealing with repulsive forces.
    This is a structural change from active-constraint hill climbing to a global-ish local search.
    """
    start_time = time.time()
    best_points = points.copy()
    best_score = _get_score(best_points)
    
    # Initial temperature
    T = 0.01
    T_min = 1e-6
    cooling_rate = 0.9995
    
    # Pre-allocate arrays for performance
    forces = np.zeros_like(points)
    
    while time.time() < start_time + time_limit and T > T_min:
        # 1. Compute Forces
        # Repulsion between all pairs
        # Using KDTree for efficiency, but for N=300, O(N^2) is fine and simpler to vectorize
        # Let's use a soft repulsion potential to avoid singularities
        d = pdist(points)
        # Avoid division by zero
        d_safe = np.maximum(d, 1e-6)
        
        # Force magnitude: 1/d^2 (Coulomb-like) or just 1/d
        # We want to maximize the minimum distance.
        # Gradient of min_dist is tricky. 
        # Instead, we minimize a potential energy E = sum(1/d_ij^2) + sum((r_i - 1)^2)
        # Or simpler: Move each point away from its nearest neighbor.
        
        # Let's stick to a simpler heuristic: 
        # Move each point slightly away from all other points, weighted by 1/d^2.
        # And move all points slightly towards the center if they are too far out.
        
        # Reconstruct full distance matrix for vectorized force calc
        D = squareform(d)
        np.fill_diagonal(D, 1.0) # Avoid div by 0
        
        # Vectorized force calculation
        # F_i = sum_{j != i} (p_i - p_j) / |p_i - p_j|^3 * strength
        # This is O(N^2) operations.
        
        # Distance matrix D
        # Vector matrix V where V[i,j] = p_i - p_j
        # This creates a (N, N, 2) array. 300*300*2 = 180k floats. Fine.
        
        p_i = points[:, np.newaxis, :]  # (N, 1, 2)
        p_j = points[np.newaxis, :, :]  # (1, N, 2)
        vecs = p_i - p_j                # (N, N, 2)
        
        # Distances
        dists_3d = np.linalg.norm(vecs, axis=2) # (N, N)
        np.fill_diagonal(dists_3d, 1.0)
        
        # Force weights: 1/d^2
        weights = 1.0 / (dists_3d ** 2)
        np.fill_diagonal(weights, 0.0)
        
        # Force on i is sum of weights[i,j] * vecs[i,j]
        # vecs[i,j] is p_i - p_j. So this pushes i away from j.
        forces = np.sum(weights[:, :, np.newaxis] * vecs, axis=1)
        
        # Boundary force: Push inward if outside or near boundary
        norms = np.linalg.norm(points, axis=1)
        # Soft boundary potential: if r > 0.9, push in
        boundary_force = np.zeros_like(points)
        mask_out = norms > 0.9
        if np.any(mask_out):
            # Direction to center
            dir_center = -points[mask_out] / norms[mask_out, np.newaxis]
            # Strength proportional to how far out
            strength = (norms[mask_out] - 0.9)
            boundary_force[mask_out] = dir_center * strength[:, np.newaxis] * 10.0 # Stronger push
            
        total_forces = forces + boundary_force
        
        # 2. Update positions
        # Simulated Annealing step
        # delta = T * random_noise + step_size * normalized_forces
        # Normalize forces to have consistent magnitude
        force_norms = np.linalg.norm(total_forces, axis=1)
        max_force = np.max(force_norms)
        if max_force > 0:
            total_forces = total_forces / max_force
        
        step_size = 0.01 * T
        noise = np.random.normal(0, step_size * 0.5, points.shape)
        points = points + step_size * total_forces + noise
        
        # 3. Project to valid domain (inside unit circle)
        norms_new = np.linalg.norm(points, axis=1)
        mask = norms_new > 1.0
        if np.any(mask):
            points[mask] = points[mask] / norms_new[mask]
            
        # 4. Evaluate and Accept/Reject
        current_score = _get_score(points)
        
        if current_score > best_score:
            best_points = points.copy()
            best_score = current_score
            # If improved, don't cool as fast? Or keep cooling.
            
        # Cooling
        T *= cooling_rate

    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    candidates = []
    
    # 1. Generate initial candidates
    # Hex lattices
    spacings = np.linspace(0.05, 0.12, 15)
    for spacing in spacings:
        pts = _hexagonal_lattice(spacing)
        if pts is not None:
            r = _get_score(pts)
            if r > 0:
                candidates.append((pts, r))
                
    # Random starts
    for _ in range(5):
        angles = 2 * np.pi * np.random.rand(N)
        radii = np.sqrt(np.random.rand(N)) * 0.95
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        r = _get_score(pts)
        if r > 0:
            candidates.append((pts, r))

    if not candidates:
        # Fallback
        angles = 2 * np.pi * np.random.rand(N)
        radii = np.sqrt(np.random.rand(N)) * 0.9
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        return pts
        
    # Sort candidates by initial score
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    best_points = candidates[0][0].copy()
    best_score = candidates[0][1]
    
    # 2. Refine the top candidates
    # Allocate time to the top 3 candidates
    time_per_candidate = 30.0
    for i in range(min(3, len(candidates))):
        if time.time() > deadline - 10:
            break
            
        pts_to_refine = candidates[i][0].copy()
        time_remaining = deadline - time.time() - 10
        if time_remaining > 5:
            refined_points, refined_score = _relax(pts_to_refine, time_remaining)
            if refined_score > best_score:
                best_score = refined_score
                best_points = refined_points

    return best_points

# EVOLVE-BLOCK-END
