"""Rewritten program for hexagon packing; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time
import math

N = 12
DEADLINE = 140.0  # safety margin before 150s kill

# Precompute vertices of unit regular hexagon at various rotations
def hex_vertices(angle_deg):
    """Return 6 vertices of a unit regular hexagon centered at origin, rotated by angle_deg."""
    angles = np.radians(angle_deg) + np.pi / 6  # offset so one vertex is at top
    # Actually, let's use the standard: vertices at angles 0, 60, 120, 180, 240, 300 degrees
    # For a regular hexagon with side length 1, vertices are at distance 1 from center
    angles = np.radians(angle_deg) + np.arange(6) * np.pi / 3
    return np.column_stack([np.cos(angles), np.sin(angles)])


def compute_min_outer_side(inner_positions, inner_angles, outer_center, outer_angle):
    """
    Compute the minimum outer hexagon side length needed to contain all inner hexagons.
    
    inner_positions: (N, 2) array of centers
    inner_angles: (N,) array of rotation angles in degrees
    outer_center: (2,) array
    outer_angle: float, rotation of outer hexagon in degrees
    
    Returns: minimum side length s such that all inner hexagons fit inside outer hexagon
    """
    # For each inner hexagon, we need to find the maximum "distance" from outer center
    # in the direction of each of the 6 normals of the outer hexagon.
    
    # The outer hexagon with side length s, centered at outer_center, rotated by outer_angle:
    # Its 6 face normals point at angles: outer_angle + k*60 degrees, k=0..5
    # The distance from center to each face is s * cos(30 deg) = s * sqrt(3)/2 (apothem)
    
    # For a point to be inside the outer hexagon, its projection onto each normal 
    # must be <= apothem = s * sqrt(3)/2
    
    # For an inner hexagon centered at c with rotation a, the farthest point in direction n
    # is at distance |dot(c - outer_center, n)| + max over vertices v of |dot(v, n)|
    # where v are the inner hexagon vertices (relative to its center)
    
    outer_normal_angles = np.radians(outer_angle) + np.arange(6) * np.pi / 3
    outer_normals = np.column_stack([np.cos(outer_normal_angles), np.sin(outer_normal_angles)])
    
    apothem_factor = math.sqrt(3) / 2  # s * apothem_factor = apothem
    
    min_s = 0.0
    
    for i in range(N):
        center = inner_positions[i]
        rel_center = center - outer_center
        inner_hex_verts = hex_vertices(inner_angles[i])
        
        for j in range(6):
            n = outer_normals[j]
            # Farthest extent of inner hexagon in direction n
            proj_center = np.dot(rel_center, n)
            proj_verts = np.dot(inner_hex_verts, n)
            max_extent = abs(proj_center) + np.max(np.abs(proj_verts))
            # Actually we need the max projection, not abs
            # The hexagon is on one side of its center, so:
            # max projection of (center + v) in direction n = proj_center + max(proj_verts)
            # But we also need to check the other side: min projection
            # For containment, we need: max_proj <= apothem AND -min_proj <= apothem
            # i.e., |proj_center + proj_verts| <= apothem for all v
            
            # More precisely: the inner hexagon fits if for all vertices v:
            # |(center + v - outer_center) dot n| <= apothem
            # So the required apothem for this face is max over v of |(center+v-outer_center) dot n|
            
            rel_verts = rel_center + inner_hex_verts
            proj_rel_verts = np.dot(rel_verts, n)
            required_apothem = np.max(np.abs(proj_rel_verts))
            s_needed = required_apothem / apothem_factor
            if s_needed > min_s:
                min_s = s_needed
    
    return min_s


def objective(params):
    """Objective function: minimize outer side length."""
    if time.time() > DEADLINE:
        return 1e10
    
    n_params = len(params)
    inner_positions = params[:2*N].reshape(N, 2)
    inner_angles = params[2*N:3*N]
    outer_center = params[3*N:3*N+2]
    outer_angle = params[3*N+2]
    
    # Check for overlaps between inner hexagons
    # For each pair, check if hexagons intersect
    for i in range(N):
        for j in range(i+1, N):
            # Distance between centers
            d = np.linalg.norm(inner_positions[i] - inner_positions[j])
            # Two unit hexagons (side 1) have circumradius 1
            # If distance < 2*1 = 2, they might overlap
            # Actually, for regular hexagons, the minimum distance for non-overlap depends on orientation
            # Simple check: if d < 2.0, they definitely might overlap
            # For a better check, we could do polygon intersection, but for now use d < 2.0 as penalty
            if d < 2.0:
                # Penalty for overlap
                overlap_penalty = (2.0 - d) * 100.0
                return 1e6 + overlap_penalty
    
    s = compute_min_outer_side(inner_positions, inner_angles, outer_center, outer_angle)
    
    # Also add penalty if inner hexagons are too close (within 0.1 of touching)
    # This helps avoid degenerate configurations
    
    return s


def construct():
    start_time = time.time()
    
    # Initial configuration: place hexagons in a roughly hexagonal arrangement
    # Center hexagon at origin, 6 around it, 5 more in outer ring
    positions = np.zeros((N, 2))
    angles = np.zeros(N)
    
    # Place 7 hexagons in a flower pattern (center + 6 neighbors)
    # Center
    positions[0] = [0.0, 0.0]
    # 6 neighbors at distance 2.0 (side length 1 hexagons, distance between centers for tiling is 2)
    for k in range(6):
        a = k * np.pi / 3
        positions[k+1] = [2.0 * np.cos(a), 2.0 * np.sin(a)]
    
    # 5 more hexagons in the next ring
    # Place them at distance ~3.5 from center, at various angles
    for k in range(5):
        a = (k + 0.5) * np.pi / 3  # offset by 30 degrees
        positions[k+7] = [3.5 * np.cos(a), 3.5 * np.sin(a)]
    
    # Build initial parameter vector
    params0 = np.concatenate([
        positions.flatten(),
        angles,
        [0.0, 0.0],  # outer center
        [0.0]  # outer angle
    ])
    
    # Simulated annealing
    best_params = params0.copy()
    best_val = objective(params0)
    
    T = 1.0
    T_min = 0.001
    cooling = 0.9995
    rng = np.random.default_rng(42)
    
    while time.time() < DEADLINE - 10.0:
        if T < T_min:
            break
        
        # Perturb
        step_size = 0.2 * T + 0.05
        new_params = best_params + rng.normal(0, step_size, size=best_params.shape)
        
        # Keep outer center near origin and angle bounded
        new_params[3*N:3*N+2] = np.clip(new_params[3*N:3*N+2], -3, 3)
        new_params[3*N+2] = np.clip(new_params[3*N+2], -30, 30)
        
        new_val = objective(new_params)
        
        if new_val < best_val or rng.random() < np.exp(-(new_val - best_val) / T):
            best_params = new_params
            best_val = new_val
        
        T *= cooling
    
    # Local optimization with Nelder-Mead
    if time.time() < DEADLINE - 5.0:
        result = minimize(objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-7, 'fatol': 1e-9})
        if result.fun < best_val:
            best_params = result.x
            best_val = result.fun
    
    # Final polish with tighter tolerance
    if time.time() < DEADLINE - 2.0:
        result = minimize(objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 300, 'xatol': 1e-9, 'fatol': 1e-11})
        if result.fun < best_val:
            best_params = result.x
            best_val = result.fun
    
    # Extract final parameters
    inner_positions = best_params[:2*N].reshape(N, 2)
    inner_angles = best_params[2*N:3*N]
    outer_center = best_params[3*N:3*N+2]
    outer_angle = best_params[3*N+2]
    
    inner = [[float(x), float(y), float(a)] for (x, y), a in zip(inner_positions, inner_angles)]
    return inner, (float(outer_center[0]), float(outer_center[1])), float(best_val), float(outer_angle)
