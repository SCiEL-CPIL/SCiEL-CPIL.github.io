# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon.
    Uses L-BFGS-B to minimize a penalty function that enforces containment and non-overlap
    while minimizing the outer hexagon size.
    Returns
        inner_hex_data: np.ndarray of shape (12,3)
        outer_hex_data: np.ndarray of shape (3,)
        outer_hex_side_length: float
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin for 600s limit
    
    sqrt3 = math.sqrt(3)
    N = 12
    
    # Helper: Get vertices of a hexagon
    def get_hex_vertices(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
        verts = []
        for i in range(6):
            a = angle_rad + i * math.pi / 3
            x = cx + side * math.cos(a)
            y = cy + side * math.sin(a)
            verts.append((x, y))
        return np.array(verts)

    # Helper: Calculate separation distance between two convex polygons using SAT
    # Returns the minimum distance (positive if separated, 0 if touching/overlapping)
    # For optimization, we want to maximize separation.
    def sat_separation(poly1, poly2):
        # poly1, poly2 are (6, 2) arrays of vertices
        min_separation = float('inf')
        
        # Get normals for both polygons
        for poly in [poly1, poly2]:
            edges = np.roll(poly, -1, axis=0) - poly
            # Normals
            normals = np.column_stack((edges[:, 1], -edges[:, 0]))
            
            for n in normals:
                # Project all vertices of poly1 onto n
                proj1 = poly1 @ n
                # Project all vertices of poly2 onto n
                proj2 = poly2 @ n
                min1, max1 = np.min(proj1), np.max(proj1)
                min2, max2 = np.min(proj2), np.max(proj2)
                
                # Calculate gap
                gap = max(min1 - max2, min2 - max1)
                if gap < min_separation:
                    min_separation = gap
                    
        return min_separation

    # Helper: Calculate minimum side length s for a set of inner hexes to fit in outer hex (centered at origin, angle 0)
    # Outer hex with side s, angle 0 has vertices at distance s from origin.
    # The "apothem" (center to edge midpoint) is s * sqrt(3)/2.
    # Normals of outer hex edges are at angles 30, 90, 150, 210, 270, 330 degrees.
    normal_angles = [math.radians(30 + i*60) for i in range(6)]
    normals = np.array([[math.cos(a), math.sin(a)] for a in normal_angles])
    
    # Vertex offsets for unit hexagon centered at origin, angle 0
    v_offsets = np.array([
        [1.0, 0.0],
        [0.5, sqrt3/2],
        [-0.5, sqrt3/2],
        [-1.0, 0.0],
        [-0.5, -sqrt3/2],
        [0.5, -sqrt3/2]
    ])

    def calc_min_s(centers):
        # centers: (N, 2) array
        max_proj = 0.0
        for c in centers:
            verts = c + v_offsets
            # Project onto normals
            projs = verts @ normals.T
            max_p = np.max(projs)
            if max_p > max_proj:
                max_proj = max_p
        return max_proj * 2.0 / sqrt3

    def objective(params):
        # params: flat array of size 24 (12 centers * 2 coords)
        centers = params.reshape(N, 2)
        
        # Recenter to keep origin at center of mass (reduces degrees of freedom effectively)
        centers = centers - np.mean(centers, axis=0)
        
        # Calculate required outer side length
        s = calc_min_s(centers)
        
        # Penalty for containment: if s is too small, vertices go outside
        # We minimize s, so we don't need a separate containment penalty if we just minimize s?
        # No, s is derived from the max projection. If we minimize s, we are effectively enforcing containment.
        # However, we need to ensure non-overlap.
        
        # Penalty for non-overlap
        overlap_penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                v1 = centers[i] + v_offsets
                v2 = centers[j] + v_offsets
                sep = sat_separation(v1, v2)
                if sep < 0:
                    # Overlapping
                    overlap_penalty += (-sep) ** 2
                elif sep < 1e-4:
                    # Close, add small penalty to encourage separation
                    overlap_penalty += (1e-4 - sep) ** 2 * 0.01
                    
        # Total cost: s + lambda * overlap_penalty
        # We want to minimize s, but s is the primary objective.
        # If there is overlap, we must fix it before reducing s further.
        # Use a large weight for overlap to prioritize feasibility.
        weight = 100.0
        return s + weight * overlap_penalty

    # Initial guess: Lattice
    centers = []
    for i in range(3):
        for j in range(4):
            x = j * 2.5
            y = i * sqrt3
            if i % 2 == 1:
                x += 1.25
            centers.append([x, y])
    centers = np.array(centers)
    centers -= np.mean(centers, axis=0)
    
    initial_params = centers.flatten()
    
    # Try multiple restarts with different initial perturbations to avoid local minima
    best_cost = float('inf')
    best_params = initial_params.copy()
    
    # Number of restarts based on time
    # L-BFGS-B is fast, so we can afford several restarts
    num_restarts = 5
    
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        if r == 0:
            current_params = initial_params.copy()
        else:
            # Perturb the best solution found so far or the initial lattice
            base = best_params if r == 1 else initial_params
            noise = np.random.normal(0, 0.1, size=2*N)
            current_params = base + noise
        
        try:
            res = minimize(
                objective, 
                current_params, 
                method='L-BFGS-B', 
                options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8, 'disp': False}
            )
            
            if res.success or res.fun < best_cost:
                best_cost = res.fun
                best_params = res.x
        except Exception as e:
            pass

    # Extract best centers
    best_centers = best_params.reshape(N, 2)
    best_centers -= np.mean(best_centers, axis=0)
    best_s = calc_min_s(best_centers)
    
    # Construct return values
    inner_ret = np.zeros((12, 3))
    for i in range(12):
        inner_ret[i, 0] = best_centers[i, 0]
        inner_ret[i, 1] = best_centers[i, 1]
        inner_ret[i, 2] = 0.0
        
    outer_ret = np.array([0.0, 0.0, 0.0])
    
    return inner_ret, outer_ret, best_s

# EVOLVE-BLOCK-END
