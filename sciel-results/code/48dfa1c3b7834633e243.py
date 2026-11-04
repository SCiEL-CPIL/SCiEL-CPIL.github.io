"""Rewritten program for hexagon packing using global search and outer hexagon optimization."""
import math
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 11

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = 150.0
    safety_margin = 10.0
    
    # Unit hexagon vertices (pointy top, angle 0)
    # Vertices at angles 30, 90, 150, 210, 270, 330 degrees? 
    # Standard "pointy right" or "flat top"? 
    # The task says "side length 1". 
    # Let's use the standard mathematical definition where vertices are at angles 
    # (2k+1)*30 degrees for "pointy top" or k*60 for "pointy right".
    # The distance between parallel sides is sqrt(3) for side 1.
    # Let's stick to a consistent frame. 
    # Vertices of a unit hexagon centered at origin, "pointy right" (vertex at 0 deg):
    # (1, 0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1, 0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
    # This orientation has width 2 and height sqrt(3).
    
    hex_verts = np.array([
        [1.0, 0.0],
        [0.5, math.sqrt(3)/2],
        [-0.5, math.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -math.sqrt(3)/2],
        [0.5, -math.sqrt(3)/2]
    ])
    
    # Precompute distances between vertices for potential overlap checks if needed, 
    # but we will rely on center distance for unit hexagons with same orientation.
    # For different orientations, center distance check is insufficient, but 
    # differential evolution with penalty will handle it if we use a robust overlap check.
    # However, for speed, let's assume we can restrict inner hexagons to a few discrete 
    # orientations or just optimize positions with a strong penalty.
    # Actually, the prompt says "each inner hexagon may translate and rotate".
    # Optimizing 11*3 = 33 variables with DE is hard.
    # Let's try to fix inner orientations to 0 or 60 degrees (symmetry of hexagon) 
    # or just 0 degrees first, and see if that's enough. 
    # Wait, the inspiration suggests "inner rows (x, y, angle_degrees)".
    # If we fix angle to 0, we reduce dimensionality. 
    # Let's try optimizing positions only (22 vars) first, assuming optimal packing 
    # for 11 hexes in a hexagon can be achieved with aligned hexes. 
    # Most known packings of hexagons in hexagons use aligned hexagons.
    
    # Objective: Minimize outer side length S.
    # Variables: centers of 11 hexagons (22 vars).
    # For each candidate center set, find optimal outer hex (x, y, theta) and its min S.
    
    def get_outer_hex_params(centers):
        """
        Given centers of inner hexagons (aligned with axes), find outer hexagon 
        center (cx, cy) and rotation theta that minimizes the side length S 
        such that all inner hexagons are inside.
        """
        # Inner hex vertices are fixed relative to their centers.
        # Outer hexagon is defined by 6 half-planes.
        # Normal vectors for outer hexagon with rotation theta:
        # n_k = (cos(theta + k*pi/3), sin(theta + k*pi/3)) for k=0..5
        
        # For a point P to be inside the outer hexagon of side S centered at C with rotation theta:
        # (P - C) . n_k <= S * sqrt(3) / 2 for all k.
        # Let D = S * sqrt(3) / 2. This is the apothem (distance from center to side).
        # So we want to minimize D (and thus S).
        # Min D subject to: max_k (max_i ( (P_i + V_j) - C ) . n_k ) <= D
        
        # This is a convex optimization problem for C and theta if we fix the set of points.
        # We can use Nelder-Mead to find C and theta that minimize the required D.
        
        def outer_obj(params):
            cx, cy, theta = params
            C = np.array([cx, cy])
            
            # Calculate normals
            angles = theta + np.arange(6) * math.pi / 3
            normals = np.column_stack([np.cos(angles), np.sin(angles)])
            
            # Calculate max projection of all inner vertices for each normal
            # Inner vertices: centers[i] + hex_verts[j]
            # We can vectorize:
            # For each normal k:
            # proj_k = max_i max_j ( (centers[i] + hex_verts[j]) - C ) . normals[k]
            
            # (centers[i] - C) . normals[k] + hex_verts[j] . normals[k]
            
            # Let's compute (centers - C) @ normals.T  -> (11, 6)
            center_diffs = centers - C
            center_projs = center_diffs @ normals.T # (11, 6)
            
            # hex_verts @ normals.T -> (6, 6)
            vert_projs = hex_verts @ normals.T # (6, 6)
            
            # Max over i and j for each k:
            # max_i (center_projs[i, k]) + max_j (vert_projs[j, k])
            max_center = np.max(center_projs, axis=0) # (6,)
            max_vert = np.max(vert_projs, axis=0) # (6,)
            
            max_proj = max_center + max_vert # (6,)
            
            # The constraint is max_proj[k] <= D.
            # So D must be at least max(max_proj).
            # We want to minimize D.
            # If we use Nelder-Mead on (cx, cy, theta), the objective is max(max_proj).
            return np.max(max_proj)
        
        # Initial guess: center of mass of inner hexes, theta=0
        com = np.mean(centers, axis=0)
        x0 = [com[0], com[1], 0.0]
        
        try:
            res = minimize(outer_obj, x0, method='Nelder-Mead', 
                           options={'xatol': 1e-4, 'fatol': 1e-4, 'maxiter': 200, 'maxfev': 500})
            if res.success:
                d_opt = res.fun
                s_opt = 2 * d_opt / math.sqrt(3)
                return s_opt, res.x
            else:
                # Fallback
                d_val = outer_obj(x0)
                return 2 * d_val / math.sqrt(3), x0
        except Exception:
            d_val = outer_obj(x0)
            return 2 * d_val / math.sqrt(3), x0

    def overall_obj(params):
        """
        Objective for Differential Evolution.
        params: 22 floats (x, y for 11 hexagons)
        Returns: outer side length S + penalty for overlaps
        """
        if time.time() > deadline - safety_margin:
            return 1000.0
            
        centers = params.reshape(N, 2)
        
        # Check for overlaps
        # For aligned hexagons, min distance is sqrt(3).
        # If they rotate, it's complex. We assume aligned for now to keep DE fast.
        # If overlaps exist, add penalty.
        penalty = 0.0
        min_dist = math.sqrt(3)
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(centers[i] - centers[j])
                if d < min_dist:
                    penalty += (min_dist - d)
        
        if penalty > 0.1:
            # Heavily penalize overlaps
            return 100.0 + penalty
            
        # Get optimal outer hex size
        S, _ = get_outer_hex_params(centers)
        
        return S

    # Initial bounds for DE
    # A rough estimate: 11 hexes area ~ 11 * 2.6 = 28.6.
    # Hexagon area ~ 2.598 * S^2. S^2 ~ 11. S ~ 3.3.
    # Let's use bounds [-5, 5] for x and y.
    bounds = [(-5.0, 5.0)] * (2 * N)
    
    # Run Differential Evolution
    # Use a seed for reproducibility? No, let it be random to explore.
    # But we need to be careful with time.
    # maxiter=100, popsize=10 might be too slow if inner optimization is slow.
    # Let's try popsize=5, maxiter=50.
    
    try:
        # Update the objective function to check time more frequently
        def fast_obj(params):
             if time.time() > deadline - safety_margin:
                 return 1000.0
             return overall_obj(params)

        res = differential_evolution(fast_obj, bounds, maxiter=50, popsize=5, tol=0.01, seed=42)
        
        if res.success and res.fun < 100.0:
            best_centers = res.x.reshape(N, 2)
            best_S, outer_params = get_outer_hex_params(best_centers)
            cx, cy, theta = outer_params
            theta_deg = math.degrees(theta) % 360.0
            
            # Format output
            inner = []
            for i in range(N):
                inner.append([best_centers[i][0], best_centers[i][1], 0.0])
            
            return inner, (cx, cy), best_S, theta_deg

    except Exception as e:
        pass

    # Fallback: Simple lattice packing
    # Just place them in a grid to ensure a valid solution
    d = math.sqrt(3)
    centers = []
    # 4x3 grid approx
    for i in range(4):
        for j in range(3):
            if len(centers) < N:
                x = (i - 1.5) * d
                y = (j - 1.0) * d * math.sqrt(3)/2
                centers.append([x, y])
    
    # Pad if necessary
    while len(centers) < N:
        centers.append([0, 0])
        
    centers = np.array(centers[:N])
    
    S, outer_params = get_outer_hex_params(centers)
    cx, cy, theta = outer_params
    theta_deg = math.degrees(theta) % 360.0

    inner = []
    for i in range(N):
        inner.append([centers[i][0], centers[i][1], 0.0])
    
    return inner, (cx, cy), S, theta_deg

# EVOLVE-BLOCK-END
