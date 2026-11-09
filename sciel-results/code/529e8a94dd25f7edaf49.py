"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    sqrt3 = math.sqrt(3.0)
    
    # Precompute edge normals for a standard hexagon (vertices at 0, 60, ...)
    # Edge i is perpendicular to radius at 30 + i*60
    edge_normals = np.array([
        [math.cos(math.radians(30 + i*60)), math.sin(math.radians(30 + i*60))] 
        for i in range(6)
    ])
    
    # Precompute vertices of a unit hexagon centered at origin (angle 0)
    # Vertices at 0, 60, ...
    unit_verts = np.array([
        [math.cos(math.radians(i*60)), math.sin(math.radians(i*60))] 
        for i in range(6)
    ])

    def get_vertices(cx, cy, angle_rad, side=1.0):
        """Get vertices of a regular hexagon."""
        c, s = math.cos(angle_rad), math.sin(angle_rad)
        R = np.array([[c, -s], [s, c]])
        return np.array([cx, cy]) + side * (unit_verts @ R.T)

    def check_overlap_fast(centers, angles_rad, eps=1e-7):
        """
        Check if any two hexagons overlap using Separating Axis Theorem (SAT).
        centers: list of (x, y)
        angles_rad: list of angles in radians
        Returns True if any overlap.
        """
        n = len(centers)
        # Precompute all vertices and edge normals
        all_verts = []
        all_normals = []
        
        for i in range(n):
            cx, cy = centers[i]
            ang = angles_rad[i]
            verts = get_vertices(cx, cy, ang)
            all_verts.append(verts)
            
            # Normals are just the edge_normals rotated by ang
            c, s = math.cos(ang), math.sin(ang)
            R = np.array([[c, -s], [s, c]])
            normals = edge_normals @ R.T
            all_normals.append(normals)

        for i in range(n):
            for j in range(i + 1, n):
                # Quick bounding box / distance check
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                dist_sq = dx*dx + dy*dy
                # Max distance between centers for non-overlap is 2 (side length 1)
                # Actually, if dist > 2, they definitely don't overlap? 
                # No, if they are aligned, they can touch at dist 2.
                # If dist > 2 + epsilon, no overlap.
                if dist_sq > 4.5: 
                    continue

                v_i = all_verts[i]
                v_j = all_verts[j]
                n_i = all_normals[i]
                n_j = all_normals[j]
                
                # SAT: Check all 12 potential separating axes (6 from i, 6 from j)
                overlap = True
                for k in range(6):
                    # Axis from i
                    axis = n_i[k]
                    proj_i = v_i @ axis
                    min_i, max_i = proj_i.min(), proj_i.max()
                    proj_j = v_j @ axis
                    min_j, max_j = proj_j.min(), proj_j.max()
                    
                    if max_i < min_j + eps or max_j < min_i + eps:
                        overlap = False
                        break
                    
                    # Axis from j
                    axis = n_j[k]
                    proj_i = v_i @ axis
                    min_i, max_i = proj_i.min(), proj_i.max()
                    proj_j = v_j @ axis
                    min_j, max_j = proj_j.min(), proj_j.max()
                    
                    if max_i < min_j + eps or max_j < min_i + eps:
                        overlap = False
                        break
                
                if overlap:
                    return True
        return False

    def calculate_L(centers, angles_rad, outer_angle_rad=0.0):
        """Calculate minimum outer side length L to enclose all inner hexagons."""
        c, s = math.cos(-outer_angle_rad), math.sin(-outer_angle_rad)
        R_outer_inv = np.array([[c, -s], [s, c]])
        
        max_proj = 0.0
        for i in range(N):
            verts = get_vertices(centers[i][0], centers[i][1], angles_rad[i])
            # Rotate vertices to outer frame
            verts_rot = verts @ R_outer_inv.T
            # Project onto outer edge normals (which are fixed in outer frame)
            projs = verts_rot @ edge_normals.T
            # max_proj is the max over all points and all 6 normals
            # Since hexagon is convex and centered, we just need max projection on any normal
            # Note: projs can be negative, but max_proj starts at 0, and we want the "radius"
            # The distance to edge is apothem. 
            # Point (x,y) is inside if dot(p, n_k) <= apothem for all k.
            # So we need apothem >= max_k dot(p, n_k).
            # Wait, if p is in direction of -n_k, dot is negative. 
            # The constraint is effectively |projection|? No.
            # The hexagon is the intersection of half planes n_k . x <= apothem.
            # So for a point to be inside, n_k . p <= apothem for ALL k.
            # So apothem must be >= max_k (n_k . p).
            
            local_max = projs.max(axis=0).max() # Max over vertices and normals
            if local_max > max_proj:
                max_proj = local_max
        
        return 2.0 * max_proj / sqrt3

    def objective(params):
        """
        Objective function for optimization.
        params: [cx0, cy0, cx1, cy1, ..., cx11, cy11, ang0, ..., ang11]
        """
        centers = []
        angles_rad = []
        for i in range(N):
            centers.append((params[2*i], params[2*i+1]))
            angles_rad.append(params[2*N + i])
        
        # Check overlap
        if check_overlap_fast(centers, angles_rad):
            # Penalty for overlap
            # We want to minimize L, but overlap is invalid.
            # Return a large value to push out of overlap regions.
            return 100.0
        
        L = calculate_L(centers, angles_rad, 0.0)
        return L

    # Multi-start strategy
    best_L = float('inf')
    best_params = None
    best_centers = None
    best_angles = None
    
    # Symmetry reduction idea:
    # The outer hexagon has 6-fold symmetry.
    # We can fix the first hexagon's position and angle to break symmetry.
    # However, Nelder-Mead handles symmetry breaking via multi-starts better than fixed constraints.
    # Let's use multiple random starts, biased towards symmetric positions.
    
    num_starts = 20
    
    for start_idx in range(num_starts):
        if time.time() > deadline:
            break
            
        # Generate initial guess
        # Strategy: Place hexagons in a hexagonal grid pattern or circle
        if start_idx % 2 == 0:
            # Circular arrangement
            r = 1.5 + 0.5 * (start_idx % 5) # Vary radius
            init_params = []
            for i in range(N):
                theta = 2 * math.pi * i / N
                cx = r * math.cos(theta)
                cy = r * math.sin(theta)
                ang = 0.0
                init_params.extend([cx, cy])
                init_params.append(ang)
        else:
            # Hexagonal grid arrangement (center + 6 neighbors + 5 more?)
            # Center at 0,0
            # Neighbors at distance 2 (side 1 hexes touch at distance 2? No, distance 2*apothem = sqrt3? No.)
            # Two unit hexes touching side-to-side have center distance 2 * apothem = sqrt(3).
            # Two unit hexes touching vertex-to-vertex have center distance 2.
            # To pack tightly, distance should be ~ sqrt(3) to 2.
            
            # Let's try a grid of hexes.
            # Hex grid coordinates (axial):
            # (0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1) -> 7 hexes
            # We need 12.
            # Add (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0)?
            
            hex_coords = [
                (0, 0),
                (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
                (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1)
            ]
            
            # Convert axial to cartesian
            # x = 2 * q + r
            # y = sqrt(3) * r
            # This is for pointy-top. Let's use flat-top.
            # x = sqrt(3) * (q + r/2)
            # y = 3/2 * r
            
            init_params = []
            for q, r in hex_coords:
                cx = sqrt3 * (q + r/2.0)
                cy = 1.5 * r
                # Add some noise
                cx += (np.random.rand() - 0.5) * 0.2
                cy += (np.random.rand() - 0.5) * 0.2
                ang = 0.0
                init_params.extend([cx, cy])
                init_params.append(ang)
        
        # Optimize
        try:
            res = minimize(
                objective, 
                init_params, 
                method='Nelder-Mead', 
                options={
                    'maxiter': 2000, 
                    'xatol': 1e-5, 
                    'fatol': 1e-5,
                    'adaptive': True
                }
            )
            
            if res.fun < best_L:
                best_L = res.fun
                best_params = res.x
                # Extract centers and angles for final check
                best_centers = [(best_params[2*i], best_params[2*i+1]) for i in range(N)]
                best_angles = [best_params[2*N + i] for i in range(N)]
                
        except Exception:
            pass

    # If no valid solution found or best_L is still inf, return a fallback
    if best_L == float('inf') or best_params is None:
        # Fallback: simple grid, large L
        inner = []
        for i in range(N):
            inner.append([i * 3.0, 0.0, 0.0])
        return inner, (0.0, 0.0), 50.0, 0.0

    # Final verification and refinement
    # Check if the best solution is valid (no overlap)
    if check_overlap_fast(best_centers, best_angles):
        # It might have ended in an overlapping state due to penalty not being high enough
        # or local minima. Try to push them apart or return a safe fallback.
        # For robustness, let's try one more round of optimization with higher penalty
        # or just return the fallback if it's invalid.
        # Actually, if it's invalid, score is 0. So we must ensure validity.
        # Let's try to fix it by increasing L? No, L is calculated from positions.
        # If they overlap, the configuration is invalid.
        # We should have found a valid one. Let's assume the best one is valid.
        # If not, we return a known valid configuration.
        
        # Simple valid config: spread out widely
        inner = []
        for i in range(N):
            angle = 2 * math.pi * i / N
            r = 5.0
            inner.append([r * math.cos(angle), r * math.sin(angle), 0.0])
        L = calculate_L([(5.0*math.cos(2*math.pi*i/N), 5.0*math.sin(2*math.pi*i/N)) for i in range(N)], [0.0]*N)
        return inner, (0.0, 0.0), L, 0.0

    # Calculate final L
    final_L = calculate_L(best_centers, best_angles, 0.0)
    
    # Construct return value
    inner = []
    for i in range(N):
        inner.append([best_centers[i][0], best_centers[i][1], math.degrees(best_angles[i])])
        
    return inner, (0.0, 0.0), final_L, 0.0
# EVOLVE-BLOCK-END
