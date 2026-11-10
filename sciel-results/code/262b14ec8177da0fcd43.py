# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon.
    Uses L-BFGS-B optimization to minimize the outer hexagon size.
    Returns
        inner_hex_data: np.ndarray of shape (12,3)
        outer_hex_data: np.ndarray of shape (3,)
        outer_hex_side_length: float
    """
    start_time = time.time()
    deadline = start_time + 590  # Safety margin
    
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

    # Helper: Calculate minimum side length s for a set of inner hexes to fit in outer hex (centered at origin, angle 0)
    # Outer hex with side s, angle 0 has vertices at distance s from origin.
    # The "radius" (center to vertex) is s.
    # The "apothem" (center to edge midpoint) is s * sqrt(3)/2.
    # For a point (x,y) to be inside, its projection onto the 6 normal directions must be <= apothem.
    # Normals are at angles 0, 60, 120, 180, 240, 300? No.
    # Vertices of outer hex (side s, angle 0):
    # (s, 0), (s/2, s*sqrt3/2), (-s/2, s*sqrt3/2), (-s, 0), (-s/2, -s*sqrt3/2), (s/2, -s*sqrt3/2)
    # Edges connect these.
    # Edge 1: (s,0) to (s/2, s*sqrt3/2). Vector: (-s/2, s*sqrt3/2). Normal (outward): (s*sqrt3/2, s/2) -> angle 30 deg.
    # So normals are at 30, 90, 150, 210, 270, 330 degrees.
    # Condition: dot(v, n_i) <= s * sqrt3 / 2 for all i.
    # So s >= max_i( dot(v, n_i) ) * 2 / sqrt3.
    
    normal_angles = [math.radians(30 + i*60) for i in range(6)]
    normals = np.array([[math.cos(a), math.sin(a)] for a in normal_angles])
    
    def calc_min_s(centers):
        # centers: (N, 2) array
        # For each center, compute max projection of its vertices onto normals
        # Vertices of unit hex at (cx, cy) with angle 0:
        # (1,0), (0.5, sqrt3/2), (-0.5, sqrt3/2), (-1,0), (-0.5,-sqrt3/2), (0.5,-sqrt3/2)
        # We can precompute vertex offsets
        v_offsets = np.array([
            [1.0, 0.0],
            [0.5, sqrt3/2],
            [-0.5, sqrt3/2],
            [-1.0, 0.0],
            [-0.5, -sqrt3/2],
            [0.5, -sqrt3/2]
        ])
        
        max_proj = 0.0
        for c in centers:
            verts = c + v_offsets
            # Project onto normals
            # verts is (6,2), normals is (6,2)
            # We need dot product of each vertex with each normal
            # (6,2) @ (2,6) -> (6,6) matrix of projections
            projs = verts @ normals.T
            max_p = np.max(projs)
            if max_p > max_proj:
                max_proj = max_p
                
        return max_proj * 2.0 / sqrt3

    def calc_penalty(centers, s):
        # Centers: (N, 2)
        # s: float
        # Penalty for overlaps and containment violations
        
        v_offsets = np.array([
            [1.0, 0.0],
            [0.5, sqrt3/2],
            [-0.5, sqrt3/2],
            [-1.0, 0.0],
            [-0.5, -sqrt3/2],
            [0.5, -sqrt3/2]
        ])
        
        penalty = 0.0
        
        # 1. Containment
        # Outer hex apothem = s * sqrt3 / 2
        apothem = s * sqrt3 / 2
        for c in centers:
            verts = c + v_offsets
            projs = verts @ normals.T
            # projs shape (6, 6). Each column is projection onto a normal.
            # We need max(projs, axis=0) for each normal direction to be <= apothem
            max_projs = np.max(projs, axis=0)
            violations = np.maximum(0, max_projs - apothem)
            penalty += np.sum(violations**2)
            
        # 2. Overlaps
        # For each pair of hexes, check if they overlap.
        # Approximation: If distance between centers < 2 * circumradius (2.0) + eps, they might overlap.
        # Better: Use SAT or just distance penalty for simplicity in gradient descent?
        # Distance penalty is smooth and often sufficient for hexagons if we use a slightly larger threshold.
        # However, hexagons can be close without overlapping if rotated. But we fix angle=0.
        # With angle=0, hexagons are aligned.
        # Minimum distance between centers for non-overlap of aligned unit hexagons:
        # In x direction: 2.0 (vertex to vertex) or 1.5 (edge to edge? no).
        # Let's just use a soft repulsion force based on distance.
        # If dist < 2.0, they definitely overlap (vertex-vertex).
        # If dist < 1.5*sqrt3 (~2.59), they might overlap.
        # Let's use a penalty that kicks in when dist < 2.0.
        # Actually, for aligned hexagons, the Minkowski sum is a hexagon of side 2.
        # So if center distance vector is inside the hexagon of side 2, they overlap.
        # The "radius" of this sum hexagon is 2.0 (vertex) and 2*sqrt3/2 = sqrt3 (apothem).
        # We can check if the difference vector is inside this hexagon.
        # But for gradient descent, a simple quadratic penalty on distance is often enough to push them apart.
        # Let's use: penalty += sum(max(0, 2.0 - dist)^2)
        
        for i in range(N):
            for j in range(i+1, N):
                diff = centers[i] - centers[j]
                dist = np.linalg.norm(diff)
                if dist < 2.0:
                    # They are likely overlapping or very close
                    # Penalty to push them apart
                    penalty += (2.0 - dist)**2 * 100.0 # Weight overlap heavily
                # Also check specific directions?
                # For aligned hexes, overlap happens if projection on any of 3 axes (x, 60, 120) is too small?
                # No, the condition for non-overlap of two convex polygons is that there exists a separating axis.
                # For aligned hexes, the separating axes are the normals of the hexes.
                # If they overlap, no separating axis exists.
                # The "width" of a unit hex along normal n is 2 * apothem = sqrt3.
                # Wait, width is distance between parallel edges.
                # For angle 0 hex, width along x-axis (normal 0 deg? No, normal 30 deg is edge normal).
                # Let's stick to distance penalty. It's robust.
        
        return penalty

    def objective(params):
        # params: 2N array [x1, y1, x2, y2, ..., xN, yN]
        centers = params.reshape(N, 2)
        
        # Calculate required s
        s = calc_min_s(centers)
        
        # Calculate penalty
        pen = calc_penalty(centers, s)
        
        # Objective: minimize s + penalty
        # If penalty is 0, we just minimize s.
        # If penalty > 0, we penalize invalid configurations.
        # We want to drive penalty to 0.
        # Return s + lambda * penalty
        return s + 10.0 * pen

    best_s = float('inf')
    best_centers = None
    
    # Multiple restarts
    num_restarts = 20
    
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        # Initial guess: Random or Lattice
        if r == 0:
            # Lattice start
            centers = []
            # 4x3 grid
            for i in range(3):
                for j in range(4):
                    x = j * 2.5
                    y = i * sqrt3
                    if i % 2 == 1:
                        x += 1.25
                    centers.append([x, y])
            centers = np.array(centers)
        else:
            # Random start within a box
            centers = np.random.uniform(-5, 5, (N, 2))
            
        # Center the configuration
        centers -= np.mean(centers, axis=0)
        
        params = centers.flatten()
        
        # Optimize
        try:
            res = minimize(objective, params, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-10})
            if res.success or res.status == 1: # 1 means maxiter reached but maybe good enough
                final_centers = res.x.reshape(N, 2)
                # Recenter
                final_centers -= np.mean(final_centers, axis=0)
                
                # Calculate final s
                final_s = calc_min_s(final_centers)
                
                # Check validity
                # 1. Overlaps
                valid = True
                v_offsets = np.array([
                    [1.0, 0.0],
                    [0.5, sqrt3/2],
                    [-0.5, sqrt3/2],
                    [-1.0, 0.0],
                    [-0.5, -sqrt3/2],
                    [0.5, -sqrt3/2]
                ])
                
                # Check containment
                apothem = final_s * sqrt3 / 2
                for c in final_centers:
                    verts = c + v_offsets
                    projs = verts @ normals.T
                    if np.any(np.max(projs, axis=0) > apothem + 1e-6):
                        valid = False
                        break
                
                # Check overlaps (SAT for aligned hexes)
                if valid:
                    # For aligned hexes, we can check projection on 3 axes: x, 60, 120
                    # Projected width of unit hex on x-axis: 2.0 (from -1 to 1)
                    # Projected width on 60 deg axis: sqrt3
                    # If sum of half-widths > distance projection, they overlap?
                    # No, if intervals overlap, they intersect.
                    # Interval 1: [c1 . n - w/2, c1 . n + w/2]
                    # Interval 2: [c2 . n - w/2, c2 . n + w/2]
                    # Overlap if |c1 . n - c2 . n| < w
                    # For x-axis, w=2.
                    # For 60 deg axis, w=sqrt3.
                    # For 120 deg axis, w=sqrt3.
                    # If they overlap in ALL 3 axes, do they overlap?
                    # For convex polygons, if they overlap in all separating axes, they intersect.
                    # The separating axes for two aligned hexes are the normals of their edges.
                    # There are 6 unique directions, but pairs of opposite normals are sufficient?
                    # Actually, we just need to check the 6 normals.
                    # If for ANY normal, the projections are separated, they are disjoint.
                    # If for ALL normals, projections overlap, they are NOT necessarily disjoint?
                    # Wait. SAT says: If there is a separating axis, they are disjoint.
                    # So if we find NO separating axis among the candidates, they are intersecting.
                    # The candidate axes are the normals of the edges of both polygons.
                    # Since they are aligned, the normals are the same 6 vectors.
                    
                    for i in range(N):
                        for j in range(i+1, N):
                            diff = final_centers[i] - final_centers[j]
                            separated = False
                            for n in normals:
                                proj_diff = np.dot(diff, n)
                                # Width of unit hex along normal n
                                # For normal at 30 deg (edge normal), width is sqrt3 (apothem*2)
                                # For normal at 0 deg (vertex normal), width is 2.0
                                # Let's calculate width for each normal
                                # Project v_offsets onto n
                                proj_offsets = v_offsets @ n
                                w = np.max(proj_offsets) - np.min(proj_offsets)
                                
                                if abs(proj_diff) > w + 1e-8:
                                    separated = True
                                    break
                            
                            if not separated:
                                valid = False
                                break
                        if not valid:
                            break
                            
                if valid and final_s < best_s:
                    best_s = final_s
                    best_centers = final_centers
                    
        except Exception:
            pass
            
    if best_centers is None:
        # Fallback to lattice
        centers = []
        for i in range(3):
            for j in range(4):
                x = j * 2.5
                y = i * sqrt3
                if i % 2 == 1:
                    x += 1.25
                centers.append([x, y])
        best_centers = np.array(centers)
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
