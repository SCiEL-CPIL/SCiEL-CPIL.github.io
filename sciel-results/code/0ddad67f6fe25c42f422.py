"""Stronger algorithm: hexagonal lattice packing with optimized outer hexagon."""
import time
import math
import numpy as np
from scipy.optimize import linprog

N = 11

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Step 1: Generate candidate inner hexagon positions on hexagonal lattice
    # We'll try different shapes: hexagonal cluster, and variations
    candidates = _generate_candidate_shapes()
    
    best_result = None
    best_side = float('inf')
    
    for idx, (positions, angles) in enumerate(candidates):
        if time.time() > deadline - 20:
            break
        
        # For this shape, find the optimal outer hexagon
        # The outer hexagon must contain all inner hexagons
        # Inner hexagons are unit side length, so their vertices are at distance 1 from center
        # (for a regular hexagon of side 1, the circumradius is 1)
        
        # For each inner hexagon with center (x,y) and rotation angle,
        # its 6 vertices are at (x + cos(a+60k), y + sin(a+60k)) for k=0..5
        
        # We need to find outer hexagon (center (cx,cy), side L, rotation phi)
        # such that all inner vertices are inside the outer hexagon.
        
        # A point (px,py) is inside a regular hexagon with center (cx,cy), side L, 
        # rotation phi if and only if for all 6 face normals n_i:
        #   dot((px-cx, py-cy), n_i) <= L * cos(30°) = L * sqrt(3)/2
        # where n_i are unit normals to the 6 faces.
        
        # Actually, for a regular hexagon with side length L, the inradius is L*sqrt(3)/2.
        # The distance from center to each face is L*sqrt(3)/2.
        
        # So the constraint for each point p_i is:
        #   max over 6 directions n_j of dot(p_i - c, n_j) <= L * sqrt(3)/2
        
        # We want to minimize L, so L >= max_i (max_j dot(p_i - c, n_j)) / (sqrt(3)/2)
        
        # This is equivalent to finding c that minimizes max_i (max_j dot(p_i - c, n_j))
        
        # Let's use a different approach: for a fixed outer rotation phi,
        # the 6 face normals are fixed. Then we want to find c that minimizes
        # max over all points p and all normals n of dot(p - c, n)
        
        # This is a convex optimization problem. We can solve it with LP.
        # Minimize t
        # subject to: dot(p_i - c, n_j) <= t  for all i, j
        #             -dot(p_i - c, n_j) <= t  for all i, j  (to make it symmetric? No)
        
        # Actually, the hexagon constraint is: for each face normal n_j,
        #   dot(p - c, n_j) <= R_in  where R_in = L * sqrt(3)/2
        # So L >= max over all (i,j) of dot(p_i - c, n_j) / (sqrt(3)/2)
        
        # To minimize L, we minimize max over all (i,j) of dot(p_i - c, n_j)
        
        # Variables: cx, cy, t
        # Minimize t
        # Subject to:
        #   For each point p_i and each normal n_j:
        #     dot(p_i - c, n_j) <= t
        #     i.e., p_ix * n_jx + p_iy * n_jy - cx * n_jx - cy * n_jy <= t
        
        # Let's collect all points first
        
        all_points = _get_all_vertices(positions, angles)
        
        if len(all_points) == 0:
            continue
        
        # Try several outer rotations
        outer_rotations = np.linspace(0, 60, 25)  # 0 to 60 degrees, 25 points
        
        best_local_side = float('inf')
        best_local_center = (0, 0)
        
        for phi_deg in outer_rotations:
            if time.time() > deadline - 5:
                break
            
            phi = math.radians(phi_deg)
            # Face normals: perpendicular to the edges. For a hexagon with vertices at angles 
            # phi + 60*k, the face normals point at angles phi + 30 + 60*k
            normals = []
            for k in range(6):
                angle = phi + math.pi/6 + math.pi*k/3  # 30 + 60k degrees offset
                normals.append((math.cos(angle), math.sin(angle)))
            normals = np.array(normals)  # shape (6, 2)
            
            # Points shape: (M, 2)
            M = len(all_points)
            P = np.array(all_points)  # (M, 2)
            
            # LP: minimize t
            # Variables: x = [cx, cy, t]
            # Constraints: for each i, j: 
            #   P[i,0]*n_jx + P[i,1]*n_jy - cx*n_jx - cy*n_jy <= t
            #   i.e., (-n_jx)*cx + (-n_jy)*cy - t <= -P[i,0]*n_jx - P[i,1]*n_jy
            
            # A matrix: (M*6, 3)
            A_rows = []
            b_rows = []
            for i in range(M):
                for j in range(6):
                    njx, njy = normals[j]
                    # -njx*cx - njy*cy - 1*t <= -(P[i,0]*njx + P[i,1]*njy)
                    A_rows.append([-njx, -njy, -1.0])
                    b_rows.append(-(P[i,0]*njx + P[i,1]*njy))
            
            A = np.array(A_rows)
            b = np.array(b_rows)
            
            # Objective: minimize t, so c = [0, 0, 1]
            c_obj = np.array([0.0, 0.0, 1.0])
            
            try:
                res = linprog(c_obj, A_ub=A, b_ub=b, bounds=[(None, None), (None, None), (0, None)], method='highs')
                if res.success:
                    t_val = res.x[2]
                    side_val = t_val / (math.sqrt(3) / 2)
                    if side_val < best_local_side:
                        best_local_side = side_val
                        best_local_center = (res.x[0], res.x[1])
            except:
                pass
        
        if best_local_side < best_side and best_local_side > 0:
            best_side = best_local_side
            best_result = (positions, angles, best_local_center, best_local_side)
    
    # If no valid result found, fall back to a simple arrangement
    if best_result is None:
        # Simple line arrangement with safe bounding
        positions = [[float(k), 0.0] for k in range(-N//2, N - N//2)]
        angles = [0.0] * N
        # Compute bounding box
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        max_x = max(xs) + 1
        max_y = 1
        # For a line of hexagons, need hexagon with width >= 2*max_x, height >= 2
        # Width of hexagon with side L (flat-top) = 2L, height = L*sqrt(3)
        # Or pointy-top: width = L*sqrt(3), height = 2L
        L = max(max_x, max_y)
        # Be safe
        L = max(L, 5.0)
        return positions, (0.0, 0.0), L, 0.0
    
    positions, angles, center, side = best_result
    
    # Final validity check and safety margin
    # Add small margin to avoid numerical issues
    side *= 1.0001
    
    return positions, center, side, 0.0


def _generate_candidate_shapes():
    """Generate different candidate arrangements of N unit hexagons."""
    candidates = []
    
    # Shape 1: Hexagonal cluster (center + rings)
    # For N=11: center + 6 (ring 1) + 4 from ring 2
    # Full hex cluster of radius 2 has 19 hexagons. We need 11.
    # Center + ring1 (6) = 7. Need 4 more from ring 2.
    # Place 4 from ring 2 symmetrically: e.g., at (2,0), (0,2), (-2,0), (0,-2) in axial coords
    # But that might not be symmetric. Let's use axial coordinates.
    
    # Axial to Cartesian: x = 1.5 * q, y = sqrt(3) * (r + q/2)
    # For a unit hexagon, adjacent centers are distance sqrt(3) apart? No.
    # For side length 1, the distance between centers of adjacent hexagons is sqrt(3).
    # Wait, for regular hexagons of side 1 tiled together, the center-to-center distance is sqrt(3).
    # Let me recalculate: a regular hexagon of side 1 has inradius sqrt(3)/2, circumradius 1.
    # When tiling, the distance between centers of adjacent hexagons is 2 * inradius = sqrt(3).
    # In axial coordinates (q, r), the 6 neighbors are at:
    # (q+1, r), (q-1, r), (q, r+1), (q, r-1), (q+1, r-1), (q-1, r+1)
    # Cartesian: x = sqrt(3) * (q + r/2), y = 3/2 * r  ... let me use standard formula.
    # Actually for pointy-top hexagons: x = sqrt(3) * (q + r/2), y = 3*r/2
    # For flat-top: different.
    
    # Let's use axial coords with x = sqrt(3) * (q + r/2), y = 3*r/2 for pointy-top hexagons.
    # But the inner hexagons can rotate. Let's keep them all with angle 0 for simplicity in the lattice.
    
    # Generate hex cluster positions in axial coords
    def axial_to_cartesian(q, r):
        x = math.sqrt(3) * (q + r / 2)
        y = 3 * r / 2
        return (x, y)
    
    # Shape 1: Compact cluster
    # Center (0,0), ring1: 6 neighbors, then pick 4 more from ring2 for symmetry
    axial_positions = [(0, 0)]
    # Ring 1
    ring1 = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    axial_positions.extend(ring1)
    # Need 4 more from ring 2. Pick symmetrically: (2,0), (0,2), (-2,0), (0,-2) in axial?
    # Axial ring 2 has 12 positions. Let's pick 4 that are spread out.
    # Actually, for 11 hexagons, a better shape might be:
    # Center + 6 + 4 = 11. The 4 should be at "corners" to make it compact.
    ring2_corners = [(2, -1), (1, -2), (-1, -1), (1, 1)]  # Hmm, not sure
    # Let me just try a few sets
    ring2_set1 = [(2, 0), (0, 2), (-2, 0), (0, -2)]
    ring2_set2 = [(2, -1), (-1, -2), (-1, 2), (1, 2)]
    
    for ring2_set in [ring2_set1, ring2_set2]:
        axial_pos = list(axial_positions) + list(ring2_set)
        cart = [axial_to_cartesian(q, r) for q, r in axial_pos]
        angles = [0.0] * len(cart)
        candidates.append((cart, angles))
    
    # Shape 2: Line
    positions = [[math.sqrt(3) * k, 0.0] for k in range(-N//2, N - N//2)]
    angles = [0.0] * N
    candidates.append((positions, angles))
    
    # Shape 3: Two rows
    # 6 in bottom, 5 in top, offset
    positions = []
    for i in range(6):
        positions.append((math.sqrt(3) * (i - 2.5), -math.sqrt(3)/2))
    for i in range(5):
        positions.append((math.sqrt(3) * (i - 2.0), math.sqrt(3)/2))
    angles = [0.0] * N
    candidates.append((positions, angles))
    
    # Shape 4: Cross-like
    positions = []
    # Center
    positions.append((0, 0))
    # 4 horizontal/vertical
    for k in [1, 2]:
        positions.append((math.sqrt(3)*k, 0))
        positions.append((-math.sqrt(3)*k, 0))
    for k in [1]:
        positions.append((0, 3*k/2))
        positions.append((0, -3*k/2))
    # That's 7. Need 4 more
    positions.append((math.sqrt(3), 3/2))
    positions.append((math.sqrt(3), -3/2))
    positions.append((-math.sqrt(3), 3/2))
    positions.append((-math.sqrt(3), -3/2))
    angles = [0.0] * len(positions)
    if len(positions) == N:
        candidates.append((positions, angles))
    
    return candidates


def _get_all_vertices(positions, angles):
    """Get all 6 vertices of each inner hexagon."""
    all_points = []
    for (x, y), ang_deg in zip(positions, angles):
        ang = math.radians(ang_deg)
        for k in range(6):
            theta = ang + math.pi * k / 3
            px = x + math.cos(theta)
            py = y + math.sin(theta)
            all_points.append((px, py))
    return all_points

# EVOLVE-BLOCK-END
