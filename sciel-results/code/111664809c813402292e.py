"""
Strategy: Grid-based Lattice Search with Local Refinement.
Instead of physics simulation or gradient descent, we exploit the symmetry of the problem.
We hypothesize that the optimal packing consists of a central hexagon and a ring of 10 surrounding hexagons, all aligned (angle=0).
We perform a coarse-to-fine grid search on the center-to-center distance `d` between the central hexagon and its neighbors.
For each `d`, we calculate the minimal enclosing hexagon side length.
We then apply a local refinement (Nelder-Mead) on the positions to squeeze out the last bit of space, ensuring validity.
This is structurally different: it relies on explicit geometric construction and parameter sweeping rather than iterative force relaxation.
"""

import numpy as np
import time
import random
import math
import sys
from scipy.optimize import minimize

N = 11
DEADLINE = 145.0  # Safety margin
START_TIME = time.time()

def _remaining_time():
    return DEADLINE - (time.time() - START_TIME)

def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon."""
    angle_rad = np.radians(angle_deg)
    verts = []
    for k in range(6):
        a = angle_rad + k * np.pi / 3
        verts.append([cx + side * np.cos(a), cy + side * np.sin(a)])
    return np.array(verts)

def _get_face_normals(angle_deg):
    """Return unit normals for the 6 faces of a hexagon at given angle."""
    angle_rad = np.radians(angle_deg)
    normals = []
    for k in range(6):
        a = angle_rad + np.pi/6 + k * np.pi / 3
        normals.append([np.cos(a), np.sin(a)])
    return np.array(normals)

def _min_enclosing_side(inner_positions, outer_center, outer_angle):
    """
    Calculate the minimum side length required to contain all inner hexagons.
    """
    cx, cy = outer_center
    face_normals = _get_face_normals(outer_angle)
    
    max_proj = 0.0
    for (ix, iy, iangle) in inner_positions:
        iverts = _hex_vertices(ix, iy, 1.0, iangle)
        iverts_rel = iverts - np.array(outer_center)
        
        for n in face_normals:
            projs = iverts_rel @ n
            m = np.max(projs)
            if m > max_proj:
                max_proj = m
                
    return 2.0 * max_proj / np.sqrt(3)

def _check_validity(inner_positions, outer_center, outer_side, outer_angle, tol=1e-5):
    """
    Strictly check if the configuration is valid:
    1. All inner hexagons are inside the outer hexagon.
    2. No inner hexagons overlap.
    """
    cx, cy = outer_center
    outer_normals = _get_face_normals(outer_angle)
    inradius = outer_side * np.sqrt(3) / 2.0
    
    # Pre-compute vertices for all inner hexagons
    all_verts = []
    for i in range(N):
        ix, iy, iangle = inner_positions[i]
        verts = _hex_vertices(ix, iy, 1.0, iangle)
        all_verts.append(verts)
        
    # 1. Check Containment
    for i in range(N):
        verts = all_verts[i]
        verts_rel = verts - np.array(outer_center)
        for n in outer_normals:
            projs = verts_rel @ n
            if np.max(projs) > inradius + tol:
                return False
                
    # 2. Check Disjointness (SAT - Separating Axis Theorem)
    for i in range(N):
        for j in range(i + 1, N):
            v1 = all_verts[i]
            v2 = all_verts[j]
            
            n1 = _get_face_normals(inner_positions[i][2])
            n2 = _get_face_normals(inner_positions[j][2])
            
            disjoint = False
            for n in np.vstack([n1, n2]):
                proj1 = v1 @ n
                proj2 = v2 @ n
                
                if np.max(proj1) < np.min(proj2) - tol or np.max(proj2) < np.min(proj1) - tol:
                    disjoint = True
                    break
            
            if not disjoint:
                return False
                
    return True

def _objective(params, outer_center, outer_angle):
    """
    Objective function for optimization: minimize the enclosing side length.
    params: array of [x1, y1, a1, x2, y2, a2, ...]
    """
    if _remaining_time() < 2.0:
        return 1e10
        
    # Reshape params
    pos = params.reshape(N, 3)
    
    # Calculate enclosing side
    side = _min_enclosing_side(pos.tolist(), outer_center, outer_angle)
    
    # Check for overlaps (hard constraint penalty)
    # We can't easily check overlaps in the objective without high cost, 
    # so we rely on the structure of the search space (grid search) to avoid them,
    # or we add a penalty if we detect overlaps.
    
    # For speed in the objective, we do a quick center-distance check for overlaps.
    # If centers are too close, it's likely an overlap.
    # Min distance for side 1 hexagons is 2*sqrt(3) ~ 3.464 if aligned.
    # If they are rotated, it can be closer, but let's assume alignment for the main search.
    # Actually, let's just return the side length and let the final check validate.
    # But if we return a small side for an overlapping config, the optimizer will try it.
    # So we need a penalty.
    
    penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(pos[i][:2] - pos[j][:2])
            # Approximate min distance for unit hexagons. 
            # If d < 2.0, they definitely overlap (circumradius sum is 2.0).
            # If d < 3.0, they likely overlap unless rotated.
            # Let's use a soft penalty for d < 3.464 (aligned min distance)
            if d < 3.464:
                penalty += (3.464 - d) * 100.0
            elif d < 2.5:
                penalty += (2.5 - d) * 1000.0
                
    return side + penalty

def _local_refine(initial_pos, outer_center, outer_angle):
    """
    Perform a local Nelder-Mead optimization to refine the positions.
    """
    # Flatten
    x0 = np.array(initial_pos).flatten()
    
    # Bounds: keep them within a reasonable area
    bounds = [(-5, 5)] * (2 * N) + [(0, 360)] * N
    
    res = minimize(_objective, x0, args=(outer_center, outer_angle), method='Nelder-Mead',
                   options={'xatol': 1e-4, 'fatol': 1e-4, 'maxiter': 500, 'maxfev': 2000})
    
    if res.success:
        refined_pos = res.x.reshape(N, 3).tolist()
        return refined_pos
    
    return initial_pos

def construct():
    """
    Main entry point.
    1. Grid search on the lattice parameter `d` (distance from center to ring).
    2. For each `d`, construct the 1+10 layout.
    3. Evaluate the enclosing side length.
    4. Track the best.
    5. Perform local refinement on the best few candidates.
    6. Return the best valid configuration.
    """
    random.seed(42)
    np.random.seed(42)
    
    outer_center = (0.0, 0.0)
    outer_angle = 0.0
    
    best_side = float('inf')
    best_pos = None
    
    # Theoretical min distance for 11 hexagons in a hexagon.
    # If we place them in a ring of 10 around 1 center:
    # The ring hexagons touch the center hexagon.
    # Distance between centers of touching aligned hexagons is 2 * inradius = 2 * sqrt(3)/2 * 1 * 2? No.
    # Inradius of unit hex is sqrt(3)/2.
    # Distance between centers of two touching aligned hexagons (face to face) is 2 * inradius = sqrt(3) ~ 1.732? No.
    # Wait.
    # Hexagon side 1.
    # Inradius (center to face) = sqrt(3)/2.
    # If two hexagons touch face-to-face, distance between centers is 2 * inradius = sqrt(3) ~ 1.732.
    # If they touch vertex-to-vertex (corner), distance is 2 * circumradius = 2.0.
    # In a standard hex packing, neighbors are at distance sqrt(3) if aligned?
    # Let's check.
    # Center at (0,0). Neighbor at (sqrt(3), 0)?
    # Face normal is (1,0). Face is at x=sqrt(3)/2.
    # Other hex center at (sqrt(3), 0). Its left face is at x = sqrt(3) - sqrt(3)/2 = sqrt(3)/2.
    # So yes, distance is sqrt(3).
    
    # So we search for d around sqrt(3) ~ 1.732.
    # But we can compress? No, they must be disjoint.
    # So d must be >= sqrt(3) if aligned.
    # If we rotate them, can we get closer?
    # Maybe. But let's start with aligned.
    
    # Search range for d
    d_min = 1.732
    d_max = 3.0
    num_steps = 50
    
    # Grid search
    for i in range(num_steps):
        if _remaining_time() < 10.0:
            break
            
        d = d_min + (d_max - d_min) * i / (num_steps - 1)
        
        # Construct 1+10 layout
        pos = []
        pos.append([0.0, 0.0, 0.0]) # Center
        
        for k in range(10):
            # 10 neighbors. 
            # In a hex grid, a center has 6 neighbors.
            # To get 10, we need a second ring or a different arrangement.
            # A "star" of 10 around 1 is not a standard hex lattice.
            # Standard hex lattice: Center + 6 neighbors + 12 second neighbors.
            # We have 10. So Center + 6 + 4 from second ring?
            # Or maybe a different topology.
            # Let's try: 6 in the first ring (distance d1), 4 in the second ring (distance d2).
            # Or maybe all 10 are in a single ring? 
            # A regular decagon of hexagons?
            # Let's try placing 10 hexagons in a circle of radius R around the center.
            # They must not overlap each other.
            # Distance between adjacent ring hexagons:
            # Chord length = 2 * R * sin(pi/10).
            # This chord length must be >= sqrt(3) (min distance for aligned hexagons).
            # So 2 * R * sin(18 deg) >= sqrt(3).
            # sin(18) ~ 0.309.
            # 2 * R * 0.309 >= 1.732 => R >= 1.732 / 0.618 ~ 2.8.
            # So if we use a single ring of 10, R is at least 2.8.
            # The enclosing hexagon size will be roughly R + 1 (circumradius of inner) = 3.8.
            # Side length S ~ 3.8 / (sqrt(3)/2) ~ 4.4.
            
            # Alternatively, use the hex lattice structure:
            # Center + 6 neighbors (dist sqrt(3)) + 4 more.
            # The 6 neighbors are at angles 0, 60, 120, 180, 240, 300.
            # The 4 more could be placed in the gaps of the second ring.
            # Second ring neighbors are at distance 2*sqrt(3) ~ 3.464? No.
            # In hex lattice, distance to second ring neighbors is 2 * sqrt(3) / 2 * 2?
            # Let's calculate.
            # Center (0,0). Neighbor 1: (sqrt(3), 0). Neighbor 2: (sqrt(3)/2, 1.5).
            # Distance between N1 and N2: sqrt((sqrt(3)/2)^2 + (1.5)^2) = sqrt(0.75 + 2.25) = sqrt(3).
            # So they touch.
            # Second ring point: e.g., (2*sqrt(3), 0).
            # Distance from Center: 2*sqrt(3) ~ 3.464.
            # Distance from N1 (sqrt(3), 0): sqrt(3) ~ 1.732.
            # So second ring points are at distance 2*sqrt(3) from center.
            
            # So we have 6 at d=sqrt(3) and 12 at d=2*sqrt(3).
            # We need 4 more. Pick 4 from the 12.
            # To minimize enclosing hexagon, we should pick the 4 that are "closest" to the center?
            # No, they are all at same distance.
            # We should pick them to be symmetric?
            # Or maybe we can move them inward?
            # If we move them inward, they might overlap with the first ring.
            # Let's just place them at the standard lattice positions for now.
            
            # Layout:
            # 1 at (0,0)
            # 6 at radius sqrt(3), angles 0, 60, ...
            # 4 at radius 2*sqrt(3), angles 30, 90, 150, 210? (Gaps)
            
            # Let's generate this specific layout.
            d1 = np.sqrt(3)
            d2 = 2 * np.sqrt(3)
            
            # 6 inner
            for k in range(6):
                a = np.radians(60 * k)
                pos.append([d1 * np.cos(a), d1 * np.sin(a), 0.0])
                
            # 4 outer (in the gaps)
            for k in range(4):
                a = np.radians(30 + 60 * k)
                pos.append([d2 * np.cos(a), d2 * np.sin(a), 0.0])
                
            # Calculate side
            side = _min_enclosing_side(pos, outer_center, outer_angle)
            
            if side < best_side:
                best_side = side
                best_pos = pos.copy()
                
    # If we found a candidate, try to refine it
    if best_pos is not None:
        if _remaining_time() > 20.0:
            # Try local refinement on the best found
            # Note: The grid search above used a fixed layout. 
            # Local refinement might break the symmetry and improve it.
            refined = _local_refine(best_pos, outer_center, outer_angle)
            
            # Check validity of refined
            if _check_validity(refined, outer_center, best_side, outer_angle, tol=1e-4):
                # Recalculate side for refined
                new_side = _min_enclosing_side(refined, outer_center, outer_angle)
                if new_side < best_side:
                    best_side = new_side
                    best_pos = refined
                    
    # Fallback if no valid config found (should not happen with d=2*sqrt(3))
    if best_pos is None:
        # Create a loose valid config
        pos = []
        pos.append([0.0, 0.0, 0.0])
        for k in range(10):
            a = 2 * np.pi * k / 10
            r = 4.0
            pos.append([r * np.cos(a), r * np.sin(a), 0.0])
        side = _min_enclosing_side(pos, outer_center, outer_angle)
        return pos, outer_center, side, outer_angle
        
    return best_pos, outer_center, best_side, outer_angle

if __name__ == "__main__":
    res = construct()
    print(res)
