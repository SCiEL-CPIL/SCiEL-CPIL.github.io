"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin
    
    # Constants
    SQRT3 = math.sqrt(3)
    INRADIUS = SQRT3 / 2.0
    
    def axial_to_cartesian(q, r, spacing=1.75):
        # Flat-top hexagon grid
        # Standard spacing is SQRT3 for touching hexagons.
        # We use slightly larger spacing to ensure disjointness (gap).
        x = SQRT3 * spacing * (q + r / 2.0)
        y = 1.5 * spacing * r
        return x, y

    def min_outer_side_centers(centers, outer_angle_deg=0.0):
        """
        Calculate the minimum side length S of a regular hexagon with given 
        orientation such that it contains all disks of radius 1 centered at 'centers'.
        This guarantees the inner hexagons (radius 1) are contained.
        Assumes the outer hexagon is centered at the origin.
        """
        # Normals to the sides of a flat-top hexagon
        # Sides are perpendicular to 30, 90, 150, 210, 270, 330 degrees
        base_angles = [30, 90, 150, 210, 270, 330]
        normals = []
        for a in base_angles:
            rad = math.radians(a + outer_angle_deg)
            normals.append((math.cos(rad), math.sin(rad)))
        
        max_proj = 0.0
        for x, y in centers:
            for nx, ny in normals:
                # The disk extends 1 unit in any direction from the center.
                # So the max projection of the disk is |center_proj| + 1.
                proj = abs(x * nx + y * ny) + 1.0
                if proj > max_proj:
                    max_proj = proj
        
        # The apothem (distance center to side) of outer hex is S * SQRT3 / 2
        # So max_proj <= S * SQRT3 / 2
        # S >= 2 * max_proj / SQRT3
        return 2.0 * max_proj / SQRT3

    def evaluate_configuration(axial_coords, inner_angle=0.0):
        """
        Evaluate a configuration of inner hexagons given by axial coordinates.
        Finds the best outer angle and returns (side_length, inner_list, best_outer_angle).
        """
        centers = []
        inner_list = []
        
        for q, r in axial_coords:
            cx, cy = axial_to_cartesian(q, r)
            inner_list.append([cx, cy, inner_angle])
            centers.append((cx, cy))
        
        # Center the configuration at the origin
        cx_avg = sum(c[0] for c in centers) / len(centers)
        cy_avg = sum(c[1] for c in centers) / len(centers)
        
        centered_centers = [(x - cx_avg, y - cy_avg) for x, y in centers]
        
        best_side = float('inf')
        best_outer_angle = 0.0
        
        # Try several outer angles to find the optimal orientation
        for oa_deg in range(0, 60, 5):
            side = min_outer_side_centers(centered_centers, oa_deg)
            if side < best_side:
                best_side = side
                best_outer_angle = oa_deg
        
        # Shift inner list to match centered centers
        shifted_inner = []
        for i, (x, y, a) in enumerate(inner_list):
            shifted_inner.append([x - cx_avg, y - cy_avg, a])
            
        return best_side, shifted_inner, best_outer_angle

    # Candidate configurations for N=11
    # Base 7-hex cluster (radius 1)
    base_7 = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    
    candidates = []
    
    # Candidate 1: Add 4 to the right
    # Neighbors of the right side (1,0), (1,-1), (0,-1)
    # (2,0) is right of (1,0)
    # (2,-1) is right of (1,-1)
    # (2,1) is right of (0,1)? No, (1,1) is right of (0,1) and (1,0)?
    # Let's just append specific coordinates that form a compact shape.
    
    # Shape A: Elongated to the right
    # (2,0), (2,-1), (2,1), (1,1) ? 
    # Let's check connectivity.
    # (1,1) is neighbor of (0,1) and (1,0).
    # (2,0) is neighbor of (1,0) and (1,-1)? No, (1,-1) neighbor is (2,-1).
    # (2,0) neighbors: (1,0), (1,1), (2,1), (2,-1), (3,-1), (3,0).
    # So (2,0), (2,-1), (2,1), (1,1) seems like a valid block.
    cand_A = base_7 + [(2,0), (2,-1), (2,1), (1,1)]
    candidates.append(cand_A)
    
    # Shape B: Symmetric cross (2 right, 2 left)
    # (2,0), (-2,0), (0,2), (0,-2)
    cand_B = base_7 + [(2,0), (-2,0), (0,2), (0,-2)]
    candidates.append(cand_B)
    
    # Shape C: Symmetric diagonal (2 right-up, 2 left-down)
    # (1,1), (2,1) and (-1,-1), (-2,-1)
    cand_C = base_7 + [(1,1), (2,1), (-1,-1), (-2,-1)]
    candidates.append(cand_C)
    
    # Shape D: Fill corners more evenly
    # (1,1), (-1,1) and (1,-1), (-1,-1) are already in base?
    # Base: (0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Missing "corners" of the radius 2 hexagon are:
    # (2,0), (0,2), (-2,2), (-2,0), (0,-2), (2,-2)
    # And the "edge" ones: (1,1), (-1,2), (-2,1), (-1,-1), (1,-2), (2,-1)
    # Let's try adding (1,1), (-1,2), (-2,1), (2,-1)
    cand_D = base_7 + [(1,1), (-1,2), (-2,1), (2,-1)]
    candidates.append(cand_D)
    
    # Shape E: Another variation
    # (2,0), (1,1), (-1,2), (-2,0)
    cand_E = base_7 + [(2,0), (1,1), (-1,2), (-2,0)]
    candidates.append(cand_E)
    
    best_side = float('inf')
    best_inner = None
    best_outer_angle = 0.0
    best_inner_angle = 0.0
    
    for cand in candidates:
        if len(cand) != N:
            continue
        side, inner, oa = evaluate_configuration(cand, 0.0)
        if side < best_side:
            best_side = side
            best_inner = inner
            best_outer_angle = oa
            best_inner_angle = 0.0
                
    # If no candidate worked (shouldn't happen), fallback to linear
    if best_inner is None:
        # Fallback: place in a line with spacing 2.0 (safe for disjoint circles)
        best_inner = [[2.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        # Calculate side for this fallback
        centers = [(p[0], p[1]) for p in best_inner]
        best_side = min_outer_side_centers(centers, 0.0)
        best_outer_angle = 0.0
        best_inner_angle = 0.0

    return best_inner, (0.0, 0.0), best_side, best_outer_angle

# EVOLVE-BLOCK-END
