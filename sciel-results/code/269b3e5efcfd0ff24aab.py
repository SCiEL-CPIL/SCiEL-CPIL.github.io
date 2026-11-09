"""Packing 12 unit hexagons in minimal regular hexagon."""
import math
import time
import numpy as np
from itertools import combinations, product

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill
    
    sqrt3 = math.sqrt(3.0)
    
    def hex_contains(cx, cy, side, angle_deg, points):
        """Check if all points are inside a regular hexagon centered at (cx,cy) with given side and angle."""
        angle = math.radians(angle_deg)
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        
        # Regular hexagon vertices (flat-top, before rotation) at radius = side
        # Vertices at angles 0, 60, 120, 180, 240, 300 degrees
        # For containment, we can check using the 6 half-planes
        # A regular hexagon with center (cx,cy), circumradius R=side, rotated by angle
        # The 6 sides are at distance apothem = side * sqrt(3)/2 from center
        # Each side normal is at angle angle + i*60 degrees
        
        apothem = side * sqrt3 / 2.0
        for i in range(6):
            normal_angle = angle + math.radians(60 * i)
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            for (px, py) in points:
                # Project point onto normal, must be <= apothem
                proj = (px - cx) * nx + (py - cy) * ny
                if proj > apothem + 1e-9:
                    return False
        return True
    
    def enclosing_hexagon_side(points, cx, cy, angle_deg):
        """Find minimum side length of regular hexagon centered at (cx,cy) with rotation angle_deg containing all points."""
        angle = math.radians(angle_deg)
        apothem = 0.0
        for i in range(6):
            normal_angle = angle + math.radians(60 * i)
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            for (px, py) in points:
                proj = (px - cx) * nx + (py - cy) * ny
                if proj > apothem:
                    apothem = proj
        # apothem = side * sqrt(3)/2, so side = 2*apothem/sqrt(3)
        return 2.0 * apothem / sqrt3
    
    def compute_bounding_hexagon(points, center_offset=(0.0, 0.0)):
        """Compute the minimal enclosing regular hexagon for given points."""
        cx, cy = center_offset
        best_side = float('inf')
        best_angle = 0.0
        
        # Try multiple angles
        for angle_deg in np.linspace(0, 60, 61):
            side = enclosing_hexagon_side(points, cx, cy, angle_deg)
            if side < best_side:
                best_side = side
                best_angle = angle_deg
        
        return best_side, best_angle
    
    # Generate hex grid positions for a side-3 hexagonal cluster (19 positions)
    # Axial coordinates (q, r) with max(|q|, |r|, |q+r|) <= 2
    all_positions = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if abs(q + r) <= 2:
                # Flat-top hex grid to cartesian
                x = 1.5 * q
                y = sqrt3 * (r + q / 2.0)
                all_positions.append((x, y, q, r))
    
    # We have 19 positions, need to choose 12
    # Try several subsets: remove 7 positions
    # Good strategy: keep the center and a compact shape
    
    # Let's try: keep all except 7 corner/edge positions
    # The 19 positions in a hex cluster of side 3:
    # Center: (0,0)
    # Ring 1 (6 positions): distance 1
    # Ring 2 (12 positions): distance 2
    
    # For 12 hexagons, a good compact arrangement:
    # Option 1: Center + Ring 1 (6) + 5 from Ring 2 = 12
    # Option 2: Center + 5 from Ring 1 + 6 from Ring 2 = 12
    
    # Let's try several arrangements and pick the best
    
    best_result = None
    best_score = float('inf')
    
    # Try arrangement: center + all 6 ring-1 + 5 from ring-2
    # Ring 2 positions (12 total), pick 5 that give most compact shape
    
    # First, let's just try a simple arrangement: 
    # Place 12 hexagons in a 3x4 rectangular-ish block or similar
    
    # Actually, let's use the full 19-hex cluster and remove 7 strategically
    # The most compact 12-subset would minimize the bounding hexagon
    
    # Let's try: remove the 6 outermost corners of ring 2 + 1 more
    # Or: keep a "thick" shape
    
    # Simple approach: try several predefined subsets
    
    # Subset 1: All 19 minus 7 specific positions
    # Let's index the positions
    pos_list = [(x, y) for (x, y, q, r) in all_positions]
    
    # Try removing 7 positions from the 19
    # For efficiency, try symmetric removals
    
    # Arrangement A: Keep center, all ring-1, and 5 ring-2 (skip 7 ring-2)
    # Ring-2 positions are those with max(|q|,|r|,|q+r|) == 2
    
    center_pos = (0.0, 0.0)
    ring1 = []
    ring2 = []
    for (x, y, q, r) in all_positions:
        dist = max(abs(q), abs(r), abs(q + r))
        if dist == 0:
            pass  # center
        elif dist == 1:
            ring1.append((x, y))
        elif dist == 2:
            ring2.append((x, y))
    
    # Try all combinations of 5 from ring2 (C(12,5) = 792)
    # For each, compute bounding hexagon
    
    from itertools import combinations
    
    ring2_indices = list(range(len(ring2)))
    best_subset_side = float('inf')
    best_subset = None
    best_subset_angle = 0.0
    
    count = 0
    for combo in combinations(range(len(ring2)), 5):
        if time.time() > deadline - 10:
            break
        count += 1
        selected = [center_pos] + ring1 + [ring2[i] for i in combo]
        
        # Compute bounding hexagon centered at origin
        side, angle = compute_bounding_hexagon(selected, (0.0, 0.0))
        if side < best_subset_side:
            best_subset_side = side
            best_subset = selected
            best_subset_angle = angle
    
    if best_subset is not None:
        # Also try shifting the center slightly
        # The bounding hexagon center might not be at origin
        # Try a few offsets
        for dx in np.linspace(-0.5, 0.5, 11):
            for dy in np.linspace(-0.5, 0.5, 11):
                if time.time() > deadline - 5:
                    break
                side, angle = compute_bounding_hexagon(best_subset, (dx, dy))
                if side < best_subset_side:
                    best_subset_side = side
                    best_subset_angle = angle
        
        # Build the inner hexagons
        inner = []
        for (x, y) in best_subset:
            inner.append([x, y, 0.0])
        
        # The outer hexagon is centered at the computed center
        # We need to find the optimal center
        # Re-compute with the best center
        best_center = (0.0, 0.0)
        best_final_side = best_subset_side
        best_final_angle = best_subset_angle
        
        # Try more center offsets
        for dx in np.linspace(-1.0, 1.0, 21):
            for dy in np.linspace(-1.0, 1.0, 21):
                if time.time() > deadline - 2:
                    break
                side, angle = compute_bounding_hexagon(best_subset, (dx, dy))
                if side < best_final_side:
                    best_final_side = side
                    best_final_angle = angle
                    best_center = (dx, dy)
        
        return inner, best_center, best_final_side, best_final_angle
    else:
        # Fallback: simple arrangement
        inner = []
        for q in range(-2, 3):
            for r in range(-2, 3):
                if abs(q + r) <= 2 and len(inner) < N:
                    x = 1.5 * q
                    y = sqrt3 * (r + q / 2.0)
                    inner.append([x, y, 0.0])
        return inner, (0.0, 0.0), 10.0, 0.0

# EVOLVE-BLOCK-END
