# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import math

N = 12

def _hex_verts(cx, cy, side, angle_deg):
    """Vertices of a regular hexagon centered at (cx, cy) with given side and rotation."""
    a = np.radians(angle_deg)
    return np.array([
        [cx + side * np.cos(a + np.radians(60 * i)),
         cy + side * np.sin(a + np.radians(60 * i))]
        for i in range(6)
    ])

def _polygon_overlap_depth(h1, h2):
    """Calculate the minimum penetration depth between two convex polygons.
    Returns 0 if disjoint.
    """
    min_depth = float('inf')
    axes = []
    for poly in [h1, h2]:
        n = len(poly)
        for i in range(n):
            p1 = poly[i]
            p2 = poly[(i + 1) % n]
            edge = p2 - p1
            norm = np.linalg.norm(edge)
            if norm < 1e-10:
                continue
            normal = np.array([-edge[1], edge[0]]) / norm
            axes.append(normal)
    
    for axis in axes:
        proj1 = np.dot(h1, axis)
        proj2 = np.dot(h2, axis)
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 <= min2 or max2 <= min1:
            return 0.0
        
        overlap = min(max1, max2) - max(min1, min2)
        if overlap < min_depth:
            min_depth = overlap
    
    return min_depth

def _containment_penalty(inner_verts, outer_verts):
    """Penalty for inner hexagon not being fully inside outer hexagon."""
    penalty = 0.0
    n_out = len(outer_verts)
    for iv in inner_verts:
        for i in range(n_out):
            v1 = outer_verts[i]
            v2 = outer_verts[(i + 1) % n_out]
            edge = v2 - v1
            edge_len = np.linalg.norm(edge)
            if edge_len < 1e-12:
                continue
            rel = iv - v1
            cross = edge[0] * rel[1] - edge[1] * rel[0]
            if cross < 0:
                penalty += (-cross) / edge_len
    return penalty

def _generate_positions_from_params(params):
    """
    Generate 12 hexagon positions (x, y, angle) from a reduced set of parameters.
    We assume a configuration with 6-fold rotational symmetry around the origin.
    
    Structure:
    - 1 central hexagon at (0,0) with rotation r0.
    - 5 unique hexagons in the first ring, rotated by 60 degrees to generate 6.
    - 1 unique hexagon in the second ring? No, 12 = 1 + 6 + 5? 
    Let's try: 1 center + 5 unique in ring 1 (generates 6) + 5 unique in ring 2 (generates 6)? Total 1+6+6=13. Too many.
    
    Alternative Symmetric Structure:
    12 hexagons can be arranged as:
    - 6 around a center? No, that's 7.
    - 2 rings of 6? That's 12.
    Let's model 2 rings of 6.
    Ring 1: 6 hexagons. Defined by 1 unique position (x1, y1, a1) and rotated 6 times.
    Ring 2: 6 hexagons. Defined by 1 unique position (x2, y2, a2) and rotated 6 times.
    Total unique params: 3 for Ring 1, 3 for Ring 2. Total 6 params.
    This is very restrictive but might find a good symmetric solution.
    
    Let's allow a bit more flexibility:
    Ring 1: 6 hexagons. Param (r1, theta1, rot1).
    Ring 2: 6 hexagons. Param (r2, theta2, rot2).
    
    Wait, 12 hexagons.
    Maybe:
    1 central? No, if we have 1 central, we have 11 left. 11 is not divisible by 6.
    
    Let's stick to 2 rings of 6.
    Params:
    r1, theta1, rot1  (Radius, Angle, Rotation of ring 1)
    r2, theta2, rot2  (Radius, Angle, Rotation of ring 2)
    
    Total 6 parameters.
    """
    positions = []
    # Ring 1
    r1, theta1, rot1 = params[0], params[1], params[2]
    x1 = r1 * np.cos(theta1)
    y1 = r1 * np.sin(theta1)
    for i in range(6):
        ang_offset = np.radians(60 * i)
        x = x1 * np.cos(ang_offset) - y1 * np.sin(ang_offset)
        y = x1 * np.sin(ang_offset) + y1 * np.cos(ang_offset)
        a = rot1 + 60 * i
        positions.append((x, y, a))
        
    # Ring 2
    r2, theta2, rot2 = params[3], params[4], params[5]
    x2 = r2 * np.cos(theta2)
    y2 = r2 * np.sin(theta2)
    for i in range(6):
        ang_offset = np.radians(60 * i)
        x = x2 * np.cos(ang_offset) - y2 * np.sin(ang_offset)
        y = x2 * np.sin(ang_offset) + y2 * np.cos(ang_offset)
        a = rot2 + 60 * i
        positions.append((x, y, a))
        
    return positions

def _objective(params, outer_side):
    """
    Objective function: minimize overlap and containment violations.
    """
    positions = _generate_positions_from_params(params)
    
    inner_verts_list = []
    for x, y, a in positions:
        inner_verts_list.append(_hex_verts(x, y, 1.0, a))
    
    outer_verts = _hex_verts(0.0, 0.0, outer_side, 0.0)
    
    # Containment penalty
    cont_pen = 0.0
    for iv in inner_verts_list:
        cont_pen += _containment_penalty(iv, outer_verts)
    
    # Disjointness penalty
    # We only need to check unique pairs due to symmetry?
    # Actually, it's safer to check all pairs if N=12 is small, but we can optimize.
    # Pairs:
    # Within Ring 1: (0,1), (0,2) are enough due to symmetry? 
    # Between Ring 1 and Ring 2: (0,6), (0,7)
    # Within Ring 2: (6,7), (6,8)
    
    disj_pen = 0.0
    # Check specific representative pairs to save time, but for robustness in optimization, 
    # checking a few key pairs is usually sufficient for symmetric constraints.
    # However, to be safe, let's check all pairs in the first few iterations or just all pairs since N=12 is small.
    # 12*11/2 = 66 pairs. This is fast.
    for i in range(N):
        for j in range(i + 1, N):
            depth = _polygon_overlap_depth(inner_verts_list[i], inner_verts_list[j])
            if depth > 0:
                disj_pen += depth**2 # Quadratic penalty for smoother gradient
    
    return cont_pen + disj_pen

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    best_score = float('inf')
    best_params = None
    best_side = None
    
    # Initial guess for parameters
    # Ring 1: radius ~2.0, angle 0, rot 0
    # Ring 2: radius ~4.0, angle 30, rot 0
    init_params = np.array([2.0, 0.0, 0.0, 4.0, 30.0, 0.0])
    
    # Binary search on outer side length
    # Lower bound: Theoretical minimum is slightly larger than the span of the packing.
    # Span of 2 rings of 6 unit hexagons:
    # Ring 1 centers at dist 2. Hex width 2. Extent ~ 2 + 1 = 3?
    # Ring 2 centers at dist 4. Hex width 2. Extent ~ 4 + 1 = 5?
    # So side length must be at least ~5.0 / cos(30) ? No, hexagon side S has width 2S.
    # If extent is R, side S must be >= R / sqrt(3) * 2? 
    # Let's start with a generous range.
    low_side = 3.0
    high_side = 6.0
    
    # First, find a valid configuration with a large side to get a good starting point
    # Optimize for a large side to ensure we find a valid packing
    current_side = 5.5
    
    try:
        # Optimize parameters for a fixed large side
        result = minimize(
            lambda p: _objective(p, current_side),
            init_params,
            method='L-BFGS-B',
            bounds=[
                (1.0, 5.0), # r1
                (0.0, 60.0), # theta1
                (0.0, 60.0), # rot1
                (2.0, 6.0), # r2
                (0.0, 60.0), # theta2
                (0.0, 60.0)  # rot2
            ],
            options={'maxiter': 200, 'maxfun': 10000}
        )
        
        if result.fun < 1e-6:
            best_params = result.x
            best_side = current_side
            best_score = current_side
    except Exception:
        pass
    
    # If we found a valid packing, try to shrink the side length
    if best_params is not None:
        # Binary search for minimum side
        low = 3.0
        high = best_side
        
        for _ in range(15):
            if time.time() > deadline - 10.0:
                break
                
            mid = (low + high) / 2.0
            if mid <= low + 1e-4:
                break
            
            # Try to optimize for this smaller side
            try:
                result = minimize(
                    lambda p: _objective(p, mid),
                    best_params,
                    method='L-BFGS-B',
                    bounds=[
                        (1.0, 5.0),
                        (0.0, 60.0),
                        (0.0, 60.0),
                        (2.0, 6.0),
                        (0.0, 60.0),
                        (0.0, 60.0)
                    ],
                    options={'maxiter': 100, 'maxfun': 5000}
                )
                
                if result.fun < 1e-6:
                    # Valid packing found for smaller side
                    high = mid
                    best_params = result.x
                    best_side = mid
                    best_score = mid
                else:
                    low = mid
            except Exception:
                low = mid
        
        # Final refinement
        if time.time() < deadline - 5.0:
            try:
                result = minimize(
                    lambda p: _objective(p, best_side),
                    best_params,
                    method='L-BFGS-B',
                    bounds=[
                        (1.0, 5.0),
                        (0.0, 60.0),
                        (0.0, 60.0),
                        (2.0, 6.0),
                        (0.0, 60.0),
                        (0.0, 60.0)
                    ],
                    options={'maxiter': 200, 'maxfun': 10000, 'ftol': 1e-15, 'gtol': 1e-12}
                )
                if result.fun < 1e-6:
                    best_params = result.x
                    # Try to shrink a tiny bit more
                    tiny_shrink = best_side - 0.01
                    result2 = minimize(
                        lambda p: _objective(p, tiny_shrink),
                        best_params,
                        method='L-BFGS-B',
                        bounds=[
                            (1.0, 5.0),
                            (0.0, 60.0),
                            (0.0, 60.0),
                            (2.0, 6.0),
                            (0.0, 60.0),
                            (0.0, 60.0)
                        ],
                        options={'maxiter': 50, 'maxfun': 2000}
                    )
                    if result2.fun < 1e-6:
                        best_params = result2.x
                        best_side = tiny_shrink
                        best_score = tiny_shrink
            except Exception:
                pass
    
    # If the symmetric approach failed or didn't find a good solution, fallback to a known valid loose packing
    if best_params is None or best_score > 5.0:
        # Fallback: 3x4 grid
        inner = []
        # Generate 12 positions in a 3x4 grid, centered
        # Spacing 2.0 x, 1.732 y
        positions = []
        for r in range(3):
            for c in range(4):
                x = (c - 1.5) * 2.0
                y = (r - 1.0) * 1.732
                positions.append((x, y, 0.0))
        
        # Calculate bounding box to determine outer side
        # Max extent in x: 3.0 + 1.0 (half width) = 4.0?
        # Hex width is 2. So extent from center is 1.5 (grid) + 1 (hex radius) = 2.5?
        # Let's just use a safe side length 5.0
        best_side = 5.0
        best_score = 5.0
        # Return the grid positions directly, ignoring the symmetric param logic for the return value
        # We need to return the list of 12 tuples.
        return positions, (0.0, 0.0), best_side, 0.0
    
    # Construct final result from best_params
    positions = _generate_positions_from_params(best_params)
    inner = [(float(x), float(y), float(a)) for x, y, a in positions]
    
    return inner, (0.0, 0.0), float(best_side), 0.0

# EVOLVE-BLOCK-END
