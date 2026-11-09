import numpy as np
from scipy.optimize import minimize
import time

N = 12

def _hex_verts(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon."""
    a = np.radians(angle_deg)
    return np.array([
        [cx + side * np.cos(a + np.radians(60 * i)),
         cy + side * np.sin(a + np.radians(60 * i))]
        for i in range(6)
    ])

def _polygon_overlap_depth(A, B):
    """SAT overlap depth between convex polygons A and B. 0 if disjoint."""
    min_depth = float('inf')
    axes = []
    for poly in (A, B):
        n = len(poly)
        for i in range(n):
            edge = poly[(i + 1) % n] - poly[i]
            length = np.linalg.norm(edge)
            if length > 1e-12:
                axes.append(np.array([-edge[1], edge[0]]) / length)
    
    for ax in axes:
        pa = np.dot(A, ax)
        pb = np.dot(B, ax)
        if pa.max() <= pb.min() or pb.max() <= pa.min():
            return 0.0
        ov = min(pa.max(), pb.max()) - max(pa.min(), pb.min())
        if ov < min_depth:
            min_depth = ov
    return min_depth

def _containment_violation(inner_verts, outer_verts):
    """Sum of distances by which inner verts fall outside outer."""
    n_out = len(outer_verts)
    pen = 0.0
    for iv in inner_verts:
        for i in range(n_out):
            v1 = outer_verts[i]
            v2 = outer_verts[(i + 1) % n_out]
            edge = v2 - v1
            el = np.linalg.norm(edge)
            if el < 1e-12:
                continue
            nrm = np.array([-edge[1], edge[0]]) / el
            # Normal points inward? 
            # For a CCW polygon, edge x->y, normal (-dy, dx) points to the left (inside).
            # So if dot(iv - v1, nrm) < 0, it is outside.
            sd = np.dot(iv - v1, nrm)
            if sd < 0:
                pen += -sd
    return pen

def _generate_symmetric_hexagons(params, S):
    """
    Generate 12 hexagons using 6-fold symmetry.
    params: [r1, a1, rot1, r2, a2, rot2]
    S: outer side length
    Returns: list of 12 hexagon vertex arrays
    """
    r1, a1, rot1 = params[0], params[1], params[2]
    r2, a2, rot2 = params[3], params[4], params[5]
    
    hexagons = []
    for k in range(6):
        base_angle = np.radians(60 * k)
        
        # Hexagon 1 in this sector
        x1 = r1 * np.cos(base_angle + np.radians(a1))
        y1 = r1 * np.sin(base_angle + np.radians(a1))
        hex1 = _hex_verts(x1, y1, 1.0, rot1 + 60 * k)
        hexagons.append(hex1)
        
        # Hexagon 2 in this sector
        x2 = r2 * np.cos(base_angle + np.radians(a2))
        y2 = r2 * np.sin(base_angle + np.radians(a2))
        hex2 = _hex_verts(x2, y2, 1.0, rot2 + 60 * k)
        hexagons.append(hex2)
        
    return hexagons

def _objective_symmetric(params, S):
    """
    Objective: S + lambda * (Overlap Penalty + Containment Penalty)
    params: [r1, a1, rot1, r2, a2, rot2]
    """
    # We optimize S as part of the vector? No, S is separate in the outer loop or included in params.
    # Let's include S in params for the optimizer to handle bounds easily.
    # params: [S, r1, a1, rot1, r2, a2, rot2]
    S_val = params[0]
    inner_params = params[1:]
    
    hexagons = _generate_symmetric_hexagons(inner_params, S_val)
    outer = _hex_verts(0, 0, S_val, 0.0)
    
    total_pen = 0.0
    
    # Containment
    for h in hexagons:
        pen = _containment_violation(h, outer)
        if pen > 0:
            total_pen += pen * pen
            
    # Overlaps
    for i in range(12):
        for j in range(i + 1, 12):
            d = _polygon_overlap_depth(hexagons[i], hexagons[j])
            if d > 0:
                total_pen += d * d
                
    # Large penalty if invalid to guide optimizer
    return S_val + 10.0 * total_pen

def construct():
    t0 = time.time()
    DEADLINE = t0 + 115.0
    
    # Initial guesses for [S, r1, a1, rot1, r2, a2, rot2]
    # r1, r2 are radii from center. a1, a2 are angular offsets in the sector.
    # rot1, rot2 are rotations of the hexagons themselves.
    
    # Guess 1: Standard flower packing
    # Inner ring of 6 around center? No, we have 2 per sector.
    # One close to center, one further out.
    # r1 ~ 2.0, a1 ~ 0 (aligned with axis), rot1 ~ 0
    # r2 ~ 3.5, a2 ~ 30 (between axes), rot2 ~ 30
    # S ~ 4.5
    
    inits = [
        [4.5, 2.0, 0.0, 0.0, 3.5, 30.0, 30.0],
        [4.5, 1.8, 15.0, 15.0, 3.2, 15.0, 15.0],
        [4.5, 2.2, 0.0, 30.0, 3.8, 0.0, 0.0],
        [4.0, 1.5, 0.0, 0.0, 3.0, 0.0, 0.0],
        [5.0, 2.5, 0.0, 0.0, 4.0, 30.0, 30.0]
    ]
    
    best_S = float('inf')
    best_params = None
    
    for init in inits:
        if time.time() > DEADLINE - 5:
            break
            
        try:
            res = minimize(
                _objective_symmetric, 
                init, 
                method='L-BFGS-B',
                bounds=[(3.0, 10.0), (0.5, 5.0), (-30, 30), (-180, 180), (0.5, 5.0), (-30, 30), (-180, 180)],
                options={'maxiter': 500, 'maxfun': 50000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res.success or res.fun < best_S:
                # Verify validity
                S_val = res.x[0]
                inner_params = res.x[1:]
                hexagons = _generate_symmetric_hexagons(inner_params, S_val)
                outer = _hex_verts(0, 0, S_val, 0.0)
                
                valid = True
                for h in hexagons:
                    if _containment_violation(h, outer) > 1e-5:
                        valid = False
                        break
                if valid:
                    for i in range(12):
                        for j in range(i + 1, 12):
                            if _polygon_overlap_depth(hexagons[i], hexagons[j]) > 1e-5:
                                valid = False
                                break
                        if not valid:
                            break
                
                if valid:
                    if S_val < best_S:
                        best_S = S_val
                        best_params = res.x
                        
        except Exception:
            continue

    # Fallback if symmetry assumption failed or no valid solution found
    if best_params is None:
        # Use a simple grid fallback
        # 3x4 grid of hexagons
        # Width: 4 * 2 + 1 = 9? No.
        # Hex width is 2. Height is sqrt(3).
        # 3 rows, 4 cols.
        # x: 0, 2, 4, 6 (offset by 1 for odd rows)
        # y: 0, sqrt(3), 2*sqrt(3)
        # This is roughly 7x3.5. Bounding hex side ~ 4.5.
        # Just return a known valid config with S=5.0
        grid = []
        for r in range(3):
            for c in range(4):
                x = c * 2.0 + (1.0 if r % 2 else 0.0)
                y = r * np.sqrt(3.0)
                grid.append([x, y, 0.0])
        # Center it
        xs = [p[0] for p in grid]
        ys = [p[1] for p in grid]
        cx = (max(xs) + min(xs)) / 2
        cy = (max(ys) + min(ys)) / 2
        for p in grid:
            p[0] -= cx
            p[1] -= cy
        return grid, (0.0, 0.0), 5.0, 0.0

    # Extract result
    S_val = best_params[0]
    inner_params = best_params[1:]
    hexagons = _generate_symmetric_hexagons(inner_params, S_val)
    
    inner_list = []
    for h in hexagons:
        # Recover center and angle from vertices
        cx = np.mean(h[:, 0])
        cy = np.mean(h[:, 1])
        # Angle is the angle of the first vertex relative to center
        # Or we can just store the params we used to generate it.
        # The interface requires (x, y, angle_degrees).
        # We generated them with rot1 + 60*k and rot2 + 60*k.
        # We need to map back to the specific hexagon in the list.
        # The list order is:
        # k=0: h1 (rot1), h2 (rot2)
        # k=1: h1 (rot1+60), h2 (rot2+60)
        # ...
        
    # Re-construct list in the order generated by _generate_symmetric_hexagons
    # which is interleaved: h1_k, h2_k, h1_{k+1}, h2_{k+1}...
    # Wait, my _generate_symmetric_hexagons appends h1 then h2 for each k.
    # So index 0: k=0, type 1
    # index 1: k=0, type 2
    # index 2: k=1, type 1
    # ...
    
    # We need x, y, angle for each.
    # We can calculate x, y from center of vertices.
    # Angle can be calculated from the orientation of the first edge or vertex.
    # Since we know the generation params, we can just use them.
    
    result_inner = []
    for k in range(6):
        # Hex 1
        r1, a1, rot1 = inner_params[0], inner_params[1], inner_params[2]
        x1 = r1 * np.cos(np.radians(60 * k + a1))
        y1 = r1 * np.sin(np.radians(60 * k + a1))
        ang1 = rot1 + 60 * k
        
        # Hex 2
        r2, a2, rot2 = inner_params[3], inner_params[4], inner_params[5]
        x2 = r2 * np.cos(np.radians(60 * k + a2))
        y2 = r2 * np.sin(np.radians(60 * k + a2))
        ang2 = rot2 + 60 * k
        
        result_inner.append([float(x1), float(y1), float(ang1)])
        result_inner.append([float(x2), float(y2), float(ang2)])
        
    return result_inner, (0.0, 0.0), float(S_val), 0.0

# EVOLVE-BLOCK-END
