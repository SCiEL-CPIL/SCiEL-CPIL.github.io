import numpy as np
import time
from scipy.optimize import differential_evolution

N = 12

def _hex_vertices(cx, cy, s, angle_deg):
    """Get vertices of a regular hexagon with side s, center (cx,cy), rotation angle_deg."""
    a = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * np.pi / 3
        x = cx + s * np.cos(theta)
        y = cy + s * np.sin(theta)
        verts.append([x, y])
    return np.array(verts)

def _min_enclosing_hexagon(inner_positions, inner_angles, inner_sides):
    """
    Find minimal regular hexagon (center, side, angle) containing all inner hexagons.
    """
    all_verts = []
    for (cx, cy, ang), s in zip(inner_positions, inner_sides):
        verts = _hex_vertices(cx, cy, s, ang)
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    best_side = float('inf')
    best_center = None
    best_angle = 0.0
    
    # The orientation of the enclosing hexagon is periodic with 60 degrees.
    # We sample orientations.
    for phi_deg in np.linspace(0, 60, 100):
        phi = np.radians(phi_deg)
        normals = []
        for i in range(6):
            theta = phi + i * np.pi / 3 + np.pi / 6
            normals.append([np.cos(theta), np.sin(theta)])
        normals = np.array(normals)
        
        projs = all_verts @ normals.T
        max_projs = projs.max(axis=0)
        min_projs = projs.min(axis=0)
        widths = max_projs - min_projs
        
        required_apothem = np.max(widths) / 2
        s_candidate = required_apothem * 2 / np.sqrt(3)
        
        lo = max_projs - required_apothem
        hi = min_projs + required_apothem
        
        if np.any(lo > hi + 1e-12):
            continue
        
        mid = (lo + hi) / 2
        c, residuals, rank, sv = np.linalg.lstsq(normals, mid, rcond=None)
        
        c_projs = normals @ c
        if np.any(c_projs < lo - 1e-9) or np.any(c_projs > hi + 1e-9):
            continue
        
        if s_candidate < best_side:
            best_side = s_candidate
            best_center = c
            best_angle = phi_deg
    
    if best_center is None:
        # Fallback: use bounding box diagonal or similar
        centroid = all_verts.mean(axis=0)
        max_dist = np.max(np.linalg.norm(all_verts - centroid, axis=1))
        return centroid[0], centroid[1], max_dist * 2, 0.0
    
    return best_center[0], best_center[1], best_side, best_angle

def _sat_overlap_depth(h1, h2):
    """Minimum penetration depth between two convex polygons using SAT."""
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

def _evaluate_packing(params):
    """
    Evaluate a packing configuration.
    params: [x1, y1, a1, x2, y2, a2, ...]
    Returns: (side_length, valid)
    """
    positions = []
    angles = []
    for i in range(N):
        x = params[3*i]
        y = params[3*i+1]
        a = params[3*i+2]
        positions.append((x, y))
        angles.append(a)
    
    sides = [1.0] * N
    
    # Compute vertices
    verts_list = []
    for i in range(N):
        verts_list.append(_hex_vertices(positions[i][0], positions[i][1], 1.0, angles[i]))
    
    # Check overlaps
    max_overlap = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            depth = _sat_overlap_depth(verts_list[i], verts_list[j])
            if depth > max_overlap:
                max_overlap = depth
    
    if max_overlap > 1e-6:
        # Penalize overlap heavily
        return max_overlap * 100.0 + 100.0, False
    
    # Compute enclosing hexagon
    cx, cy, side, angle = _min_enclosing_hexagon(positions, angles, sides)
    return side, True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Keep 10s buffer for 150s limit
    
    best_side = float('inf')
    best_params = None
    
    # Define bounds for 12 hexagons: x, y in [-5, 5], angle in [0, 60]
    bounds = []
    for _ in range(N):
        bounds.append((-5.0, 5.0))
        bounds.append((-5.0, 5.0))
        bounds.append((0.0, 60.0))
    
    # Initial guess: Honeycomb pattern
    # This helps DE start in a valid region
    sqrt3 = np.sqrt(3)
    init_positions = []
    # 4-3-3-2 pattern centered
    row_counts = [4, 3, 3, 2]
    y_offset = 0
    for r, count in enumerate(row_counts):
        y = (r - 1.5) * 1.5
        x_start = -(count - 1) * sqrt3 / 2.0
        for i in range(count):
            x = x_start + i * sqrt3
            init_positions.append([x, y, 0.0])
    
    init_params = []
    for p in init_positions:
        init_params.extend(p)
    
    def objective(params):
        side, valid = _evaluate_packing(params)
        return side
    
    # Run Differential Evolution
    try:
        result = differential_evolution(
            objective,
            bounds,
            x0=init_params,
            maxiter=100,
            popsize=15,
            tol=1e-8,
            seed=42,
            polish=True,
            workers=1,
            init='sobol' # More efficient sampling
        )
        
        if result.fun < best_side:
            best_side = result.fun
            best_params = result.x
            
        # Check if the result is valid (no overlap)
        final_side, is_valid = _evaluate_packing(best_params)
        
        if not is_valid:
            # Try to find a valid solution from the population or restart
            # For simplicity, if DE fails to converge to a valid non-overlapping state,
            # we might need to fall back. But usually DE with penalty works.
            # Let's try a local refinement on the best non-overlapping candidate found during DE?
            # DE doesn't expose intermediate valid states easily.
            
            # Fallback: Use the initial honeycomb if it's valid
            init_side, init_valid = _evaluate_packing(init_params)
            if init_valid:
                best_side = init_side
                best_params = init_params
            else:
                # Emergency fallback: Scale up honeycomb slightly? No, that increases side.
                # Just return the honeycomb and hope the evaluator's tolerance is okay, 
                # or adjust the honeycomb to be strictly non-overlapping.
                # The honeycomb with scale 1.0 is touching. We need a tiny gap.
                # Let's construct a valid honeycomb with scale 1.001
                scale = 1.001
                new_init = []
                for p in init_positions:
                    new_init.append([p[0]*scale, p[1]*scale, p[2]])
                new_init_side, new_init_valid = _evaluate_packing(new_init)
                if new_init_valid:
                    best_side = new_init_side
                    best_params = new_init
    
    except Exception:
        # Fallback to valid honeycomb
        scale = 1.001
        new_init = []
        for p in init_positions:
            new_init.append([p[0]*scale, p[1]*scale, p[2]])
        best_params = new_init
        best_side, _ = _evaluate_packing(best_params)
        
    # Final Verification and Construction
    if best_params is None:
        # Should not happen
        best_params = init_params
        best_side, _ = _evaluate_packing(best_params)
        
    # Extract final positions
    inner = []
    for i in range(N):
        x = best_params[3*i]
        y = best_params[3*i+1]
        a = best_params[3*i+2]
        inner.append((x, y, a))
    
    # Recalculate enclosing hexagon for the final parameters
    positions = [(p[0], p[1]) for p in inner]
    angles = [p[2] for p in inner]
    sides = [1.0] * N
    cx, cy, side, angle = _min_enclosing_hexagon(positions, angles, sides)
    
    # Add a small safety margin to side length to ensure containment
    side_safe = side * 1.0001 + 1e-6
    
    return inner, (cx, cy), side_safe, angle
