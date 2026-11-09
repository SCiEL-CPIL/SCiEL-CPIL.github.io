import numpy as np
import time
import scipy.optimize as opt

N = 12

def _hex_verts(cx, cy, s, angle_deg):
    a = np.radians(angle_deg)
    return np.array([
        [cx + s * np.cos(a + i * np.pi / 3), cy + s * np.sin(a + i * np.pi / 3)] 
        for i in range(6)
    ])

def _check_validity(inner, outer_side, outer_angle=0.0):
    """
    Check if the configuration is valid.
    inner: list of (x, y, angle_deg)
    outer_side: float
    """
    # Outer hexagon parameters
    A = outer_side * np.sqrt(3) / 2
    outer_normals = np.array([
        [np.cos(np.radians(outer_angle) + i * np.pi / 3 + np.pi / 6), 
         np.sin(np.radians(outer_angle) + i * np.pi / 3 + np.pi / 6)] 
        for i in range(6)
    ])
    
    # Check containment
    for x, y, ang in inner:
        verts = _hex_verts(x, y, 1.0, ang)
        # Project vertices onto outer normals
        projs = verts @ outer_normals.T
        # Max projection for each normal
        max_projs = projs.max(axis=0)
        if np.any(max_projs > A + 1e-9):
            return False
            
    # Check disjointness
    # Pre-compute vertices for all inner hexagons
    all_verts = [_hex_verts(x, y, 1.0, ang) for x, y, ang in inner]
    
    for i in range(N):
        for j in range(i + 1, N):
            v1 = all_verts[i]
            v2 = all_verts[j]
            # Get normals for v1 and v2
            a1 = np.radians(inner[i][2])
            a2 = np.radians(inner[j][2])
            n1 = np.array([[np.cos(a1 + k * np.pi / 3 + np.pi / 6), np.sin(a1 + k * np.pi / 3 + np.pi / 6)] for k in range(6)])
            n2 = np.array([[np.cos(a2 + k * np.pi / 3 + np.pi / 6), np.sin(a2 + k * np.pi / 3 + np.pi / 6)] for k in range(6)])
            normals = np.vstack([n1, n2])
            
            sep_min = float('inf')
            for n in normals:
                p1 = v1 @ n
                p2 = v2 @ n
                min1, max1 = p1.min(), p1.max()
                min2, max2 = p2.min(), p2.max()
                if max1 < min2 or max2 < min1:
                    sep = min(max1, max2) - max(min1, min2) # This logic is wrong for separation
                    # If disjoint on this axis, separation is positive.
                    # We want the MINIMUM separation over all axes.
                    # If they overlap on any axis, sep will be negative.
                    # If they are disjoint, sep is positive.
                    # We need to check if there is ANY axis where they are disjoint?
                    # No, SAT says they are disjoint if there exists an axis where projections don't overlap.
                    # They overlap if for ALL axes, projections overlap.
                    # So overlap depth is the minimum "gap" (which is negative if overlapping) over all axes?
                    # No. The overlap depth is the minimum translation distance to separate them.
                    # For convex polygons, the overlap depth is indeed the minimum over all normals of the "penetration".
                    # Let's define penetration on axis n as:
                    # If intervals overlap: pen = min(max1, max2) - max(min1, min2) (this is positive if overlapping? No, max1-max2 is negative if disjoint).
                    # Let's use:
                    # gap = min(max1, max2) - max(min1, min2)
                    # If gap < 0, they overlap on this axis by -gap.
                    # If gap >= 0, they are disjoint on this axis.
                    # The actual overlap depth of the two polygons is the MINIMUM gap over all axes?
                    # Wait. If they are disjoint, there is at least one axis with gap > 0.
                    # If they overlap, all axes have gap < 0.
                    # The "depth" of overlap is the smallest amount you need to push them apart.
                    # This is equal to the minimum gap (which is negative) over all axes?
                    # Yes, for convex bodies, the translation distance to separate is the minimum over all directions of the penetration depth in that direction.
                    # So, if we find the minimum gap (most negative), that is the depth.
                    # If min_gap > 0, they are disjoint.
                    # If min_gap < 0, they overlap by -min_gap.
                    
                    gap = min(max1, max2) - max(min1, min2)
                    if gap < sep_min:
                        sep_min = gap
            
            if sep_min > 1e-9:
                # Disjoint
                pass
            else:
                # Overlapping
                return False
    return True

def _total_penalty(inner_params, outer_side, outer_angle=0.0):
    """
    inner_params: flat array [x1, y1, a1, x2, y2, a2, ...]
    """
    x = inner_params[0::3]
    y = inner_params[1::3]
    a = inner_params[2::3]
    
    A = outer_side * np.sqrt(3) / 2
    outer_normals = np.array([
        [np.cos(np.radians(outer_angle) + i * np.pi / 3 + np.pi / 6), 
         np.sin(np.radians(outer_angle) + i * np.pi / 3 + np.pi / 6)] 
        for i in range(6)
    ])
    
    penalty = 0.0
    all_verts = []
    
    # Containment
    for i in range(N):
        verts = _hex_verts(x[i], y[i], 1.0, a[i])
        all_verts.append(verts)
        projs = verts @ outer_normals.T
        max_projs = projs.max(axis=0)
        violations = np.maximum(0, max_projs - A)
        penalty += np.sum(violations ** 2)
        
    # Overlap
    for i in range(N):
        for j in range(i + 1, N):
            v1 = all_verts[i]
            v2 = all_verts[j]
            a1 = a[i]
            a2 = a[j]
            n1 = np.array([[np.cos(np.radians(a1) + k * np.pi / 3 + np.pi / 6), np.sin(np.radians(a1) + k * np.pi / 3 + np.pi / 6)] for k in range(6)])
            n2 = np.array([[np.cos(np.radians(a2) + k * np.pi / 3 + np.pi / 6), np.sin(np.radians(a2) + k * np.pi / 3 + np.pi / 6)] for k in range(6)])
            normals = np.vstack([n1, n2])
            
            sep_min = float('inf')
            for n in normals:
                p1 = v1 @ n
                p2 = v2 @ n
                min1, max1 = p1.min(), p1.max()
                min2, max2 = p2.min(), p2.max()
                gap = min(max1, max2) - max(min1, min2)
                if gap < sep_min:
                    sep_min = gap
            
            if sep_min < 0:
                penalty += sep_min ** 2
                
    return penalty

def _make_honeycomb_init():
    sqrt3 = np.sqrt(3)
    # 3-4-3-2 pattern
    positions = []
    pattern = [3, 4, 3, 2]
    for j, row_len in enumerate(pattern):
        for i in range(row_len):
            x = (i + 0.5 * (j % 2)) * sqrt3
            y = j * 1.5
            positions.append((x, y, 0.0))
    # Center it
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    return [(x - cx, y - cy, 0.0) for x, y, _ in positions]

def _make_ring_init():
    positions = []
    # 1 center, 5 ring, 6 outer? No, 12 total.
    # 1 center, 6 inner ring, 5 outer ring?
    # Let's do 1 center, 6 at radius 2*sqrt(3)/2? No, touching.
    # Distance between centers of touching hexagons is 2 * apothem = sqrt(3).
    # So ring 1 radius sqrt(3).
    # Ring 2 radius 2*sqrt(3)?
    # Let's just use a grid for simplicity and variety.
    # 4x3 grid
    for i in range(3):
        for j in range(4):
            x = i * 2 * np.sqrt(3)
            y = j * 3
            positions.append((x, y, 0.0))
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    return [(x - cx, y - cy, 0.0) for x, y, _ in positions]

def _make_random_init():
    return [(np.random.uniform(-3, 3), np.random.uniform(-3, 3), np.random.uniform(0, 60)) for _ in range(N)]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_score = float('inf')
    best_result = None
    
    # Binary search on S
    lo = 2.0
    hi = 10.0
    
    inits = [
        _make_honeycomb_init,
        _make_ring_init,
        _make_random_init
    ]
    
    # We will try to find the minimal S
    # For each S, we try to minimize penalty to ~0
    
    # To save time, we can do a few iterations of binary search
    for _ in range(20):
        if time.time() > deadline:
            break
            
        mid = (lo + hi) / 2
        
        success = False
        best_params_for_mid = None
        best_penalty_for_mid = float('inf')
        
        # Try a few initializations
        for init_func in inits:
            if time.time() > deadline:
                break
                
            init_positions = init_func()
            x0 = []
            for x, y, a in init_positions:
                x0.extend([x, y, a])
            x0 = np.array(x0)
            
            # Optimize
            res = opt.minimize(
                _total_penalty, 
                x0, 
                args=(mid, 0.0), 
                method='L-BFGS-B', 
                options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if res.fun < best_penalty_for_mid:
                best_penalty_for_mid = res.fun
                best_params_for_mid = res.x
                
            if res.fun < 1e-6:
                success = True
                break
        
        if success and best_params_for_mid is not None:
            hi = mid
            # Store result
            inner = []
            for i in range(N):
                inner.append((best_params_for_mid[i*3], best_params_for_mid[i*3+1], best_params_for_mid[i*3+2]))
            # Verify
            if _check_validity(inner, hi):
                if hi < best_score:
                    best_score = hi
                    best_result = (inner, (0.0, 0.0), hi, 0.0)
        else:
            lo = mid
            
    if best_result is None:
        # Fallback to a large hexagon
        inner = _make_honeycomb_init()
        return inner, (0.0, 0.0), 10.0, 0.0
        
    return best_result
