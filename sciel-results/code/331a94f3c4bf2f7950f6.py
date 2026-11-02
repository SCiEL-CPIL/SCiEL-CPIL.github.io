import numpy as np
import time
import random
from scipy.optimize import minimize

N = 12
DEADLINE = 150.0
TIME_MARGIN = 5.0  # seconds to spare before the 150s kill

def _hex_vertices(cx, cy, rot, side=1.0):
    """Return 6 vertices of a regular hexagon (counter-clockwise)."""
    verts = []
    for k in range(6):
        a = rot + k * np.pi / 3.0
        verts.append((cx + side * np.cos(a), cy + side * np.sin(a)))
    return np.array(verts)

def _polygon_area(poly):
    """Shoelace area."""
    n = len(poly)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0

def _clip_polygon(poly, ax, ay, bx, by, inside_fn):
    """Sutherland-Hodgman clip of poly against line ax*x+ay*y <= b*bx+b*by."""
    if len(poly) == 0:
        return []
    out = []
    for i in range(len(poly)):
        cur = poly[i]
        nxt = poly[(i + 1) % len(poly)]
        cur_in = inside_fn(cur[0], cur[1])
        nxt_in = inside_fn(nxt[0], nxt[1])
        if cur_in:
            out.append(cur)
        if cur_in != nxt_in:
            # intersect
            f1 = ax * cur[0] + ay * cur[1] - (bx * bx + by * by)
            f2 = ax * nxt[0] + ay * nxt[1] - (bx * bx + by * by)
            t = f1 / (f1 - f2) if (f1 - f2) != 0 else 0.0
            ix = cur[0] + t * (nxt[0] - cur[0])
            iy = cur[1] + t * (nxt[1] - cur[1])
            out.append((ix, iy))
    return out

def _convex_intersection_depth(poly_a, poly_b):
    """
    Return the maximum depth of intersection (penetration) between two convex polygons.
    If they don't intersect (or touch), return 0.
    If they overlap, return a positive value representing how much they are interpenetrating.
    This is computed as the maximum over all separating axes of the signed distance
    such that when projected, the intervals overlap.
    Actually, simpler: compute area of intersection. If > 0, they overlap.
    But we want a smooth penalty. Let's use the maximum overlap depth.
    
    Actually, for a smooth penalty, let's compute the area of intersection.
    """
    # Clip poly_a against each edge of poly_b
    poly = np.array(poly_a, dtype=float)
    poly_b = np.array(poly_b, dtype=float)
    
    n = len(poly_b)
    for i in range(n):
        v1 = poly_b[i]
        v2 = poly_b[(i + 1) % n]
        # Normal pointing inward (toward center of poly_b roughly)
        dx = v2[0] - v1[0]
        dy = v2[1] - v1[1]
        # Left normal
        nx, ny = -dy, dx
        # Normalize
        norm = np.sqrt(nx*nx + ny*ny)
        if norm == 0:
            continue
        nx, ny = nx/norm, ny/norm
        # Check if center of poly_b is on the left side
        # For a convex polygon, the inward normal should point toward the interior
        # We can determine this by checking a point inside, e.g., centroid
        cx, cy = poly_b.mean(axis=0)
        val = nx * (cx - v1[0]) + ny * (cy - v1[1])
        if val < 0:
            nx, ny = -nx, -ny
        
        def inside_fn(x, y):
            return nx * (x - v1[0]) + ny * (y - v1[1]) <= 1e-10
        
        poly = _clip_polygon(poly, nx, ny, v1[0], v1[1], inside_fn)
        if len(poly) == 0:
            return 0.0
    
    if len(poly) < 3:
        return 0.0
    
    area = _polygon_area(poly)
    return area

def _containment_penalty(inner_verts, outer_verts):
    """
    Compute penalty for inner polygon not being inside outer polygon.
    Returns 0 if fully inside, else area of part outside.
    """
    # Clip inner against outer (outer is the container)
    # We want the area of inner that is OUTSIDE outer
    # This is area(inner) - area(inner ∩ outer)
    inner_area = _polygon_area(inner_verts)
    intersection_area = _convex_intersection_depth(inner_verts, outer_verts)
    # Wait, _convex_intersection_depth returns area of intersection
    outside_area = max(0, inner_area - intersection_area)
    return outside_area

def _distance_penalty(inner_verts, other_verts):
    """
    Compute penalty for two polygons overlapping.
    Returns 0 if not overlapping, else area of intersection.
    """
    return _convex_intersection_depth(inner_verts, other_verts)

def _get_cost(inner, outer_center, outer_side, outer_rot):
    """
    inner: list of (x, y, rot_degrees)
    Returns (S, penalty) where penalty is sum of overlap and containment penalties.
    """
    outer_verts = _hex_vertices(outer_center[0], outer_center[1], np.radians(outer_rot), outer_side)
    
    total_pen = 0.0
    
    inner_verts_list = []
    for i in range(N):
        x, y, rot_deg = inner[i]
        verts = _hex_vertices(x, y, np.radians(rot_deg), 1.0)
        inner_verts_list.append(verts)
    
    # Containment penalty
    for i in range(N):
        pen = _containment_penalty(inner_verts_list[i], outer_verts)
        total_pen += pen * 10.0  # weight
    
    # Overlap penalty
    for i in range(N):
        for j in range(i + 1, N):
            pen = _distance_penalty(inner_verts_list[i], inner_verts_list[j])
            total_pen += pen * 10.0
    
    return total_pen

def _objective(flat_params):
    """
    flat_params: [x1, y1, rot1, ..., x12, y12, rot12, S, outer_rot_deg]
    """
    params = flat_params
    S = params[N*3]
    outer_rot_deg = params[N*3 + 1]
    outer_center = (0.0, 0.0)
    
    if S < 1.0:
        S = 1.0
    
    inner = []
    for i in range(N):
        x = params[i*3]
        y = params[i*3 + 1]
        rot_deg = params[i*3 + 2]
        inner.append((x, y, rot_deg))
    
    pen = _get_cost(inner, outer_center, S, outer_rot_deg)
    
    # We want to minimize S + penalty
    # But S is the objective, penalty should be zero for valid solution
    # To encourage valid solutions, we use a large multiplier
    return S + 100.0 * pen

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - TIME_MARGIN
    
    # Initial configuration: honeycomb-like arrangement
    # Hexagonal tiling: place hexagons in a hexagonal pattern
    # Side length 1, distance between centers in tiling is sqrt(3)
    # Let's place 7 in a hexagonal cluster and 5 more around
    
    # Actually, let's use a simple grid-like start
    # For N=12, we can do 2 rows of 6, or 3 rows of 4
    
    # Let's try a compact arrangement
    # Row 0: 4 hexagons
    # Row 1: 4 hexagons (offset)
    # Row 2: 4 hexagons
    
    dx = np.sqrt(3)  # horizontal distance in tiling
    dy = 1.5  # vertical distance in tiling (for pointy top)
    
    # Let's use flat-top hexagons for tiling
    # Flat top: width = 2, height = sqrt(3)
    # Distance between centers:
    # Horizontal: 1.5 (3/2 * side)
    # Vertical: sqrt(3)
    
    # Actually, let's just use a random start and let the optimizer work
    # But a good start helps.
    
    # Hexagonal packing:
    # Centers at (i * 1.5 + j * 0.75, j * sqrt(3) * 0.5)
    # No, let's think again.
    
    # For flat-top hexagons (vertices at 0, 60, 120, 180, 240, 300 degrees):
    # Width = 2, Height = sqrt(3)
    # Tiling: 
    # Horizontal distance: 1.5
    # Vertical distance: sqrt(3)
    # Odd columns offset by sqrt(3)/2
    
    # Let's place them in a roughly hexagonal shape
    # Center at (0,0)
    # Ring 0: 1 hex
    # Ring 1: 6 hex
    # Ring 2: 12 hex (we only need 5 more)
    
    # Total 1 + 6 + 5 = 12
    
    inner = []
    
    # Center
    inner.append((0.0, 0.0, 0.0))
    
    # Ring 1 (6 hexagons)
    for k in range(6):
        angle = k * np.pi / 3
        r = np.sqrt(3)  # distance between centers in tiling
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        inner.append((x, y, 0.0))
    
    # Ring 2 (5 more hexagons, skip one to make 12 total)
    # Actually, ring 2 has 12 positions. We need 5 more.
    # Let's place them at alternating positions
    for k in range(0, 12, 2):  # every other position
        if len(inner) >= N:
            break
        angle = k * np.pi / 6
        r = 2 * np.sqrt(3)
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        inner.append((x, y, 0.0))
    
    # Trim to N
    inner = inner[:N]
    
    # Estimate S
    # Bounding box of initial config
    xs = [p[0] for p in inner]
    ys = [p[1] for p in inner]
    max_r = max(np.sqrt(x*x + y*y) for x, y in zip(xs, ys))
    S_init = max_r + 1.5  # rough estimate
    
    # Parameters: [x1, y1, rot1, ..., x12, y12, rot12, S, outer_rot_deg]
    x0 = []
    for x, y, rot in inner:
        x0.extend([x, y, rot])
    x0.extend([S_init, 0.0])
    
    # Bounds
    bounds = []
    for i in range(N):
        bounds.append((-6.0, 6.0))  # x
        bounds.append((-6.0, 6.0))  # y
        bounds.append((0.0, 180.0))  # rot_deg
    bounds.append((2.0, 12.0))  # S
    bounds.append((0.0, 60.0))  # outer_rot_deg
    
    # Try L-BFGS-B
    best_params = x0
    best_cost = _objective(x0)
    
    try:
        result = minimize(_objective, x0, method='L-BFGS-B', bounds=bounds,
                         options={'maxiter': 200, 'ftol': 1e-8})
        if result.fun < best_cost:
            best_params = result.x
            best_cost = result.fun
    except Exception:
        pass
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > deadline - start_time:
        pass  # just proceed to extract
    
    # If time permits, do a second round with different initializations
    if time.time() < deadline:
        # Try a few random restarts
        for trial in range(5):
            if time.time() > deadline:
                break
            
            # Random initialization
            x0_random = []
            for i in range(N):
                x = random.uniform(-4, 4)
                y = random.uniform(-4, 4)
                rot = random.uniform(0, 180)
                x0_random.extend([x, y, rot])
            S_est = random.uniform(3, 8)
            outer_rot = random.uniform(0, 60)
            x0_random.extend([S_est, outer_rot])
            
            try:
                result2 = minimize(_objective, x0_random, method='L-BFGS-B', bounds=bounds,
                                 options={'maxiter': 100, 'ftol': 1e-8})
                if result2.fun < best_cost:
                    best_params = result2.x
                    best_cost = result2.fun
            except Exception:
                pass
    
    # Extract results
    params = best_params
    S = params[N*3]
    outer_rot_deg = params[N*3 + 1]
    
    inner_final = []
    for i in range(N):
        x = params[i*3]
        y = params[i*3 + 1]
        rot_deg = params[i*3 + 2]
        inner_final.append([x, y, rot_deg])
    
    # Verify and adjust S if needed
    # If there are violations, increase S
    for _ in range(20):
        pen = _get_cost(inner_final, (0.0, 0.0), S, outer_rot_deg)
        if pen < 1e-6:
            break
        S *= 1.05  # increase S to accommodate
    
    return inner_final, (0.0, 0.0), S, outer_rot_deg

# EVOLVE-BLOCK-END
