import numpy as np
import time
import math
from scipy.optimize import minimize

N = 12
_START = time.time()
_DEADLINE = 140.0

def _elapsed():
    return time.time() - _START

def _hex_vertices(cx, cy, angle_deg, side=1.0):
    a = np.radians(angle_deg)
    angles = a + np.radians(60.0 * np.arange(6))
    return np.column_stack([cx + side * np.cos(angles),
                            cy + side * np.sin(angles)])

def _point_in_hexagon(x, y, cx, cy, angle_deg, side=1.0):
    """Check if point (x,y) is inside regular hexagon centered at (cx,cy)."""
    dx, dy = x - cx, y - cy
    # Rotate point into hexagon's local frame
    a = np.radians(angle_deg)
    lx = dx * np.cos(-a) - dy * np.sin(-a)
    ly = dx * np.sin(-a) + dy * np.cos(-a)
    # Regular hexagon with circumradius = side (since side = circumradius for regular hex)
    # In local frame, hexagon has vertices at angles 0, 60, 120, ...
    # The hexagon is the intersection of 6 half-planes.
    # For a regular hex with "flat top" (vertex at angle 0):
    # Actually with circumradius R, the inradius (apothem) is R * cos(30) = R * sqrt(3)/2
    # The half-width in each of the 6 directions is the apothem.
    # Let's use the standard test: for each edge, check dot product.
    edges = np.array([[math.cos(np.radians(30 + 60*k)), math.sin(np.radians(30 + 60*k))] for k in range(6)])
    apothem = side * math.sqrt(3.0) / 2.0
    dots = np.array([lx * edges[i,0] + ly * edges[i,1] for i in range(6)])
    return np.all(dots <= apothem) + 1e-12

def _hexagon_intersection_area(v1, v2):
    """Approximate intersection of two convex hexagons using Sutherland-Hodgman."""
    # Clip polygon v1 by each edge of v2
    def clip_polygon(subject, clip_edge):
        # Sutherland-Hodgman clip
        output = []
        for i in range(len(subject)):
            current = subject[i]
            next_pt = subject[(i+1) % len(subject)]
            cur_in = _point_in_hexagon(current[0], current[1], 0, 0, 0, 1)  # dummy, not used
            # Actually we need to clip against a specific edge of v2
            # Let me rewrite this properly
            pass
        return output
    
    # Better approach: use shapely if available, otherwise use a simpler method
    # For now, let's use a Monte Carlo estimate or a grid-based estimate
    # Actually, let's implement proper convex polygon clipping
    
    def point_in_halfplane(px, py, ax, ay, bx, by):
        """Check if point is on the same side as the interior of the edge (a->b)."""
        # Normal pointing inward (assuming CCW polygon)
        ex, ey = bx - ax, by - ay
        nx, ny = -ey, ex  # left normal
        return (px - ax) * nx + (py - ay) * ny >= -1e-12
    
    def clip_by_edge(subject, ax, ay, bx, by):
        output = []
        for i in range(len(subject)):
            cur = subject[i]
            nxt = subject[(i+1) % len(subject)]
            cur_in = point_in_halfplane(cur[0], cur[1], ax, ay, bx, by)
            nxt_in = point_in_halfplane(nxt[0], nxt[1], ax, ay, bx, by)
            if cur_in and nxt_in:
                output.append(cur)
            elif cur_in and not nxt_in:
                # Exit: compute intersection
                t = ((ax - cur[0]) * (by - ay) - (ay - cur[1]) * (bx - ax)) / ((nxt[0]-cur[0]) * (by-ay) - (nxt[1]-cur[1]) * (bx-ax) + 1e-15)
                t = max(0, min(1, t))
                ix = cur[0] + t * (nxt[0] - cur[0])
                iy = cur[1] + t * (nxt[1] - cur[1])
                output.append((ix, iy))
            elif not cur_in and nxt_in:
                # Enter: compute intersection
                t = ((ax - cur[0]) * (by - ay) - (ay - cur[1]) * (bx - ax)) / ((nxt[0]-cur[0]) * (by-ay) - (nxt[1]-cur[1]) * (bx-ax) + 1e-15)
                t = max(0, min(1, t))
                ix = cur[0] + t * (nxt[0] - cur[0])
                iy = cur[1] + t * (nxt[1] - cur[1])
                output.append((ix, iy))
                output.append(nxt)
        return output
    
    # Ensure v2 is CCW
    area2 = sum(v2[i][0]*v2[(i+1)%6][1] - v2[(i+1)%6][0]*v2[i][1] for i in range(6)) / 2.0
    if area2 < 0:
        v2 = v2[::-1]
    
    subject = [tuple(p) for p in v1]
    for i in range(6):
        ax, ay = v2[i]
        bx, by = v2[(i+1)%6]
        subject = clip_by_edge(subject, ax, ay, bx, by)
        if len(subject) < 3:
            return 0.0
    
    # Compute area of intersection polygon
    n = len(subject)
    area = 0.0
    for i in range(n):
        x1, y1 = subject[i]
        x2, y2 = subject[(i+1)%n]
        area += x1*y2 - x2*y1
    return abs(area) / 2.0

def _overlap_penalty(inner, outer_center, outer_side, outer_angle, weight=100.0):
    """Compute total overlap area of inner hexagons with each other and with outer boundary."""
    penalty = 0.0
    # Overlaps between inner hexagons
    for i in range(N):
        x1, y1, a1 = inner[i]
        v1 = _hex_vertices(x1, y1, a1)
        for j in range(i+1, N):
            x2, y2, a2 = inner[j]
            v2 = _hex_vertices(x2, y2, a2)
            # Quick reject: distance between centers
            dx, dy = x2 - x1, y2 - y1
            if dx*dx + dy*dy > 4.0:  # max distance for two unit hexagons to overlap
                continue
            pen = _hexagon_intersection_area(v1, v2)
            penalty += weight * pen * pen  # squared penalty for smoothness
    
    # Overlaps with outer hexagon (inner hexagons should be INSIDE outer)
    ov_center = outer_center
    ov_angle = outer_angle
    # Vertices of outer hexagon (negated for "inside" test)
    # For each inner hexagon, check if it's fully inside outer
    for i in range(N):
        x1, y1, a1 = inner[i]
        v1 = _hex_vertices(x1, y1, a1)
        # Check each vertex of inner hex is inside outer hex
        # If not, add penalty based on how far outside
        # Simple: compute distance of each vertex to outer boundary
        for vi in v1:
            if not _point_in_hexagon(vi[0], vi[1], ov_center[0], ov_center[1], ov_angle, outer_side):
                # Compute approximate penetration depth
                penalty += weight * 0.01
    return penalty

def _min_enclosing_side(inner, center, angle):
    """Find minimum side length of outer hexagon centered at 'center' with given angle
    that contains all inner hexagons."""
    cx, cy = center
    a = np.radians(angle)
    # For each inner hex vertex, compute the required 'radius' in the outer hex's frame
    # Outer hex with side s, center (cx,cy), rotation angle:
    # A point (px, py) is inside iff for all 6 edge normals n_k:
    #   dot((px-cx, py-cy), R_a * n_k) <= s * sqrt(3)/2
    # where n_k are the outward normals of a unit hexagon's edges.
    
    # Edge normals of unit regular hexagon (apothem direction for each edge)
    # For a hex with vertex at angle 0, edges are at angles 30, 90, 150, 210, 270, 330
    # The outward normal for edge k points at angle (30 + 60*k) degrees
    base_normals = np.array([[math.cos(math.radians(30 + 60*k)), 
                              math.sin(math.radians(30 + 60*k))] for k in range(6)])
    
    max_required = 0.0
    for i in range(N):
        xi, yi, ai = inner[i]
        vi = _hex_vertices(xi, yi, ai)
        for v in vi:
            dx, dy = v[0] - cx, v[1] - cy
            # Rotate into outer hex frame
            lx = dx * math.cos(-a) - dy * math.sin(-a)
            ly = dx * math.sin(-a) + dy * math.cos(-a)
            for nk in base_normals:
                d = lx * nk[0] + ly * nk[1]
                # Required apothem >= d, so required side >= d / (sqrt(3)/2) = 2*d/sqrt(3)
                required = 2.0 * d / math.sqrt(3.0)
                if required > max_required:
                    max_required = required
    return max(1.0, max_required)

def _evaluate(inner, center, angle):
    """Compute the side length of the minimum enclosing hexagon."""
    return _min_enclosing_side(inner, center, angle)

def construct():
    global _START, _DEADLINE
    _START = time.time()
    _DEADLINE = 140.0
    
    best_result = None
    best_score = float('inf')
    
    # Strategy: Use gradient-based optimization
    # Variables: positions (x,y) and angles for each of 12 hexagons
    # Objective: minimize enclosing hexagon side length
    # We'll use a smooth surrogate: sum of max(0, violation) or log-sum-exp
    
    def _smooth_enclosing_side(inner, center, angle, beta=50.0):
        """Smooth surrogate for min enclosing side using log-sum-exp."""
        cx, cy = center
        a = np.radians(angle)
        base_normals = np.array([[math.cos(math.radians(30 + 60*k)), 
                                  math.sin(math.radians(30 + 60*k))] for k in range(6)])
        
        all_required = []
        for i in range(N):
            xi, yi, ai = inner[i]
            vi = _hex_vertices(xi, yi, ai)
            for v in vi:
                dx, dy = v[0] - cx, v[1] - cy
                lx = dx * math.cos(-a) - dy * math.sin(-a)
                ly = dx * math.sin(-a) + dy * math.cos(-a)
                for nk in base_normals:
                    d = lx * nk[0] + ly * nk[1]
                    required = 2.0 * d / math.sqrt(3.0)
                    all_required.append(required)
        
        all_required = np.array(all_required)
        if len(all_required) == 0:
            return 1.0
        
        # Log-sum-exp for smooth max
        m = np.max(all_required)
        if m < 0:
            m = 0.0
        lse = m + np.log(np.sum(np.exp((all_required - m) / beta))) * beta
        return max(1.0, lse)
    
    def _objective(params):
        if _elapsed() > _DEADLINE - 10:
            return 1e6
        # params: 12 * 3 + 3 = 39 variables
        # x1,y1,a1, ..., x12,y12,a12, cx, cy, ca
        inner = []
        for k in range(N):
            x, y, ang = params[3*k], params[3*k+1], params[3*k+2]
            inner.append([x, y, ang])
        cx, cy, ca = params[36], params[37], params[38]
        
        # Smooth enclosing side
        s = _smooth_enclosing_side(inner, (cx, cy), ca, beta=30.0)
        
        # Overlap penalty (smooth)
        pen = 0.0
        for i in range(N):
            for j in range(i+1, N):
                x1, y1, a1 = inner[i]
                x2, y2, a2 = inner[j]
                dx, dy = x2-x1, y2-y1
                dist_sq = dx*dx + dy*dy
                # Soft repulsion when too close
                min_dist = 1.732  # sqrt(3), inradius * 2
                if dist_sq < (2.5)**2:
                    overlap = max(0, min_dist - math.sqrt(dist_sq))
                    pen += 50.0 * overlap * overlap
        
        return s + pen
    
    def _obj_grad(params):
        # Numerical gradient
        eps = 1e-5
        grad = np.zeros(len(params))
        f0 = _objective(params)
        for i in range(len(params)):
            params[i] += eps
            f1 = _objective(params)
            params[i] -= eps
            grad[i] = (f1 - f0) / eps
        return grad
    
    # Initial guess: spread hexagons in a ring
    def _random_init(rng):
        params = np.zeros(39)
        # Place in a loose ring
        for k in range(N):
            angle_pos = 2 * math.pi * k / N
            r = 3.0 + rng.uniform(0, 1.5)
            params[3*k] = r * math.cos(angle_pos) + rng.uniform(-0.5, 0.5)
            params[3*k+1] = r * math.sin(angle_pos) + rng.uniform(-0.5, 0.5)
            params[3*k+2] = rng.uniform(-30, 30)
        params[36] = 0.0
        params[37] = 0.0
        params[38] = 0.0
        return params
    
    def _extract(params):
        inner = []
        for k in range(N):
            inner.append([float(params[3*k]), float(params[3*k+1]), float(params[3*k+2])])
        cx, cy, ca = float(params[36]), float(params[37]), float(params[38])
        return inner, (cx, cy), ca
    
    rng = np.random.default_rng(42)
    
    n_restarts = 8
    per_restart_budget = 130.0 / n_restarts
    
    for restart in range(n_restarts):
        if _elapsed() > _DEADLINE - 15:
            break
        
        params = _random_init(rng)
        
        # Run L-BFGS
        try:
            result = minimize(_objective, params, method='L-BFGS-B', 
                             options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6,
                                      'maxfun': 5000},
                             jac=_obj_grad)
        except Exception:
            continue
        
        if _elapsed() > _DEADLINE - 10:
            break
        
        # Evaluate with exact (non-smooth) enclosing side
        inner, center, angle = _extract(result.x)
        side = _min_enclosing_side(inner, center, angle)
        
        if side < best_score:
            best_score = side
            best_result = (inner, center, side, angle)
        
        # Local refinement with tighter beta
        try:
            params2 = result.x.copy()
            result2 = minimize(lambda p: _smooth_enclosing_side(
                [[p[3*k], p[3*k+1], p[3*k+2]] for k in range(N)],
                (p[36], p[37]), p[38], beta=10.0),
                params2, method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-10, 'maxfun': 2000})
            inner2, center2, angle2 = _extract(result2.x)
            side2 = _min_enclosing_side(inner2, center2, angle2)
            if side2 < best_score:
                best_score = side2
                best_result = (inner2, center2, side2, angle2)
        except Exception:
            pass
    
    if best_result is None:
        # Fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_result
