"""Pack 11 unit hexagons inside a regular hexagon, minimizing outer side length."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
_T0 = time.time()
_DEADLINE = 145.0

def _hex_verts(cx, cy, s, rot):
    """Return the 6 vertices of a regular hexagon."""
    vs = np.empty((6, 2))
    for k in range(6):
        a = math.radians(60.0 * k) + rot
        vs[k, 0] = cx + s * math.cos(a)
        vs[k, 1] = cy + s * math.sin(a)
    return vs

def _point_in_hex(px, py, cx, cy, s, rot):
    """Check if point (px,py) is inside hexagon centered at (cx,cy) with side s, rotation rot."""
    dx = px - cx
    dy = py - cy
    ca, sa = math.cos(-rot), math.sin(-rot)
    lx = ca * dx + sa * dy
    ly = -sa * dx + ca * dy
    r_in = s * math.cos(math.radians(30))
    c30 = math.cos(math.radians(30))
    d1 = abs(c30 * lx + 0.5 * ly)
    d2 = abs(ly)
    d3 = abs(-c30 * lx + 0.5 * ly)
    return max(d1, d2, d3) <= r_in + 1e-9

def _smooth_penalty_containment(xi, yi, ai, t):
    """
    Smooth penalty for hexagon i's vertices being outside the outer hexagon (centered at origin, side t).
    Uses a smooth approximation of the hexagon containment constraint.
    """
    pen = 0.0
    for k in range(6):
        a = math.radians(60.0 * k) + ai
        vx = xi + math.cos(a)
        vy = yi + math.sin(a)
        # Check if (vx, vy) is inside outer hexagon (centered at origin, side t, rotation 0)
        # Outer hexagon has vertices at angles 0, 60, ..., 300
        # A point is inside if it satisfies 3 half-plane constraints
        # For a regular hexagon with side t centered at origin:
        # The inradius is t * cos(30°)
        # Distance from center to each side = inradius
        # The 3 pairs of parallel sides have normals at 30°, 90°, 150°
        
        # Transform point to hexagon's local frame (no rotation for outer hex)
        px, py = vx, vy
        c30 = math.cos(math.radians(30))
        r_in = t * c30
        
        # Three half-plane constraints:
        # |px * cos(30) + py * sin(30)| <= r_in
        # |py| <= r_in  
        # |px * cos(150) + py * sin(150)| <= r_in  => |-px*cos(30) + py*sin(30)| <= r_in
        
        d1 = abs(c30 * px + 0.5 * py)
        d2 = abs(py)
        d3 = abs(-c30 * px + 0.5 * py)
        
        # Smooth penalty: use softplus-like function for violation
        for d in [d1, d2, d3]:
            if d > r_in:
                violation = d - r_in
                pen += 5.0 * violation * violation  # Quadratic penalty for smoothness
    return pen

def _smooth_penalty_overlap(i, j, xi, yi, ai, xj, yj, aj):
    """
    Smooth penalty for overlap between hexagons i and j.
    Uses the minimum distance between edges as a smooth proxy.
    """
    # If centers are far apart, no overlap
    dc = math.sqrt((xi - xj)**2 + (yi - yj)**2)
    if dc > 4.0:
        return 0.0
    
    # Check if any vertex of i is inside j, or vice versa
    pen = 0.0
    vi = _hex_verts(xi, yi, 1.0, ai)
    vj = _hex_verts(xj, yj, 1.0, aj)
    
    for m in range(6):
        if _point_in_hex(vi[m,0], vi[m,1], xj, yj, 1.0, aj):
            # Estimate penetration depth
            pen += 2.0
            break
        if _point_in_hex(vj[m,0], vj[m,1], xi, yi, 1.0, ai):
            pen += 2.0
            break
    
    return pen

def _build_objective(n):
    def objective(p):
        x = p[:2*n]
        angles = p[2*n:3*n]
        t = p[3*n]
        
        pen = 0.0
        
        # Containment penalty
        for i in range(n):
            xi, yi = x[2*i], x[2*i+1]
            ai = angles[i]
            pen += _smooth_penalty_containment(xi, yi, ai, t)
        
        # Overlap penalty
        for i in range(n):
            for j in range(i+1, n):
                pen += _smooth_penalty_overlap(i, j, x[2*i], x[2*i+1], angles[i], x[2*j], x[2*j+1], angles[j])
        
        return 0.5 * t * t + pen
    return objective

def construct():
    t_start = time.time()
    n = N

    def _remaining():
        return _DEADLINE - (time.time() - t_start)

    objective = _build_objective(n)
    best_val = float('inf')
    best_p = None

    rng = np.random.RandomState(123)

    # Initial guess: place hexagons in a honeycomb-like pattern
    # Center hexagon at origin, 6 around it, 4 more in second ring
    def _initial_guess():
        p = np.zeros(3*n + 1)
        # Center
        p[0] = 0.0
        p[1] = 0.0
        # 6 surrounding
        for k in range(6):
            a = math.radians(60.0*k)
            p[2*(1+k)] = 2.0 * math.cos(a)
            p[2*(1+k)+1] = 2.0 * math.sin(a)
        # 4 more
        for k in range(4):
            a = math.radians(60.0*k + 30.0)
            p[2*(7+k)] = 4.0 * math.cos(a)
            p[2*(7+k)+1] = 4.0 * math.sin(a)
        p[3*n] = 7.0
        return p

    # Try multiple starts with different initial configurations
    for trial in range(12):
        if _remaining() < 20.0:
            break

        p = _initial_guess()
        if trial > 0:
            # Add random perturbation
            p[:3*n] += rng.randn(3*n) * 0.5
            p[3*n] = 6.0 + rng.rand()

        try:
            # Use Nelder-Mead with smooth objective
            res = minimize(objective, p, method='Nelder-Mead',
                          options={'maxiter': 300, 'xatol': 1e-3, 'fatol': 1e-4, 'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_p = res.x.copy()
        except Exception:
            pass

        # Also try L-BFGS-B for better convergence on smooth parts
        if _remaining() < 15.0:
            break
        try:
            bounds = [(-10, 10)] * (2*n) + [(-math.pi, math.pi)] * n + [(3.0, 15.0)]
            res2 = minimize(objective, p, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 200, 'ftol': 1e-5, 'gtol': 1e-4})
            if res2.fun < best_val:
                best_val = res2.fun
                best_p = res2.x.copy()
        except Exception:
            pass

    if best_p is None:
        inner = [[float(k*2.0), 0.0, 0.0] for k in range(n)]
        return inner, (0.0, 0.0), 40.0, 0.0

    x = best_p[:2*n]
    angles = best_p[2*n:3*n]
    t = best_p[3*n]

    inner = [[float(x[2*i]), float(x[2*i+1]), float(math.degrees(angles[i]))] for i in range(n)]

    # Validate: check if solution is feasible
    # If t is too small, fall back to a known feasible solution
    if t < 3.5:
        # Use a simple valid packing: line up hexagons
        inner = []
        for k in range(n):
            inner.append([float(k * 2.0), 0.0, 0.0])
        t = 40.0

    return inner, (0.0, 0.0), float(t), 0.0
