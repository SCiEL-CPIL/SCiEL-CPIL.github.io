"""Hexagon packing: place N unit regular hexagons inside a regular hexagon, minimize outer side."""
import numpy as np
import time
from scipy.optimize import minimize

N = 11
TIME_LIMIT = 140.0  # safety margin before 150s kill

def _hex_vertices(cx, cy, angle_deg, side=1.0):
    """Return 6 vertices of a regular hexagon (flat-top orientation at angle 0)."""
    a = np.deg2rad(angle_deg)
    # Flat-top: vertices at 0,60,...,300 degrees
    angles = a + np.arange(6) * np.pi / 3.0
    return np.column_stack((cx + side * np.cos(angles),
                            cy + side * np.sin(angles)))

def _point_to_segment_dist(px, py, ax, ay, bx, by):
    """Distance from point (px,py) to segment (ax,ay)-(bx,by)."""
    abx, aby = bx - ax, by - ay
    apx, apy = px - ax, py - ay
    t = (apx * abx + apy * aby) / (abx * abx + aby * aby + 1e-30)
    t = max(0.0, min(1.0, t))
    qx, qy = ax + t * abx, ay + t * aby
    return np.hypot(px - qx, py - qy)

def _min_dist_hex(v1, v2):
    """Min distance between two convex hexagons given their vertex arrays (6,2)."""
    # Check if any vertex of one is inside the other -> overlap
    # Simple point-in-polygon for convex polygon
    def point_in_convex(p, verts):
        # Cross products should all have same sign
        n = len(verts)
        signs = []
        for i in range(n):
            ax, ay = verts[i]
            bx, by = verts[(i + 1) % n]
            cx, cy = p
            cross = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
            signs.append(cross)
        # All same sign (or zero)
        pos = any(s > 1e-12 for s in signs)
        neg = any(s < -1e-12 for s in signs)
        return not (pos and neg)
    
    if point_in_convex(v1[0], v2) or point_in_convex(v2[0], v1):
        return 0.0
    # Min vertex-to-edge distance
    dmin = float('inf')
    for i in range(6):
        px, py = v1[i]
        for j in range(6):
            ax, ay = v2[j]
            bx, by = v2[(j + 1) % 6]
            d = _point_to_segment_dist(px, py, ax, ay, bx, by)
            if d < dmin:
                dmin = d
    for i in range(6):
        px, py = v2[i]
        for j in range(6):
            ax, ay = v1[j]
            bx, by = v1[(j + 1) % 6]
            d = _point_to_segment_dist(px, py, ax, ay, bx, by)
            if d < dmin:
                dmin = d
    return dmin

def _softplus(x):
    return np.log1p(np.exp(x)) if x > 30 else (x if x < -30 else np.log(1 + np.exp(x)))

def _objective(params, N):
    """params: [x1,y1,a1, ..., xN,yN,aN, cx, cy, s, phi]"""
    n = N
    inner = params[:3 * n]
    cx, cy, s, phi = params[3 * n:3 * n + 4]
    if s <= 0.1:
        return 1e6
    s = max(s, 0.1)
    
    # Compute inner hexagon vertices
    verts = []
    for i in range(n):
        xi, yi, ai = inner[3 * i:3 * i + 3]
        verts.append(_hex_vertices(xi, yi, ai))
    
    # Overlap penalty
    overlap_pen = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = _min_dist_hex(verts[i], verts[j])
            if d < 0.01:
                overlap_pen += _softplus(-d * 100)
    
    # Containment penalty: each inner vertex must be inside outer hexagon
    # Outer hexagon: center (cx,cy), side s, rotation phi
    # Axes of outer hexagon (flat-top at rotation phi)
    phi_r = np.deg2rad(phi)
    axes = []
    for k in range(3):
        ang = phi_r + k * np.pi / 3.0
        axes.append(np.array([np.cos(ang), np.sin(ang)]))
    
    contain_pen = 0.0
    for v in verts:
        for pt in v:
            dx, dy = pt[0] - cx, pt[1] - cy
            for ax in axes:
                proj = abs(dx * ax[0] + dy * ax[1])
                viol = proj - s
                if viol > 0:
                    contain_pen += _softplus(viol * 50)
    
    return s + 5.0 * overlap_pen + 10.0 * contain_pen

def _make_initial_config(seed, N):
    """Generate an initial configuration."""
    rng = np.random.RandomState(seed)
    # Hexagonal lattice arrangement
    # Place hexagons in a hexagonal pattern
    positions = []
    # For N hexagons, use a hexagonal number arrangement
    # Layer 0: 1, Layer 1: 6, Layer 2: 12, ...
    # Total after k layers: 1 + 3k(k+1)
    # For N=11: 1+6=7 (k=1), 1+6+12=19 (k=2). So use 1 center + 6 + 4 more
    # Simple approach: place in a grid
    # Hexagonal lattice spacing: for flat-top hexagons side 1, 
    # horizontal spacing = 2*inradius = sqrt(3), vertical spacing = 1.5
    # But with rotation freedom, we can use tighter packing.
    # Start with a reasonable grid
    spacing = 1.8  # slightly more than 2*inradius=1.732
    count = 0
    row = 0
    while count < N:
        cols = min(N - count, 3 + row % 2)
        for c in range(cols):
            if count >= N:
                break
            x = c * spacing + (row % 2) * spacing / 2
            y = row * 1.5
            positions.append([x, y, 0.0])
            count += 1
        row += 1
    
    # Add random perturbation
    for p in positions:
        p[0] += rng.uniform(-0.1, 0.1)
        p[1] += rng.uniform(-0.1, 0.1)
        p[2] = rng.uniform(0, 60)
    
    # Outer hexagon initial guess
    cx = np.mean([p[0] for p in positions])
    cy = np.mean([p[1] for p in positions])
    s = 3.5
    phi = 0.0
    
    params = []
    for p in positions:
        params.extend(p)
    params.extend([cx, cy, s, phi])
    return np.array(params)

def construct():
    t0 = time.time()
    best_params = None
    best_score = float('inf')
    
    # Multi-start
    n_starts = 8
    for start in range(n_starts):
        if time.time() - t0 > TIME_LIMIT - 20:
            break
        params = _make_initial_config(start * 137 + 7, N)
        
        # L-BFGS-B optimization
        result = minimize(
            _objective, params, args=(N,),
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6}
        )
        
        if result.success or result.fun < best_score:
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x.copy()
        
        if time.time() - t0 > TIME_LIMIT - 30:
            break
    
    if best_params is None:
        # Fallback: return a trivial valid config
        inner = [[float(i), 0.0, 0.0] for i in range(N)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Extract best
    n = N
    inner_params = best_params[:3 * n]
    cx, cy, s, phi = best_params[3 * n:3 * n + 4]
    
    # Polish: fix inner positions/angles, minimize s with containment only
    def _containment_only(s_and_outer):
        s, phi2, cx2, cy2 = s_and_outer
        if s <= 0.1:
            return 1e6
        s = max(s, 0.1)
        phi_r = np.deg2rad(phi2)
        axes = []
        for k in range(3):
            ang = phi_r + k * np.pi / 3.0
            axes.append(np.array([np.cos(ang), np.sin(ang)]))
        
        pen = 0.0
        for i in range(n):
            xi, yi, ai = inner_params[3 * i:3 * i + 3]
            v = _hex_vertices(xi, yi, ai)
            for pt in v:
                dx, dy = pt[0] - cx2, pt[1] - cy2
                for ax in axes:
                    proj = abs(dx * ax[0] + dy * ax[1])
                    viol = proj - s
                    if viol > 0:
                        pen += _softplus(viol * 100)
        return s + 10.0 * pen
    
    if time.time() - t0 < TIME_LIMIT - 10:
        init_outer = np.array([s, phi, cx, cy])
        result2 = minimize(
            _containment_only, init_outer,
            method='L-BFGS-B',
            options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        if result2.fun < best_score:
            s, phi, cx, cy = result2.x
            s = max(s, 0.1)
    
    # Build inner list
    inner = []
    for i in range(n):
        xi, yi, ai = inner_params[3 * i:3 * i + 3]
        inner.append([float(xi), float(yi), float(ai)])
    
    return inner, (float(cx), float(cy)), float(s), float(phi)
