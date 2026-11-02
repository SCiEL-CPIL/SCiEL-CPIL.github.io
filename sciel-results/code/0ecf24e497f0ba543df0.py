"""Optimization-based hexagon packing using gradient descent on smooth distances."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time

N = 11

def hex_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon centered at (cx, cy) with given side and rotation."""
    a = np.radians(angle_deg)
    angles = a + np.pi / 3 * np.arange(6)
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def point_to_hexagon_distance(px, py, hx, hy, hangle):
    """Smooth distance from point (px,py) to hexagon centered at (hx,hy) with angle hangle.
    Uses distance to edges (line segments) via a smooth approximation."""
    verts = hex_vertices(hx, hy, 1.0, hangle)
    min_dist = np.inf
    for i in range(6):
        ax, ay = verts[i]
        bx, by = verts[(i + 1) % 6]
        # Distance from point to line segment
        abx, aby = bx - ax, by - ay
        apx, apy = px - ax, py - ay
        t = (apx * abx + apy * aby) / (abx * abx + aby * aby + 1e-12)
        t = np.clip(t, 0.0, 1.0)
        cx, cy = ax + t * abx, ay + t * aby
        d = np.sqrt((px - cx) ** 2 + (py - cy) ** 2)
        min_dist = min(min_dist, d)
    return min_dist

def smooth_min(values, beta=20.0):
    """Log-sum-exp approximation of min, differentiable."""
    m = np.min(values)
    return m + np.log(np.sum(np.exp(-beta * (values - m)))) / beta

def clearance_function(params):
    """
    params: [x0,y0,a0, x1,y1,a1, ..., outer_cx, outer_cy, outer_side, outer_angle]
    Returns negative of minimum clearance (to minimize = maximize clearance).
    """
    n = N
    inner_params = params[:3 * n]
    outer_cx = params[3 * n]
    outer_cy = params[3 * n + 1]
    outer_side = params[3 * n + 2]
    outer_angle = params[3 * n + 3]
    
    if outer_side < 0.5:
        return 1e6
    
    outer_verts = hex_vertices(outer_cx, outer_cy, outer_side, outer_angle)
    
    # For each inner hexagon vertex, compute distance to outer hexagon boundary
    # and distance to other inner hexagons
    min_clearance = np.inf
    
    for i in range(n):
        ix = inner_params[3 * i]
        iy = inner_params[3 * i + 1]
        iangle = inner_params[3 * i + 2]
        
        iverts = hex_vertices(ix, iy, 1.0, iangle)
        
        for j in range(6):
            vx, vy = iverts[j]
            
            # Distance to outer boundary (must be >= 0, i.e., inside)
            # Compute distance from point to outer hexagon edges
            outer_d = np.inf
            for k in range(6):
                ax, ay = outer_verts[k]
                bx, by = outer_verts[(k + 1) % 6]
                abx, aby = bx - ax, by - ay
                apx, apy = vx - ax, vy - ay
                t = (apx * abx + apy * aby) / (abx * abx + aby * aby + 1e-12)
                t = np.clip(t, 0.0, 1.0)
                cx, cy = ax + t * abx, ay + t * aby
                d = np.sqrt((vx - cx) ** 2 + (vy - cy) ** 2)
                outer_d = min(outer_d, d)
            
            # Check if vertex is inside outer hexagon using winding/point-in-poly
            # Simple: check all edge cross products have same sign
            inside = True
            for k in range(6):
                ax, ay = outer_verts[k]
                bx, by = outer_verts[(k + 1) % 6]
                cross = (bx - ax) * (vy - ay) - (by - ay) * (vx - ax)
                if cross < -1e-9:
                    inside = False
                    break
            if not inside:
                # Penalty: distance to boundary is negative
                min_clearance = min(min_clearance, -outer_d)
                continue
            
            # Distance to other inner hexagons
            for j2 in range(n):
                if j2 == i:
                    continue
                jx = inner_params[3 * j2]
                jy = inner_params[3 * j2 + 1]
                jangle = inner_params[3 * j2 + 2]
                
                jverts = hex_vertices(jx, jy, 1.0, jangle)
                # Distance from vertex to j hexagon
                for k in range(6):
                    ax, ay = jverts[k]
                    bx, by = jverts[(k + 1) % 6]
                    abx, aby = bx - ax, by - ay
                    apx, apy = vx - ax, vy - ay
                    t = (apx * abx + apy * aby) / (abx * abx + aby * aby + 1e-12)
                    t = np.clip(t, 0.0, 1.0)
                    cx, cy = ax + t * abx, ay + t * aby
                    d = np.sqrt((vx - cx) ** 2 + (vy - cy) ** 2)
                    min_clearance = min(min_clearance, d - 2.0)  # subtract 2 for two unit hexes touching
    
    return -min_clearance  # We want to maximize min_clearance, so minimize negative

def construct():
    start_time = time.time()
    
    # Initial guess: spread hexagons in a grid pattern
    best_result = None
    best_score = np.inf
    
    n_starts = 5
    
    for s in range(n_starts):
        if time.time() - start_time > 100:
            break
        
        np.random.seed(42 + s)
        
        # Initial positions: hexagonal grid with some randomness
        params = np.zeros(3 * N + 4)
        
        # Place in a rough hexagonal arrangement
        for i in range(N):
            row = i // 4
            col = i % 4
            params[3 * i] = (col - 1.5) * 2.2 + np.random.randn() * 0.5
            params[3 * i + 1] = (row - 1) * 2.0 + np.random.randn() * 0.5
            params[3 * i + 2] = np.random.randn() * 10  # random angle
        
        params[3 * N] = 0.0       # outer cx
        params[3 * N + 1] = 0.0   # outer cy
        params[3 * N + 2] = 8.0   # outer side (initial guess)
        params[3 * N + 3] = 0.0   # outer angle
        
        # Bounds
        bounds = []
        for i in range(N):
            bounds.append((-15, 15))
            bounds.append((-15, 15))
            bounds.append((-180, 180))
        bounds.append((-5, 5))
        bounds.append((-5, 5))
        bounds.append((1.0, 15.0))
        bounds.append((-180, 180))
        
        try:
            res = minimize(
                clearance_function,
                params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            
            if time.time() - start_time > 110:
                break
            
            if res.success:
                p = res.x
                outer_side = p[3 * N + 2]
                if outer_side < best_score:
                    best_score = outer_side
                    best_result = p
        except Exception:
            continue
    
    if best_result is None:
        # Fallback: simple grid
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    p = best_result
    inner = []
    for i in range(N):
        inner.append([p[3 * i], p[3 * i + 1], p[3 * i + 2]])
    
    outer_cx = p[3 * N]
    outer_cy = p[3 * N + 1]
    outer_side = p[3 * N + 2]
    outer_angle = p[3 * N + 3]
    
    return inner, (outer_cx, outer_cy), outer_side, outer_angle

# EVOLVE-BLOCK-START
# (construct is defined above; this block kept for interface compatibility)
# EVOLVE-BLOCK-END
