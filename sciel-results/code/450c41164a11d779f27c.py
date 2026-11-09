import numpy as np
from scipy.optimize import minimize
import time

N = 11
DEADLINE = 140.0  # safety margin before 150s kill

def _hex_vertices(cx, cy, angle_deg, side=1.0):
    """Get 6 vertices of a regular hexagon."""
    a = np.radians(angle_deg)
    angles = a + np.arange(6) * np.pi / 3.0
    return np.column_stack([cx + side * np.cos(angles),
                            cy + side * np.sin(angles)])

def _point_in_hex(px, py, cx, cy, angle_deg, side=1.0):
    """Check if point (px,py) is inside or on boundary of hexagon."""
    verts = _hex_vertices(cx, cy, angle_deg, side)
    # Use cross product to check if point is on the same side of all edges
    for i in range(6):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % 6]
        # Edge vector
        ex, ey = x2 - x1, y2 - y1
        # Vector from edge start to point
        px_, py_ = px - x1, py - y1
        # Cross product (z-component)
        cross = ex * py_ - ey * px_
        if cross > 1e-9:  # strictly outside this edge
            return False
    return True

def _hex_disjoint(cx1, cy1, a1, cx2, cy2, a2, side=1.0, eps=1e-7):
    """Check if two unit hexagons are disjoint (no interior overlap)."""
    v1 = _hex_vertices(cx1, cy1, a1, side)
    v2 = _hex_vertices(cx2, cy2, a2, side)
    # Check vertices of hex1 against hex2
    for px, py in v1:
        if _point_in_hex(px, py, cx2, cy2, a2, side) and \
           not _point_on_boundary(px, py, cx2, cy2, a2, side):
            return False
    # Check vertices of hex2 against hex1
    for px, py in v2:
        if _point_in_hex(px, py, cx1, cy1, a1, side) and \
           not _point_on_boundary(px, py, cx1, cy1, a1, side):
            return False
    return True

def _point_on_boundary(px, py, cx, cy, angle_deg, side=1.0, tol=1e-6):
    """Check if point is on the boundary of the hexagon."""
    verts = _hex_vertices(cx, cy, angle_deg, side)
    for i in range(6):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % 6]
        # Distance from point to line segment
        dx, dy = x2 - x1, y2 - y1
        length_sq = dx*dx + dy*dy
        if length_sq < 1e-12:
            continue
        t = max(0, min(1, ((px-x1)*dx + (py-y1)*dy) / length_sq))
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy
        dist_sq = (px - proj_x)**2 + (py - proj_y)**2
        if dist_sq < tol*tol:
            return True
    return False

def _hex_fits_in_outer(cx, cy, a_inner, S, ox, oy, a_outer):
    """Check if inner hexagon (at cx,cy with angle a_inner) fits inside outer hexagon."""
    verts = _hex_vertices(cx, cy, a_inner, 1.0)
    for px, py in verts:
        if not _point_in_hex(px, py, ox, oy, a_outer, S):
            return False
    return True

def construct():
    start_time = time.time()
    
    # Initial configuration: honeycomb-like arrangement
    # 11 hexagons: 1 center + 6 around + 4 more
    # Use a compact initial layout
    hex_positions = []
    # Center
    hex_positions.append([0.0, 0.0, 0.0])
    # 6 around center (hexagonal neighbors)
    for i in range(6):
        angle = i * np.pi / 3.0
        x = 2.0 * np.cos(angle)
        y = 2.0 * np.sin(angle)
        hex_positions.append([x, y, 0.0])
    # 4 more in a ring
    for i in range(4):
        angle = np.pi/6 + i * np.pi / 2.0
        x = 4.0 * np.cos(angle)
        y = 4.0 * np.sin(angle)
        hex_positions.append([x, y, 0.0])
    
    # State vector: [x0, y0, a0, x1, y1, a1, ..., x10, y10, a10, ox, oy, a_outer, S]
    # 11*3 + 4 = 37 variables
    x0 = np.array([p[0] for p in hex_positions] + 
                   [p[1] for p in hex_positions] + 
                   [p[2] for p in hex_positions] +
                   [0.0, 0.0, 0.0, 8.0])
    
    def objective(params):
        if time.time() - start_time > DEADLINE:
            return 1e10  # timeout penalty
        
        # Unpack
        inner = []
        for i in range(N):
            inner.append((params[3*i], params[3*i+1], params[3*i+2]))
        ox, oy, a_outer, S = params[3*N], params[3*N+1], params[3*N+2], params[3*N+3]
        
        if S <= 0:
            return 1e10
        
        penalty = 0.0
        
        # Check each inner hexagon fits in outer
        for cx, cy, a in inner:
            if not _hex_fits_in_outer(cx, cy, a, S, ox, oy, a_outer):
                # Penalty: how far outside the vertices are
                verts = _hex_vertices(cx, cy, a, 1.0)
                for px, py in verts:
                    if not _point_in_hex(px, py, ox, oy, a_outer, S):
                        penalty += 1.0
        
        # Check pairwise disjointness
        for i in range(N):
            for j in range(i+1, N):
                cx1, cy1, a1 = inner[i]
                cx2, cy2, a2 = inner[j]
                if not _hex_disjoint(cx1, cy1, a1, cx2, cy2, a2, 1.0):
                    penalty += 1.0
        
        return penalty + 0.1 * S  # minimize S with penalty for violations
    
    # Try multiple random restarts
    best_S = 1e10
    best_params = None
    
    n_restarts = 10
    for restart in range(n_restarts):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        # Perturb initial solution
        if restart == 0:
            params = x0.copy()
        else:
            params = x0.copy()
            # Random perturbations
            params[:3*N] += np.random.normal(0, 0.5, 3*N)
            params[3*N+3] = 6.0 + np.random.uniform(0, 3)  # S
        
        # Optimize
        try:
            result = minimize(objective, params, method='Nelder-Mead',
                            options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-6})
            if result.fun < objective(best_params) if best_params is not None else result.fun < 1e10:
                if result.fun < 0.1 + 0.1 * result.x[3*N+3]:  # roughly no violations
                    S_candidate = result.x[3*N+3]
                    if S_candidate < best_S:
                        best_S = S_candidate
                        best_params = result.x.copy()
        except Exception:
            continue
    
    if best_params is None:
        # Fallback: return seed-like solution
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Extract best solution
    inner = []
    for i in range(N):
        inner.append([best_params[3*i], best_params[3*i+1], best_params[3*i+2]])
    ox, oy, a_outer, S = best_params[3*N], best_params[3*N+1], best_params[3*N+2], best_params[3*N+3]
    
    return inner, (ox, oy), S, a_outer
