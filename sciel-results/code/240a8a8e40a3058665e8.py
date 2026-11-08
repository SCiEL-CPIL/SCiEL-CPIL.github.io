import numpy as np
from scipy.optimize import minimize
import time

N = 11

# EVOLVE-BLOCK-START
def _hex_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon."""
    a = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * np.pi / 3
        x = cx + side * np.cos(theta)
        y = cy + side * np.sin(theta)
        verts.append((x, y))
    return np.array(verts)

def _hex_distance(p1, a1, p2, a2):
    """Approximate minimum distance between two unit hexagons."""
    v1 = _hex_vertices(p1[0], p1[1], 1.0, a1)
    v2 = _hex_vertices(p2[0], p2[1], 1.0, a2)
    
    # Sample edges
    def sample(verts, n=2):
        pts = []
        for i in range(6):
            for j in range(n):
                t = j / n
                pts.append(verts[i] + t * (verts[(i+1)%6] - verts[i]))
        return np.array(pts)
    
    s1 = sample(v1)
    s2 = sample(v2)
    
    dmin = np.inf
    for p in s1:
        d = np.sqrt(np.sum((s2 - p)**2, axis=1))
        dmin = min(dmin, np.min(d))
    return dmin

def _bounding_hex_side(pts, center, outer_angle):
    """Minimum side of regular hexagon containing all points."""
    a = np.radians(outer_angle)
    cos_a, sin_a = np.cos(a), np.sin(a)
    dx = pts[:, 0] - center[0]
    dy = pts[:, 1] - center[1]
    rx = dx * cos_a + dy * sin_a
    ry = -dx * sin_a + dy * cos_a
    
    # 6 edge normals (for flat-top hexagon, normals at 0, 60, 120, 180, 240, 300)
    s = 0
    for i in range(6):
        theta = i * np.pi / 3
        nx, ny = np.cos(theta), np.sin(theta)
        proj = rx * nx + ry * ny
        s = max(s, np.max(proj))
    return s

def construct():
    deadline = time.time() + 100
    
    sqrt3 = np.sqrt(3)
    
    # Honeycomb initial configuration
    # 1 center + 6 first ring + 4 from second ring
    positions = np.zeros((N, 2))
    angles = np.zeros(N)
    
    # Center
    positions[0] = [0, 0]
    
    # First ring (6 hexagons at distance 2)
    for i in range(6):
        theta = i * np.pi / 3
        positions[i+1] = [2.0 * np.cos(theta), 2.0 * np.sin(theta)]
    
    # Second ring - pick 4 positions
    # Second ring has 12 positions. Let's use 4 that are spread out.
    # Positions at distance 2*sqrt(3) in directions 30, 90, 150, 210, 270, 330
    # and at distance 4 in directions 0, 60, 120, 180, 240, 300
    # Use the 4 at distance 4: angles 0, 90, 180, 270 (but 90 and 270 aren't lattice points)
    # Actually for hexagonal lattice, second shell:
    # 6 at distance 2*sqrt(3) (corners) and 6 at distance 4 (edge midpoints)
    
    # Let's use 4 from the 12 second-shell positions
    second_shell = []
    for i in range(6):
        theta = i * np.pi / 3
        # Corner position (distance 2*sqrt(3))
        second_shell.append([2*sqrt3 * np.cos(theta + np.pi/6), 2*sqrt3 * np.sin(theta + np.pi/6)])
        # Edge midpoint (distance 4)
        second_shell.append([4.0 * np.cos(theta), 4.0 * np.sin(theta)])
    
    # Pick 4 spread out: indices 0, 3, 6, 9 (corners at 30, 150, 270, 390=30... let me just pick 0, 3, 6, 9)
    for i, idx in enumerate([0, 3, 6, 9]):
        positions[i+7] = second_shell[idx]
    
    def objective(params):
        nonlocal deadline
        if time.time() > deadline:
            return 1e10
        
        n = len(params) // 3
        pos = params[:2*n].reshape(n, 2)
        ang = params[2*n:]
        
        # Check overlaps
        for i in range(n):
            for j in range(i+1, n):
                d = _hex_distance(pos[i], ang[i], pos[j], ang[j])
                if d < -0.005:
                    return 1e8 + abs(d) * 1e5
        
        # Get all vertices
        all_pts = []
        for i in range(n):
            v = _hex_vertices(pos[i, 0], pos[i, 1], 1.0, ang[i])
            all_pts.append(v)
        all_pts = np.vstack(all_pts)
        
        # Try a few outer angles and centers
        best_s = np.inf
        for oa in [0, 30]:
            # Centroid as center estimate
            c = np.mean(all_pts, axis=0)
            s = _bounding_hex_side(all_pts, c, oa)
            best_s = min(best_s, s)
        
        return best_s
    
    x0 = np.concatenate([positions.flatten(), angles])
    
    best_x = x0.copy()
    best_val = objective(x0)
    
    # Local search with Nelder-Mead
    try:
        res = minimize(objective, x0, method='Nelder-Mead',
                      options={'maxiter': 100, 'xatol': 0.02, 'fatol': 0.002, 'adaptive': True})
        if res.fun < best_val:
            best_x = res.x
            best_val = res.fun
    except:
        pass
    
    if time.time() > deadline:
        pass
    
    n = len(best_x) // 3
    pos = best_x[:2*n].reshape(n, 2)
    ang = best_x[2*n:]
    
    # Compute final center
    all_pts = []
    for i in range(n):
        v = _hex_vertices(pos[i, 0], pos[i, 1], 1.0, ang[i])
        all_pts.append(v)
    all_pts = np.vstack(all_pts)
    center = np.mean(all_pts, axis=0)
    
    inner = []
    for i in range(n):
        inner.append([float(pos[i, 0]), float(pos[i, 1]), float(ang[i])])
    
    return inner, (float(center[0]), float(center[1])), float(best_val), 0.0

# EVOLVE-BLOCK-END
