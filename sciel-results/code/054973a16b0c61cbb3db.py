# EVOLVE-BLOCK-START
import numpy as np
import time

def _hex_vertices(cx, cy, s, a_deg):
    """Compute 6 vertices of a regular hexagon."""
    a = np.radians(a_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def _check_separation(hex1, hex2, tol=1e-6):
    """Check if two unit hexagons are separated on at least one of their 12 edge normals."""
    v1 = _hex_vertices(hex1[0], hex1[1], 1.0, hex1[2])
    v2 = _hex_vertices(hex2[0], hex2[1], 1.0, hex2[2])
    
    # Check all 12 edge normals (6 from each hex)
    for v, w in [(v1, v2), (v2, v1)]:
        for i in range(6):
            e = w[(i+1) % 6] - w[i]
            n = np.array([-e[1], e[0]])
            n = n / (np.linalg.norm(n) + 1e-15)
            
            proj_v = v @ n
            proj_w = w @ n
            
            gap1 = max(proj_v) - min(proj_w)
            gap2 = max(proj_w) - min(proj_v)
            
            if gap1 > tol or gap2 > tol:
                return True
    
    return False

def _check_containment(centers, outer_side, outer_angle, tol=1e-6):
    """Check all inner hex vertices (unit side, angle 0) are inside outer hexagon."""
    outer_v = _hex_vertices(0, 0, outer_side, outer_angle)
    
    for c in centers:
        v = _hex_vertices(c[0], c[1], 1.0, 0)
        for p in v:
            inside = True
            for i in range(6):
                p1 = outer_v[i]
                p2 = outer_v[(i+1) % 6]
                e = p2 - p1
                cross = np.cross(e, p - p1)
                if cross < -tol:
                    inside = False
                    break
            if not inside:
                return False
    return True


def hexagon_packing_11():
    start_time = time.time()
    
    # Hex lattice with small epsilon gap for separation
    d_base = np.sqrt(3)
    eps = 1e-3  # small gap
    
    v1 = np.array([d_base + eps, 0.0])
    v2 = np.array([(d_base + eps)/2, 1.5 + eps/2])
    
    # Generate lattice points
    points = []
    max_range = 5
    for i in range(-max_range, max_range + 1):
        for j in range(-max_range, max_range + 1):
            pt = i * v1 + j * v2
            points.append(pt)
    points = np.array(points)
    
    # Sort by distance and take 11 closest
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    centers = points[idx[:11]]
    
    # Find minimal enclosing hexagon
    best_side = float('inf')
    best_angle = 0.0
    
    for angle_deg in np.arange(0, 30, 0.5):
        all_vertices = []
        for c in centers:
            v = _hex_vertices(c[0], c[1], 1.0, 0)
            all_vertices.append(v)
        all_vertices = np.vstack(all_vertices)
        max_dist = np.max(np.linalg.norm(all_vertices, axis=1))
        
        lo = 0.0
        hi = max_dist + 2.0
        
        for _ in range(50):
            mid = (lo + hi) / 2
            if _check_containment(centers, mid, angle_deg):
                hi = mid
            else:
                lo = mid
        
        if hi < best_side:
            best_side = hi
            best_angle = angle_deg
    
    inner_hex_data = np.zeros((11, 3))
    for i, c in enumerate(centers):
        inner_hex_data[i] = [c[0], c[1], 0.0]
    
    outer_hex_data = np.array([0.0, 0.0, float(best_angle)])
    
    return inner_hex_data, outer_hex_data, float(best_side)


# EVOLVE-BLOCK-END
