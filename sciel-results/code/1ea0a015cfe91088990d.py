import numpy as np
import time
from scipy.optimize import linprog

N = 11
_START = time.time()
_DEADLINE = 115.0  # seconds; keep safety margin under 150s kill

def _remaining():
    return _DEADLINE - (time.time() - _START)

def _hex_vertices(cx, cy, side, angle_rad):
    """Return 6 vertices of a regular hexagon centered at (cx, cy)."""
    angle = angle_rad
    verts = np.empty((6, 2))
    for i in range(6):
        theta = angle + np.pi * i / 3.0
        verts[i, 0] = cx + side * np.cos(theta)
        verts[i, 1] = cy + side * np.sin(theta)
    return verts

def _hex_edge_normals(angle_rad):
    """Return 6 outward normal vectors of a regular hexagon."""
    angle = angle_rad
    normals = np.empty((6, 2))
    for i in range(6):
        mid_angle = angle + np.pi / 6.0 + np.pi * i / 3.0
        normals[i, 0] = np.cos(mid_angle)
        normals[i, 1] = np.sin(mid_angle)
    return normals

def _sat_overlap_depth(v1, v2):
    """
    Returns penetration depth if v1 and v2 overlap, else 0.
    v1, v2: (6, 2) arrays.
    """
    # Axes from v1
    axes = []
    for i in range(6):
        p0 = v1[i]
        p1 = v1[(i + 1) % 6]
        edge = p1 - p0
        n = np.array([-edge[1], edge[0]])
        nl = np.linalg.norm(n)
        if nl > 1e-12:
            axes.append(n / nl)
    # Axes from v2
    for i in range(6):
        p0 = v2[i]
        p1 = v2[(i + 1) % 6]
        edge = p1 - p0
        n = np.array([-edge[1], edge[0]])
        nl = np.linalg.norm(n)
        if nl > 1e-12:
            axes.append(n / nl)
    
    min_pen = np.inf
    for n in axes:
        proj1 = v1 @ n
        proj2 = v2 @ n
        overlap = min(proj1.max(), proj2.max()) - max(proj1.min(), proj2.min())
        if overlap <= 0:
            return 0.0
        min_pen = min(min_pen, overlap)
    return min_pen

def _min_enclosing_side_lp(centers, inner_angles, outer_angle_rad):
    """
    Compute minimum side length of a regular hexagon (fixed orientation)
    enclosing all inner hexagons using Linear Programming.
    Returns (side, center).
    """
    sqrt3 = np.sqrt(3)
    normals = _hex_edge_normals(outer_angle_rad)
    
    # Collect all vertices
    all_points = np.vstack([_hex_vertices(centers[i][0], centers[i][1], 1.0, inner_angles[i]) 
                            for i in range(len(centers))])
    
    # A_i = max projection in direction n_i
    projections = all_points @ normals.T
    A = projections.max(axis=0)
    
    # Variables: x = [cx, cy, r]
    # Minimize r
    c_obj = np.array([0.0, 0.0, 1.0])
    
    # Constraints: n_i . c - r <= -A_i
    A_mat = np.zeros((6, 3))
    b_vec = np.zeros(6)
    
    for i in range(6):
        A_mat[i, 0] = normals[i, 0]
        A_mat[i, 1] = normals[i, 1]
        A_mat[i, 2] = -1.0
        b_vec[i] = -A[i]
        
    try:
        res = linprog(c_obj, A_ub=A_mat, b_ub=b_vec, method='highs')
        if res.status != 0:
            return float('inf'), np.array([0.0, 0.0])
            
        cx, cy, r = res.x
        side = 2 * r / sqrt3
        return side, np.array([cx, cy])
    except:
        return float('inf'), np.array([0.0, 0.0])

def _optimize_outer(centers, inner_angles):
    """Find optimal outer hexagon for fixed inner positions."""
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # Sweep angle finely
    # Using 61 steps (every 1 degree) is fast enough
    for outer_angle_deg in np.linspace(0, 60, 61):
        outer_angle_rad = np.radians(outer_angle_deg)
        side, center = _min_enclosing_side_lp(centers, inner_angles, outer_angle_rad)
        if side < best_side:
            best_side = side
            best_center = center
            best_angle = outer_angle_rad
            
    return best_side, best_center, best_angle

def _check_validity(centers, inner_angles, outer_center, outer_side, outer_angle_rad, tol=1e-7):
    """Check containment and non-overlap."""
    sqrt3 = np.sqrt(3)
    outer_normals = _hex_edge_normals(outer_angle_rad)
    inradius = outer_side * sqrt3 / 2
    
    # Check containment
    for i in range(len(centers)):
        verts = _hex_vertices(centers[i][0], centers[i][1], 1.0, inner_angles[i])
        for vx, vy in verts:
            for nx, ny in outer_normals:
                dist = (vx - outer_center[0]) * nx + (vy - outer_center[1]) * ny
                if dist > inradius + tol:
                    return False
    
    # Check overlap
    for i in range(N):
        v1 = _hex_vertices(centers[i][0], centers[i][1], 1.0, inner_angles[i])
        for j in range(i + 1, N):
            v2 = _hex_vertices(centers[j][0], centers[j][1], 1.0, inner_angles[j])
            if _sat_overlap_depth(v1, v2) > tol:
                return False
                
    return True

def _get_valid_fallback():
    """Return a known valid, non-overlapping configuration."""
    centers = []
    angles = [0.0] * N
    x = 0
    y = 0
    for i in range(N):
        centers.append([x, y])
        x += 4.0
        if i % 2 == 1:
            y += 4.0
            x = -4.0 + (i // 2) * 8.0
    
    side, center, angle = _optimize_outer(centers, angles)
    return centers, angles, center, side, angle

def _force_directed_relax(centers, angles, outer_side, outer_center, outer_angle_rad, max_iter=1000):
    """
    Force-directed relaxation: push overlapping hexagons apart,
    pull them toward center if outside boundary.
    """
    centers = [list(c) for c in centers]
    angles = list(angles)
    sqrt3 = np.sqrt(3)
    outer_normals = _hex_edge_normals(outer_angle_rad)
    inradius = outer_side * sqrt3 / 2
    
    for it in range(max_iter):
        if _remaining() < 2.0:
            break
        displacements = [[0.0, 0.0] for _ in range(N)]
        
        # Repulsion between overlapping hexagons
        for i in range(N):
            v1 = _hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i])
            for j in range(i + 1, N):
                v2 = _hex_vertices(centers[j][0], centers[j][1], 1.0, angles[j])
                pen = _sat_overlap_depth(v1, v2)
                if pen > 0:
                    # Push apart along center-to-center vector
                    dx = centers[j][0] - centers[i][0]
                    dy = centers[j][1] - centers[i][1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < 1e-12:
                        dx, dy = 1.0, 0.0
                        dist = 1.0
                    else:
                        dx /= dist
                        dy /= dist
                    
                    # Use SAT to find minimum penetration axis? 
                    # For speed, just use center vector direction.
                    # A more precise method would find the separating axis.
                    # Let's use a simple gradient step.
                    step = 0.05 * pen
                    displacements[i][0] -= dx * step
                    displacements[i][1] -= dy * step
                    displacements[j][0] += dx * step
                    displacements[j][1] += dy * step
        
        # Attraction to center if outside
        for i in range(N):
            verts = _hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i])
            max_dist = 0
            for vx, vy in verts:
                for nx, ny in outer_normals:
                    dist = (vx - outer_center[0]) * nx + (vy - outer_center[1]) * ny
                    if dist > max_dist:
                        max_dist = dist
            
            if max_dist > inradius:
                # Pull towards center
                dx = centers[i][0] - outer_center[0]
                dy = centers[i][1] - outer_center[1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist > 1e-12:
                    dx /= dist
                    dy /= dist
                else:
                    dx, dy = 0, 0
                step = 0.05 * (max_dist - inradius)
                displacements[i][0] -= dx * step
                displacements[i][1] -= dy * step
        
        # Apply displacements
        for i in range(N):
            centers[i][0] += displacements[i][0]
            centers[i][1] += displacements[i][1]
            
        # Check convergence
        max_disp = max(np.linalg.norm(d) for d in displacements)
        if max_disp < 1e-6:
            break
            
    return centers, angles

def construct():
    t0 = time.time()
    rng = np.random.default_rng(42)
    
    best_side = float('inf')
    best_result = None
    
    # Initial layout: hexagonal cluster
    def init_layout():
        pts = [(0.0, 0.0, 0.0)]
        for k in range(6):
            a = k * np.pi / 3.0
            pts.append((2.05 * np.cos(a), 2.05 * np.sin(a), 0.0))
        for k in range(5):
            a = (k + 0.5) * np.pi / 3.0
            pts.append((4.1 * np.cos(a), 4.1 * np.sin(a), 0.0))
        pts = pts[:N]
        return pts

    # Try a few random starts with force-directed relaxation
    for start in range(20):
        if _remaining() < 5.0:
            break
        
        positions = init_layout()
        # Add some noise
        for i in range(N):
            positions[i][0] += rng.standard_normal() * 0.5
            positions[i][1] += rng.standard_normal() * 0.5
            positions[i][2] += rng.standard_normal() * 0.2
        
        centers = [list(p[:2]) for p in positions]
        angles = [p[2] for p in positions]
        
        # Initial outer hexagon
        side, center, angle = _optimize_outer(centers, angles)
        
        # Force-directed relaxation
        centers, angles = _force_directed_relax(centers, angles, side, center, angle, max_iter=500)
        
        # Re-optimize outer hexagon
        side, center, angle = _optimize_outer(centers, angles)
        
        # Verify validity
        if _check_validity(centers, angles, center, side, angle):
            if side < best_side:
                best_side = side
                best_result = (centers, angles, center, side, angle)
        
        # If we have a good result, try to refine with more iterations
        if side < 6.0 and _remaining() > 10.0:
            centers2, angles2 = _force_directed_relax(centers, angles, side, center, angle, max_iter=1000)
            side2, center2, angle2 = _optimize_outer(centers2, angles2)
            if _check_validity(centers2, angles2, center2, side2, angle2):
                if side2 < best_side:
                    best_side = side2
                    best_result = (centers2, angles2, center2, side2, angle2)

    if best_result is None:
        # Fallback
        centers, angles, center, side, angle = _get_valid_fallback()
        best_result = (centers, angles, center, side, angle)
        best_side = side

    centers, angles, center, side, angle = best_result
    
    # Convert to required output format
    inner = []
    for i in range(N):
        x, y = centers[i]
        a = angles[i]
        inner.append([float(x), float(y), float(np.degrees(a))])
    
    return inner, (float(center[0]), float(center[1])), float(side), float(np.degrees(angle))

if __name__ == "__main__":
    construct()
