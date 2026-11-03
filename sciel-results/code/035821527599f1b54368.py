import numpy as np
import time
from scipy.optimize import minimize

N = 12
TIME_LIMIT = 145.0  # Safety margin for 150s limit
START_TIME = time.time()

def _hex_vertices(center, side, angle_deg):
    """Return vertices of a regular hexagon."""
    x, y = center
    theta = np.deg2rad(angle_deg)
    angles = theta + np.arange(6) * (np.pi / 3)
    return np.array([
        [x + side * np.cos(a), y + side * np.sin(a)] 
        for a in angles
    ])

def _separating_axis_check(v1, v2):
    """Check if two convex polygons are disjoint using separating axis theorem."""
    n1, n2 = len(v1), len(v2)
    
    # Check axes from both polygons
    for i in range(n1):
        edge = v1[(i+1) % n1] - v1[i]
        normal = np.array([-edge[1], edge[0]])
        norm = np.linalg.norm(normal)
        if norm < 1e-12:
            continue
        normal = normal / norm
        
        # Project both polygons onto this normal
        proj1 = np.array([np.dot(p, normal) for p in v1])
        proj2 = np.array([np.dot(p, normal) for p in v2])
        
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 < min2 - 1e-9 or max2 < min1 - 1e-9:
            return True  # Separating axis found, no overlap
    
    return False  # No separating axis, polygons overlap

def _is_valid_fast(inner_hexes, outer_side, outer_center, outer_angle, eps=1e-9):
    """Fast validity check."""
    outer_verts = _hex_vertices(outer_center, outer_side, outer_angle)
    inner_verts = [_hex_vertices(h[0:2], 1.0, h[2]) for h in inner_hexes]
    
    # Check disjointness
    for i in range(len(inner_hexes)):
        for j in range(i + 1, len(inner_hexes)):
            if not _separating_axis_check(inner_verts[i], inner_verts[j]):
                return False, "overlap"
        
        # Check containment: all vertices of inner hex must be inside outer hex
        for p in inner_verts[i]:
            inside = True
            for k in range(6):
                a = outer_verts[k]
                b = outer_verts[(k+1)%6]
                cross_ab = (b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0])
                if cross_ab < -eps:
                    inside = False
                    break
            if not inside:
                return False, "outside"
                
    return True, "valid"

def _outer_hex_side(centers, angles, outer_angle=0.0):
    """Compute the minimum outer side length required to contain all hexagons."""
    max_proj = 0.0
    # Outer hexagon normals (6 directions)
    for k in range(6):
        angle = np.deg2rad(outer_angle + k * 60)
        normal = np.array([np.cos(angle), np.sin(angle)])
        
        for i in range(N):
            verts = _hex_vertices(centers[i], 1.0, angles[i])
            for v in verts:
                proj = np.dot(v, normal)
                if proj > max_proj:
                    max_proj = proj
                    
    # For a regular hexagon with side length S, the apothem is S * sqrt(3)/2
    # The max projection from center to any point must be <= apothem
    # So apothem >= max_proj, which means S * sqrt(3)/2 >= max_proj
    # S >= max_proj * 2 / sqrt(3)
    return max_proj * 2.0 / np.sqrt(3.0)

def _sdf_distance(centers, angles, outer_side, outer_center, outer_angle):
    """
    Compute a smooth penalty based on SDF-like distances.
    Returns a scalar penalty value.
    """
    penalty = 0.0
    outer_verts = _hex_vertices(outer_center, outer_side, outer_angle)
    inner_verts = [_hex_vertices(centers[i], 1.0, angles[i]) for i in range(N)]
    
    # Pairwise overlap penalty
    for i in range(N):
        for j in range(i + 1, N):
            v1, v2 = inner_verts[i], inner_verts[j]
            # Check if they overlap
            if not _separating_axis_check(v1, v2):
                # Compute a rough overlap measure
                # Use distance between centers as proxy
                d = np.linalg.norm(centers[i] - centers[j])
                # For two unit hexagons, minimum distance between centers when touching is ~sqrt(3)
                # But it depends on orientation. Use a simpler penalty.
                # Penalty grows as centers get closer
                min_dist = 1.732  # sqrt(3), approximate touching distance
                if d < min_dist:
                    penalty += (min_dist - d) ** 2 * 10.0
    
    # Containment penalty
    for i in range(N):
        for v in inner_verts[i]:
            # Check if v is inside outer hex
            for k in range(6):
                a = outer_verts[k]
                b = outer_verts[(k+1)%6]
                edge = b - a
                normal = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(normal)
                if norm < 1e-12:
                    continue
                normal = normal / norm
                # Ensure normal points inward
                mid = (a + b) / 2
                if np.dot(normal, mid) > 0:
                    normal = -normal
                dist = np.dot(v - a, normal)
                if dist < 0:
                    penalty += (-dist) ** 2 * 10.0
                    break
                    
    return penalty

def _combined_objective(x, outer_angle, outer_center):
    """
    Objective function for optimization.
    x: 2N vector of inner hexagon centers (x1, y1, x2, y2, ..., xN, yN)
    Returns: (value, gradient) - but we'll use numerical gradient for simplicity
    """
    centers = x.reshape(N, 2)
    # Angles are fixed at 0 for this formulation (simplification)
    angles = np.zeros(N)
    
    # Compute required outer side
    req_side = _outer_hex_side(centers, angles, outer_angle)
    
    # Add penalty for overlaps
    penalty = _sdf_distance(centers, angles, req_side, outer_center, outer_angle)
    
    value = req_side + penalty
    return value

def construct():
    start_time = time.time()
    
    best_side = float('inf')
    best_inner = None
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    np.random.seed(42)
    
    # Try different outer angles
    for outer_angle in [0.0, 30.0]:
        if time.time() - start_time > TIME_LIMIT - 15:
            break
            
        # Multiple random initializations
        for restart in range(20):
            if time.time() - start_time > TIME_LIMIT - 5:
                break
                
            # Random initial positions
            if restart == 0:
                # Hexagonal lattice initial
                pts = [[0,0]]
                for k in range(6):
                    angle = np.pi/3 * k
                    pts.append([np.sqrt(3)*np.cos(angle), np.sqrt(3)*np.sin(angle)])
                for k in range(6):
                    angle = np.pi/3 * k + np.pi/6
                    pts.append([2*np.sqrt(3)*np.cos(angle), 2*np.sqrt(3)*np.sin(angle)])
                    if len(pts) == 12:
                        break
                centers = np.array(pts)
            else:
                # Random positions in a disk
                r = np.sqrt(np.random.uniform(0, 16))
                theta = np.random.uniform(0, 2*np.pi)
                centers = np.array([[r*np.cos(theta), r*np.sin(theta)] for _ in range(N)])
            
            x0 = centers.flatten()
            
            # Optimize
            try:
                result = minimize(
                    _combined_objective,
                    x0,
                    args=(outer_angle, (0.0, 0.0)),
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                if result.success or result.fun < best_side:
                    best_centers = result.x.reshape(N, 2)
                    angles = np.zeros(N)
                    req_side = _outer_hex_side(best_centers, angles, outer_angle)
                    
                    inner_hexes = [[best_centers[i][0], best_centers[i][1], 0.0] for i in range(N)]
                    is_valid, reason = _is_valid_fast(inner_hexes, req_side, (0.0, 0.0), outer_angle)
                    
                    if is_valid and req_side < best_side:
                        best_side = req_side
                        best_inner = inner_hexes
                        best_outer_center = (0.0, 0.0)
                        best_outer_angle = outer_angle
                        
            except Exception:
                pass
                
    if best_inner is not None:
        return (best_inner, best_outer_center, best_side, best_outer_angle)
    
    # Fallback
    return [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)], (0.0, 0.0), 40.0, 0.0
