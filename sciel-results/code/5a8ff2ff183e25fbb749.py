import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

def _hex_vertices(cx, cy, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    vertices = []
    for i in range(6):
        theta = angle_rad + i * np.pi / 3
        x = cx + np.cos(theta)
        y = cy + np.sin(theta)
        vertices.append((x, y))
    return np.array(vertices)

def _get_outer_normals(angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    normals = []
    for i in range(6):
        theta = angle_rad + np.pi/6 + i * np.pi / 3
        normals.append((np.cos(theta), np.sin(theta)))
    return np.array(normals)

def _min_enclosing_hex_side(composite_verts, outer_angle_deg, outer_center):
    # composite_verts: (N, 2) array of all vertices of the composite shape
    # outer_center: (2,) array, center of outer hexagon
    # Returns the minimum side length of outer hexagon with given rotation that encloses composite shape
    normals = _get_outer_normals(outer_angle_deg)
    # Translate composite vertices relative to outer center
    rel_verts = composite_verts - outer_center
    # Project onto all 6 normals
    # dots: (N, 6)
    dots = rel_verts @ normals.T
    # For each normal, we need max(dots) - min(dots) <= S * sqrt(3)
    # Actually, the constraint is: for each normal n, max(v·n) - min(v·n) <= S * sqrt(3)
    # Wait, that's not right. The hexagon has 6 sides. Each side is defined by a normal and offset.
    # For a centered hexagon with side S, the distance from center to each side is S*sqrt(3)/2.
    # So the constraint is: max(v·n) <= S*sqrt(3)/2 AND min(v·n) >= -S*sqrt(3)/2
    # Which means: max(|v·n|) <= S*sqrt(3)/2
    # So S >= 2 * max(|v·n|) / sqrt(3)
    max_proj = np.max(np.abs(dots))
    return 2 * max_proj / np.sqrt(3)

def _compute_composite_hull(centers, angles):
    # Compute convex hull of all vertices of all inner hexagons
    all_verts = []
    for i, (cx, cy) in enumerate(centers):
        verts = _hex_vertices(cx, cy, angles[i])
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    # Compute convex hull
    hull = ConvexHull(all_verts)
    hull_verts = all_verts[hull.vertices]
    return hull_verts

def _objective(vars, centers, angles):
    # vars: [outer_cx, outer_cy, outer_angle]
    outer_cx, outer_cy, outer_angle = vars
    outer_center = np.array([outer_cx, outer_cy])
    # Compute composite hull
    composite = _compute_composite_hull(centers, angles)
    # Compute enclosing side length
    S = _min_enclosing_hex_side(composite, outer_angle, outer_center)
    return S

def construct():
    start_time = time.time()
    DEADLINE = 140.0
    
    # Initial configuration: 1 center + 6 ring + 4 outer ring
    eps = 1e-9
    centers_list = []
    angles_list = []
    
    # Center hexagon
    centers_list.append([0.0, 0.0])
    angles_list.append(30.0)
    
    # Inner ring: 6 hexagons at distance sqrt(3) from center
    for k in range(6):
        angle = np.deg2rad(k * 60)
        r = np.sqrt(3) + eps
        centers_list.append([r * np.cos(angle), r * np.sin(angle)])
        angles_list.append(30.0)
    
    # Outer ring: 4 hexagons at distance 3 from center, at 30, 90, 150, 210 degrees
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        r = 3.0 + 2 * eps
        centers_list.append([r * np.cos(angle), r * np.sin(angle)])
        angles_list.append(30.0)
        
    centers = np.array(centers_list)
    angles = np.array(angles_list)
    
    # Initial outer hexagon: centered at origin, angle 30
    x0 = np.array([0.0, 0.0, 30.0])
    
    best_S = None
    best_vars = x0
    
    # Run Nelder-Mead optimization
    try:
        res = minimize(_objective, x0, args=(centers, angles), method='Nelder-Mead', 
                       options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8})
        if time.time() < DEADLINE - 5.0:
            if res.fun < (best_S if best_S is not None else float('inf')):
                best_S = res.fun
                best_vars = res.x
    except Exception:
        pass
    
    # Try multiple initial angles for outer hexagon
    for init_angle in [0.0, 15.0, 30.0, 45.0, 60.0]:
        if time.time() > DEADLINE - 10.0:
            break
        x0_trial = np.array([0.0, 0.0, init_angle])
        try:
            res = minimize(_objective, x0_trial, args=(centers, angles), method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
            if time.time() < DEADLINE - 5.0:
                if res.fun < (best_S if best_S is not None else float('inf')):
                    best_S = res.fun
                    best_vars = res.x
        except Exception:
            pass
    
    # Final result
    outer_cx, outer_cy, outer_angle = best_vars
    outer_center = (outer_cx, outer_cy)
    
    inner = []
    for i in range(len(centers)):
        inner.append([centers[i][0], centers[i][1], angles[i]])
    
    side = best_S if best_S is not None else _min_enclosing_hex_side(
        _compute_composite_hull(centers, angles), 30.0, np.array([0.0, 0.0]))
    
    return inner, outer_center, side, outer_angle
