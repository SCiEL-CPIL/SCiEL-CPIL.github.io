import numpy as np
import time
from scipy.optimize import minimize_scalar
from itertools import combinations

N = 11

def _min_enclosing_hex_side(points, angle_deg, margin=1.0):
    angle = np.radians(angle_deg)
    normal_angles = angle + np.radians(np.array([30, 90, 150, 210, 270, 330]))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    projections = points @ normals.T
    max_proj = projections.max(axis=0)
    min_proj = projections.min(axis=0)
    widths = max_proj - min_proj + 2 * margin
    return widths.max() / np.sqrt(3)

def _find_best_hex_for_points(points, margin=1.0):
    best_side = np.inf
    best_angle = 0.0
    for angle_deg in np.linspace(0, 180, 721):
        side = _min_enclosing_hex_side(points, angle_deg, margin)
        if side < best_side:
            best_side = side
            best_angle = angle_deg
    def objective(angle_deg):
        return _min_enclosing_hex_side(points, angle_deg, margin)
    try:
        lo = max(0, best_angle - 5)
        hi = min(180, best_angle + 5)
        result = minimize_scalar(objective, bounds=(lo, hi), method='bounded', options={'xatol': 1e-8})
        if result.fun < best_side:
            best_side = result.fun
            best_angle = result.x
    except:
        pass
    return best_side, best_angle

def _get_configs():
    d = np.sqrt(3)
    center = (0.0, 0.0)
    first_ring = [
        (d, 0),
        (d/2, d*np.sqrt(3)/2),
        (-d/2, d*np.sqrt(3)/2),
        (-d, 0),
        (-d/2, -d*np.sqrt(3)/2),
        (d/2, -d*np.sqrt(3)/2),
    ]
    second_ring = [
        (2*d, 0),
        (3*d/2, d*np.sqrt(3)/2),
        (d, d*np.sqrt(3)),
        (0, d*np.sqrt(3)),
        (-d, d*np.sqrt(3)),
        (-3*d/2, d*np.sqrt(3)/2),
        (-2*d, 0),
        (-3*d/2, -d*np.sqrt(3)/2),
        (-d, -d*np.sqrt(3)),
        (0, -d*np.sqrt(3)),
        (d, -d*np.sqrt(3)),
        (3*d/2, -d*np.sqrt(3)/2),
    ]
    return center, first_ring, second_ring

def construct():
    start_time = time.time()
    deadline = 145.0

    center, first_ring, second_ring = _get_configs()
    
    # Initial config: symmetric selection
    best_combo = (0, 3, 6, 9)
    centers = [center] + first_ring + [second_ring[i] for i in best_combo]
    inner = [[cx, cy, 0.0] for cx, cy in centers]
    points = np.array(centers)
    best_side, best_angle = _find_best_hex_for_points(points, margin=1.0)
    center_x, center_y = points[:, 0].mean(), points[:, 1].mean()

    # Search over all C(12,4)=495 combinations
    for combo in combinations(range(12), 4):
        if time.time() - start_time > deadline - 10:
            break
        centers_alt = [center] + first_ring + [second_ring[i] for i in combo]
        points_alt = np.array(centers_alt)
        side_alt, angle_alt = _find_best_hex_for_points(points_alt, margin=1.0)
        if side_alt < best_side:
            best_side = side_alt
            best_angle = angle_alt
            inner = [[cx, cy, 0.0] for cx, cy in centers_alt]
            center_x, center_y = points_alt[:, 0].mean(), points_alt[:, 1].mean()

    return inner, (center_x, center_y), best_side, best_angle
