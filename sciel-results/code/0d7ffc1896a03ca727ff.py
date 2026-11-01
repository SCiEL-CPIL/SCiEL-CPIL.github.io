"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize_scalar

N = 12

# EVOLVE-BLOCK-START
def _hexagon_vertices(cx, cy, side, angle_deg=0.0):
    """Return vertices of a regular hexagon centered at (cx, cy) with given side and rotation."""
    angle = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = angle + i * np.pi / 3
        verts.append([cx + side * np.cos(theta), cy + side * np.sin(theta)])
    return np.array(verts)

def _point_in_hexagon(pt, center, side, angle_deg):
    """Check if a point is inside a regular hexagon."""
    center = np.array(center)
    pt = np.array(pt)
    verts = _hexagon_vertices(center[0], center[1], side, angle_deg)
    
    for i in range(6):
        v1 = verts[i]
        v2 = verts[(i + 1) % 6]
        edge = v2 - v1
        normal = np.array([-edge[1], edge[0]])
        mid_edge = (v1 + v2) / 2
        center_to_mid = mid_edge - center
        if np.dot(normal, center_to_mid) < 0:
            normal = -normal
        if np.dot(pt - v1, normal) > 1e-9:
            return False
    return True

def _hexagon_fits(inner_verts, outer_center, outer_side, outer_angle_deg):
    """Check if all vertices of inner hexagon are inside outer hexagon."""
    for pt in inner_verts:
        if not _point_in_hexagon(pt, outer_center, outer_side, outer_angle_deg):
            return False
    return True

def _honeycomb_positions():
    """Generate positions for 12 unit hexagons in a honeycomb pattern."""
    d = np.sqrt(3)  # flat-to-flat distance for unit hexagons
    
    positions = []
    # Center
    positions.append((0.0, 0.0))
    
    # First ring: 6 hexagons at distance d
    for i in range(6):
        angle = i * np.pi / 3
        x = d * np.cos(angle)
        y = d * np.sin(angle)
        positions.append((x, y))
    
    # Second ring: 5 more hexagons at distance 2*d
    for i in range(6):
        angle = i * np.pi / 3
        x = 2 * d * np.cos(angle)
        y = 2 * d * np.sin(angle)
        positions.append((x, y))
        if len(positions) == 12:
            break
    
    return np.array(positions)

def _min_enclosing_hex_side_for_hexagons(inner_positions, inner_angle_deg, outer_angle_deg):
    """Find minimum side length of outer hexagon enclosing all inner hexagons."""
    # Generate all inner hexagon vertices
    all_verts = []
    for pos in inner_positions:
        verts = _hexagon_vertices(pos[0], pos[1], 1.0, inner_angle_deg)
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    # Binary search on outer side length
    def check(side):
        # Center outer hexagon at centroid of inner hexagons
        center = np.mean(inner_positions, axis=0)
        return _hexagon_fits(all_verts, center, side, outer_angle_deg)
    
    lo, hi = 0.0, 20.0
    for _ in range(50):
        mid = (lo + hi) / 2
        if check(mid):
            hi = mid
        else:
            lo = mid
    
    return hi

def construct():
    base_positions = _honeycomb_positions()
    
    # Optimize over rotation of the cluster
    def objective(theta):
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        rotated = np.column_stack([
            base_positions[:, 0] * cos_t - base_positions[:, 1] * sin_t,
            base_positions[:, 0] * sin_t + base_positions[:, 1] * cos_t
        ])
        
        # Try several outer hexagon orientations
        min_side = float('inf')
        for outer_angle in range(0, 30, 3):
            side = _min_enclosing_hex_side_for_hexagons(rotated, 0.0, float(outer_angle))
            min_side = min(min_side, side)
        
        return min_side
    
    result = minimize_scalar(objective, bounds=(0, np.pi), method='bounded', 
                            options={'xatol': 1e-4, 'maxiter': 100})
    
    best_theta = result.x
    cos_t, sin_t = np.cos(best_theta), np.sin(best_theta)
    rotated = np.column_stack([
        base_positions[:, 0] * cos_t - base_positions[:, 1] * sin_t,
        base_positions[:, 0] * sin_t + base_positions[:, 1] * cos_t
    ])
    
    # Fine-tune outer angle
    best_side = float('inf')
    best_outer_angle = 0
    for outer_angle in range(0, 30, 1):
        side = _min_enclosing_hex_side_for_hexagons(rotated, 0.0, float(outer_angle))
        if side < best_side:
            best_side = side
            best_outer_angle = outer_angle
    
    # Refine
    def obj_outer(theta):
        return _min_enclosing_hex_side_for_hexagons(rotated, 0.0, float(theta))
    
    result2 = minimize_scalar(obj_outer, bounds=(best_outer_angle - 2, best_outer_angle + 2), 
                             method='bounded', options={'xatol': 1e-4})
    best_outer_angle = result2.x
    
    # Create inner hexagons
    inner = []
    for i in range(len(rotated)):
        x, y = rotated[i]
        inner.append([float(x), float(y), 0.0])
    
    center = (float(np.mean(rotated[:, 0])), float(np.mean(rotated[:, 1])))
    
    return inner, center, float(best_side), float(best_outer_angle)
# EVOLVE-BLOCK-END
