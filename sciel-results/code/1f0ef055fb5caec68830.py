"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 12

# EVOLVE-BLOCK-START
def _hexagon_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon with center (cx,cy), side length, and rotation."""
    angle = np.radians(angle_deg)
    # Regular hexagon vertices at angles 0, 60, 120, 180, 240, 300 degrees
    angles = angle + np.arange(6) * np.pi / 3
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])


def _min_enclosing_hexagon(vertices_all):
    """
    Given a set of points (vertices of all inner hexagons), find the minimum side length
    of a regular hexagon (with arbitrary center and rotation) that contains all points.
    
    Returns (center_x, center_y, side_length, rotation_degrees).
    """
    points = np.vstack(vertices_all)
    n_pts = len(points)
    
    if n_pts == 0:
        return (0.0, 0.0, 0.0, 0.0)
    
    # For a regular hexagon with side length s and rotation theta, centered at (cx, cy),
    # the constraint is that for each of the 6 face normals, the projection of each point
    # onto that normal must be within [-s*sqrt(3)/2, s*sqrt(3)/2] relative to center.
    
    # Actually, for a regular hexagon with side length s, the distance from center to each
    # face is s * sqrt(3) / 2 (the apothem). The 6 face normals are at angles theta + k*60.
    
    # So for each point p, and each face normal n_k (unit vector), we need:
    #   -apothem <= dot(p - c, n_k) <= apothem
    # where apothem = s * sqrt(3) / 2
    
    # This means: for each k, max_i dot(p_i, n_k) - min_i dot(p_i, n_k) <= 2 * apothem = s * sqrt(3)
    # And the center c must satisfy: for each k, (max_i dot(p_i, n_k) + min_i dot(p_i, n_k)) / 2 
    #   is the projection of c onto n_k... wait, that's not quite right.
    
    # Let me reconsider. For a fixed rotation theta, the hexagon is defined by 6 half-planes.
    # The minimal side length for a given theta is determined by:
    #   For each face direction k (normal n_k), the width of the point set in direction n_k
    #   must be <= s * sqrt(3)
    #   So s >= max_k (width_k) / sqrt(3)
    #   where width_k = max_i dot(p_i, n_k) - min_i dot(p_i, n_k)
    
    # And the center is at the midpoint of the bounding box in each direction:
    #   dot(c, n_k) = (max_i dot(p_i, n_k) + min_i dot(p_i, n_k)) / 2
    
    # For a given theta, we can compute s(theta) = max_k width_k(theta) / sqrt(3)
    # and find the theta that minimizes this.
    
    # The width in direction n(theta + k*pi/3) is a periodic function of theta with period pi/3.
    # We can optimize over theta in [0, pi/3).
    
    def side_length_for_rotation(theta):
        normals = []
        for k in range(6):
            angle = theta + k * np.pi / 3
            normals.append(np.array([np.cos(angle), np.sin(angle)]))
        normals = np.array(normals)  # shape (6, 2)
        
        # Project all points onto each normal
        projections = points @ normals.T  # shape (n_pts, 6)
        
        widths = projections.max(axis=0) - projections.min(axis=0)  # shape (6,)
        return widths.max() / np.sqrt(3)
    
    # Optimize theta
    theta0 = 0.0
    result = minimize(side_length_for_rotation, theta0, method='Nelder-Mead',
                      options={'xatol': 1e-12, 'fatol': 1e-12, 'maxiter': 10000})
    
    # Also try multiple starting points for robustness
    best_theta = theta0
    best_side = result.fun
    for start in np.linspace(0, np.pi/3, 13)[:-1]:
        r = minimize(side_length_for_rotation, start, method='Nelder-Mead',
                     options={'xatol': 1e-12, 'fatol': 1e-12, 'maxiter': 5000})
        if r.fun < best_side:
            best_side = r.fun
            best_theta = r.x[0]
    
    # Compute center from the optimal rotation
    theta = best_theta
    normals = []
    for k in range(6):
        angle = theta + k * np.pi / 3
        normals.append(np.array([np.cos(angle), np.sin(angle)]))
    normals = np.array(normals)
    
    projections = points @ normals.T
    midpoints = (projections.max(axis=0) + projections.min(axis=0)) / 2  # shape (6,)
    
    # Solve for center: normals @ c = midpoints
    # normals is 6x2, midpoints is 6x1, c is 2x1
    # Least squares solution
    c, _, _, _ = np.linalg.lstsq(normals, midpoints, rcond=None)
    
    return (c[0], c[1], best_side, np.degrees(theta))


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Pre-pack 12 unit hexagons in a 3x4 grid arrangement
    # A regular hexagon with side 1 has:
    # - Width (flat-to-flat) = sqrt(3)
    # - Height (point-to-point) = 2
    # 
    # For a honeycomb packing, adjacent hexagons in a row are separated by sqrt(3) in x,
    # and rows are offset by sqrt(3)/2 in x and 1.5 in y (for pointy-top) or similar.
    #
    # Let's use a flat-top orientation (angle=0 means flat side on top/bottom).
    # Actually, let's use pointy-top: vertices at 30, 90, 150, 210, 270, 330 degrees.
    # For pointy-top hexagons with side 1:
    # - Width = sqrt(3)
    # - Height = 2
    # - Horizontal spacing between centers in same row: sqrt(3)
    # - Vertical spacing between rows: 1.5
    # - Odd rows offset by sqrt(3)/2
    
    # Let's create a 3-column by 4-row grid
    sqrt3 = np.sqrt(3)
    
    inner_positions = []
    for row in range(4):
        for col in range(3):
            x = col * sqrt3 + (row % 2) * (sqrt3 / 2)
            y = row * 1.5
            inner_positions.append([x, y, 0.0])
    
    # Now optimize the global transform: translate, rotate, and scale
    # The outer hexagon will be centered at origin, and we'll find the minimal enclosing hexagon.
    # We can fix the outer hexagon's rotation and center, and optimize the inner cluster's
    # position and rotation.
    
    # Actually, by symmetry, we can fix the outer hexagon at origin with rotation 0,
    # and optimize the inner cluster's rotation and translation.
    
    def objective(params):
        if time.time() > deadline:
            return 1e10
        rot_deg, tx, ty = params
        rot_rad = np.radians(rot_deg)
        cos_r, sin_r = np.cos(rot_rad), np.sin(rot_rad)
        
        # Transform inner positions
        transformed = []
        for px, py, _ in inner_positions:
            rx = cos_r * px - sin_r * py + tx
            ry = sin_r * px + cos_r * py + ty
            transformed.append([rx, ry, 0.0])
        
        # Compute vertices of all transformed hexagons
        all_vertices = []
        for x, y, ang in transformed:
            verts = _hexagon_vertices(x, y, 1.0, ang)
            all_vertices.append(verts)
        
        # Find minimum enclosing hexagon
        cx, cy, side, theta = _min_enclosing_hexagon(all_vertices)
        return side
    
    # Initial guess
    x0 = [0.0, 0.0, 0.0]
    
    # Use differential evolution for global search
    bounds = [(-180, 180), (-10, 10), (-10, 10)]
    
    best_result = None
    best_side = 1e10
    
    # Quick local optimization first
    try:
        result = minimize(objective, x0, method='Nelder-Mead',
                          options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 2000})
        if result.fun < best_side:
            best_side = result.fun
            best_result = result.x
    except:
        pass
    
    if time.time() > deadline - 30:
        # Not enough time for DE, just return best
        if best_result is not None:
            rot_deg, tx, ty = best_result
            rot_rad = np.radians(rot_deg)
            cos_r, sin_r = np.cos(rot_rad), np.sin(rot_rad)
            final_inner = []
            for px, py, _ in inner_positions:
                rx = cos_r * px - sin_r * py + tx
                ry = sin_r * px + cos_r * py + ty
                final_inner.append([rx, ry, 0.0])
            # Compute final enclosing hexagon
            all_vertices = []
            for x, y, ang in final_inner:
                verts = _hexagon_vertices(x, y, 1.0, ang)
                all_vertices.append(verts)
            cx, cy, side, theta = _min_enclosing_hexagon(all_vertices)
            return final_inner, (cx, cy), side, theta
        else:
            return inner_positions, (0.0, 0.0), 10.0, 0.0
    
    # Differential evolution
    try:
        de_result = differential_evolution(objective, bounds, seed=42, maxiter=50,
                                            tol=1e-8, polish=False, workers=1)
        if de_result.fun < best_side:
            best_side = de_result.fun
            best_result = de_result.x
    except:
        pass
    
    # Final polish
    if best_result is not None:
        try:
            polish_result = minimize(objective, best_result, method='Nelder-Mead',
                                     options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 5000})
            if polish_result.fun < best_side:
                best_side = polish_result.fun
                best_result = polish_result.x
        except:
            pass
    
    if best_result is None:
        best_result = x0
        best_side = 10.0
    
    rot_deg, tx, ty = best_result
    rot_rad = np.radians(rot_deg)
    cos_r, sin_r = np.cos(rot_rad), np.sin(rot_rad)
    final_inner = []
    for px, py, _ in inner_positions:
        rx = cos_r * px - sin_r * py + tx
        ry = sin_r * px + cos_r * py + ty
        final_inner.append([rx, ry, 0.0])
    
    # Compute final enclosing hexagon
    all_vertices = []
    for x, y, ang in final_inner:
        verts = _hexagon_vertices(x, y, 1.0, ang)
        all_vertices.append(verts)
    cx, cy, side, theta = _min_enclosing_hexagon(all_vertices)
    
    return final_inner, (cx, cy), side, theta
# EVOLVE-BLOCK-END
