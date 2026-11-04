"""Optimized packing of 11 unit hexagons in minimal enclosing hexagon."""
import numpy as np
from scipy.optimize import minimize
import time

N = 11

# EVOLVE-BLOCK-START

def _hex_vertices(cx, cy, s, angle_rad):
    """Return 6 vertices of a regular hexagon centered at (cx,cy), side s, rotation angle_rad."""
    # Regular hexagon vertices at angles 0, 60, 120, 180, 240, 300 degrees from center
    # Circumradius = s for a regular hexagon
    angles = angle_rad + np.array([0, 1, 2, 3, 4, 5]) * np.pi / 3.0
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])


def _min_enclosing_hexagon_side(points, outer_center, outer_angle_rad):
    """
    Given a set of 2D points and an outer hexagon center/angle,
    compute the minimum side length S of the outer hexagon (regular, same angle)
    such that all points are inside.
    
    For a regular hexagon with center (cx,cy), side S, rotation phi:
    The 6 outward normals are at angles phi + k*pi/3 for k=0..5
    The apothem (distance from center to each side) = S * sqrt(3) / 2
    A point p is inside iff for all 6 normals n_k:
        dot(p - center, n_k) <= apothem
    So apothem >= max over all points and all normals of dot(p-center, n_k)
    Thus S = 2/sqrt(3) * max_k max_i dot(p_i - center, n_k)
    """
    cx, cy = outer_center
    # 6 outward normals
    normal_angles = outer_angle_rad + np.arange(6) * np.pi / 3.0
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    # Points relative to center
    rel = points - np.array([cx, cy])
    
    # Projections: shape (n_points, 6)
    proj = rel @ normals.T
    
    # For each normal direction, the max projection
    max_proj = np.max(proj, axis=0)  # shape (6,)
    
    # The apothem must be at least max(max_proj)
    # But we also need to handle negative projections (points on the "inside" side)
    # Actually, the constraint is dot(p - c, n_k) <= apothem for ALL k
    # So apothem >= max over k of max_i dot(p_i - c, n_k)
    # But if all points are on one side, some normals will have negative max_proj,
    # which is fine - we just need apothem >= the maximum of all 6 max_projs
    
    apothem = np.max(max_proj)
    # Apothem must be non-negative
    apothem = max(apothem, 0.0)
    
    S = 2.0 / np.sqrt(3.0) * apothem
    return S


def _all_inner_hexagon_points(positions, angles_deg, side=1.0):
    """
    positions: (N, 2) array of centers
    angles_deg: (N,) array of rotation angles in degrees
    Returns all vertices of all N hexagons as (6*N, 2) array.
    """
    all_pts = []
    for i in range(len(positions)):
        cx, cy = positions[i]
        ang = np.radians(angles_deg[i])
        verts = _hex_vertices(cx, cy, side, ang)
        all_pts.append(verts)
    return np.vstack(all_pts)


def _cost(params, outer_center, outer_angle_rad, inner_points):
    """
    Cost function: given inner hexagon centers/angles (already encoded in inner_points),
    return the side length of the enclosing hexagon.
    """
    return _min_enclosing_hexagon_side(inner_points, outer_center, outer_angle_rad)


def _params_to_points(params):
    """
    params: flat array of length 3*N = [x0,y0,ang0, x1,y1,ang1, ...]
    Returns (positions, angles_deg, all_points)
    """
    N_local = len(params) // 3
    positions = np.zeros((N_local, 2))
    angles_deg = np.zeros(N_local)
    for i in range(N_local):
        positions[i, 0] = params[3*i]
        positions[i, 1] = params[3*i+1]
        angles_deg[i] = params[3*i+2]
    all_points = _all_inner_hexagon_points(positions, angles_deg)
    return positions, angles_deg, all_points


def construct():
    start_time = time.time()
    DEADLINE = 145.0  # 5s safety margin
    
    # Initial arrangement: hexagonal cluster
    # Center + 6 around it + 4 more
    # For unit hexagons in a hexagonal lattice:
    # Nearest neighbor distance = 2 * inradius = 2 * sqrt(3)/2 = sqrt(3)
    # But actually, two unit hexagons side-by-side (sharing a side) have centers
    # distance = 2 * apothem = 2 * sqrt(3)/2 = sqrt(3)
    # Wait: apothem of unit hexagon = sqrt(3)/2, so center-to-center for adjacent = sqrt(3)
    
    # Let's use a standard hexagonal packing
    # Row 0: 3 hexagons
    # Row 1: 4 hexagons  
    # Row 2: 3 hexagons
    # Row 3: 1 hexagon
    # Total: 11
    
    # In a hexagonal lattice of unit hexagons:
    # Horizontal spacing: 2 * apothem = sqrt(3)
    # Vertical spacing between rows: 1.5 (for flat-top hexagons)
    # Actually, let me think more carefully.
    
    # For a regular hexagon with side 1, oriented with flat sides top/bottom:
    # Width (left-right) = 2 * apothem = sqrt(3)
    # Height (top-bottom) = 2 * circumradius = 2
    # 
    # In a hexagonal tiling:
    # Horizontal distance between centers in same row: sqrt(3)
    # Vertical distance between rows: 1.5
    # Offset between rows: sqrt(3)/2
    
    # Let me use pointy-top hexagons for easier lattice
    # Actually, let's just use a reasonable initial guess and optimize.
    
    # Simple initial: all in a line
    initial_params = np.zeros(3 * N)
    for i in range(N):
        initial_params[3*i] = i * 1.8  # spread out
        initial_params[3*i+1] = 0.0
        initial_params[3*i+2] = 0.0
    
    # Better initial: hexagonal cluster
    # Positions in a hexagonal pattern
    hex_positions = []
    # Center
    hex_positions.append((0.0, 0.0))
    # 6 neighbors at distance sqrt(3)
    d = np.sqrt(3)
    for k in range(6):
        ang = k * np.pi / 3.0
        hex_positions.append((d * np.cos(ang), d * np.sin(ang)))
    # 4 more: extend the hexagon
    # Second ring positions (select 4 from 12)
    d2 = 2 * d  # distance to second ring... actually second ring is at distance 2*sqrt(3)?
    # No, second ring in hex lattice: distance = 2 * sqrt(3) * cos(30°) = 2*sqrt(3)*sqrt(3)/2 = 3
    # Hmm, let me just place them reasonably
    # Add 4 more around the outside
    for k in range(4):
        ang = k * np.pi / 2.0 + np.pi/4.0
        hex_positions.append((2.5 * np.cos(ang), 2.5 * np.sin(ang)))
    
    # Take first 11
    hex_positions = hex_positions[:N]
    
    initial_params2 = np.zeros(3 * N)
    for i in range(N):
        initial_params2[3*i] = hex_positions[i][0]
        initial_params2[3*i+1] = hex_positions[i][1]
        initial_params2[3*i+2] = 0.0
    
    # Try multiple initial conditions
    best_S = float('inf')
    best_params = None
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    initial_conditions = [
        initial_params2,  # hexagonal cluster
        initial_params,   # line
        np.random.randn(3*N) * 2.0,  # random
    ]
    
    for init_p in initial_conditions:
        if time.time() - start_time > DEADLINE:
            break
        
        # Optimize inner positions/angles and outer center/angle simultaneously
        # Full parameter vector: 3*N + 2 (outer center) + 1 (outer angle) = 3*N + 3
        full_init = np.concatenate([init_p, [0.0, 0.0, 0.0]])
        
        def full_cost(full_params):
            inner_p = full_params[:3*N]
            ocx, ocy, oang = full_params[3*N], full_params[3*N+1], full_params[3*N+2]
            _, _, all_pts = _params_to_points(inner_p)
            S = _min_enclosing_hexagon_side(all_pts, (ocx, ocy), np.radians(oang))
            return S
        
        try:
            result = minimize(full_cost, full_init, method='Powell',
                            options={'maxiter': 500, 'maxfev': 2000, 'xtol': 1e-10, 'ftol': 1e-10})
            if result.fun < best_S:
                best_S = result.fun
                best_params = result.x
                best_outer_center = (result.x[3*N], result.x[3*N+1])
                best_outer_angle = result.x[3*N+2]
        except Exception:
            pass
        
        # Also try fixing outer center at origin and optimizing only inner + outer angle
        if time.time() - start_time > DEADLINE:
            break
        
        fixed_center_init = np.concatenate([init_p, [0.0, 0.0]])
        
        def fixed_center_cost(p):
            inner_p = p[:3*N]
            oang = p[3*N+1]
            _, _, all_pts = _params_to_points(inner_p)
            S = _min_enclosing_hexagon_side(all_pts, (0.0, 0.0), np.radians(oang))
            return S
        
        try:
            result2 = minimize(fixed_center_cost, fixed_center_init, method='Powell',
                              options={'maxiter': 300, 'maxfev': 1500, 'xtol': 1e-10, 'ftol': 1e-10})
            if result2.fun < best_S:
                best_S = result2.fun
                best_params = np.concatenate([result2.x[:3*N], [0.0, 0.0, result2.x[3*N+1]]])
                best_outer_center = (0.0, 0.0)
                best_outer_angle = result2.x[3*N+1]
        except Exception:
            pass
    
    # If we still have time, do a local refinement
    if time.time() - start_time < DEADLINE - 10 and best_params is not None:
        def full_cost_refine(full_params):
            inner_p = full_params[:3*N]
            ocx, ocy, oang = full_params[3*N], full_params[3*N+1], full_params[3*N+2]
            _, _, all_pts = _params_to_points(inner_p)
            S = _min_enclosing_hexagon_side(all_pts, (ocx, ocy), np.radians(oang))
            return S
        
        try:
            result3 = minimize(full_cost_refine, best_params, method='Nelder-Mead',
                             options={'maxiter': 200, 'xatol': 1e-12, 'fatol': 1e-12})
            if result3.fun < best_S:
                best_S = result3.fun
                best_params = result3.x
                best_outer_center = (result3.x[3*N], result3.x[3*N+1])
                best_outer_angle = result3.x[3*N+2]
        except Exception:
            pass
    
    # Extract final result
    if best_params is None:
        # Fallback to initial
        best_params = initial_params2
        best_outer_center = (0.0, 0.0)
        best_outer_angle = 0.0
        best_S = full_cost(np.concatenate([initial_params2, [0.0, 0.0, 0.0]]))
    
    inner_p = best_params[:3*N]
    positions, angles_deg, _ = _params_to_points(inner_p)
    
    inner = []
    for i in range(N):
        inner.append([positions[i, 0], positions[i, 1], angles_deg[i]])
    
    return inner, best_outer_center, best_S, np.degrees(best_outer_angle)

# EVOLVE-BLOCK-END
