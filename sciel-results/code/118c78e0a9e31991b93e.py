import numpy as np
import time
from scipy.optimize import minimize

N = 11

def _hex_vertices_batch(centers, angles_deg):
    """
    Vectorized: get vertices of multiple unit regular hexagons.
    centers: (N, 2) array, angles_deg: (N,) array
    Returns: (N, 6, 2) array of vertices
    """
    n = len(centers)
    base_angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180.0  # (6,)
    angle_rad = np.deg2rad(angles_deg)  # (N,)
    # angles: (N, 6)
    angles = angle_rad[:, np.newaxis] + base_angles[np.newaxis, :]
    cos_a = np.cos(angles)  # (N, 6)
    sin_a = np.sin(angles)  # (N, 6)
    # centers: (N, 1, 2)
    cx = centers[:, 0:1, np.newaxis]  # (N, 1)
    cy = centers[:, 1:2, np.newaxis]  # (N, 1)
    x = cx + cos_a  # (N, 6)
    y = cy + sin_a  # (N, 6)
    return np.stack([x, y], axis=-1)  # (N, 6, 2)

def _compute_max_proj(centers, angles, outer_angle_deg):
    """
    Compute max projection of all inner hexagon vertices onto the 6 outer normals.
    Returns max_proj (float).
    """
    theta_rad = np.deg2rad(outer_angle_deg)
    normals = np.array([[np.cos(theta_rad + np.deg2rad(30 + k * 60)),
                         np.sin(theta_rad + np.deg2rad(30 + k * 60))] for k in range(6)])
    
    centers_arr = np.array(centers)
    angles_arr = np.array(angles)
    verts = _hex_vertices_batch(centers_arr, angles_arr)  # (N, 6, 2)
    # Project: for each hexagon, each vertex, each normal
    # verts: (N, 6, 2), normals.T: (2, 6)
    projs = np.einsum('nij,jk->nik', verts, normals.T)  # (N, 6, 6)
    max_proj = np.max(projs)
    return max_proj

def _min_outer_side(inner_centers, inner_angles, outer_angle_deg=0.0):
    """
    Calculate the minimum side length S of the outer hexagon (centered at 0,0, rotated by outer_angle_deg)
    that contains all inner hexagons.
    """
    max_proj = _compute_max_proj(inner_centers, inner_angles, outer_angle_deg)
    S = 2.0 * max_proj / np.sqrt(3.0)
    return S

def _objective_full(x):
    """
    Full objective including outer angle.
    x = [x1,y1,a1, x2,y2,a2, ..., outer_angle_rad]
    """
    n = (len(x) - 1) // 3
    centers = [(x[3*i], x[3*i+1]) for i in range(n)]
    angles_deg = [np.rad2deg(x[3*i+2]) for i in range(n)]
    outer_ang_deg = np.rad2deg(x[-1])
    return _min_outer_side(centers, angles_deg, outer_ang_deg)

def _build_flower_config():
    """
    Construct the optimal 'flower' packing of 11 unit hexagons.
    Center at (0,0).
    6 neighbors at distance 2*sqrt(3).
    4 next-neighbors at distance 4*sqrt(3) (specifically at 0, 120, 240, 0? No, 4 specific spots).
    
    Lattice vectors for hex grid (pointy top or flat top? 
    Unit hex side 1.
    Distance between centers of touching hexes is 2*sqrt(3).
    Basis vectors for hex lattice (flat top):
    v1 = (2*sqrt(3), 0)
    v2 = (sqrt(3), 3)
    
    Wait, standard hex packing distance is 2*sqrt(3) if side is 1?
    Area of hex = 3*sqrt(3)/2 * s^2.
    Distance from center to vertex = s = 1.
    Distance from center to edge = sqrt(3)/2.
    Distance between centers of touching hexes = 2 * (sqrt(3)/2) * 2 ? No.
    Distance between centers = 2 * apothem = sqrt(3).
    
    Let's re-verify.
    Hexagon side s=1.
    Apothem a = sqrt(3)/2.
    Two hexes touching share an edge. The distance between their centers is 2 * apothem = sqrt(3).
    
    So neighbors are at distance sqrt(3).
    Basis vectors for hex lattice (flat top, horizontal rows):
    u = (sqrt(3), 0)
    v = (sqrt(3)/2, 1.5)
    
    Let's check distance v. |v|^2 = 3/4 + 2.25 = 3. |v| = sqrt(3). Correct.
    
    Flower config:
    Center: (0,0)
    Ring 1 (6 neighbors):
    (sqrt(3), 0)
    (sqrt(3)/2, 1.5)
    (-sqrt(3)/2, 1.5)
    (-sqrt(3), 0)
    (-sqrt(3)/2, -1.5)
    (sqrt(3)/2, -1.5)
    
    Ring 2 (4 neighbors to make 11 total):
    To minimize bounding box, we pick 4 from the 12 available next-neighbors.
    The 12 next-neighbors are at distance 2*sqrt(3) or sqrt(12)?
    Next neighbors are at 2*u, 2*v, 2*u-v, etc.
    Actually, just pick 4 that form a compact shape.
    Common choice: (2*sqrt(3), 0), (sqrt(3), 3), (-sqrt(3), 3), (-2*sqrt(3), 0) ?
    Let's check (2*u, 0) -> (2*sqrt(3), 0).
    (u+v) -> (1.5*sqrt(3), 1.5).
    (u-v) -> (0.5*sqrt(3), -1.5).
    
    Let's just use the lattice coordinates (i,j) and convert.
    (0,0)
    (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    Next ring:
    (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1), (2,1), (1,2)
    
    We need 4 more.
    Symmetric choice: (2,0), (1,1), (-1,1), (-2,0) ?
    Let's check bounding box of this set.
    Points:
    (0,0)
    (1.732, 0)
    (0.866, 1.5)
    (-0.866, 1.5)
    (-1.732, 0)
    (-0.866, -1.5)
    (0.866, -1.5)
    
    Next 4:
    (2,0) -> (3.464, 0)
    (1,1) -> (2.598, 1.5)
    (-1,1) -> (-2.598, 1.5)
    (-2,0) -> (-3.464, 0)
    
    Max X: 3.464 + 1 (vertex extent) = 4.464?
    Min X: -3.464 - 1 = -4.464?
    
    Let's check the enclosing hexagon side length.
    If we align the outer hexagon with the lattice (angle 0),
    The width is determined by the max projection on the 6 normals.
    
    Let's just return the coordinates and let the optimizer fine tune.
    """
    sqrt3 = np.sqrt(3)
    u = np.array([sqrt3, 0])
    v = np.array([sqrt3/2, 1.5])
    
    coords = [
        (0,0),
        (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
        (2,0), (1,1), (-1,1), (-2,0)
    ]
    
    centers = []
    for i, j in coords:
        c = i * u + j * v
        centers.append(tuple(c))
        
    angles = [0.0] * 11
    return centers, angles

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin for 150s limit
    
    # 1. Initialize with the known optimal "flower" configuration
    centers, angles = _build_flower_config()
    
    # 2. Evaluate initial score
    best_S = _min_outer_side(centers, angles, 0.0)
    best_config = (centers, angles, 0.0)
    
    # 3. Local optimization to fine-tune positions, inner angles, and outer angle
    # x = [x1, y1, a1, ..., xN, yN, aN, outer_angle]
    x0 = []
    for c, a in zip(centers, angles):
        x0.extend([c[0], c[1], np.deg2rad(a)])
    x0.append(0.0) # outer angle
    
    # Use Nelder-Mead for non-smooth objective
    if time.time() < deadline - 10:
        try:
            result = minimize(_objective_full, x0, method='Nelder-Mead',
                             options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            if result.fun < best_S:
                opt_centers = [(result.x[3*i], result.x[3*i+1]) for i in range(N)]
                opt_angles = [np.rad2deg(result.x[3*i+2]) for i in range(N)]
                opt_outer = np.rad2deg(result.x[-1])
                best_S = result.fun
                best_config = (opt_centers, opt_angles, opt_outer)
        except Exception:
            pass
            
    # 4. Try a few random perturbations of the flower config to ensure we aren't stuck in a local min
    # (Though flower is likely global min)
    if time.time() < deadline - 5:
        for _ in range(10):
            if time.time() > deadline - 2:
                break
            # Small random perturbation
            pert_centers = [tuple(c + np.random.normal(0, 0.1, 2)) for c in centers]
            pert_angles = [a + np.random.uniform(-10, 10) for a in angles]
            
            x0_pert = []
            for c, a in zip(pert_centers, pert_angles):
                x0_pert.extend([c[0], c[1], np.deg2rad(a)])
            x0_pert.append(np.random.uniform(-15, 15))
            
            try:
                result = minimize(_objective_full, x0_pert, method='Nelder-Mead',
                                 options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
                if result.fun < best_S:
                    opt_centers = [(result.x[3*i], result.x[3*i+1]) for i in range(N)]
                    opt_angles = [np.rad2deg(result.x[3*i+2]) for i in range(N)]
                    opt_outer = np.rad2deg(result.x[-1])
                    best_S = result.fun
                    best_config = (opt_centers, opt_angles, opt_outer)
            except Exception:
                pass

    centers, angles, outer_ang = best_config
    inner = [[c[0], c[1], a] for c, a in zip(centers, angles)]
    return inner, (0.0, 0.0), best_S, outer_ang

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
