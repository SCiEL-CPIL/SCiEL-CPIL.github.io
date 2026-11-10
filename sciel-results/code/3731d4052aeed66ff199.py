# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _hex_vertices(cx, cy, s, a_deg):
    """
    Returns vertices of a regular hexagon centered at (cx, cy) with side s and angle a_deg.
    Vertices are at angles a + 60*i degrees.
    """
    a_rad = np.deg2rad(a_deg)
    angles = a_rad + np.deg2rad(60 * np.arange(6))
    xs = cx + s * np.cos(angles)
    ys = cy + s * np.sin(angles)
    return np.column_stack((xs, ys))

def _check_pair_separation(v1, v2):
    """
    Returns the maximum gap between hexagons v1 and v2 over all 12 edge normals.
    If max_gap > 1e-6, they are validly separated.
    """
    def get_normals(verts):
        normals = []
        for i in range(6):
            p1 = verts[i]
            p2 = verts[(i + 1) % 6]
            edge = p2 - p1
            n = np.array([-edge[1], edge[0]])
            n_norm = np.linalg.norm(n)
            if n_norm > 1e-9:
                n = n / n_norm
                normals.append(n)
        return np.array(normals)
        
    n1 = get_normals(v1)
    n2 = get_normals(v2)
    all_normals = np.vstack((n1, n2))
    
    max_gap = -np.inf
    for n in all_normals:
        proj1 = np.dot(v1, n)
        proj2 = np.dot(v2, n)
        
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 < min2:
            gap = min2 - max1
        elif max2 < min1:
            gap = min1 - max2
        else:
            gap = 0.0
            
        if gap > max_gap:
            max_gap = gap
            
    return max_gap

def _min_outer_side_for_inner(inner_hexes):
    """
    inner_hexes: array of shape (11, 3) (x, y, angle_deg). Unit side.
    Returns (min_side, best_angle) for an outer hexagon centered at origin.
    """
    all_verts = []
    for i in range(inner_hexes.shape[0]):
        v = _hex_vertices(inner_hexes[i, 0], inner_hexes[i, 1], 1.0, inner_hexes[i, 2])
        all_verts.append(v)
    all_verts = np.vstack(all_verts)
    
    # Sample alpha in [0, 60)
    # Normals are at alpha + 30, alpha + 150, alpha + 270
    alphas = np.linspace(0, 60, 1000, endpoint=False)
    min_d = np.inf
    best_alpha = 0
    
    for a in alphas:
        a_rad = np.deg2rad(a)
        n_angles = a_rad + np.deg2rad([30, 150, 270])
        
        cos_n = np.cos(n_angles)
        sin_n = np.sin(n_angles)
        
        projs = all_verts[:, 0:1] * cos_n[None, :] + all_verts[:, 1:2] * sin_n[None, :]
        abs_projs = np.abs(projs)
        max_per_axis = np.max(abs_projs, axis=0)
        current_d = np.max(max_per_axis)
        
        if current_d < min_d:
            min_d = current_d
            best_alpha = a
            
    S = 2.0 * min_d / np.sqrt(3.0)
    return S, best_alpha

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 550.0
    
    n = 11
    
    # Honeycomb topology:
    # 0: center
    # 1-6: ring around center
    # 7-10: outer ring (4 hexes)
    # Distances in a perfect honeycomb (side 1):
    # Center to ring 1: 2.0
    # Ring 1 to adjacent ring 1: 2.0
    # Ring 1 to outer (adjacent to 2 ring 1s): sqrt(3) * sqrt(2)? No.
    # Let's use standard coordinates.
    # Center (0,0)
    # Ring 1: (2,0), (1,sqrt3), (-1,sqrt3), (-2,0), (-1,-sqrt3), (1,-sqrt3)
    # Outer 1: (3, sqrt3) -> dist from (2,0) is sqrt(1+3)=2, dist from (1,sqrt3) is sqrt(4+0)=2.
    # So outer hexes are at distance 2 from two ring 1 hexes.
    # The 4 outer hexes are at:
    # (3, sqrt3), (0, 2*sqrt3), (-3, sqrt3), (-3, -sqrt3), (0, -2*sqrt3), (3, -sqrt3)
    # Wait, that's 6 outer hexes. We only have 4.
    # So we pick 4 of these 6 positions.
    # By symmetry, any 4 consecutive? Or 2 opposite and 2 adjacent?
    # To minimize bounding box, we should probably pick ones that are "spread out" but not too much.
    # Actually, the "flower" of 7 hexes has a bounding hexagon of side 3 (if aligned).
    # Adding 4 more will increase it.
    
    # Let's define the 7 core positions.
    core = np.array([
        [0, 0, 0],
        [2, 0, 0],
        [1, np.sqrt(3), 0],
        [-1, np.sqrt(3), 0],
        [-2, 0, 0],
        [-1, -np.sqrt(3), 0],
        [1, -np.sqrt(3), 0]
    ])
    
    # Candidate outer positions (indices 7-10)
    # The 6 neighbors of the ring 1 hexes (excluding the center and other ring 1 hexes)
    # These are the "second ring" positions.
    # Positions:
    # 7: (3, sqrt3)
    # 8: (0, 2*sqrt3)
    # 9: (-3, sqrt3)
    # 10: (-3, -sqrt3)
    # 11: (0, -2*sqrt3)
    # 12: (3, -sqrt3)
    
    # We need to choose 4 out of these 6.
    # There are C(6,4) = 15 combinations.
    # Due to symmetry, there are only a few distinct configurations.
    # 1. 4 consecutive (e.g. 7,8,9,10). This creates a "C" shape.
    # 2. 2 opposite, 2 adjacent?
    
    # Let's try a few specific configurations.
    # Config 1: 4 consecutive (7,8,9,10) -> angles 60, 120, 180, 240 roughly?
    # Config 2: 2 opposite, 2 adjacent (7, 11, 8, 9)?
    
    # Let's just try a few and optimize.
    
    configs = []
    
    # Config A: 7, 8, 9, 10
    outer_A = np.array([
        [3, np.sqrt(3), 0],
        [0, 2*np.sqrt(3), 0],
        [-3, np.sqrt(3), 0],
        [-3, -np.sqrt(3), 0]
    ])
    configs.append(np.vstack((core, outer_A)))
    
    # Config B: 7, 8, 9, 11 (missing 10 and 12? No, 7,8,9,11)
    outer_B = np.array([
        [3, np.sqrt(3), 0],
        [0, 2*np.sqrt(3), 0],
        [-3, np.sqrt(3), 0],
        [0, -2*np.sqrt(3), 0]
    ])
    configs.append(np.vstack((core, outer_B)))
    
    # Config C: 7, 9, 10, 12 (2 opposite, 2 adjacent? 7 and 10 are not opposite. 7 and 12 are adjacent? No.)
    # 7(3,s), 9(-3,s) are opposite. 10(-3,-s), 12(3,-s) are opposite.
    # So 7,9,10,12 is 2 pairs of opposites.
    outer_C = np.array([
        [3, np.sqrt(3), 0],
        [-3, np.sqrt(3), 0],
        [-3, -np.sqrt(3), 0],
        [3, -np.sqrt(3), 0]
    ])
    configs.append(np.vstack((core, outer_C)))
    
    best_S = np.inf
    best_hexes = configs[0].copy()
    
    # Objective function
    def objective(params):
        if time.time() > deadline:
            return 1e9
            
        hexes = params.reshape(n, 3)
        
        # Check separation
        min_sep = np.inf
        for i in range(n):
            for j in range(i + 1, n):
                v1 = _hex_vertices(hexes[i, 0], hexes[i, 1], 1.0, hexes[i, 2])
                v2 = _hex_vertices(hexes[j, 0], hexes[j, 1], 1.0, hexes[j, 2])
                gap = _check_pair_separation(v1, v2)
                if gap < min_sep:
                    min_sep = gap
                    
        if min_sep < 1e-6:
            return 10.0 + (1e-6 - min_sep) * 100.0
            
        S, _ = _min_outer_side_for_inner(hexes)
        return S
        
    # Optimize each config
    for config in configs:
        if time.time() > deadline:
            break
            
        # Try Nelder-Mead
        try:
            res = minimize(objective, config.flatten(), method='Nelder-Mead',
                           options={'xatol': 1e-5, 'fatol': 1e-5, 'maxiter': 2000, 'maxfev': 2000})
            
            if res.fun < best_S:
                best_S = res.fun
                best_hexes = res.x.reshape(n, 3)
                
        except Exception:
            pass
            
    # If we have time, try random perturbations of the best found
    if time.time() < deadline - 100:
        for _ in range(3):
            if time.time() > deadline:
                break
            perturbed = best_hexes + np.random.normal(0, 0.1, best_hexes.shape)
            try:
                res = minimize(objective, perturbed.flatten(), method='Nelder-Mead',
                               options={'xatol': 1e-5, 'fatol': 1e-5, 'maxiter': 1000, 'maxfev': 1000})
                if res.fun < best_S:
                    best_S = res.fun
                    best_hexes = res.x.reshape(n, 3)
            except Exception:
                pass

    # Finalize
    best_hexes[:, 2] = best_hexes[:, 2] % 60.0
    
    final_S, final_angle = _min_outer_side_for_inner(best_hexes)
    
    inner = best_hexes
    outer = np.array([0.0, 0.0, final_angle])
    side = final_S
    
    return inner, outer, side

# EVOLVE-BLOCK-END
