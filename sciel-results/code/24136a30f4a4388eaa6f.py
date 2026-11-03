import numpy as np
import time
from scipy.optimize import minimize

N = 11

def _hex_vertices(cx, cy, angle_deg=0.0):
    """
    Get vertices of a unit regular hexagon centered at (cx, cy) rotated by angle_deg.
    Vertices are at distance 1 from center.
    """
    angle_rad = np.deg2rad(angle_deg)
    base_angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180.0
    angles = base_angles + angle_rad
    x = cx + np.cos(angles)
    y = cy + np.sin(angles)
    return np.column_stack((x, y))

def _compute_max_proj(centers, angles, outer_angle_deg):
    """
    Compute max projection of all inner hexagon vertices onto the 6 outer normals.
    Returns max_proj (float).
    """
    theta_rad = np.deg2rad(outer_angle_deg)
    normals = np.array([[np.cos(theta_rad + np.deg2rad(30 + k * 60)),
                         np.sin(theta_rad + np.deg2rad(30 + k * 60))] for k in range(6)])
    max_proj = 0.0
    for (cx, cy), ang in zip(centers, angles):
        verts = _hex_vertices(cx, cy, ang)
        # Project each vertex onto each normal: (6,6) matrix
        projs = verts @ normals.T
        local_max = np.max(projs)
        if local_max > max_proj:
            max_proj = local_max
    return max_proj

def _min_outer_side(inner_centers, inner_angles, outer_angle_deg=0.0):
    """
    Calculate the minimum side length S of the outer hexagon (centered at 0,0, rotated by outer_angle_deg)
    that contains all inner hexagons.
    """
    max_proj = _compute_max_proj(inner_centers, inner_angles, outer_angle_deg)
    S = 2.0 * max_proj / np.sqrt(3.0)
    return S

def _smooth_objective(x, beta=50.0):
    """
    Smooth surrogate objective using log-sum-exp approximation of max.
    x = [x1, y1, a1, x2, y2, a2, ..., outer_angle]
    """
    n = (len(x) - 1) // 3
    centers = [(x[3*i], x[3*i+1]) for i in range(n)]
    angles = [x[3*i+2] for i in range(n)]
    outer_ang = x[-1]
    
    theta_rad = np.deg2rad(outer_ang)
    normals = np.array([[np.cos(theta_rad + np.deg2rad(30 + k * 60)),
                         np.sin(theta_rad + np.deg2rad(30 + k * 60))] for k in range(6)])
    
    all_projs = []
    for (cx, cy), ang in zip(centers, angles):
        verts = _hex_vertices(cx, cy, ang)
        projs = verts @ normals.T
        all_projs.append(projs.flatten())
    
    all_projs = np.concatenate(all_projs)
    # log-sum-exp for smooth max
    max_p = np.max(all_projs)
    if max_p > 0:
        smooth_max = max_p + np.log(np.sum(np.exp((all_projs - max_p) / beta))) * beta / 1.0
        # Actually: soft_max = (1/beta) * log(sum(exp(beta * p)))
        # Let me redo this properly
        smooth_max = (1.0 / beta) * np.log(np.sum(np.exp(beta * all_projs)))
    else:
        smooth_max = np.max(all_projs)
    
    S = 2.0 * smooth_max / np.sqrt(3.0)
    return S

def _objective_nm(x, outer_angle_deg=0.0):
    """
    Objective for Nelder-Mead: x = [x1,y1,a1, x2,y2,a2, ...] (angles in radians)
    """
    n = len(x) // 3
    centers = [(x[3*i], x[3*i+1]) for i in range(n)]
    angles_deg = [np.rad2deg(x[3*i+2]) for i in range(n)]
    return _min_outer_side(centers, angles_deg, outer_angle_deg)

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

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin for 150s limit
    
    # Lattice basis vectors for hexagonal packing
    A = np.array([np.sqrt(3), 0.0])
    B = np.array([np.sqrt(3)/2, 1.5])
    
    # Generate candidate topologies
    t1 = [(0,0)] + [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)] + [(2,0), (0,2), (-2,0), (0,-2)]
    t2 = [(0,0)] + [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)] + [(1,1), (1,-1), (-1,1), (-1,-1)]
    t3 = [(0,0)] + [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)] + [(2,0), (1,1), (0,2), (-1,1)]
    t4 = [(0,0)] + [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)] + [(2,0), (1,1), (-1,1), (-2,0)]
    t5 = [(0,0)] + [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)] + [(0,2), (1,1), (1,-1), (0,-2)]
    t6 = [(0,0)] + [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)] + [(-2,0), (-1,1), (-1,-1), (0,-2)]
    # Additional topologies
    t7 = [(0,0)] + [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)] + [(2,0), (1,1), (0,2), (1,-1)]
    t8 = [(0,0)] + [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)] + [(2,0), (-1,1), (0,-2), (1,-1)]
    t9 = [(0,0)] + [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)] + [(-2,0), (-1,1), (0,2), (0,-2)]
    
    topologies = [t1, t2, t3, t4, t5, t6, t7, t8, t9]
    
    best_S = float('inf')
    best_config = None
    best_init_configs = []  # Store multiple good initial configs
    
    for topo in topologies:
        centers = []
        for i, j in topo:
            c = i * A + j * B
            centers.append(tuple(c))
        
        for rot_deg in range(0, 180, 10):
            if time.time() > deadline - 60:
                break
            rot_rad = np.deg2rad(rot_deg)
            cos_r = np.cos(rot_rad)
            sin_r = np.sin(rot_rad)
            rotated_centers = []
            for (x, y) in centers:
                rx = x * cos_r - y * sin_r
                ry = x * sin_r + y * cos_r
                rotated_centers.append((rx, ry))
            
            for inner_ang in [0, 30, 60]:
                S = _min_outer_side(rotated_centers, [inner_ang]*N, outer_angle_deg=0.0)
                if S < best_S:
                    best_S = S
                    best_config = (rotated_centers, [inner_ang]*N, 0.0)
                if S < best_S + 0.5:
                    best_init_configs.append((rotated_centers, [inner_ang]*N, 0.0))
    
    if best_config is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Local optimization passes
    # Pass 1: Nelder-Mead on positions + inner angles (outer angle fixed)
    centers, angles, outer_ang = best_config
    x0 = []
    for (cx, cy), ang in zip(centers, angles):
        x0.extend([cx, cy, np.deg2rad(ang)])
    
    if time.time() < deadline - 40:
        try:
            result = minimize(_objective_nm, x0, method='Nelder-Mead',
                             options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-7, 'adaptive': True})
            if result.fun < best_S:
                opt_centers = [(result.x[3*i], result.x[3*i+1]) for i in range(N)]
                opt_angles = [np.rad2deg(result.x[3*i+2]) for i in range(N)]
                best_S = result.fun
                best_config = (opt_centers, opt_angles, 0.0)
        except Exception:
            pass
    
    # Pass 2: Optimize including outer angle
    if time.time() < deadline - 30:
        centers, angles, outer_ang = best_config
        x0_full = []
        for (cx, cy), ang in zip(centers, angles):
            x0_full.extend([cx, cy, np.deg2rad(ang)])
        x0_full.append(np.deg2rad(outer_ang))
        
        try:
            result = minimize(_objective_full, x0_full, method='Nelder-Mead',
                             options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-7, 'adaptive': True})
            if result.fun < best_S:
                opt_centers = [(result.x[3*i], result.x[3*i+1]) for i in range(N)]
                opt_angles = [np.rad2deg(result.x[3*i+2]) for i in range(N)]
                opt_outer = np.rad2deg(result.x[-1])
                best_S = result.fun
                best_config = (opt_centers, opt_angles, opt_outer)
        except Exception:
            pass
    
    # Pass 3: Try smooth surrogate with different beta values from best config
    if time.time() < deadline - 25:
        centers, angles, outer_ang = best_config
        for beta in [20.0, 50.0, 100.0]:
            if time.time() > deadline - 20:
                break
            x0_smooth = []
            for (cx, cy), ang in zip(centers, angles):
                x0_smooth.extend([cx, cy, np.deg2rad(ang)])
            x0_smooth.append(np.deg2rad(outer_ang))
            
            def _smooth_obj(x, b=beta):
                n = (len(x) - 1) // 3
                c = [(x[3*i], x[3*i+1]) for i in range(n)]
                a = [x[3*i+2] for i in range(n)]
                oa = x[-1]
                theta_rad = oa
                normals = np.array([[np.cos(theta_rad + np.deg2rad(30 + k * 60)),
                                     np.sin(theta_rad + np.deg2rad(30 + k * 60))] for k in range(6)])
                all_projs = []
                for (cx, cy), ang in zip(c, a):
                    verts = _hex_vertices(cx, cy, np.rad2deg(ang))
                    projs = verts @ normals.T
                    all_projs.append(projs.flatten())
                all_projs = np.concatenate(all_projs)
                # Smooth max via log-sum-exp
                m = np.max(all_projs)
                smooth_max = m + np.log(np.sum(np.exp((all_projs - m) / b))) * b / 1.0
                # Correct: soft_max = (1/b)*log(sum(exp(b*p)))
                smooth_max = (1.0/b) * np.log(np.sum(np.exp(b * all_projs)))
                return 2.0 * smooth_max / np.sqrt(3.0)
            
            try:
                result = minimize(_smooth_obj, x0_smooth, method='Nelder-Mead',
                                 options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-8, 'adaptive': True})
                if result.fun < best_S:
                    # Verify with exact objective
                    opt_centers = [(result.x[3*i], result.x[3*i+1]) for i in range(N)]
                    opt_angles = [np.rad2deg(result.x[3*i+2]) for i in range(N)]
                    opt_outer = np.rad2deg(result.x[-1])
                    exact_S = _min_outer_side(opt_centers, opt_angles, opt_outer)
                    if exact_S < best_S:
                        best_S = exact_S
                        best_config = (opt_centers, opt_angles, opt_outer)
            except Exception:
                pass
    
    # Pass 4: Try other good initial configs with Nelder-Mead
    if time.time() < deadline - 15 and len(best_init_configs) > 1:
        for cfg in best_init_configs[1:5]:
            if time.time() > deadline - 10:
                break
            c, a, oa = cfg
            x0_try = []
            for (cx, cy), ang in zip(c, a):
                x0_try.extend([cx, cy, np.deg2rad(ang)])
            x0_try.append(np.deg2rad(oa))
            try:
                result = minimize(_objective_full, x0_try, method='Nelder-Mead',
                                 options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-7, 'adaptive': True})
                if result.fun < best_S:
                    opt_centers = [(result.x[3*i], result.x[3*i+1]) for i in range(N)]
                    opt_angles = [np.rad2deg(result.x[3*i+2]) for i in range(N)]
                    opt_outer = np.rad2deg(result.x[-1])
                    best_S = result.fun
                    best_config = (opt_centers, opt_angles, opt_outer)
            except Exception:
                pass
    
    # Final polish: high-precision Nelder-Mead
    if time.time() < deadline - 5:
        centers, angles, outer_ang = best_config
        x0_final = []
        for (cx, cy), ang in zip(centers, angles):
            x0_final.extend([cx, cy, np.deg2rad(ang)])
        x0_final.append(np.deg2rad(outer_ang))
        try:
            result = minimize(_objective_full, x0_final, method='Nelder-Mead',
                             options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
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
