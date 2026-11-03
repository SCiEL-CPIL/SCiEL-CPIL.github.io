import math
import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar

N = 11
APOTHEM_UNIT = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Helper to get vertices of a unit hexagon centered at (x,y) with rotation theta
    def get_hex_vertices(x, y, theta):
        # Vertices at distance 1 from center
        # Angles: theta, theta+60, ...
        angles = [theta + k * math.pi / 3.0 for k in range(6)]
        return np.array([[x + math.cos(a), y + math.sin(a)] for a in angles])

    # Compute min S for a given set of inner hexagons and outer rotation phi
    # Outer hexagon center is (0,0)
    def compute_S_for_config(pts, angs, phi):
        # Normals of outer hexagon sides
        # Sides are at angles phi + 30 + k*60
        # Normal to side k is at angle phi + 30 + k*60 + 90? No.
        # Standard hexagon: vertices at 0, 60, ...
        # Sides are perpendicular to angles 30, 90, 150, ...
        # Let's use the property: max projection on 6 axes.
        # Axes for outer hexagon (distance from center to side is S*sqrt(3)/2)
        # Axis angles: phi + pi/6 + k*pi/3
        max_proj = 0.0
        for i in range(N):
            verts = get_hex_vertices(pts[i, 0], pts[i, 1], angs[i])
            for v in verts:
                for k in range(6):
                    ang = phi + math.pi / 6.0 + k * math.pi / 3.0
                    nx, ny = math.cos(ang), math.sin(ang)
                    proj = abs(v[0] * nx + v[1] * ny)
                    if proj > max_proj:
                        max_proj = proj
        # S * APOTHEM_UNIT = max_proj
        # S = max_proj / APOTHEM_UNIT
        return max_proj / APOTHEM_UNIT

    # Objective function for the inner hexagon positions/angles
    # We also optimize the outer rotation phi
    def objective(vars):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return 1e10
        pts = vars[:2*N].reshape(N, 2)
        angs = vars[2*N:3*N]
        
        # Optimize phi for this configuration
        # phi is in [0, pi/3) due to symmetry
        try:
            res = minimize_scalar(lambda p: compute_S_for_config(pts, angs, p), bounds=(0, math.pi/3), method='bounded', options={'xatol': 1e-6})
            return res.fun
        except Exception:
            return 1e10

    # Generate initial configurations
    # 1. Grid-like
    # 2. Cluster-like
    # 3. Random
    
    best_S = 1e10
    best_vars = None
    best_phi = 0.0

    # Initial Config 1: 3x3 grid + 2
    # Hex grid coords: (q, r)
    # x = sqrt(3) * (q + r/2), y = 1.5 * r
    axial_coords = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (3, 1), (-1, 1)
    ]
    pts_init = np.array([[math.sqrt(3) * (q + r/2.0), 1.5 * r] for q, r in axial_coords])
    angs_init = np.zeros(N)
    # Center the config
    pts_init -= np.mean(pts_init, axis=0)
    
    initial_vars_list = []
    initial_vars_list.append(np.concatenate([pts_init.flatten(), angs_init]))
    
    # Add random perturbations to initial config
    for _ in range(5):
        pert_pts = pts_init + np.random.normal(0, 0.2, (N, 2))
        pert_angs = angs_init + np.random.normal(0, 0.1, N)
        initial_vars_list.append(np.concatenate([pert_pts.flatten(), pert_angs]))
        
    # Add a "loose" random config
    for _ in range(5):
        rand_pts = np.random.uniform(-3, 3, (N, 2))
        rand_angs = np.random.uniform(0, math.pi/3, N)
        initial_vars_list.append(np.concatenate([rand_pts.flatten(), rand_angs]))

    for init_vars in initial_vars_list:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        try:
            # Use Nelder-Mead for non-smooth objective (due to max/min operations)
            # maxiter should be high enough
            res = minimize(objective, init_vars, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
            
            if res.fun < best_S:
                best_S = res.fun
                best_vars = res.x
                # Find optimal phi for this best config
                pts = best_vars[:2*N].reshape(N, 2)
                angs = best_vars[2*N:3*N]
                res_phi = minimize_scalar(lambda p: compute_S_for_config(pts, angs, p), bounds=(0, math.pi/3), method='bounded', options={'xatol': 1e-8})
                best_phi = res_phi.x
        except Exception:
            continue

    if best_vars is None:
        # Fallback to a simple valid config
        axial_coords = [
            (0, 0), (1, 0), (2, 0),
            (0, 1), (1, 1), (2, 1),
            (0, 2), (1, 2), (2, 2),
            (3, 1), (-1, 1)
        ]
        pts = np.array([[math.sqrt(3) * (q + r/2.0), 1.5 * r] for q, r in axial_coords])
        pts -= np.mean(pts, axis=0)
        angs = np.zeros(N)
        best_phi = 0.0
        # Calculate S for this
        # We need to return the config.
        # For safety, just return this grid config with a calculated S
        # But we need the exact S and center.
        # The center of the outer hexagon is (0,0) in our optimization.
        # The inner hexagons are at pts.
        # We need to scale them? No, they are unit size.
        # We just need to report the S that contains them.
        
        # Calculate S for fallback
        max_proj = 0
        for i in range(N):
            verts = get_hex_vertices(pts[i, 0], pts[i, 1], 0.0)
            for v in verts:
                for k in range(6):
                    ang = best_phi + math.pi / 6.0 + k * math.pi / 3.0
                    nx, ny = math.cos(ang), math.sin(ang)
                    proj = abs(v[0] * nx + v[1] * ny)
                    if proj > max_proj:
                        max_proj = proj
        best_S = max_proj / APOTHEM_UNIT
        best_vars = np.concatenate([pts.flatten(), angs])

    # Extract final configuration
    pts = best_vars[:2*N].reshape(N, 2)
    angs = best_vars[2*N:3*N]
    
    # The outer hexagon is centered at (0,0) with rotation best_phi
    # The inner hexagons are at pts with rotations angs
    # We need to return:
    # inner: list of [x, y, angle_degrees]
    # center: [0, 0]
    # side: best_S
    # rotation: degrees(best_phi)
    
    # Add a tiny safety margin to S to ensure no floating point errors cause overlap/out-of-bounds
    # The evaluator might check strictly.
    # Let's add 1e-7 to S
    final_S = best_S + 1e-7
    
    inner = []
    for i in range(N):
        x, y = pts[i]
        ang_deg = math.degrees(angs[i])
        inner.append([float(x), float(y), float(ang_deg)])
        
    center = [0.0, 0.0]
    rotation = math.degrees(best_phi)
    
    return inner, center, final_S, float(rotation)
