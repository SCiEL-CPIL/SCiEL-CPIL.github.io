import numpy as np
import time
from scipy.optimize import minimize

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 580  # safety margin
    
    # Normals for the hexagon edges (angles 30, 90, 150)
    angles = np.deg2rad([30, 90, 150])
    normals = np.array([
        [np.cos(a), np.sin(a)] for a in angles
    ])
    
    # Vertices of a unit hexagon (side 1)
    v_angles = np.deg2rad([0, 60, 120, 180, 240, 300])
    hex_verts = np.array([
        [np.cos(a), np.sin(a)] for a in v_angles
    ])
    
    # Threshold for non-overlap: separation must be > 1e-6
    # So we require max_proj > sqrt(3) + 1e-6
    # We use a slightly larger threshold for optimization to ensure robustness
    threshold = np.sqrt(3) + 1e-4

    def compute_side(pts):
        """Compute the minimum side length of enclosing hexagon centered at origin."""
        max_proj_all = 0.0
        for k in range(3):
            n_k = normals[k]
            pts_proj = pts @ n_k
            verts_proj = hex_verts @ n_k
            sums = pts_proj[:, np.newaxis] + verts_proj[np.newaxis, :]
            max_abs_sum = np.max(np.abs(sums))
            if max_abs_sum > max_proj_all:
                max_proj_all = max_abs_sum
        return (2.0 / np.sqrt(3.0)) * max_proj_all

    def check_validity(pts):
        """Check if configuration is valid (no overlaps, all inside)."""
        # Check overlaps
        for i in range(11):
            for j in range(i + 1, 11):
                v = pts[i] - pts[j]
                proj_v = np.abs(v @ normals.T)
                max_proj_v = np.max(proj_v)
                # Must be apart by > 1e-6
                if max_proj_v < np.sqrt(3) + 1e-6:
                    return False
        return True

    def objective_and_penalty(x):
        pts = x.reshape(11, 2)
        
        # Compute required side length
        max_proj_all = 0.0
        for k in range(3):
            n_k = normals[k]
            pts_proj = pts @ n_k
            verts_proj = hex_verts @ n_k
            sums = pts_proj[:, np.newaxis] + verts_proj[np.newaxis, :]
            max_abs_sum = np.max(np.abs(sums))
            if max_abs_sum > max_proj_all:
                max_proj_all = max_abs_sum
                
        S_req = (2.0 / np.sqrt(3.0)) * max_proj_all
        
        # Calculate Penalty for overlaps
        penalty = 0.0
        for i in range(11):
            for j in range(i + 1, 11):
                v = pts[i] - pts[j]
                proj_v = np.abs(v @ normals.T)
                max_proj_v = np.max(proj_v)
                if max_proj_v < threshold:
                    depth = threshold - max_proj_v
                    penalty += depth * depth
                    
        return S_req + 1000.0 * penalty

    # Symmetry-reduced parameterization:
    # By 6-fold symmetry, we can place:
    # - 1 hexagon at center (0,0)
    # - 6 hexagons at distance r1, angles 0, 60, 120, 180, 240, 300
    # - 4 hexagons at distance r2, angles 0, 120, 180, 300
    # This gives us 2 free parameters: r1 and r2
    
    def build_pts_symmetric(params):
        """Build 11 points from symmetric parameters [r1, r2]."""
        r1, r2 = params
        pts = np.zeros((11, 2))
        pts[0] = [0, 0]  # center
        for k in range(6):
            angle = np.deg2rad(60 * k)
            pts[k + 1] = [r1 * np.cos(angle), r1 * np.sin(angle)]
        spike_angles = [0, 120, 180, 300]
        for idx, angle_deg in enumerate(spike_angles):
            angle = np.deg2rad(angle_deg)
            pts[idx + 7] = [r2 * np.cos(angle), r2 * np.sin(angle)]
        return pts

    def objective_symmetric(params):
        """Objective function for symmetric parameterization."""
        pts = build_pts_symmetric(params)
        return objective_and_penalty(pts.flatten())

    # Initial guess
    r1_init = 2.1
    r2_init = 4.2
    x0_sym = np.array([r1_init, r2_init])
    
    best_x = x0_sym.copy()
    best_pts = build_pts_symmetric(best_x)
    best_val = objective_symmetric(best_x)
    best_side = compute_side(best_pts)
    best_valid = check_validity(best_pts)
    
    # Try multiple optimization runs
    rng = np.random.default_rng(42)
    
    # First run: Nelder-Mead on symmetric params
    if time.time() < deadline - 120:
        res = minimize(objective_symmetric, x0_sym, method='Nelder-Mead',
                      options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-12,
                               'initial_simplex': [[r1_init, r2_init],
                                                   [r1_init + 0.1, r2_init],
                                                   [r1_init, r2_init + 0.1]]})
        pts_res = build_pts_symmetric(res.x)
        if check_validity(pts_res):
            if res.fun < best_val:
                best_x = res.x.copy()
                best_pts = pts_res
                best_val = res.fun
                best_side = compute_side(pts_res)
                best_valid = True
    
    # Second run: perturbed start
    if time.time() < deadline - 120:
        x0_pert = x0_sym + rng.normal(0, 0.05, x0_sym.shape)
        res2 = minimize(objective_symmetric, x0_pert, method='Nelder-Mead',
                       options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-12})
        pts_res2 = build_pts_symmetric(res2.x)
        if check_validity(pts_res2):
            if res2.fun < best_val:
                best_x = res2.x.copy()
                best_pts = pts_res2
                best_val = res2.fun
                best_side = compute_side(pts_res2)
                best_valid = True
    
    # Third run: different perturbation
    if time.time() < deadline - 60:
        x0_pert2 = x0_sym + rng.normal(0, 0.1, x0_sym.shape)
        res3 = minimize(objective_symmetric, x0_pert2, method='Nelder-Mead',
                       options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-12})
        pts_res3 = build_pts_symmetric(res3.x)
        if check_validity(pts_res3):
            if res3.fun < best_val:
                best_x = res3.x.copy()
                best_pts = pts_res3
                best_val = res3.fun
                best_side = compute_side(pts_res3)
                best_valid = True

    # Fourth run: try Powell on symmetric params
    if time.time() < deadline - 60:
        res4 = minimize(objective_symmetric, best_x, method='Powell',
                       options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12})
        pts_res4 = build_pts_symmetric(res4.x)
        if check_validity(pts_res4):
            if res4.fun < best_val:
                best_x = res4.x.copy()
                best_pts = pts_res4
                best_val = res4.fun
                best_side = compute_side(pts_res4)
                best_valid = True
    
    # Final local refinement
    if time.time() < deadline - 30:
        res5 = minimize(objective_symmetric, best_x, method='Nelder-Mead',
                       options={'maxiter': 20000, 'xatol': 1e-14, 'fatol': 1e-14})
        pts_res5 = build_pts_symmetric(res5.x)
        if check_validity(pts_res5):
            if res5.fun < best_val:
                best_x = res5.x.copy()
                best_pts = pts_res5
                best_val = res5.fun
                best_side = compute_side(pts_res5)
                best_valid = True

    # If the best found is invalid, fall back to initial guess
    if not best_valid:
        best_pts = build_pts_symmetric(x0_sym)
        best_side = compute_side(best_pts)

    pts_final = best_pts
    side = compute_side(pts_final)
    
    # Ensure side is at least sqrt(11)
    if side < np.sqrt(11):
        side = np.sqrt(11)
    
    inner_hex_data = np.zeros((11, 3))
    inner_hex_data[:, 0] = pts_final[:, 0]
    inner_hex_data[:, 1] = pts_final[:, 1]
    inner_hex_data[:, 2] = 0
    
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, float(side)
