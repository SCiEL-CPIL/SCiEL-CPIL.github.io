import numpy as np
import time
from scipy.optimize import minimize
import math

def construct():
    """
    Pack 11 unit regular hexagons inside a regular hexagon, minimizing the outer side length.
    Uses a flower packing (1 center, 6 ring 1, 4 ring 2) with optimized rotations.
    Combines reduced-variable symmetry exploitation, smooth log-sum-exp surrogate,
    multi-start Powell, Nelder-Mead polish, and coordinate descent refinement.
    """
    start_time = time.time()
    deadline = 148.0  # Safety margin before 150s limit

    # Centers of the 11 hexagons in the standard hex lattice
    eps = 1e-10
    centers = np.array([[0.0, 0.0]])
    
    r1 = np.sqrt(3) + eps
    for k in range(6):
        angle = np.deg2rad(k * 60)
        centers = np.vstack([centers, [r1 * np.cos(angle), r1 * np.sin(angle)]])
        
    r2 = 3.0 + 2 * eps
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        centers = np.vstack([centers, [r2 * np.cos(angle), r2 * np.sin(angle)]])
        
    n_hex = len(centers)
    
    # Base vertices of unit hexagon (pointy-right orientation, angle 0)
    base_vertices = np.array([
        [np.cos(np.deg2rad(60 * k)), np.sin(np.deg2rad(60 * k))] for k in range(6)
    ])
    
    def _get_rotated_vertices(phi):
        """Get rotated vertices for a unit hexagon with rotation phi."""
        c, s = np.cos(phi), np.sin(phi)
        rot = np.array([[c, -s], [s, c]])
        return base_vertices @ rot.T

    def compute_outer_side_full(theta_out, phis):
        """
        Compute the minimum outer hexagon side length given outer rotation and inner rotations.
        """
        normals = np.array([
            [np.cos(np.deg2rad(30 + k * 60) + theta_out),
             np.sin(np.deg2rad(30 + k * 60) + theta_out)]
            for k in range(6)
        ])
        
        all_verts = []
        for i in range(n_hex):
            verts = _get_rotated_vertices(phis[i]) + centers[i]
            all_verts.append(verts)
        all_verts = np.array(all_verts)  # shape (n_hex, 6, 2)
        
        projs = np.einsum('ijk,jl->ikl', all_verts, normals)
        max_proj = np.max(np.abs(projs))
        
        return 2.0 * max_proj / np.sqrt(3)

    def objective_full(params):
        """
        params: [theta_out, phi_0, ..., phi_10]
        """
        theta_out = params[0]
        phis = params[1:]
        return compute_outer_side_full(theta_out, phis)

    def compute_outer_side_reduced(theta_out, phi_center, phi_ring1, phi_ring2):
        """
        Compute the minimum outer hexagon side length given outer rotation and symmetric inner rotations.
        """
        normals = np.array([
            [np.cos(np.deg2rad(30 + k * 60) + theta_out),
             np.sin(np.deg2rad(30 + k * 60) + theta_out)]
            for k in range(6)
        ])
        
        max_proj = 0.0
        
        # Center hexagon
        verts = _get_rotated_vertices(phi_center) + centers[0]
        projs = np.max(np.abs(verts @ normals.T))
        max_proj = max(max_proj, projs)
        
        # Ring 1 hexagons
        for i in range(6):
            verts = _get_rotated_vertices(phi_ring1) + centers[i+1]
            projs = np.max(np.abs(verts @ normals.T))
            max_proj = max(max_proj, projs)
            
        # Ring 2 hexagons
        for i in range(4):
            verts = _get_rotated_vertices(phi_ring2) + centers[i+7]
            projs = np.max(np.abs(verts @ normals.T))
            max_proj = max(max_proj, projs)

        return 2.0 * max_proj / np.sqrt(3)

    def objective_reduced(params):
        """
        params: [theta_out, phi_center, phi_ring1, phi_ring2]
        """
        return compute_outer_side_reduced(params[0], params[1], params[2], params[3])

    def smooth_objective_reduced(params, beta=100.0):
        """
        Smooth log-sum-exp approximation of the reduced objective.
        params: [theta_out, phi_center, phi_ring1, phi_ring2]
        """
        theta_out = params[0]
        phi_center = params[1]
        phi_ring1 = params[2]
        phi_ring2 = params[3]
        
        normals = np.array([
            [np.cos(np.deg2rad(30 + k * 60) + theta_out),
             np.sin(np.deg2rad(30 + k * 60) + theta_out)]
            for k in range(6)
        ])
        
        all_dots = []
        
        # Center hexagon
        verts = _get_rotated_vertices(phi_center) + centers[0]
        dots = verts @ normals.T
        all_dots.extend(np.abs(dots).ravel())
        
        # Ring 1 hexagons
        for i in range(6):
            verts = _get_rotated_vertices(phi_ring1) + centers[i+1]
            dots = verts @ normals.T
            all_dots.extend(np.abs(dots).ravel())
            
        # Ring 2 hexagons
        for i in range(4):
            verts = _get_rotated_vertices(phi_ring2) + centers[i+7]
            dots = verts @ normals.T
            all_dots.extend(np.abs(dots).ravel())
        
        all_dots = np.array(all_dots)
        max_val = np.max(all_dots)
        if max_val == 0:
            return 0.0
        shifted = (all_dots - max_val) * beta
        shifted = np.clip(shifted, -500, 500)
        log_sum_exp = max_val + np.log(np.sum(np.exp(shifted)) / len(all_dots)) / beta
        
        return 2.0 * log_sum_exp / np.sqrt(3)

    def smooth_objective_full(params, beta=100.0):
        """
        Smooth log-sum-exp approximation of the full objective.
        params: [theta_out, phi_0, ..., phi_10]
        """
        theta_out = params[0]
        phis = params[1:]
        
        normals = np.array([
            [np.cos(np.deg2rad(30 + k * 60) + theta_out),
             np.sin(np.deg2rad(30 + k * 60) + theta_out)]
            for k in range(6)
        ])
        
        all_dots = []
        for i in range(n_hex):
            verts = _get_rotated_vertices(phis[i]) + centers[i]
            dots = verts @ normals.T
            all_dots.extend(np.abs(dots).ravel())
        
        all_dots = np.array(all_dots)
        max_val = np.max(all_dots)
        if max_val == 0:
            return 0.0
        shifted = (all_dots - max_val) * beta
        shifted = np.clip(shifted, -500, 500)
        log_sum_exp = max_val + np.log(np.sum(np.exp(shifted)) / len(all_dots)) / beta
        
        return 2.0 * log_sum_exp / np.sqrt(3)

    best_score = float('inf')
    best_params_full = None
    
    # Strategy 1: Grid search over theta_out with reduced variables
    theta_steps = np.arange(0, np.deg2rad(30) + 1e-6, np.deg2rad(0.5))
    
    for theta in theta_steps:
        if time.time() > deadline - 40:
            break
            
        x0 = np.array([theta, 0.0, 0.0, 0.0])
        x1 = np.array([theta, np.deg2rad(30), np.deg2rad(30), np.deg2rad(30)])
        x2 = np.array([theta, 0.0, 0.0, np.deg2rad(30)])
        x3 = np.array([theta, np.deg2rad(30), 0.0, 0.0])
        
        starts = [x0, x1, x2, x3]
        
        for start in starts:
            if time.time() > deadline - 35:
                break
            try:
                res = minimize(objective_reduced, start, method='Nelder-Mead',
                               options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 200, 'disp': False})
                
                if res.fun < best_score:
                    best_score = res.fun
                    # Convert reduced to full params
                    theta_out = res.x[0]
                    phi_c = res.x[1]
                    phi_r1 = res.x[2]
                    phi_r2 = res.x[3]
                    
                    full_params = np.zeros(n_hex + 1)
                    full_params[0] = theta_out
                    full_params[1] = phi_c
                    for i in range(6):
                        full_params[i+2] = phi_r1
                    for i in range(4):
                        full_params[i+8] = phi_r2
                        
                    best_params_full = full_params
            except Exception:
                pass

    if best_params_full is None:
        return _fallback_solution()

    # Strategy 2: Smooth surrogate optimization on reduced variables
    if time.time() < deadline - 25:
        for beta_val in [50.0, 100.0, 200.0]:
            if time.time() > deadline - 20:
                break
            try:
                # Convert best full params to reduced
                theta_out = best_params_full[0]
                phi_c = best_params_full[1]
                phi_r1 = best_params_full[2]  # representative ring1
                phi_r2 = best_params_full[8]  # representative ring2
                
                start_red = np.array([theta_out, phi_c, phi_r1, phi_r2])
                
                res = minimize(lambda p: smooth_objective_reduced(p, beta=beta_val), start_red,
                              method='Nelder-Mead',
                              options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})
                
                true_score = objective_reduced(res.x)
                if true_score < best_score - 1e-15:
                    best_score = true_score
                    theta_out = res.x[0]
                    phi_c = res.x[1]
                    phi_r1 = res.x[2]
                    phi_r2 = res.x[3]
                    
                    full_params = np.zeros(n_hex + 1)
                    full_params[0] = theta_out
                    full_params[1] = phi_c
                    for i in range(6):
                        full_params[i+2] = phi_r1
                    for i in range(4):
                        full_params[i+8] = phi_r2
                    best_params_full = full_params
            except Exception:
                pass

    # Strategy 3: Multi-start Powell on full parameters
    if time.time() < deadline - 15:
        # Create diverse starting points
        start_variants = [best_params_full.copy()]
        
        # Add some perturbed variants
        np.random.seed(42)
        for _ in range(5):
            perturbed = best_params_full.copy()
            perturbed[1:] += np.random.uniform(-0.1, 0.1, n_hex)
            start_variants.append(perturbed)
        
        for start in start_variants:
            if time.time() > deadline - 10:
                break
            try:
                res = minimize(objective_full, start, method='Powell',
                              options={'maxiter': 300, 'disp': False, 'ftol': 1e-14, 'xtol': 1e-14})
                if res.fun < best_score - 1e-15:
                    best_score = res.fun
                    best_params_full = res.x.copy()
            except Exception:
                pass

    # Strategy 4: Full Nelder-Mead polish
    if time.time() < deadline - 8:
        try:
            res2 = minimize(objective_full, best_params_full, method='Nelder-Mead',
                           options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 1000, 'disp': False})
            if res2.fun < best_score:
                best_score = res2.fun
                best_params_full = res2.x.copy()
        except Exception:
            pass

    # Strategy 5: Smooth surrogate polish on full parameters
    if time.time() < deadline - 5:
        for beta_val in [200.0, 500.0]:
            if time.time() > deadline - 3:
                break
            try:
                res = minimize(lambda p: smooth_objective_full(p, beta=beta_val), best_params_full,
                              method='Nelder-Mead',
                              options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})
                true_score = objective_full(res.x)
                if true_score < best_score - 1e-15:
                    best_score = true_score
                    best_params_full = res.x.copy()
            except Exception:
                pass

    # Strategy 6: Coordinate descent refinement
    if time.time() < deadline - 3:
        for iteration in range(50):
            if time.time() > deadline - 2:
                break

            improved = False
            for j in range(n_hex + 1):
                if time.time() > deadline - 2:
                    break

                current_val = best_params_full[j]
                best_local = best_score
                best_local_val = current_val

                # Try multiple step sizes
                for delta in [1e-7, -1e-7, 1e-6, -1e-6, 1e-5, -1e-5,
                             1e-4, -1e-4, 1e-3, -1e-3]:
                    test_val = current_val + delta
                    test_param = best_params_full.copy()
                    test_param[j] = test_val
                    try:
                        test_score = objective_full(test_param)
                        if test_score < best_local - 1e-16:
                            best_local = test_score
                            best_local_val = test_val
                    except Exception:
                        pass

                if best_local < best_score - 1e-16:
                    best_score = best_local
                    best_params_full[j] = best_local_val
                    improved = True

            if not improved:
                break

    # Strategy 7: Final high-precision polish
    if time.time() < deadline - 1:
        try:
            res3 = minimize(objective_full, best_params_full, method='Nelder-Mead',
                           options={'xatol': 1e-12, 'fatol': 1e-14, 'maxiter': 2000, 'disp': False})
            if res3.fun < best_score:
                best_score = res3.fun
                best_params_full = res3.x.copy()
        except Exception:
            pass
            
    theta_out = best_params_full[0]
    phis = best_params_full[1:]
    
    inner = []
    for i in range(n_hex):
        inner.append([float(centers[i][0]), float(centers[i][1]), float(np.rad2deg(phis[i]))])
        
    side = best_score + 1e-9
    
    return inner, (0.0, 0.0), float(side), float(np.rad2deg(theta_out))

def _fallback_solution():
    """
    Valid fallback: standard hex lattice with all hexagons aligned.
    """
    eps = 1e-10
    centers = []
    centers.append((0.0, 0.0))
    
    r1 = np.sqrt(3) + eps
    for k in range(6):
        angle = np.deg2rad(k * 60)
        centers.append((r1 * np.cos(angle), r1 * np.sin(angle)))
    
    r2 = 3.0 + 2 * eps
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        centers.append((r2 * np.cos(angle), r2 * np.sin(angle)))
    
    inner = []
    for x, y in centers:
        inner.append([float(x), float(y), 30.0])
    
    # For the standard lattice with all hexagons at 30 degrees,
    # the outer hexagon needs side length 4 (apothem = 2*sqrt(3))
    side = 4.0 + 1e-8
    return inner, (0.0, 0.0), float(side), 30.0
