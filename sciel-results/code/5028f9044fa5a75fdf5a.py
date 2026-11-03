import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import ConvexHull
import itertools

N = 13
TIME_LIMIT = 145.0

def construct():
    start_time = time.time()
    
    def check_time():
        return time.time() - start_time < TIME_LIMIT - 5.0

    # Precompute triple indices for fast area calculation
    tri_i, tri_j, tri_k = [], [], []
    for i in range(N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
                tri_i.append(i)
                tri_j.append(j)
                tri_k.append(k)
    tri_i = np.array(tri_i, dtype=np.int32)
    tri_j = np.array(tri_j, dtype=np.int32)
    tri_k = np.array(tri_k, dtype=np.int32)

    def compute_score(points):
        """
        Computes the score: min triangle area / convex hull area.
        Returns (score, min_area, hull_area).
        """
        try:
            hull = ConvexHull(points)
            area_hull = hull.volume
        except Exception:
            return 0.0, 0.0, 0.0
        if area_hull < 1e-9:
            return 0.0, 0.0, 0.0
        
        x, y = points[:, 0], points[:, 1]
        v1x = x[tri_j] - x[tri_i]
        v1y = y[tri_j] - y[tri_i]
        v2x = x[tri_k] - x[tri_i]
        v2y = y[tri_k] - y[tri_i]
        
        areas = 0.5 * np.abs(v1x * v2y - v1y * v2x)
        min_area = float(np.min(areas))
        score = min_area / area_hull
        return score, min_area, area_hull

    def _soft_min_score(points, beta=100.0):
        """
        Smooth approximation of the negative score for gradient-based optimization.
        """
        try:
            hull = ConvexHull(points)
            area_hull = hull.volume
        except Exception:
            return 1e6
        if area_hull < 1e-9:
            return 1e6
        
        x, y = points[:, 0], points[:, 1]
        v1x = x[tri_j] - x[tri_i]
        v1y = y[tri_j] - y[tri_i]
        v2x = x[tri_k] - x[tri_i]
        v2y = y[tri_k] - y[tri_i]
        
        areas = 0.5 * np.abs(v1x * v2y - v1y * v2x)
        
        # Soft min of areas
        neg_beta_areas = -beta * areas
        max_val = np.max(neg_beta_areas)
        if max_val < -1e6:
            return 1e6
        log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
        soft_min_area = -log_sum_exp / beta
        
        return -(soft_min_area / area_hull)

    # --- Symmetric Parameterization ---
    
    def generate_symmetric_points(params):
        """
        Generates 13 points from symmetric parameters.
        params: [R1, theta1, R2, theta2, R3, theta3]
        
        Structure:
        - 9 points on outer ring (Radius R1, Angle offset theta1)
        - 3 points on middle ring (Radius R2, Angle offset theta2)
        - 1 point at center? No, let's try 9 hull + 4 interior.
        
        Alternative Symmetric Structure:
        - 9 points on hull (Regular Nonagon)
        - 4 points in interior forming a rotated square (Radius R2, Angle theta2)
        
        Let's stick to:
        - 9 points: R1 * exp(i * (2*pi*k/9 + theta1))
        - 4 points: R2 * exp(i * (2*pi*m/4 + theta2))
        
        Total 13 points.
        """
        R1, theta1, R2, theta2 = params
        
        # Outer ring: 9 points
        angles1 = np.linspace(0, 2 * np.pi, 9, endpoint=False) + theta1
        pts1 = np.column_stack([R1 * np.cos(angles1), R1 * np.sin(angles1)])
        
        # Inner ring: 4 points
        angles2 = np.linspace(0, 2 * np.pi, 4, endpoint=False) + theta2
        pts2 = np.column_stack([R2 * np.cos(angles2), R2 * np.sin(angles2)])
        
        points = np.vstack([pts1, pts2])
        return points

    def symmetric_objective(params):
        if not check_time():
            return 1e6
        points = generate_symmetric_points(params)
        # Normalize to unit area? The score is invariant to scaling, 
        # but numerical stability might benefit from keeping R1 ~ 1.
        # We fix R1=1 in the parameter space to reduce dimensions? 
        # Actually, the score is scale-invariant, so we can fix R1=1.
        # Let's fix R1=1.0 to simplify.
        # params: [theta1, R2, theta2]
        # Wait, the relative size R2/R1 matters. So we optimize R2 and angles.
        # Let's use params = [R2, theta1, theta2] with R1 fixed to 1.0.
        
        # Re-evaluating: The function above takes 4 params. 
        # Let's define a specific objective for the reduced space.
        pass

    def reduced_symmetric_objective(params):
        """
        params: [R2, theta1, theta2]
        R1 is fixed to 1.0.
        """
        if not check_time():
            return 1e6
        R2, theta1, theta2 = params
        
        # Generate points
        angles1 = np.linspace(0, 2 * np.pi, 9, endpoint=False) + theta1
        pts1 = np.column_stack([np.cos(angles1), np.sin(angles1)]) # R1 = 1
        
        angles2 = np.linspace(0, 2 * np.pi, 4, endpoint=False) + theta2
        pts2 = np.column_stack([R2 * np.cos(angles2), R2 * np.sin(angles2)])
        
        points = np.vstack([pts1, pts2])
        
        score, _, _ = compute_score(points)
        return -score

    best_points = None
    best_score = -float('inf')

    # --- Phase 1: Global Search in Symmetric Space ---
    
    # Bounds for R2 (0 to 0.8), theta1 (0 to 2pi), theta2 (0 to 2pi)
    bounds_sym = [(0.0, 0.8), (0.0, 2 * np.pi), (0.0, 2 * np.pi)]
    
    if check_time():
        try:
            # Differential Evolution for global search
            res_de = opt.differential_evolution(
                reduced_symmetric_objective, 
                bounds_sym, 
                seed=42, 
                maxiter=50, 
                tol=1e-8,
                polish=False
            )
            best_params_sym = res_de.x
            best_score_sym = -res_de.fun
            
            # Refine with L-BFGS-B
            if check_time():
                res_lbfgs = opt.minimize(
                    reduced_symmetric_objective,
                    best_params_sym,
                    method='L-BFGS-B',
                    bounds=bounds_sym,
                    options={'maxiter': 200, 'ftol': 1e-12}
                )
                best_params_sym = res_lbfgs.x
                best_score_sym = -res_lbfgs.fun
                
            # Refine with Nelder-Mead
            if check_time():
                res_nm = opt.minimize(
                    reduced_symmetric_objective,
                    best_params_sym,
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'adaptive': True, 'xatol': 1e-10, 'fatol': 1e-12}
                )
                best_params_sym = res_nm.x
                best_score_sym = -res_nm.fun
            
            # Generate points from best symmetric params
            R2, theta1, theta2 = best_params_sym
            angles1 = np.linspace(0, 2 * np.pi, 9, endpoint=False) + theta1
            pts1 = np.column_stack([np.cos(angles1), np.sin(angles1)])
            angles2 = np.linspace(0, 2 * np.pi, 4, endpoint=False) + theta2
            pts2 = np.column_stack([R2 * np.cos(angles2), R2 * np.sin(angles2)])
            candidate_points = np.vstack([pts1, pts2])
            
            cand_score, _, _ = compute_score(candidate_points)
            if cand_score > best_score:
                best_score = cand_score
                best_points = candidate_points.copy()
                
        except Exception as e:
            pass

    # --- Phase 2: Local Refinement of Raw Coordinates ---
    
    if best_points is not None and check_time():
        # Use the symmetric result as a start for raw coordinate optimization
        # This allows breaking symmetry if it improves the score
        
        start_pts = best_points.copy()
        
        # L-BFGS-B with soft min
        for beta in [50.0, 100.0, 200.0, 500.0]:
            if not check_time():
                break
            try:
                res = opt.minimize(
                    lambda x: _soft_min_score(x.reshape(-1, 2), beta),
                    start_pts.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
                )
                cand = res.x.reshape(-1, 2)
                s, _, _ = compute_score(cand)
                if s > best_score:
                    best_score = s
                    best_points = cand.copy()
                    start_pts = cand.copy()
            except Exception:
                pass

        # Powell Method
        if check_time():
            try:
                res = opt.minimize(
                    lambda x: -compute_score(x.reshape(-1, 2))[0],
                    start_pts.flatten(),
                    method='Powell',
                    options={'maxiter': 300, 'ftol': 1e-12, 'xtol': 1e-10}
                )
                cand = res.x.reshape(-1, 2)
                s, _, _ = compute_score(cand)
                if s > best_score:
                    best_score = s
                    best_points = cand.copy()
            except Exception:
                pass

        # Nelder-Mead Final Polish
        if check_time():
            try:
                res = opt.minimize(
                    lambda x: -compute_score(x.reshape(-1, 2))[0],
                    best_points.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'adaptive': True, 'xatol': 1e-12, 'fatol': 1e-14}
                )
                cand = res.x.reshape(-1, 2)
                s, _, _ = compute_score(cand)
                if s > best_score:
                    best_score = s
                    best_points = cand.copy()
            except Exception:
                pass

    # --- Fallback if Symmetric Search Failed ---
    
    if best_points is None:
        # Use a regular 13-gon as fallback
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        best_score, _, _ = compute_score(best_points)
        
        # Quick local optimization on the regular polygon
        if check_time():
            try:
                res = opt.minimize(
                    lambda x: -compute_score(x.reshape(-1, 2))[0],
                    best_points.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'adaptive': True}
                )
                cand = res.x.reshape(-1, 2)
                s, _, _ = compute_score(cand)
                if s > best_score:
                    best_score = s
                    best_points = cand.copy()
            except Exception:
                pass

    return best_points
