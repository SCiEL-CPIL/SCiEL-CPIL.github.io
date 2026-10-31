import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import pdist
import time

N = 16
D = 2
DEADLINE = 110.0  # Safety margin for 120s limit

def construct():
    start_time = time.time()
    
    def get_elapsed():
        return time.time() - start_time

    def check_time(budget=5.0):
        return get_elapsed() < DEADLINE - budget

    # Reduced parameter space:
    # Fix p0 = (0,0), p1 = (1,0).
    # Params are (x2, y2, ..., x15, y15) -> shape (28,)
    
    def params_to_points(params):
        pts = np.zeros((N, 2))
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        pts[2:] = params.reshape(14, 2)
        return pts

    def exact_score(params):
        if not check_time(1.0):
            return 1e10
        pts = params_to_points(params)
        d = pdist(pts, 'euclidean')
        min_d = np.min(d)
        if min_d < 1e-8:
            return 1e10
        max_d = np.max(d)
        return (max_d / min_d) ** 2

    def soft_score(params, T=50.0):
        """
        Smooth surrogate for (max_d / min_d)^2.
        Uses log-sum-exp for stable soft max/min.
        Objective: soft_max_d_sq / soft_min_d_sq
        """
        if not check_time(1.0):
            return 1e10
        pts = params_to_points(params)
        # Compute squared distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        d_sq = np.sum(diff**2, axis=2)
        iu = np.triu_indices(N, 1)
        d_sq = d_sq[iu]
        
        # Clip to avoid log(0) or negative issues, though d_sq >= 0
        d_sq = np.maximum(d_sq, 1e-12)
        
        # Soft Max
        m = np.max(d_sq)
        soft_max_d_sq = m + (1.0 / T) * np.log(np.sum(np.exp(T * (d_sq - m))))
        
        # Soft Min
        n = np.min(d_sq)
        soft_min_d_sq = n - (1.0 / T) * np.log(np.sum(np.exp(-T * (d_sq - n))))
        
        if soft_min_d_sq < 1e-8:
            return 1e10
            
        return soft_max_d_sq / soft_min_d_sq

    def polish(params, tag=""):
        """
        Multi-stage polishing:
        1. Nelder-Mead (robust local search)
        2. Powell (derivative-free)
        3. L-BFGS-B with temperature annealing (soft objective)
        4. Final Powell (exact objective)
        """
        if not check_time(10.0):
            return params, exact_score(params)

        best_p = params.copy()
        best_s = exact_score(params)

        # Stage 1: Nelder-Mead
        if check_time(10.0):
            try:
                res_nm = minimize(exact_score, best_p, method='Nelder-Mead', 
                                  options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8, 'disp': False})
                if res_nm.fun < best_s:
                    best_s = res_nm.fun
                    best_p = res_nm.x.copy()
            except Exception:
                pass

        # Stage 2: Powell
        if check_time(10.0):
            try:
                res_pw = minimize(exact_score, best_p, method='Powell', 
                                  options={'maxiter': 200, 'maxfev': 2000, 'disp': False})
                if res_pw.fun < best_s:
                    best_s = res_pw.fun
                    best_p = res_pw.x.copy()
            except Exception:
                pass

        # Stage 3: L-BFGS-B with Annealing
        temperatures = [20.0, 50.0, 100.0, 200.0]
        if check_time(15.0):
            for T in temperatures:
                if not check_time(5.0):
                    break
                try:
                    res_lbfgs = minimize(lambda p: soft_score(p, T=T), best_p, method='L-BFGS-B',
                                         options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False})
                    exact_s = exact_score(res_lbfgs.x)
                    if exact_s < best_s:
                        best_s = exact_s
                        best_p = res_lbfgs.x.copy()
                except Exception:
                    pass

        # Stage 4: Final Exact Polish
        if check_time(5.0):
            try:
                res_final = minimize(exact_score, best_p, method='Powell',
                                     options={'maxiter': 300, 'maxfev': 5000, 'disp': False})
                exact_s = exact_score(res_final.x)
                if exact_s < best_s:
                    best_s = exact_s
                    best_p = res_final.x.copy()
            except Exception:
                pass

        return best_p, best_s

    def prepare_params(pts):
        """
        Transform a set of 16 points into reduced parameter space.
        Finds the closest pair, aligns them to (0,0) and (1,0).
        """
        d = pdist(pts, 'euclidean')
        min_idx = np.argmin(d)
        iu = np.triu_indices(N, 1)
        i_min, j_min = iu[0][min_idx], iu[1][min_idx]
        
        p0 = pts[i_min]
        p1 = pts[j_min]
        
        v = p1 - p0
        norm_v = np.linalg.norm(v)
        if norm_v < 1e-8:
            return None

        cos_t = v[0] / norm_v
        sin_t = v[1] / norm_v
        # Rotation matrix to align v with x-axis
        # We want R such that R @ v = [norm_v, 0]
        # v = [cos_t, sin_t] * norm_v
        # R = [[cos_t, sin_t], [-sin_t, cos_t]]
        R_mat = np.array([[cos_t, sin_t], [-sin_t, cos_t]])

        # Transform all points
        transformed = (pts - p0) @ R_mat.T / norm_v
        
        # Reorder so that i_min is 0 and j_min is 1
        perm = list(range(N))
        perm[0] = i_min
        perm[1] = j_min
        remaining = [i for i in range(N) if i != i_min and i != j_min]
        perm[2:] = remaining
        
        transformed = transformed[perm]
        
        # transformed[0] is [0,0], transformed[1] is [1,0]
        params = transformed[2:].flatten()
        return params

    # --- Seed Generation ---
    
    def generate_hex_seed():
        # Hexagonal lattice points
        R = 5
        hex_pts = []
        for q in range(-R, R + 1):
            for r in range(-R, R + 1):
                x = q + 0.5 * r
                y = (np.sqrt(3) / 2) * r
                hex_pts.append((x, y))
        hex_pts = np.array(hex_pts)
        # Select 16 points closest to center
        dists = np.sum(hex_pts**2, axis=1)
        idx = np.argsort(dists)[:16]
        return hex_pts[idx]

    def generate_grid_seed():
        # 4x4 grid
        x = np.linspace(-1.5, 1.5, 4)
        y = np.linspace(-1.5, 1.5, 4)
        X, Y = np.meshgrid(x, y)
        pts = np.column_stack([X.flatten(), Y.flatten()])
        return pts

    def generate_circular_seed():
        # 3 rings: 4, 6, 6
        pts = []
        for k in range(4):
            theta = k * np.pi / 2
            pts.append([0.3 * np.cos(theta), 0.3 * np.sin(theta)])
        for k in range(6):
            theta = k * 2 * np.pi / 6 + np.pi / 6
            pts.append([0.7 * np.cos(theta), 0.7 * np.sin(theta)])
        for k in range(6):
            theta = k * 2 * np.pi / 6
            pts.append([1.2 * np.cos(theta), 1.2 * np.sin(theta)])
        return np.array(pts)

    def generate_random_seed():
        # Random points in a square
        return np.random.rand(16, 2) * 4 - 2

    seeds = [
        generate_hex_seed(),
        generate_grid_seed(),
        generate_circular_seed()
    ]
    
    # Add a few random seeds
    for _ in range(3):
        seeds.append(generate_random_seed())

    seed_params = []
    for s in seeds:
        p = prepare_params(s)
        if p is not None:
            seed_params.append(p)

    # --- Main Optimization ---
    
    best_params = None
    best_score = 1e10

    # 1. Polish Seeds
    for i, p in enumerate(seed_params):
        if not check_time(20.0):
            break
        p_polished, s_polished = polish(p, f"seed_{i}")
        if s_polished < best_score:
            best_score = s_polished
            best_params = p_polished

    # 2. Differential Evolution
    if check_time(30.0):
        bounds = [(-4.0, 4.0)] * 28
        try:
            de_result = differential_evolution(
                exact_score, 
                bounds, 
                maxiter=100, 
                popsize=15, 
                tol=1e-5, 
                seed=42,
                disp=False
            )
            if de_result.fun < best_score:
                best_score = de_result.fun
                best_params = de_result.x.copy()
        except Exception:
            pass

    # 3. Polish the Best Result
    if best_params is not None and check_time(10.0):
        p_final, s_final = polish(best_params, "final")
        if s_final < best_score:
            best_score = s_final
            best_params = p_final

    # 4. Reconstruct Points
    if best_params is not None:
        pts = params_to_points(best_params)
        # Center the points for numerical stability (optional, but good practice)
        pts -= pts.mean(axis=0)
        return pts
    else:
        # Fallback to hex seed
        pts = generate_hex_seed()
        pts -= pts.mean(axis=0)
        return pts
