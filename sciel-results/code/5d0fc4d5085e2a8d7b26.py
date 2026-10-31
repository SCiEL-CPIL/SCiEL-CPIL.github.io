import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import pdist
import time

N = 16
D = 2

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s buffer for 150s limit

    def exact_score(params):
        """
        Exact objective function in reduced parameter space.
        params: (28,) array of 14 points (x,y) for indices 2..15.
        Points 0 and 1 are fixed at (0,0) and (1,0).
        """
        if time.time() > deadline:
            return 1e10
            
        pts = np.zeros((N, 2))
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        pts[2:] = params.reshape(14, 2)
        
        # Check for distinct points
        # pdist handles this, but if min is 0, score is inf
        d = pdist(pts, 'euclidean')
        min_d = np.min(d)
        if min_d < 1e-8:
            return 1e10
        max_d = np.max(d)
        return (max_d / min_d) ** 2

    def soft_score(params, beta=50.0):
        """
        Smooth surrogate for L-BFGS-B.
        """
        if time.time() > deadline:
            return 1e10
            
        pts = np.zeros((N, 2))
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        pts[2:] = params.reshape(14, 2)
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        iu = np.triu_indices(N, 1)
        d_sq = dists_sq[iu]
        
        d_sq = np.maximum(d_sq, 1e-12)
        
        # Soft max
        max_d_sq = np.max(d_sq)
        soft_max_d_sq = max_d_sq + (1.0 / beta) * np.log(np.sum(np.exp(beta * (d_sq - max_d_sq))))
        
        # Soft min
        min_d_sq = np.min(d_sq)
        soft_min_d_sq = min_d_sq - (1.0 / beta) * np.log(np.sum(np.exp(-beta * (d_sq - min_d_sq))))
        
        if soft_min_d_sq < 1e-8:
            return 1e10
            
        return soft_max_d_sq / soft_min_d_sq

    def prepare_params(pts):
        """
        Transform a set of 16 points into reduced parameter space.
        Finds closest pair, aligns to (0,0) and (1,0).
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
        R_mat = np.array([[cos_t, sin_t], [-sin_t, cos_t]])

        transformed = (pts - p0) @ R_mat.T / norm_v
        
        perm = list(range(N))
        perm[0] = i_min
        perm[1] = j_min
        remaining = [i for i in range(N) if i != i_min and i != j_min]
        perm[2:] = remaining
        
        transformed = transformed[perm]
        
        # transformed[0] is [0,0], transformed[1] is [1,0]
        params = transformed[2:].flatten()
        return params, perm

    def polish(params, tag=""):
        """
        Apply Powell then L-BFGS-B to a candidate parameter vector.
        Returns the best params and score found.
        """
        if time.time() > deadline - 10:
            return params, exact_score(params)
            
        best_p = params.copy()
        best_s = exact_score(params)
        
        # Phase 1: Powell (Derivative-free, robust for non-smooth)
        try:
            res_powell = minimize(exact_score, best_p, method='Powell', 
                                  options={'maxiter': 200, 'maxfev': 2000, 'disp': False})
            if res_powell.fun < best_s:
                best_s = res_powell.fun
                best_p = res_powell.x.copy()
        except Exception:
            pass

        # Phase 2: L-BFGS-B on Soft Objective (Gradient-based polish)
        if time.time() < deadline - 5:
            try:
                # Use high beta for accuracy
                res_lbfgs = minimize(lambda p: soft_score(p, beta=100.0), best_p, method='L-BFGS-B',
                                     options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False})
                exact_s = exact_score(res_lbfgs.x)
                if exact_s < best_s:
                    best_s = exact_s
                    best_p = res_lbfgs.x.copy()
            except Exception:
                pass
                
        return best_p, best_s

    # 1. Generate Seeds
    def generate_hex_seed():
        R = 5
        hex_pts = []
        for q in range(-R, R + 1):
            for r in range(-R, R + 1):
                x = q + 0.5 * r
                y = (np.sqrt(3) / 2) * r
                hex_pts.append((x, y))
        hex_pts = np.array(hex_pts)
        dists = np.sum(hex_pts**2, axis=1)
        idx = np.argsort(dists)[:16]
        return hex_pts[idx]

    def generate_circular_seed():
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

    def generate_rings_seed():
        inner_angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
        outer_angles = np.linspace(0, 2*np.pi, 8, endpoint=False) + np.pi/8
        r_in, r_out = 0.6, 1.3
        inner_pts = np.column_stack([r_in * np.cos(inner_angles), r_in * np.sin(inner_angles)])
        outer_pts = np.column_stack([r_out * np.cos(outer_angles), r_out * np.sin(outer_angles)])
        return np.vstack([inner_pts, outer_pts])

    def generate_sqrt_radial_seed():
        """
        4x4 grid with square-root radial scaling to mitigate corner-center disparity.
        """
        grid_pts = []
        for i in range(4):
            for j in range(4):
                x = (i - 1.5) * 0.8
                y = (j - 1.5) * 0.8
                grid_pts.append([x, y])
        grid_pts = np.array(grid_pts)
        
        # Apply square-root radial scaling: r_new = r_old * sqrt(r_old / r_max)
        center = grid_pts.mean(axis=0)
        grid_pts = grid_pts - center
        radii = np.linalg.norm(grid_pts, axis=1)
        max_r = np.max(radii)
        if max_r > 1e-8:
            scale_factors = np.sqrt(radii / max_r)
            # Only scale outward, not inward
            scale_factors = np.maximum(scale_factors, 0.5)
            for k in range(16):
                if radii[k] > 1e-8:
                    grid_pts[k] *= scale_factors[k]
        
        # Small angular perturbation per quadrant to break symmetry
        for k in range(16):
            x, y = grid_pts[k]
            if abs(x) < 1e-8 and abs(y) < 1e-8:
                continue
            quadrant = 0
            if x >= 0 and y >= 0:
                quadrant = 0
            elif x < 0 and y >= 0:
                quadrant = 1
            elif x < 0 and y < 0:
                quadrant = 2
            else:
                quadrant = 3
            angle_offset = quadrant * np.pi / 16
            cos_a = np.cos(angle_offset)
            sin_a = np.sin(angle_offset)
            new_x = x * cos_a - y * sin_a
            new_y = x * sin_a + y * cos_a
            grid_pts[k] = [new_x, new_y]
        
        return grid_pts

    seeds = [
        generate_hex_seed(),
        generate_circular_seed(),
        generate_rings_seed(),
        generate_sqrt_radial_seed()
    ]
    
    # Convert seeds to reduced space
    seed_params = []
    for s in seeds:
        p, _ = prepare_params(s)
        if p is not None:
            seed_params.append(p)

    # 2. Global Search with Differential Evolution
    best_params = None
    best_score = 1e10
    
    # Bounds for DE: Points should be reasonably close.
    # Min dist is 1 (scaled). Max dist is likely < 5.
    bounds = [(-4.0, 4.0)] * 28
    
    # Allocate time for DE based on remaining budget
    de_budget = min(50.0, max(20.0, deadline - time.time() - 30))
    if de_budget > 10:
        try:
            de_result = differential_evolution(
                exact_score, 
                bounds, 
                maxiter=150, 
                popsize=20, 
                tol=1e-6, 
                seed=42,
                disp=False
            )
            if de_result.fun < best_score:
                best_score = de_result.fun
                best_params = de_result.x.copy()
        except Exception:
            pass

    # 3. Polish the DE result
    if best_params is not None:
        p_polished, s_polished = polish(best_params, "de")
        if s_polished < best_score:
            best_score = s_polished
            best_params = p_polished

    # 4. Polish the Seeds (if time remains)
    if time.time() < deadline - 20:
        for i, p in enumerate(seed_params):
            if time.time() > deadline - 10:
                break
            p_polished, s_polished = polish(p, f"seed_{i}")
            if s_polished < best_score:
                best_score = s_polished
                best_params = p_polished

    # 5. Final Polish on the absolute best
    if best_params is not None and time.time() < deadline - 10:
        # Phase 5a: Tight L-BFGS pass
        try:
            res_final = minimize(lambda p: soft_score(p, beta=200.0), best_params, method='L-BFGS-B',
                                 options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-10, 'disp': False})
            exact_s = exact_score(res_final.x)
            if exact_s < best_score:
                best_score = exact_s
                best_params = res_final.x.copy()
        except Exception:
            pass
        
        # Phase 5b: Final exact Powell polish
        if time.time() < deadline - 5:
            try:
                res_powell_final = minimize(exact_score, best_params, method='Powell',
                                            options={'maxiter': 300, 'maxfev': 3000, 'disp': False})
                exact_s = exact_score(res_powell_final.x)
                if exact_s < best_score:
                    best_score = exact_s
                    best_params = res_powell_final.x.copy()
            except Exception:
                pass

    # Reconstruct final points
    if best_params is not None:
        pts = np.zeros((N, 2))
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        pts[2:] = best_params.reshape(14, 2)
        
        # Center the points for numerical stability
        pts -= pts.mean(axis=0)
        
        return pts
    else:
        # Fallback
        pts = generate_hex_seed()
        pts -= pts.mean(axis=0)
        return pts
