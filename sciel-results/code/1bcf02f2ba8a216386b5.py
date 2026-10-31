import numpy as np
from scipy.optimize import minimize
import time

N = 16
D = 2

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    def raw_score(pts):
        """Compute the raw score: (max pairwise distance / min pairwise distance)^2."""
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        iu = np.triu_indices(N, 1)
        d_sq = dists_sq[iu]
        min_d_sq = np.min(d_sq)
        if min_d_sq < 1e-12:
            return 1e10
        max_d_sq = np.max(d_sq)
        return max_d_sq / min_d_sq

    def smooth_score(pts, beta=100.0):
        """
        Smooth approximation of the score using log-sum-exp on distances.
        This helps with gradient-based optimization.
        """
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        iu = np.triu_indices(N, 1)
        d = dists[iu]
        
        if np.any(d < 1e-8):
            return 1e10
            
        # Soft max
        max_d = np.max(d)
        d_shifted_max = d - max_d
        exp_terms_max = np.exp(beta * d_shifted_max)
        sum_exp_max = np.sum(exp_terms_max)
        if sum_exp_max < 1e-300:
            return 1e10
        max_d_soft = max_d + (1.0 / beta) * np.log(sum_exp_max)
        
        # Soft min
        min_d = np.min(d)
        d_shifted_min = d - min_d
        exp_terms_min = np.exp(-beta * d_shifted_min)
        sum_exp_min = np.sum(exp_terms_min)
        if sum_exp_min < 1e-300:
            return 1e10
        min_d_soft = min_d - (1.0 / beta) * np.log(sum_exp_min)
        
        if min_d_soft < 1e-8:
            return 1e10
            
        ratio = max_d_soft / min_d_soft
        return ratio**2

    def make_hexagonal_cluster(n):
        """
        Generate a hexagonal lattice cluster of n points centered at origin.
        """
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.5, np.sqrt(3) / 2])
        
        # Generate a large enough grid of lattice points
        pts = []
        for i in range(-5, 6):
            for j in range(-5, 6):
                p = i * v1 + j * v2
                pts.append(p)
        pts = np.array(pts)
        
        # Center the cluster
        pts -= pts.mean(axis=0)
        
        # Sort by distance to center and pick the closest n
        dists = np.linalg.norm(pts, axis=1)
        order = np.argsort(dists)
        return pts[order[:n]]

    def make_grid(n=4):
        """Generate an n x n grid centered at origin."""
        x, y = np.meshgrid(np.linspace(-1.5, 1.5, n), np.linspace(-1.5, 1.5, n))
        return np.column_stack([x.ravel(), y.ravel()])

    def make_circles():
        """Generate two concentric circles of 8 points each."""
        angles1 = np.linspace(0, 2*np.pi, 8, endpoint=False)
        circle1 = np.column_stack([np.cos(angles1), np.sin(angles1)])
        angles2 = np.linspace(0, 2*np.pi, 8, endpoint=False) + np.pi/8
        circle2 = np.column_stack([np.cos(angles2), np.sin(angles2)]) * 0.5
        return np.vstack([circle1, circle2])

    def normalize_and_fix(pts):
        """
        Translate, rotate, and scale the points such that:
        - Point 0 is at (0,0)
        - Point 1 is at (1,0)
        Returns the parameters for the remaining N-2 points.
        """
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        iu = np.triu_indices(N, 1)
        d = dists[iu]
        
        min_idx = np.argmin(d)
        pair = np.unravel_index(min_idx, (N, N))
        i0, i1 = pair[0], pair[1]
        if i0 > i1:
            i0, i1 = i1, i0
        
        p0 = pts[i0]
        p1 = pts[i1]
        
        v = p1 - p0
        norm_v = np.linalg.norm(v)
        
        if norm_v < 1e-8:
            return np.zeros((N-2)*2)

        cos_t = v[0] / norm_v
        sin_t = v[1] / norm_v
        R_mat = np.array([[cos_t, sin_t], [-sin_t, cos_t]])
        
        transformed = (pts - p0) @ R_mat.T / norm_v
        
        perm = list(range(N))
        perm[0] = i0
        perm[1] = i1
        remaining = [i for i in range(N) if i != i0 and i != i1]
        perm[2:] = remaining
        
        transformed = transformed[perm]
        
        init_params = transformed[2:].flatten()
        return init_params

    def get_pts_from_params(params):
        """Reconstruct points from fixed points and params."""
        pts = np.zeros((N, 2))
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        pts[2:] = params.reshape(-1, 2)
        return pts

    def objective_from_params(params, beta=None):
        """Objective function for optimization in parameter space."""
        pts = get_pts_from_params(params)
        if beta is None:
            return raw_score(pts)
        else:
            return smooth_score(pts, beta)

    # Generate initial candidates
    candidates = []
    
    # 1. Hexagonal cluster
    hex_pts = make_hexagonal_cluster(N)
    candidates.append(hex_pts)
    
    # 2. Rotated hexagonal clusters
    for angle in [10, 20, 30, 45, 60, 90]:
        rad = np.deg2rad(angle)
        rot = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
        candidates.append(hex_pts @ rot.T)
        
    # 3. Perturbed hexagonal clusters
    np.random.seed(42)
    for _ in range(3):
        noise = np.random.normal(0, 0.1, hex_pts.shape)
        candidates.append(hex_pts + noise)
        
    # 4. 4x4 grid
    grid_pts = make_grid(4)
    candidates.append(grid_pts)
    
    # 5. Circular distribution
    circ_pts = make_circles()
    candidates.append(circ_pts)
    
    # 6. Randomized hexagonal
    for _ in range(2):
        noise = np.random.normal(0, 0.2, hex_pts.shape)
        candidates.append(hex_pts + noise)

    bnds = [(-5, 5)] * (2 * (N - 2))
    best_val = np.inf
    best_pts = None

    # Optimize each candidate
    for idx, cand in enumerate(candidates):
        if time.time() > deadline - 20:
            break
            
        init_params = normalize_and_fix(cand)
        
        # Check initial score
        current_score = raw_score(get_pts_from_params(init_params))
        
        # Phase 1: Coarse optimization with smooth objective
        if time.time() < deadline - 40:
            try:
                res1 = minimize(objective_from_params, init_params, method='L-BFGS-B', 
                               bounds=bnds, args=(10.0,),
                               options={'maxiter': 100, 'ftol': 1e-6, 'gtol': 1e-4})
                if res1.fun < 1e9:
                    candidate_params = res1.x
                    cand_score = raw_score(get_pts_from_params(candidate_params))
                    if cand_score < current_score:
                        current_score = cand_score
                        init_params = candidate_params
            except Exception:
                pass

        # Phase 2: Medium optimization
        if time.time() < deadline - 20:
            try:
                res2 = minimize(objective_from_params, init_params, method='L-BFGS-B', 
                               bounds=bnds, args=(50.0,),
                               options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6})
                if res2.fun < 1e9:
                    candidate_params = res2.x
                    cand_score = raw_score(get_pts_from_params(candidate_params))
                    if cand_score < current_score:
                        current_score = cand_score
                        init_params = candidate_params
            except Exception:
                pass

        # Phase 3: Fine optimization
        if time.time() < deadline - 10:
            try:
                res3 = minimize(objective_from_params, init_params, method='L-BFGS-B', 
                               bounds=bnds, args=(100.0,),
                               options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8})
                if res3.fun < 1e9:
                    candidate_params = res3.x
                    cand_score = raw_score(get_pts_from_params(candidate_params))
                    if cand_score < current_score:
                        current_score = cand_score
                        init_params = candidate_params
            except Exception:
                pass
        
        # Final exact polish
        if time.time() < deadline - 5:
            try:
                res4 = minimize(objective_from_params, init_params, method='Powell',
                               options={'maxiter': 2000, 'maxfev': 20000})
                if res4.fun < current_score:
                    current_score = res4.fun
                    init_params = res4.x
            except Exception:
                pass
        
        # Update best
        if current_score < best_val:
            best_val = current_score
            best_pts = get_pts_from_params(init_params)

    # Multiple local restarts from the best solution found
    np.random.seed(43)
    for _ in range(10):
        if time.time() > deadline - 15:
            break
        
        # Perturb the best solution
        noise = np.random.normal(0, 0.1, best_pts.shape)
        perturbed_pts = best_pts + noise
        
        # Normalize
        init_params = normalize_and_fix(perturbed_pts)
        
        # Local optimization
        try:
            res = minimize(objective_from_params, init_params, method='Powell',
                           options={'maxiter': 500, 'maxfev': 5000})
            if res.fun < best_val:
                best_val = res.fun
                best_pts = get_pts_from_params(res.x)
        except Exception:
            pass

    # Final local polish using Powell on the best candidate found
    if time.time() < deadline - 5:
        try:
            current = best_pts.copy()
            result = minimize(
                lambda x: raw_score(x),
                current,
                method='Powell',
                options={'maxiter': 5000, 'maxfev': 50000}
            )
            
            final_score = raw_score(result.x)
            current_raw_score = raw_score(current)
            
            if final_score < current_raw_score:
                best_pts = result.x.copy()
        except Exception:
            pass

    # Ensure we return valid distinct points
    return best_pts
