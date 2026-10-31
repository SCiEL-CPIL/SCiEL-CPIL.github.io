import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import pdist
import time

N = 16
D = 2

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit

    def score_points(pts):
        d = pdist(pts, 'euclidean')
        min_d = np.min(d)
        if min_d < 1e-8:
            return 1e10
        max_d = np.max(d)
        return (max_d / min_d) ** 2

    def prepare_params(pts):
        d = pdist(pts, 'euclidean')
        min_idx = np.argmin(d)
        iu = np.triu_indices(N, 1)
        i_min, j_min = iu[0][min_idx], iu[1][min_idx]
        
        p0 = pts[i_min]
        p1 = pts[j_min]
        
        v = p1 - p0
        norm_v = np.linalg.norm(v)
        if norm_v < 1e-8:
            return None, None

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
        
        params = transformed[2:].flatten()
        return params, perm

    def objective(params):
        if time.time() > deadline:
            return 1e10
        pts = np.zeros((N, 2))
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        pts[2:] = params.reshape(-1, 2)
        return score_points(pts)

    def soft_objective(params, beta=80.0):
        if time.time() > deadline:
            return 1e10
        pts = np.zeros((N, 2))
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        pts[2:] = params.reshape(-1, 2)
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        iu = np.triu_indices(N, 1)
        d_sq = dists_sq[iu]
        
        d_sq = np.maximum(d_sq, 1e-12)
        
        max_d_sq = np.max(d_sq)
        soft_max_d_sq = max_d_sq + (1.0 / beta) * np.log(np.sum(np.exp(beta * (d_sq - max_d_sq))))
        
        min_d_sq = np.min(d_sq)
        soft_min_d_sq = min_d_sq - (1.0 / beta) * np.log(np.sum(np.exp(-beta * (d_sq - min_d_sq))))
        
        if soft_min_d_sq < 1e-8:
            return 1e10
            
        return soft_max_d_sq / soft_min_d_sq

    def generate_triangular_seed():
        """Generate optimal subset of triangular lattice."""
        # Triangular lattice points
        s = np.sqrt(3) / 2
        pts = []
        for i in range(-3, 4):
            for j in range(-3, 4):
                x = i + 0.5 * j
                y = s * j
                pts.append((x, y))
        pts = np.array(pts)
        
        # Select 16 points that form the most compact cluster
        # Use a hexagonal number pattern: center + rings
        # Ring 0: 1 point (center)
        # Ring 1: 6 points
        # Ring 2: 12 points -> total 19, take 16
        
        # Better: select based on distance from centroid of a hexagonal arrangement
        # Center at origin
        center = np.array([0.0, 0.0])
        dists = np.linalg.norm(pts - center, axis=1)
        idx = np.argsort(dists)[:16]
        return pts[idx]

    def generate_ring_seed():
        """Two concentric rings: 8 inner, 8 outer."""
        angles_inner = np.linspace(0, 2*np.pi, 8, endpoint=False)
        angles_outer = np.linspace(0, 2*np.pi, 8, endpoint=False) + np.pi/8
        r_inner = 0.577  # ~1/sqrt(3)
        r_outer = 1.5
        inner_pts = np.column_stack([r_inner * np.cos(angles_inner), r_inner * np.sin(angles_inner)])
        outer_pts = np.column_stack([r_outer * np.cos(angles_outer), r_outer * np.sin(angles_outer)])
        return np.vstack([inner_pts, outer_pts])

    def generate_hex_seed():
        """Hexagonal lattice subset."""
        R = 3
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

    best_score = 1e10
    best_params = None
    best_perm = None

    # Generate multiple seeds
    seeds = [
        generate_triangular_seed(),
        generate_hex_seed(),
        generate_ring_seed()
    ]
    
    # Add perturbed versions
    rng = np.random.default_rng(42)
    for base in [generate_triangular_seed(), generate_hex_seed()]:
        for scale in [0.03, 0.08, 0.15]:
            perturbed = base + rng.normal(0, scale, size=(N, 2))
            seeds.append(perturbed)

    # Process each seed
    for seed_pts in seeds:
        if time.time() > deadline - 20:
            break
            
        params, perm = prepare_params(seed_pts)
        if params is None:
            continue
            
        curr_score = objective(params)
        if curr_score < best_score:
            best_score = curr_score
            best_params = params.copy()
            best_perm = perm.copy()
            
        # Stage 1: Powell on hard objective
        try:
            res = minimize(objective, params, method='Powell',
                           options={'maxiter': 150, 'maxfev': 1500, 'disp': False})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
                best_perm = perm.copy()
        except Exception:
            pass

        # Stage 2: L-BFGS-B on soft objective for gradient-based refinement
        if time.time() < deadline - 15:
            try:
                res2 = minimize(soft_objective, best_params, method='L-BFGS-B',
                                options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10, 'disp': False})
                exact_score = objective(res2.x)
                if exact_score < best_score:
                    best_score = exact_score
                    best_params = res2.x.copy()
                    best_perm = perm.copy()
            except Exception:
                pass

    # Final high-precision polish
    if best_params is not None and time.time() < deadline - 8:
        try:
            def final_soft(params, beta=200.0):
                return soft_objective(params, beta)
            
            res3 = minimize(final_soft, best_params, method='L-BFGS-B',
                            options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-12, 'disp': False})
            exact_score = objective(res3.x)
            if exact_score < best_score:
                best_score = exact_score
                best_params = res3.x.copy()
                best_perm = best_perm.copy()
        except Exception:
            pass

    # Reconstruct final points
    if best_params is not None:
        pts = np.zeros((N, 2))
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        pts[2:] = best_params.reshape(-1, 2)
        
        inv_perm = np.argsort(best_perm)
        final_pts = pts[inv_perm]
        
        final_pts -= final_pts.mean(axis=0)
        
        # Verify distinctness
        d = pdist(final_pts, 'euclidean')
        if np.min(d) < 1e-8:
            final_pts += rng.normal(0, 1e-6, final_pts.shape)
        
        return final_pts
    else:
        pts = generate_triangular_seed()
        pts -= pts.mean(axis=0)
        return pts
