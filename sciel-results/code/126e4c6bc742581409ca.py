import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import time

N = 16
D = 2

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit

    def score_points(pts):
        """Calculate the squared ratio of max to min pairwise distance."""
        d = pdist(pts, 'euclidean')
        min_d = np.min(d)
        if min_d < 1e-8:
            return 1e10
        max_d = np.max(d)
        return (max_d / min_d) ** 2

    def prepare_params(pts):
        """
        Reduces dimensionality by fixing the first two points (closest pair)
        to (0,0) and (1,0).
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

    def reconstruct(pts_fixed, params):
        """Reconstruct points from fixed points and free params."""
        pts = np.zeros((N, 2))
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        pts[2:] = params.reshape(-1, 2)
        return pts

    def soft_objective(params, beta=100.0):
        """
        Smooth surrogate using log-sum-exp for soft max/min of squared distances.
        """
        if time.time() > deadline:
            return 1e10
            
        pts = reconstruct(None, params)
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        iu = np.triu_indices(N, 1)
        d_sq = dists_sq[iu]
        
        d_sq = np.maximum(d_sq, 1e-12)
        
        # Soft max via log-sum-exp
        max_d_sq = np.max(d_sq)
        soft_max_d_sq = max_d_sq + (1.0 / beta) * np.log(np.sum(np.exp(beta * (d_sq - max_d_sq))))
        
        # Soft min via log-sum-exp (negated)
        min_d_sq = np.min(d_sq)
        soft_min_d_sq = min_d_sq - (1.0 / beta) * np.log(np.sum(np.exp(-beta * (d_sq - min_d_sq))))
        
        if soft_min_d_sq < 1e-8:
            return 1e10
            
        return soft_max_d_sq / soft_min_d_sq

    def epigraph_objective(params_t, beta=50.0):
        """
        Epigraph formulation with smooth constraints.
        params_t = [free_params..., t]
        Minimize t subject to: soft_min(d_sq) >= 1/t^2, soft_max(d_sq) <= t^2
        """
        if time.time() > deadline:
            return 1e10
            
        p = params_t[:-1]
        t = params_t[-1]
        
        if t < 1.0:
            return 1e10
            
        pts = reconstruct(None, p)
        
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
            
        ratio_sq = soft_max_d_sq / soft_min_d_sq
        
        # Penalty if ratio exceeds t^2
        penalty = 0.0
        if ratio_sq > t**2:
            penalty = 10.0 * (ratio_sq - t**2)
        
        return t**2 + penalty

    def generate_hex_seed():
        """Generate a hexagonal lattice subset of 16 points."""
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

    def generate_bubble_relax_seed():
        """Generate a seed using bubble relaxation."""
        rng = np.random.default_rng(999)
        pts = rng.uniform(-1, 1, size=(N, 2))
        
        for iteration in range(300):
            if time.time() > deadline - 20:
                break
            forces = np.zeros((N, 2))
            for i in range(N):
                for j in range(i+1, N):
                    d = pts[j] - pts[i]
                    dist = np.linalg.norm(d)
                    if dist < 1.0 and dist > 1e-8:
                        force_mag = (1.0 - dist)
                        forces[i] -= force_mag * d / dist
                        forces[j] += force_mag * d / dist
            pts += 0.05 * forces
        return pts

    def generate_ring_seed(n_rings=3):
        """Generate points on concentric rings."""
        pts = []
        remaining = N
        for ring in range(n_rings):
            if remaining <= 0:
                break
            n_in_ring = min(remaining, 6 if ring > 0 else 4)
            if ring == n_rings - 1:
                n_in_ring = remaining
            remaining -= n_in_ring
            
            radius = 0.5 + 0.5 * ring
            for k in range(n_in_ring):
                theta = k * 2 * np.pi / n_in_ring + ring * np.pi / 6
                pts.append([radius * np.cos(theta), radius * np.sin(theta)])
        
        while len(pts) < N:
            pts.append([0.0, 0.0])
        
        return np.array(pts[:N])

    def generate_diverse_seeds():
        """Generate multiple diverse seed configurations."""
        seeds = []
        
        # Hexagonal
        seeds.append(generate_hex_seed())
        
        # Bubble relaxation
        seeds.append(generate_bubble_relax_seed())
        
        # Three rings
        seeds.append(generate_ring_seed(3))
        
        # Two rings (8+8)
        pts = []
        for k in range(8):
            theta = k * 2 * np.pi / 8
            pts.append([0.5 * np.cos(theta), 0.5 * np.sin(theta)])
        for k in range(8):
            theta = k * 2 * np.pi / 8 + np.pi / 8
            pts.append([1.5 * np.cos(theta), 1.5 * np.sin(theta)])
        seeds.append(np.array(pts))
        
        # Square grid 4x4
        grid = np.array([(i, j) for i in range(4) for j in range(4)], dtype=float)
        grid = grid * 0.8 - 1.2
        seeds.append(grid)
        
        # Random
        rng = np.random.default_rng(42)
        seeds.append(rng.uniform(-1, 1, size=(16, 2)))
        
        # Perturbed hex
        rng2 = np.random.default_rng(123)
        for scale in [0.05, 0.10, 0.20]:
            perturbed = seeds[0] + rng2.normal(0, scale, size=(16, 2))
            seeds.append(perturbed)
        
        return seeds

    best_score = 1e10
    best_params = None
    best_perm = None

    def update_best(params, perm, score):
        nonlocal best_score, best_params, best_perm
        if score < best_score:
            best_score = score
            best_params = params.copy()
            best_perm = perm.copy()
            return True
        return False

    def try_optimize(seed_pts, tag=""):
        """Try to optimize from a given seed point configuration."""
        if time.time() > deadline - 10:
            return
            
        params, perm = prepare_params(seed_pts)
        if params is None:
            return
            
        curr_score = soft_objective(params, beta=50.0)
        update_best(params, perm, curr_score)
        
        # Phase 1: Powell for coarse optimization
        try:
            res1 = minimize(soft_objective, params, args=(50.0,), method='Powell',
                           options={'maxiter': 300, 'maxfev': 5000, 'disp': False})
            exact1 = soft_objective(res1.x, beta=50.0)
            update_best(res1.x, perm, exact1)
        except Exception:
            pass

        if time.time() > deadline - 20:
            return

        # Phase 2: L-BFGS-B on soft objective for medium refinement
        try:
            res2 = minimize(soft_objective, best_params, args=(100.0,), method='L-BFGS-B',
                           options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False})
            exact2 = soft_objective(res2.x, beta=100.0)
            update_best(res2.x, perm, exact2)
        except Exception:
            pass

        if time.time() > deadline - 30:
            return

        # Phase 3: Epigraph formulation with SLSQP
        try:
            t_init = np.sqrt(best_score) if best_score < 1e9 else 2.0
            init_params_t = np.append(best_params, t_init)
            
            def epigraph_wrapper(params_t):
                return epigraph_objective(params_t, beta=50.0)
            
            bounds = [(None, None)] * (N-2)*2 + [(1.0, 10.0)]
            
            res3 = minimize(epigraph_wrapper, init_params_t, method='SLSQP',
                           bounds=bounds, options={'maxiter': 500, 'ftol': 1e-12, 'disp': False})
            
            if res3.fun < best_score:
                p3 = res3.x[:-1]
                exact3 = soft_objective(p3, beta=100.0)
                update_best(p3, perm, exact3)
        except Exception:
            pass

        if time.time() > deadline - 40:
            return

        # Phase 4: Local search with perturbations
        try:
            local_rng = np.random.default_rng(hash(tag) % 2**32)
            for _ in range(5):
                if time.time() > deadline - 40:
                    break
                perturb_scale = 0.005
                perturbed_params = best_params + local_rng.normal(0, perturb_scale, size=best_params.shape)
                
                res4 = minimize(soft_objective, perturbed_params, args=(150.0,), method='L-BFGS-B',
                                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-9, 'disp': False})
                
                exact4 = soft_objective(res4.x, beta=150.0)
                update_best(res4.x, perm, exact4)
        except Exception:
            pass

    # Generate and process all seeds
    all_seeds = generate_diverse_seeds()
    
    for idx, seed_pts in enumerate(all_seeds):
        try_optimize(seed_pts, tag=f"seed_{idx}")

    # Final polish passes with increasing beta
    if best_params is not None:
        for beta in [200.0, 300.0, 500.0]:
            if time.time() > deadline - 5:
                break
            try:
                res_final = minimize(soft_objective, best_params, args=(beta,), method='L-BFGS-B',
                                    options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-10, 'disp': False})
                exact_final = soft_objective(res_final.x, beta=beta)
                update_best(res_final.x, best_perm, exact_final)
            except Exception:
                pass

    # Reconstruct final points
    if best_params is not None:
        pts = reconstruct(None, best_params)
        
        inv_perm = np.argsort(best_perm)
        final_pts = pts[inv_perm]
        
        final_pts -= final_pts.mean(axis=0)
        
        return final_pts
    else:
        pts = generate_hex_seed()
        pts -= pts.mean(axis=0)
        return pts
