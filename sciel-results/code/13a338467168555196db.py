import numpy as np
from scipy.optimize import minimize
import time

N = 16
D = 2

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    def get_sq_dists(pts):
        """Calculate squared pairwise distances."""
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        sq_dists = np.sum(diff**2, axis=2)
        iu = np.triu_indices(N, 1)
        return sq_dists[iu]

    def exact_score(pts):
        """Calculate the exact score: (max dist / min dist)^2"""
        if len(pts) < 2:
            return 1e10
        sq_dists = get_sq_dists(pts)
        min_sq = np.min(sq_dists)
        max_sq = np.max(sq_dists)
        if min_sq < 1e-12:
            return 1e10
        # (max_d / min_d)^2 = (sqrt(max_sq) / sqrt(min_sq))^2 = max_sq / min_sq
        return max_sq / min_sq

    def soft_score_sq(pts, beta):
        """
        Smooth surrogate using log-sum-exp on SQUARED distances.
        This avoids sqrt operations in the inner loop.
        """
        sq_dists = get_sq_dists(pts)
        # Ensure no negative/zero values cause log issues
        sq_dists = np.maximum(sq_dists, 1e-12)
        
        # Soft Max of squared distances
        max_sq = np.max(sq_dists)
        exp_terms_max = np.exp(beta * (sq_dists - max_sq))
        sum_exp_max = np.sum(exp_terms_max)
        if sum_exp_max < 1e-300:
            soft_max_sq = max_sq
        else:
            soft_max_sq = max_sq + (1.0 / beta) * np.log(sum_exp_max)
        
        # Soft Min of squared distances
        min_sq = np.min(sq_dists)
        exp_terms_min = np.exp(-beta * (sq_dists - min_sq))
        sum_exp_min = np.sum(exp_terms_min)
        if sum_exp_min < 1e-300:
            soft_min_sq = min_sq
        else:
            soft_min_sq = min_sq - (1.0 / beta) * np.log(sum_exp_min)
            
        if soft_min_sq < 1e-12:
            return 1e10
            
        # The objective is (max_d / min_d)^2 = max_sq / min_sq
        return soft_max_sq / soft_min_sq

    def normalize(pts):
        """
        Normalize points: center at origin and scale such that RMS distance is 1.
        This helps with numerical stability and scale invariance.
        """
        pts = pts - np.mean(pts, axis=0)
        # Scale by RMS distance from origin
        rms = np.sqrt(np.mean(np.sum(pts**2, axis=1)))
        if rms > 1e-9:
            pts = pts / rms
        return pts

    def simulated_annealing(init_pts, max_iters=20000, T_start=0.5, T_end=0.001, alpha=0.99995):
        """
        Simulated Annealing to escape local minima.
        Moves one point at a time.
        """
        pts = init_pts.copy()
        current_score = exact_score(pts)
        best_pts = pts.copy()
        best_score = current_score
        T = T_start
        
        # Precompute random indices for speed
        rng = np.random.default_rng(42)
        
        for i in range(max_iters):
            if time.time() > deadline:
                break
                
            # Pick a random point
            idx = rng.integers(0, N)
            # Gaussian displacement
            displacement = rng.normal(0, 0.1, 2)
            
            new_pts = pts.copy()
            new_pts[idx] += displacement
            new_pts = normalize(new_pts)
            
            new_score = exact_score(new_pts)
            
            # Accept if better or with probability exp(-delta/T)
            if new_score < current_score:
                current_score = new_score
                pts = new_pts
                if new_score < best_score:
                    best_score = new_score
                    best_pts = new_pts.copy()
            else:
                # Avoid overflow in exp
                delta = new_score - current_score
                if delta < 50 * T: 
                    prob = np.exp(-delta / T)
                    if rng.random() < prob:
                        current_score = new_score
                        pts = new_pts
                        if new_score < best_score:
                            best_score = new_score
                            best_pts = new_pts.copy()
            
            T *= alpha
            if T < T_end:
                break
                
        return best_pts, best_score

    def generate_seeds():
        """Generate diverse initial configurations."""
        seeds = []
        rng = np.random.default_rng(42)
        
        # 1. Hexagonal Lattice (Top 16 closest to center)
        R = 5
        hex_pts = []
        for q in range(-R, R + 1):
            for r in range(-R, R + 1):
                x = q + 0.5 * r
                y = (np.sqrt(3) / 2) * r
                hex_pts.append([x, y])
        hex_pts = np.array(hex_pts)
        dists_origin = np.sum(hex_pts**2, axis=1)
        sorted_indices = np.argsort(dists_origin)
        init_hex = hex_pts[sorted_indices[:N]]
        seeds.append(normalize(init_hex))
        
        # 2. Regular 16-gon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        init_circle = np.column_stack([np.cos(angles), np.sin(angles)])
        seeds.append(normalize(init_circle))
        
        # 3. 4x4 Grid
        grid_x = np.linspace(-1.5, 1.5, 4)
        grid_y = np.linspace(-1.5, 1.5, 4)
        gx, gy = np.meshgrid(grid_x, grid_y)
        init_grid = np.column_stack([gx.flatten(), gy.flatten()])
        seeds.append(normalize(init_grid))
        
        # 4. Golden Angle Spiral
        init_spiral = np.zeros((N, 2))
        golden_angle = np.pi * (3 - np.sqrt(5))
        for i in range(N):
            r = np.sqrt(i + 0.5)
            theta = i * golden_angle
            init_spiral[i, 0] = r * np.cos(theta)
            init_spiral[i, 1] = r * np.sin(theta)
        seeds.append(normalize(init_spiral))
        
        # 5. Random
        init_random = rng.uniform(-2, 2, (N, 2))
        seeds.append(normalize(init_random))
        
        # 6. Perturbed Hex
        seeds.append(normalize(init_hex + rng.normal(0, 0.1, size=(N, 2))))
        
        # 7. Perturbed Grid
        seeds.append(normalize(init_grid + rng.normal(0, 0.1, size=(N, 2))))

        return seeds

    # Generate seeds
    candidates = generate_seeds()
    
    # Score seeds to prioritize
    scored_seeds = [(exact_score(s), s) for s in candidates]
    scored_seeds.sort(key=lambda x: x[0])
    
    best_score = 1e10
    best_params = None
    
    bnds = [(-5, 5)] * (2 * N)
    
    # Process top seeds
    # We limit the number of seeds to process to ensure we finish in time
    num_to_process = min(6, len(scored_seeds))
    
    for idx in range(num_to_process):
        if time.time() > deadline - 30:
            break
            
        seed_score, init_pts = scored_seeds[idx]
        
        # Update best if seed is good
        if seed_score < best_score:
            best_score = seed_score
            best_params = init_pts.copy()
            
        current_params = init_pts.copy()
        
        # Phase 1: Simulated Annealing
        # SA helps escape local minima that gradient methods get stuck in
        sa_pts, sa_score = simulated_annealing(current_params, max_iters=10000)
        if sa_score < best_score:
            best_score = sa_score
            best_params = sa_pts.copy()
        
        current_params = sa_pts
        
        # Phase 2: Soft Objective Refinement (L-BFGS-B)
        # Use increasing beta to sharpen the soft min/max
        betas = [10.0, 50.0, 200.0]
        
        for beta in betas:
            if time.time() > deadline - 15:
                break
                
            try:
                def soft_obj(params):
                    return soft_score_sq(params.reshape(N, 2), beta)
                
                res = minimize(
                    soft_obj, 
                    current_params.flatten(), 
                    method='L-BFGS-B', 
                    bounds=bnds,
                    options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6}
                )
                if res.fun < 1e9:
                    candidate_pts = res.x.reshape(N, 2)
                    # Check exact score of the soft-optimized result
                    exact_s = exact_score(candidate_pts)
                    if exact_s < best_score:
                        best_score = exact_s
                        best_params = candidate_pts.copy()
                    current_params = candidate_pts
            except Exception:
                pass
        
        # Phase 3: Powell's Method on Exact Objective
        # Powell is robust for final polishing
        if time.time() < deadline - 5:
            try:
                def exact_obj(params):
                    return exact_score(params.reshape(N, 2))
                
                res_powell = minimize(
                    exact_obj, 
                    current_params.flatten(), 
                    method='Powell',
                    options={'maxiter': 200, 'maxfev': 2000, 'ftol': 1e-12, 'xtol': 1e-12}
                )
                if res_powell.fun < 1e9:
                    candidate_pts = res_powell.x.reshape(N, 2)
                    exact_s = exact_score(candidate_pts)
                    if exact_s < best_score:
                        best_score = exact_s
                        best_params = candidate_pts.copy()
            except Exception:
                pass
            
            # Phase 4: Nelder-Mead for very fine local refinement
            if time.time() < deadline - 2:
                try:
                    def exact_obj_nm(params):
                        return exact_score(params.reshape(N, 2))
                    
                    res_nm = minimize(
                        exact_obj_nm, 
                        current_params.flatten(), 
                        method='Nelder-Mead',
                        options={'maxiter': 100, 'xatol': 1e-12, 'fatol': 1e-12}
                    )
                    if res_nm.fun < 1e9:
                        candidate_pts = res_nm.x.reshape(N, 2)
                        exact_s = exact_score(candidate_pts)
                        if exact_s < best_score:
                            best_score = exact_s
                            best_params = candidate_pts.copy()
                except Exception:
                    pass

    # Fallback
    if best_params is None:
        best_params = candidates[0]
        
    final_pts = best_params.reshape(N, 2)
    
    # Ensure distinct points
    dists_sq = get_sq_dists(final_pts)
    if np.min(dists_sq) < 1e-10:
        # Simple perturb if points are too close
        final_pts += np.random.uniform(-1e-3, 1e-3, final_pts.shape)
        # Re-normalize
        final_pts = normalize(final_pts)
        
    return final_pts

# EVOLVE-BLOCK-END
