import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

N = 300

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s safety margin before 150s kill

    def get_radius_candidates(x):
        x = x.reshape(-1, 2)
        dist_boundary = 1.0 - np.linalg.norm(x, axis=1)
        d_pair = pdist(x) / 2.0
        return np.concatenate([dist_boundary, d_pair])

    def raw_score(x):
        return np.min(get_radius_candidates(x))

    def hexagonal_initial(n, spacing):
        """Generate hexagonal lattice points, return first n within unit circle."""
        points = []
        # Generate hexagonal grid
        range_val = int(1.0 / spacing) + 2
        for j in range(-range_val, range_val + 1):
            for i in range(-range_val, range_val + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                r = np.hypot(x, y)
                if r <= 1.0 - spacing / 2.0:
                    points.append([x, y])
        points = np.array(points)
        if len(points) < n:
            return None
        # Sort by distance from center, take n closest
        dists = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists, kind='stable')[:n]
        return points[idx].copy()

    def optimize_from_start(x0, beta_start=10, beta_end=500, n_betas=10):
        """Anneal beta and optimize."""
        current_x = x0.copy()
        best_x = x0.copy()
        best_score = raw_score(best_x)
        
        betas = np.geomspace(beta_start, beta_end, n_betas)
        
        for beta in betas:
            if time.time() > deadline - 2:
                break
                
            def objective(x):
                x = x.reshape(-1, 2)
                d = get_radius_candidates(x)
                A = -beta * d
                max_A = np.max(A)
                # Handle case where all A are very negative
                exp_sum = np.sum(np.exp(A - max_A))
                if exp_sum == 0:
                    return 1e10
                log_sum_exp = max_A + np.log(exp_sum)
                return log_sum_exp / beta

            bounds = [(-1.0, 1.0)] * (2 * N)
            
            try:
                res = minimize(objective, current_x, method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
                if res.fun < objective(current_x):
                    current_x = res.x
            except Exception:
                pass
            
            s = raw_score(current_x)
            if s > best_score:
                best_score = s
                best_x = current_x.copy()
        
        return best_x, best_score

    def local_refine(x, max_iter=50):
        """Local refinement focusing on closest pairs."""
        current_x = x.copy()
        best_score = raw_score(current_x)
        
        for it in range(max_iter):
            if time.time() > deadline - 2:
                break
                
            d = get_radius_candidates(current_x)
            min_idx = np.argmin(d)
            
            # Determine which constraint is active
            n_boundary = N
            if min_idx < n_boundary:
                # Boundary constraint: point i
                i = min_idx
                # Nudge inward
                norm = np.linalg.norm(current_x[i])
                if norm > 0:
                    current_x[i] *= (1.0 - 0.001) / norm
                else:
                    current_x[i] += np.array([0.001, 0.0])
            else:
                # Pair constraint
                pair_idx = min_idx - n_boundary
                # Convert pair index to (i, j)
                # pdist uses condensed form: index = i*(2N-i-1)//2 + j - i - 1 for i<j
                # Simpler: just find the pair
                i, j = _pair_from_condensed(pair_idx, N)
                mid = (current_x[i] + current_x[j]) / 2.0
                diff = current_x[j] - current_x[i]
                dist = np.linalg.norm(diff)
                if dist > 0:
                    direction = diff / dist
                    # Push apart slightly
                    push = 0.001
                    current_x[i] -= direction * push
                    current_x[j] += direction * push
            
            # Keep in bounds
            norms = np.linalg.norm(current_x, axis=1)
            mask = norms > 1.0
            if np.any(mask):
                current_x[mask] *= (0.999 / norms[mask])[:, np.newaxis]
            
            s = raw_score(current_x)
            if s > best_score:
                best_score = s
            else:
                # Revert
                current_x = x.copy()
                break
        
        return current_x, best_score

    def _pair_from_condensed(idx, n):
        """Convert condensed distance index to (i, j) pair."""
        # For n points, condensed index for pair (i, j) with i < j:
        # idx = i*(2n - i - 1)//2 + j - i - 1
        # Solve for i: find i such that i*(2n-i-1)//2 <= idx < (i+1)*(2n-i-2)//2
        i = 0
        while True:
            start = i * (2 * n - i - 1) // 2
            end = (i + 1) * (2 * n - i - 2) // 2
            if idx < end:
                j = idx - start + i + 1
                return i, j
            i += 1
            if i >= n - 1:
                break
        return 0, 1  # fallback

    # Try multiple initial configurations
    best_x = None
    best_score = -1.0

    # Configuration 1: Standard hexagonal
    spacings = [0.11, 0.115, 0.12, 0.125, 0.13]
    for sp in spacings:
        if time.time() > deadline - 10:
            break
        x0 = hexagonal_initial(N, sp)
        if x0 is None:
            continue
        
        x_opt, score = optimize_from_start(x0, beta_start=5, beta_end=300, n_betas=8)
        if score > best_score:
            best_score = score
            best_x = x_opt.copy()
        
        # Local refine
        x_ref, score_ref = local_refine(x_opt, max_iter=30)
        if score_ref > best_score:
            best_score = score_ref
            best_x = x_ref.copy()

    # Configuration 2: Slightly perturbed hexagonal
    if time.time() < deadline - 15:
        np.random.seed(42)
        sp = 0.115
        x0 = hexagonal_initial(N, sp)
        if x0 is not None:
            x0_pert = x0 + np.random.randn(N, 2) * 0.005
            # Clip to unit circle
            norms = np.linalg.norm(x0_pert, axis=1)
            mask = norms > 0.99
            x0_pert[mask] *= (0.99 / norms[mask])[:, np.newaxis]
            
            x_opt, score = optimize_from_start(x0_pert, beta_start=5, beta_end=300, n_betas=8)
            if score > best_score:
                best_score = score
                best_x = x_opt.copy()
            
            x_ref, score_ref = local_refine(x_opt, max_iter=30)
            if score_ref > best_score:
                best_score = score_ref
                best_x = x_ref.copy()

    # Configuration 3: Fibonacci sphere-like distribution projected
    if time.time() < deadline - 10:
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        x0 = np.zeros((N, 2))
        for i in range(N):
            r = np.sqrt((i + 0.5) / N)
            theta = i * golden_angle
            x0[i] = [r * np.cos(theta), r * np.sin(theta)]
        
        x_opt, score = optimize_from_start(x0, beta_start=5, beta_end=300, n_betas=8)
        if score > best_score:
            best_score = score
            best_x = x_opt.copy()
        
        x_ref, score_ref = local_refine(x_opt, max_iter=30)
        if score_ref > best_score:
            best_score = score_ref
            best_x = x_ref.copy()

    # Final validation and cleanup
    if best_x is None:
        # Fallback: simple grid
        x0 = hexagonal_initial(N, 0.12)
        best_x = x0 if x0 is not None else np.zeros((N, 2))
    
    best_x = best_x.reshape(-1, 2)
    norms = np.linalg.norm(best_x, axis=1)
    mask = norms > 1.0
    if np.any(mask):
        best_x[mask] *= (0.999 / norms[mask])[:, np.newaxis]
    
    # Ensure no coincident points
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(best_x[i] - best_x[j])
            if d < 1e-6:
                best_x[j] += np.array([1e-5, 0.0])
                norms_j = np.linalg.norm(best_x[j])
                if norms_j > 1.0:
                    best_x[j] *= 0.999 / norms_j

    return best_x
