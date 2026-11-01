import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

N = 300

def construct():
    start_time = time.time()
    deadline = 115.0  # 5s safety margin
    
    # 1. Initial Hexagonal Lattice
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            x0 = points[np.argsort(radius, kind="stable")[:N]].copy()
            break
        spacing *= 0.98
        
    def get_distances(x):
        """Returns list of boundary distances and pairwise distances."""
        x = x.reshape(-1, 2)
        dist_boundary = 1.0 - np.linalg.norm(x, axis=1)
        d_pair = pdist(x)
        return np.concatenate([dist_boundary, d_pair])

    def soft_min_score(x, beta):
        """
        Minimize this function.
        soft_min(d) = -1/beta * log(sum(exp(-beta * d)))
        We want to maximize min(d), so we minimize -soft_min(d).
        """
        x = x.reshape(-1, 2)
        d = get_distances(x)
        # Prevent underflow/overflow
        # d should be positive. If any d is negative (overlap/outside), exp(-beta*d) is large.
        # We want the score to be low (bad) if d is small.
        # f = - soft_min(d) = 1/beta * log(sum(exp(-beta * d)))
        # As beta -> inf, f -> -min(d). Minimizing f -> maximizing min(d).
        
        # Numerical stability: subtract max? No, we are summing exp(-beta*d).
        # If d is small, -beta*d is large positive -> exp is large.
        # If d is large, -beta*d is large negative -> exp is small.
        # This is fine.
        
        # Clip d to avoid extreme values if points fly away?
        # Bounds will keep them in [-1, 1].
        
        # Calculate log-sum-exp for stability
        # log(sum(exp(A))) = max(A) + log(sum(exp(A - max(A))))
        A = -beta * d
        max_A = np.max(A)
        log_sum_exp = max_A + np.log(np.sum(np.exp(A - max_A)))
        return log_sum_exp / beta

    def neg_soft_min(x, beta):
        return soft_min_score(x, beta) # We minimize this

    best_x = x0.copy()
    best_raw_score = -1.0
    
    def raw_score(x):
        x = x.reshape(-1, 2)
        d = get_distances(x)
        return np.min(d) / 2.0 # Note: boundary dist is 1-|x|, pairwise is dist.
        # Wait, the task score is min(min_pairwise/2, min_boundary).
        # My get_distances returns [boundary_dists, pairwise_dists].
        # So min(d) is the min of both.
        # But pairwise distance needs to be divided by 2 for radius?
        # Task: r = min(min_{i<j} |c_i - c_j| / 2, min_i (1 - |c_i|))
        # So yes, pairwise distances in the list need to be halved?
        # Or I can just compute the final score separately.
        # For the soft-min, I can just use the raw distances and adjust the "effective" boundary.
        # Actually, to be consistent, let's define the "constraint distances" properly.
        # Let's define d_i for boundary as 1-|x_i|.
        # Let d_ij for pairs as |x_i-x_j|/2.
        # Then r = min(all d's).
        
    # Redefine get_distances to return "radius candidates"
    def get_radius_candidates(x):
        x = x.reshape(-1, 2)
        dist_boundary = 1.0 - np.linalg.norm(x, axis=1)
        d_pair = pdist(x) / 2.0
        return np.concatenate([dist_boundary, d_pair])

    def soft_min_score_v2(x, beta):
        d = get_radius_candidates(x)
        A = -beta * d
        max_A = np.max(A)
        log_sum_exp = max_A + np.log(np.sum(np.exp(A - max_A)))
        return log_sum_exp / beta

    def neg_soft_min_v2(x, beta):
        return soft_min_score_v2(x, beta)

    bounds = [(-1.0, 1.0)] * (2 * N)
    
    # Annealing loop
    betas = [5, 20, 50, 100, 200]
    current_x = best_x.copy()
    
    for beta in betas:
        if time.time() > deadline - 5:
            break
            
        try:
            # Use L-BFGS-B for speed
            res = minimize(neg_soft_min_v2, current_x, args=(beta,), 
                           method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-6})
            if res.success or res.fun < soft_min_score_v2(current_x, beta):
                current_x = res.x
        except Exception:
            pass
            
        # Update best raw score
        raw_s = np.min(get_radius_candidates(current_x))
        if raw_s > best_raw_score:
            best_raw_score = raw_s
            best_x = current_x.copy()
            
    # Final Polish with Raw Score using Nelder-Mead (small steps)
    # Nelder-Mead is slow, but might fix the "kinks" left by soft-min.
    if time.time() < deadline - 10:
        def neg_raw_score(x):
            return -np.min(get_radius_candidates(x.reshape(-1, 2)))
            
        try:
            res = minimize(neg_raw_score, best_x, method='Nelder-Mead',
                           options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.fun < -best_raw_score:
                best_x = res.x
                best_raw_score = -res.fun
        except Exception:
            pass

    # Final validation
    best_x = best_x.reshape(-1, 2)
    norms = np.linalg.norm(best_x, axis=1)
    mask = norms > 1.0
    if np.any(mask):
        best_x[mask] *= (1.0 / norms[mask] - 0.001)[:, np.newaxis]
        
    return best_x
