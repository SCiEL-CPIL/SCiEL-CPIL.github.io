import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    
    Returns
        points: np.ndarray of shape (14,3) containing the coordinates of the 14 points.
    """
    n = 14
    d = 3
    start_time = time.time()
    deadline = start_time + 625  # 5s margin for safety
    
    best_score = 0.0
    best_points = np.random.randn(n, d)
    
    # Helper to calculate the actual metric score
    def get_score(pts):
        dists_sq = pdist(pts, metric='sqeuclidean')
        if len(dists_sq) == 0:
            return 0.0
        d_min_sq = np.min(dists_sq)
        d_max_sq = np.max(dists_sq)
        
        if d_max_sq < 1e-12:
            return 0.0
        
        return d_min_sq / d_max_sq

    # Objective: Minimize -softmin(d_sq) + barrier(d_sq > 1)
    # d_sq are the 91 pairwise squared distances.
    # softmin(x, k) approximates min(x) for large k.
    # We want to maximize min(d_sq), so we minimize -softmin(d_sq).
    # Constraint: max(d_sq) <= 1.
    # Barrier: mu * sum(log(1 - d_sq)) for d_sq < 1. If d_sq >= 1, penalty is large.
    
    def objective_and_grad(x, k=10.0, mu=1.0):
        pts = x.reshape(n, d)
        
        # Compute pairwise squared distances
        # diff: (n, n, d)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff * diff, axis=2)
        
        # We only need upper triangle (i < j)
        # Create mask
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        d_sq = dists_sq[mask]
        
        # 1. Softmin term: -1/k * log(sum(exp(-k * d_sq)))
        # To avoid overflow/underflow, shift by min
        min_d_sq = np.min(d_sq)
        # If min_d_sq is very small, exp(-k * d_sq) might be large? No, d_sq >= 0.
        # exp(-k * d_sq) <= 1. So no overflow. Underflow is fine.
        exp_terms = np.exp(-k * (d_sq - min_d_sq))
        soft_min_val = -min_d_sq - (1.0/k) * np.log(np.sum(exp_terms))
        
        obj_val = -soft_min_val
        
        # Gradient of -softmin
        # d(-softmin)/dx_i = - d(softmin)/dx_i
        # softmin = -min - (1/k) log(sum exp(-k(d - min)))
        # Let S = sum exp(-k(d - min)).
        # d(softmin)/dx = - (1/k) * (1/S) * sum( d(exp(-k(d-min)))/dx )
        # d(exp(-k(d-min)))/dx = exp(-k(d-min)) * (-k) * d(d)/dx
        # d(d_sq)/dx_i = 2 * sum_{j != i} (x_i - x_j)
        
        # We need d(d_sq)/dx for each pair.
        # Let's compute gradient of softmin directly.
        # g_softmin_i = - (1/S) * sum_{j} exp(-k(d_ij - min)) * (-k) * d(d_ij)/dx_i
        #             = (k/S) * sum_{j} exp(-k(d_ij - min)) * d(d_ij)/dx_i
        
        # Note: d_ij is squared distance.
        # d(d_ij)/dx_i = 2(x_i - x_j)
        
        # S = sum exp_terms
        S = np.sum(exp_terms)
        
        # We need to sum over j for each i.
        # exp_weights: (n, n) matrix where exp_weights[i, j] = exp(-k(d_ij - min)) if i != j else 0
        # Actually, we can compute the full matrix exp_terms_full
        # exp_full = exp(-k * (dists_sq - min_d_sq))
        # But diagonal is exp(0) = 1. We should mask it out or set to 0.
        
        exp_full = np.exp(-k * (dists_sq - min_d_sq))
        np.fill_diagonal(exp_full, 0.0) # Diagonal d=0, but we don't want self-interaction
        
        # Wait, the sum in softmin is over i<j. 
        # The gradient calculation requires summing over all j != i.
        # The term exp(-k(d_ij - min)) appears in the sum for pair (i,j).
        # In the full matrix sum, each pair (i,j) and (j,i) is counted twice.
        # The log-sum-exp was over the 91 unique pairs.
        # Let's stick to the unique pairs to be consistent with the objective value.
        
        # However, computing gradient via full matrix is easier to vectorize.
        # If we use full matrix sum, S_full = 2 * S_unique.
        # The derivative of log(S_unique) is (1/S_unique) * sum_unique ...
        # sum_unique (w_ij * grad_d_ij) = 0.5 * sum_full (w_ij * grad_d_ij) because w_ij = w_ji and grad_d_ij = -grad_d_ji?
        # No, grad_d_ij/dx_i = 2(x_i - x_j). grad_d_ji/dx_i = 2(x_i - x_j) (since d_ji = d_ij).
        # So sum_full (w_ij * 2(x_i - x_j)) = sum_{i != j} w_ij * 2(x_i - x_j).
        # sum_unique (w_ij * 2(x_i - x_j)) = 0.5 * sum_{i != j} w_ij * 2(x_i - x_j).
        # So if we use S_full, we must divide the gradient contribution by 2 to match the 0.5 factor?
        # Let's just compute the gradient using the unique pairs logic to be safe, or carefully vectorize.
        
        # Let's use the vectorized full matrix approach and correct for the factor.
        # Objective used S_unique.
        # Gradient term: (k / S_unique) * sum_{j != i, i<j?} ... 
        # Actually, the chain rule for softmin over set P (pairs):
        # d/dx sum_{p in P} f(d_p) = sum_{p in P} f'(d_p) d(d_p)/dx
        # Here f(d) = -1/k log(sum exp(-k(d-min))) is not separable.
        # Let L = log(S). dL/dx = (1/S) sum_{p} d(exp(-k(d_p-min)))/dx
        # = (1/S) sum_{p} exp(-k(d_p-min)) * (-k) * d(d_p)/dx
        # d(softmin)/dx = - (1/k) dL/dx = (1/S) sum_{p} exp(-k(d_p-min)) * d(d_p)/dx
        # d(-softmin)/dx = - (1/S) sum_{p} exp(-k(d_p-min)) * d(d_p)/dx
        
        # So we need sum_{p in P} w_p * d(d_p)/dx_i where w_p = exp(-k(d_p - min)).
        # d(d_p)/dx_i for p=(i,j) is 2(x_i - x_j).
        # Sum over p containing i.
        # This is equivalent to 0.5 * sum_{j != i} w_ij * 2(x_i - x_j) ?
        # Yes, because for each j, the pair (i,j) is in P if i<j. If i>j, (j,i) is in P.
        # In the sum over j != i, we include both (i,j) and (j,i) if we iterate all j.
        # But P only contains unique pairs.
        # So sum_{p in P, i in p} w_p * d(d_p)/dx_i = sum_{j: (i,j) in P} w_ij * 2(x_i - x_j).
        # If we compute T_i = sum_{j != i} w_ij * 2(x_i - x_j), this sums over all j.
        # If i < j, (i,j) is in P. If i > j, (j,i) is in P.
        # w_ij = w_ji.
        # d(d_ij)/dx_i = 2(x_i - x_j).
        # d(d_ji)/dx_i = 2(x_i - x_j).
        # So T_i = sum_{j != i} w_ij * 2(x_i - x_j).
        # The sum over P is exactly T_i? No.
        # T_i includes j where i>j and j where i<j.
        # For j where i>j, the pair is (j,i). w_ji = w_ij.
        # The term in T_i for this j is w_ij * 2(x_i - x_j).
        # The term in the P-sum for this pair (j,i) involving x_i is w_ji * d(d_ji)/dx_i = w_ij * 2(x_i - x_j).
        # So yes, T_i is exactly the sum we need!
        
        # So grad_softmin_i = (1/S) * T_i
        # grad_obj_i = - grad_softmin_i = - (1/S) * T_i
        
        # Calculate T
        # w_full is exp_full.
        # T_i = sum_j w_ij * 2(x_i - x_j) = 2 * (sum_j w_ij * x_i - sum_j w_ij * x_j)
        # sum_j w_ij * x_i = x_i * sum_j w_ij
        # sum_j w_ij * x_j = (w_full @ pts)_i
        
        w_row_sums = np.sum(exp_full, axis=1)
        w_pts = exp_full @ pts
        
        T = 2.0 * (w_row_sums[:, np.newaxis] * pts - w_pts)
        
        grad_softmin = T / S
        grad_obj = -grad_softmin
        
        # 2. Barrier term
        # B = mu * sum_{p in P} log(1 - d_p)
        # If d_p >= 1, this is problematic. 
        # We can use a penalty: if d_p > 1, add large penalty.
        # Or use log(1 - d_p) only if d_p < 1, else -inf.
        # For optimization, we can use a smooth approximation or just a hard penalty.
        # Let's use: if d_p > 1, penalty = mu * (d_p - 1)^2 * 1000 + mu * 10 (to push back)
        # Or simpler: use log(1 - d_p) but clip d_p to 0.999999 to avoid nan.
        # But if d_p > 1, log is nan.
        # Better: use a function that is smooth and goes to -inf at 1.
        # f(d) = log(1 - d) for d < 1.
        # If d >= 1, we can define f(d) = log(1 - 0.999999) - (d - 0.999999) * 1000?
        # Let's just clip d_sq to 0.999999 for the log, and add a linear penalty for exceeding 1.
        
        d_sq_clipped = np.minimum(d_sq, 0.999999)
        barrier_val = mu * np.sum(np.log(1 - d_sq_clipped))
        
        # Penalty for exceeding 1
        excess = np.maximum(d_sq - 1.0, 0.0)
        penalty_val = 1000.0 * mu * np.sum(excess)
        
        obj_val += barrier_val + penalty_val
        
        # Gradient of barrier
        # d(log(1-d))/dx = -1/(1-d) * d(d)/dx
        # Only for d < 1.
        # If we used clipping, the gradient is 0 for d > 0.999999?
        # Let's calculate gradient for d < 0.999999.
        # w_barrier_ij = 1 / (1 - d_ij) if d_ij < 0.999999 else 0
        # grad_barrier_i = - mu * sum_{j} w_barrier_ij * d(d_ij)/dx_i
        # d(d_ij)/dx_i = 2(x_i - x_j)
        
        # Calculate w_barrier matrix
        # Only valid where d_sq < 0.999999
        valid_mask = dists_sq < 0.999999
        # Diagonal is 0, so valid. But we don't want self interaction.
        # exp_full was 0 on diagonal. We should do same for barrier weights.
        w_barrier = np.zeros_like(dists_sq)
        w_barrier[valid_mask] = 1.0 / (1.0 - dists_sq[valid_mask])
        np.fill_diagonal(w_barrier, 0.0)
        
        w_barrier_row_sums = np.sum(w_barrier, axis=1)
        w_barrier_pts = w_barrier @ pts
        
        T_barrier = 2.0 * (w_barrier_row_sums[:, np.newaxis] * pts - w_barrier_pts)
        grad_barrier = -mu * T_barrier
        
        # Gradient of penalty
        # penalty = 1000 * mu * sum(excess)
        # excess = max(d-1, 0)
        # grad penalty = 1000 * mu * sum_{j: d_ij > 1} 2(x_i - x_j)
        
        excess_mask = dists_sq > 1.0
        w_excess = np.zeros_like(dists_sq)
        w_excess[excess_mask] = 1.0
        np.fill_diagonal(w_excess, 0.0)
        
        w_excess_row_sums = np.sum(w_excess, axis=1)
        w_excess_pts = w_excess @ pts
        
        T_excess = 2.0 * (w_excess_row_sums[:, np.newaxis] * pts - w_excess_pts)
        grad_penalty = 1000.0 * mu * T_excess
        
        grad_total = (grad_obj + grad_barrier + grad_penalty).flatten()
        
        return obj_val, grad_total

    def run_optimization(x0):
        x0 = x0.flatten()
        res = minimize(objective_and_grad, x0, method='BFGS', jac=True,
                       options={'maxiter': 200, 'gtol': 1e-6})
        return res.x.reshape(n, d)

    # Initializations
    # 1. Random points in unit cube, centered
    # 2. Icosahedron + 2 random
    # 3. Random on sphere
    
    num_restarts = 0
    while time.time() < deadline:
        # Choose initialization strategy
        if num_restarts % 3 == 0:
            # Random in cube
            pts = np.random.rand(n, d) * 2 - 1
            # Center
            pts -= np.mean(pts, axis=0, keepdims=True)
            # Scale to have max dist ~ 1
            d_max = np.max(pdist(pts))
            if d_max > 0:
                pts /= d_max
        elif num_restarts % 3 == 1:
            # Icosahedron + 2 random
            phi = (1 + np.sqrt(5)) / 2
            ico = np.array([
                [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
                [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
                [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
            ])
            ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
            v1 = np.random.randn(3)
            v1 /= np.linalg.norm(v1)
            v2 = np.random.randn(3)
            v2 /= np.linalg.norm(v2)
            pts = np.vstack([ico, [v1, v2]])
            # Normalize max dist to 1
            d_max = np.max(pdist(pts))
            if d_max > 0:
                pts /= d_max
        else:
            # Random on sphere
            pts = np.random.randn(n, d)
            pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
            d_max = np.max(pdist(pts))
            if d_max > 0:
                pts /= d_max
        
        try:
            # Run BFGS
            # We might need to adjust k and mu. 
            # Start with moderate k, increase if stuck?
            # For now, fixed k=10, mu=1.
            # If we have time, we could anneal k and mu.
            # But BFGS with fixed params is faster per restart.
            
            # Let's try a few different k values in different restarts?
            # No, keep it simple first.
            
            local_pts = run_optimization(pts)
            local_score = get_score(local_pts)
            
            if local_score > best_score:
                best_score = local_score
                best_points = local_pts
                
        except Exception:
            pass
        
        num_restarts += 1
        
        # If we are very good, break
        if best_score > 0.95:
            break

    return best_points
