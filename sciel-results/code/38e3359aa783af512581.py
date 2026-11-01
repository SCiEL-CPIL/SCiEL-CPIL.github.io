import numpy as np
import time
from scipy.optimize import minimize

N = 300

def construct():
    start_time = time.time()
    time_limit = 110.0 # Safety margin
    
    # 1. Initialization: Hexagonal Lattice
    def get_hex_lattice_points(n_points):
        best_points = None
        r_est = 0.05
        s = 2 * r_est
        
        for _ in range(10):
            k = int(2.0 / s) + 5
            rows = []
            for j in range(-k, k + 1):
                y = j * s * np.sqrt(3.0) / 2.0
                offset = 0.5 * s if j % 2 != 0 else 0.0
                for i in range(-k, k + 1):
                    x = i * s + offset
                    rows.append((x, y))
            points = np.array(rows)
            radii = np.hypot(points[:, 0], points[:, 1])
            mask = radii < 1.0
            points = points[mask]
            
            if len(points) >= n_points:
                indices = np.argsort(radii[mask])
                return points[indices[:n_points]]
            s *= 0.95
        
        return np.random.rand(n_points, 2) * 2 - 1

    points = get_hex_lattice_points(N)
    
    # Ensure points are strictly inside
    radii = np.hypot(points[:, 0], points[:, 1])
    mask = radii >= 1.0
    if np.any(mask):
        points[mask] = points[mask] * (0.99 / radii[mask, None])

    # 2. Optimization using Soft-Min Surrogate
    
    def _soft_min_obj_and_grad(pos, t):
        """
        Objective: Minimize -softmin(distances, t)
        This is equivalent to maximizing softmin(distances, t)
        Distances include:
        - Pairwise distances / 2
        - Boundary distances (1 - |p_i|)
        """
        n = len(pos)
        
        # Compute pairwise distances
        # To save memory and time, we can compute in chunks or use a clever trick, 
        # but for N=300, N^2 = 90k is small.
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dists_sq = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(dists_sq, np.inf)
        
        # We want distances d_ij = sqrt(dists_sq).
        # Objective involves exp(-t * d_ij).
        # Note: d_ij/2 is the "radius constraint" for pairs.
        # Let v_pair = dists / 2.
        # Let v_bound = 1 - norm(pos).
        
        # Compute pairwise distances
        pair_dists = np.sqrt(dists_sq)
        pair_vals = pair_dists / 2.0
        
        # Boundary distances
        norms = np.linalg.norm(pos, axis=1)
        bound_vals = 1.0 - norms
        
        # Combine all values into a single vector for soft-min
        # Number of pairs: N*(N-1)/2
        # Number of bounds: N
        # We can flatten the upper triangle of pair_vals
        
        # To be efficient, we can compute the log-sum-exp directly without materializing the huge vector if possible,
        # but N=300 is small enough.
        
        # Extract upper triangle indices
        i_idx, j_idx = np.triu_indices(n, k=1)
        v_pairs = pair_vals[i_idx, j_idx]
        v_bounds = bound_vals
        
        v_all = np.concatenate([v_pairs, v_bounds])
        
        # Compute soft-min: -1/t * log(sum(exp(-t * v)))
        # To avoid overflow/underflow, use log-sum-exp trick:
        # max_v = max(v_all)
        # log_sum_exp = max_v + log(sum(exp(-t * (v_all - max_v))))
        # soft_min = -1/t * log_sum_exp
        
        max_v = np.max(v_all)
        # If t is large, -t * (v - max_v) can be very negative, exp becomes 0. This is fine.
        # But if v - max_v is 0, exp(0)=1.
        
        exp_terms = np.exp(-t * (v_all - max_v))
        sum_exp = np.sum(exp_terms)
        
        if sum_exp == 0:
            # Fallback, shouldn't happen
            obj = -max_v
            grad = np.zeros_like(pos)
            return obj, grad
            
        log_sum_exp = max_v + np.log(sum_exp)
        soft_min_val = -log_sum_exp / t
        obj = -soft_min_val # We want to minimize -soft_min
        
        # Gradient of obj w.r.t pos
        # d(obj)/d(pos) = - d(soft_min)/d(pos)
        # d(soft_min)/d(v_k) = - (exp(-t * v_k) / sum_exp)
        # So d(obj)/d(v_k) = exp(-t * v_k) / sum_exp = weights_k
        
        weights = exp_terms / sum_exp
        
        # Now backpropagate to pos
        grad = np.zeros_like(pos)
        
        # 1. Pairwise contribution
        # v_pair_ij = 0.5 * ||p_i - p_j||
        # d(v_pair_ij)/d(p_i) = 0.5 * (p_i - p_j) / ||p_i - p_j||
        # d(obj)/d(p_i) += sum_{j != i} weights_{ij} * 0.5 * (p_i - p_j) / ||p_i - p_j||
        
        # We have weights for the flattened upper triangle.
        # We need to accumulate gradients for each point.
        
        # Create a matrix of weights corresponding to the upper triangle
        w_mat = np.zeros((n, n))
        w_mat[i_idx, j_idx] = weights[:len(i_idx)]
        w_mat[j_idx, i_idx] = weights[:len(i_idx)]
        
        # dists is symmetric.
        # We need (p_i - p_j) / dist_ij.
        # diff is p_i - p_j.
        # dists is ||diff||.
        
        # Avoid division by zero (diagonal is inf, so weight is 0, but let's be safe)
        # dists matrix has inf on diagonal.
        # We can compute unit vectors.
        
        # For each i, sum over j of w_ij * (p_i - p_j) / dist_ij * 0.5
        # Note: w_ij is the weight for the pair (i,j).
        
        # Let's compute vectorially.
        # diff is (N, N, 2).
        # dists is (N, N).
        
        # unit_diff = diff / dists[..., None]
        # But dists has inf on diagonal.
        
        # We can mask the diagonal.
        mask_diag = np.eye(n, dtype=bool)
        
        # Compute terms
        # term_ij = w_mat[i,j] * (p_i - p_j) / dist_ij * 0.5
        # grad[i] = sum_j term_ij
        
        # To avoid memory blowup, we can do this in a loop or use einsum if possible, 
        # but N=300 is small. A loop over N is fast enough.
        
        for i in range(n):
            # For point i, we sum over all j != i
            # w_mat[i, :] contains weights for pairs (i, j)
            # dists[i, :] contains distances
            # diff[i, :] contains p_i - p_j
            
            # Mask diagonal
            w_row = w_mat[i].copy()
            w_row[i] = 0.0
            
            # Avoid division by zero for very small distances? 
            # If dist is 0, weight might be non-zero? No, if dist is 0, v_pair is 0, 
            # which is likely the min, so weight is high. 
            # But if dist is 0, gradient is undefined. 
            # In practice, we start from a lattice, so no overlaps initially.
            
            # Calculate contribution
            # factor = w_row / dists[i] * 0.5
            # But dists[i] has inf at i.
            
            dist_row = dists[i].copy()
            dist_row[i] = 1.0 # Avoid div by zero, w_row[i] is 0 anyway
            
            factor = w_row / dist_row * 0.5
            
            # grad[i] = sum_j factor[j] * (p_i - p_j)
            # diff[i, :] is p_i - p_j
            grad[i] = np.sum(factor[:, None] * diff[i], axis=0)
            
        # 2. Boundary contribution
        # v_bound_i = 1 - ||p_i||
        # d(v_bound_i)/d(p_i) = - p_i / ||p_i||
        # d(obj)/d(p_i) += weights_bound_i * (- p_i / ||p_i||)
        
        w_bounds = weights[len(i_idx):]
        
        # norms is already computed
        # Avoid division by zero
        safe_norms = np.where(norms > 1e-9, norms, 1.0)
        
        # grad_boundary_i = - w_bound_i * p_i / norm_i
        grad_boundary = -w_bounds[:, None] * (pos / safe_norms[:, None])
        
        grad += grad_boundary
        
        # Wait, the sign.
        # obj = - soft_min
        # soft_min = -1/t log sum exp(-t v)
        # d(soft_min)/d(v_k) = - exp(-t v_k) / sum exp(-t v_j) = - w_k
        # d(obj)/d(v_k) = - d(soft_min)/d(v_k) = w_k
        # So grad_obj = sum w_k * d(v_k)/d(pos)
        
        # My derivation:
        # Pair: d(v_pair)/d(p_i) = 0.5 (p_i - p_j) / dist
        # Contribution: w_pair * 0.5 (p_i - p_j) / dist
        # My code: factor = w_row / dist * 0.5. grad[i] += sum factor * diff. Correct.
        
        # Boundary: d(v_bound)/d(p_i) = - p_i / norm
        # Contribution: w_bound * (- p_i / norm)
        # My code: grad_boundary = -w_bounds * pos / norms. Correct.
        
        return obj, grad

    # Optimization Loop
    # We will run L-BFGS-B for a fixed number of iterations or until time limit.
    # We can also do a few GD steps first if needed, but L-BFGS-B is usually strong.
    
    # Start with a moderate t, then increase t to sharpen the min.
    # However, increasing t makes the problem harder (more peaked).
    # Let's try a fixed t that is large enough to approximate min well, but not so large it causes numerical issues.
    # t=100 or 200 is usually good for N=300.
    
    t = 100.0
    
    # We can restart with higher t if we have time.
    # For now, one pass with t=100.
    
    # Flatten points for optimizer
    x0 = points.flatten()
    
    def obj_flat(x):
        pos = x.reshape(-1, 2)
        val, _ = _soft_min_obj_and_grad(pos, t)
        return val
        
    def grad_flat(x):
        pos = x.reshape(-1, 2)
        _, g = _soft_min_obj_and_grad(pos, t)
        return g.flatten()
        
    # Run minimize
    # maxiter needs to be reasonable.
    # N=300, 600 variables.
    
    # We can run multiple restarts or just one long one.
    # Given time constraints, let's do one robust run.
    
    best_result = None
    best_score = -np.inf
    
    # We'll run minimize, and if it fails or we have time, try again with higher t.
    
    # First pass
    try:
        res = minimize(obj_flat, x0, jac=grad_flat, method='L-BFGS-B', 
                       options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
        
        if res.success or res.status in [1, 2, 3, 4, 5, 6]: # Common success/warning statuses
            best_result = res.x
            # Calculate true score to track
            pos = best_result.reshape(-1, 2)
            # Quick score calc
            diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, np.inf)
            min_pair = np.min(dists) / 2
            min_bound = np.min(1.0 - np.linalg.norm(pos, axis=1))
            score = min(min_pair, min_bound)
            best_score = score
    except Exception as e:
        pass
        
    # If we have time, try a second pass with higher t for sharpening
    if time.time() - start_time < time_limit - 30 and best_result is not None:
        t2 = 500.0
        x_start = best_result
        
        def obj_flat2(x):
            pos = x.reshape(-1, 2)
            val, _ = _soft_min_obj_and_grad(pos, t2)
            return val
            
        def grad_flat2(x):
            pos = x.reshape(-1, 2)
            _, g = _soft_min_obj_and_grad(pos, t2)
            return g.flatten()
            
        try:
            res2 = minimize(obj_flat2, x_start, jac=grad_flat2, method='L-BFGS-B', 
                           options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
            
            if res2.success or res2.status in [1, 2, 3, 4, 5, 6]:
                pos = res2.x.reshape(-1, 2)
                diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
                dists = np.linalg.norm(diff, axis=2)
                np.fill_diagonal(dists, np.inf)
                min_pair = np.min(dists) / 2
                min_bound = np.min(1.0 - np.linalg.norm(pos, axis=1))
                score2 = min(min_pair, min_bound)
                
                if score2 > best_score:
                    best_score = score2
                    best_result = res2.x
        except Exception as e:
            pass
            
    if best_result is None:
        # Fallback to original points if optimization failed completely
        return points
        
    final_points = best_result.reshape(-1, 2)
    
    # Final cleanup: ensure inside unit circle
    radii = np.linalg.norm(final_points, axis=1)
    mask = radii > 1.0
    if np.any(mask):
        final_points[mask] = final_points[mask] * (1.0 / radii[mask, None])
        
    return final_points
