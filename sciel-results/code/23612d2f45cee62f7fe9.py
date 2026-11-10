# EVOLVE-BLOCK-START
import numpy as np
import time

def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Two-dimensional Kakeya-needle area-minimization problem for n=64.
    Uses smooth gradient descent with log-sum-exp approximation of min(),
    followed by coordinate descent polish with exact area evaluation.
    """
    n = 64
    k = int(np.log2(n))
    w = 1.0 / n
    n_heights = 20000
    y_vals = (np.arange(n_heights) + 0.5) / n_heights
    
    # Initial: Keich construction
    positions = []
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        j_arr = np.arange(1, k + 1)
        positions.append(np.sum((1 - j_arr) * eps * (2.0 ** -j_arr) / k))
    x = np.array(positions, dtype=np.float64)
    
    # Precompute a_i
    a = (np.arange(n) + 1) / n
    
    def compute_area_exact(x):
        """Exact area computation using the quadrature formula."""
        # L_i at each height: x_i + a_i * y
        # Shape: (n_heights, n)
        L = x[None, :] + a[None, :] * y_vals[:, None]
        # Sort L along axis=1
        L_sorted = np.sort(L, axis=1)
        # Consecutive differences
        diffs = L_sorted[:, 1:] - L_sorted[:, :-1]
        # Union length: w + sum min(diffs, w)
        union_lengths = w + np.sum(np.minimum(diffs, w), axis=1)
        return np.mean(union_lengths)
    
    def compute_area_smooth(x, eps=0.01):
        """Smooth approximation of area using soft min."""
        L = x[None, :] + a[None, :] * y_vals[:, None]
        L_sorted = np.sort(L, axis=1)
        diffs = L_sorted[:, 1:] - L_sorted[:, :-1]
        # Soft min: min(d, w) ≈ w - softplus(d - w) where softplus(z) = eps*log(1+exp(z/eps))
        # min(d, w) = w - max(0, d-w) ≈ w - eps*log(1 + exp((d-w)/eps))
        z = (diffs - w) / eps
        # Numerically stable softplus
        softplus = eps * np.log1p(np.exp(np.clip(z, -50, 50)))
        soft_min = w - softplus
        union_lengths = w + np.sum(soft_min, axis=1)
        return np.mean(union_lengths)
    
    def compute_grad_smooth(x, eps=0.01):
        """Gradient of smooth area w.r.t. x."""
        L = x[None, :] + a[None, :] * y_vals[:, None]
        L_sorted = np.sort(L, axis=1)
        diffs = L_sorted[:, 1:] - L_sorted[:, :-1]
        z = (diffs - w) / eps
        # d(softplus)/dz = sigmoid(z), d(soft_min)/ddiff = -sigmoid(z)
        sigmoid_z = 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))
        # d(union_length)/d(L_sorted[j+1]) = -sigmoid_z for pair j
        # d(union_length)/d(L_sorted[j]) = sigmoid_z for pair j (from min(d_j, w) where d_j = L[j+1]-L[j])
        # Actually: min(d_j, w) where d_j = L_sorted[j+1] - L_sorted[j]
        # d(min)/dL_sorted[j+1] = -sigmoid_z[j]
        # d(min)/dL_sorted[j] = +sigmoid_z[j]
        # So for each sorted position p, its coefficient is:
        # from pair p-1 (if p>0): +sigmoid_z[p-1] (as L_sorted[p] is the "right" end of pair p-1... wait)
        # Let me re-derive:
        # d_j = L_sorted[j+1] - L_sorted[j], j=0..n-2
        # union = w + sum_j min(d_j, w)
        # d(union)/dL_sorted[j] = d(min(d_j,w))/dL_sorted[j] + d(min(d_{j-1},w))/dL_sorted[j]
        # = -sigmoid_z[j] (from d_j, L_sorted[j] is left end, d(d_j)/dL_sorted[j]=-1)
        #   + sigmoid_z[j-1] (from d_{j-1}, L_sorted[j] is right end, d(d_{j-1})/dL_sorted[j]=+1)
        # So coefficient for L_sorted[j]: sigmoid_z[j-1] - sigmoid_z[j] (with boundaries)
        
        # Build gradient w.r.t. L_sorted
        grad_L_sorted = np.zeros_like(L_sorted)
        # For each pair j: d(min(d_j,w))/dL_sorted[j+1] = -sigmoid_z[j]
        #               d(min(d_j,w))/dL_sorted[j]   = +sigmoid_z[j]
        grad_L_sorted[:, 1:] += -sigmoid_z  # right end of each pair
        grad_L_sorted[:, :-1] += sigmoid_z   # left end of each pair
        
        # Now map from L_sorted back to L (unsorted)
        # L_sorted[:, j] = L[:, perm[j]] for some permutation
        # We need to track the permutation
        # Recompute permutation
        perm = np.argsort(L, axis=1)
        # grad_L[:, i] = sum over j where perm[:, j] == i of grad_L_sorted[:, j]
        # Vectorized: scatter
        grad_L = np.zeros_like(L)
        np.add.at(grad_L, (np.arange(n_heights)[:, None], perm), grad_L_sorted)
        
        # d(L)/d(x_i) = a_i
        # d(area)/d(x_i) = mean over heights of grad_L[:, i] * a_i
        grad_x = np.mean(grad_L * a[None, :], axis=0)
        return grad_x
    
    # Deadline guard
    start_time = time.time()
    time_limit = 600  # seconds, safety margin from 630
    
    # Phase 1: Smooth gradient descent with adaptive epsilon
    eps_schedule = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]
    best_x = x.copy()
    best_area = compute_area_exact(x)
    
    for eps in eps_schedule:
        if time.time() - start_time > time_limit - 120:
            break
        lr = 0.01
        for it in range(200):
            if time.time() - start_time > time_limit - 120:
                break
            grad = compute_grad_smooth(x, eps)
            x_new = x - lr * grad
            # Check if improved with exact area
            area_new = compute_area_exact(x_new)
            if area_new < best_area:
                best_x = x_new.copy()
                best_area = area_new
                x = x_new
                lr = min(lr * 1.05, 0.1)
            else:
                lr *= 0.7
                if lr < 1e-6:
                    break
    
    x = best_x
    
    # Phase 2: Coordinate descent with exact evaluation
    if time.time() - start_time < time_limit - 60:
        improved = True
        max_cd_iter = 50
        for cd_iter in range(max_cd_iter):
            if time.time() - start_time > time_limit - 60:
                break
            improved = False
            for i in range(n):
                if time.time() - start_time > time_limit - 60:
                    break
                # Try small perturbations
                base_area = best_area
                for delta in [0.01, -0.01, 0.005, -0.005, 0.001, -0.001]:
                    x_trial = best_x.copy()
                    x_trial[i] += delta
                    area_trial = compute_area_exact(x_trial)
                    if area_trial < base_area - 1e-12:
                        best_x = x_trial
                        base_area = area_trial
                        improved = True
                        break
            if not improved:
                break
    
    # Phase 3: Fine coordinate descent
    if time.time() - start_time < time_limit - 30:
        for cd_iter in range(20):
            if time.time() - start_time > time_limit - 30:
                break
            improved = False
            for i in range(n):
                if time.time() - start_time > time_limit - 30:
                    break
                base_area = best_area
                for delta in [0.001, -0.001, 0.0005, -0.0005, 0.0001, -0.0001]:
                    x_trial = best_x.copy()
                    x_trial[i] += delta
                    area_trial = compute_area_exact(x_trial)
                    if area_trial < base_area - 1e-12:
                        best_x = x_trial
                        base_area = area_trial
                        improved = True
                        break
            if not improved:
                break
    
    # Final check
    final_area = compute_area_exact(best_x)
    if final_area <= 0 or np.any(~np.isfinite(best_x)):
        # Fallback to Keich
        positions = []
        for i in range(n):
            eps = np.array(list(str(bin(i))[2:]), dtype=int)
            eps = np.pad(eps, (k - len(eps), 0), "constant")
            j_arr = np.arange(1, k + 1)
            positions.append(np.sum((1 - j_arr) * eps * (2.0 ** -j_arr) / k))
        return np.array(positions, dtype=np.float64)
    
    return best_x.astype(np.float64)

# EVOLVE-BLOCK-END
