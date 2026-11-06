"""Pack 200 equal circles in the unit square [-1/2, 1/2]^2 to maximize common radius."""
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

N = 200
START_TIME = time.time()
TIME_LIMIT = 145.0  # Safety margin before 150s kill


def _remaining_time():
    return TIME_LIMIT - (time.time() - START_TIME)


def _compute_radius(centers):
    """Compute the maximum radius for given centers."""
    if len(centers) == 0:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Wall distances
    dx = 0.5 - np.abs(centers[:, 0])
    dy = 0.5 - np.abs(centers[:, 1])
    wall_min = min(np.min(dx), np.min(dy))
    if wall_min < 0:
        return 0.0
    if N < 2:
        return float(wall_min)
    d = pdist(centers)
    min_pair = np.min(d)
    pair_term = min_pair / 2.0
    return min(float(wall_min), float(pair_term))


def _pdist_index_to_pair(k, n):
    """Convert pdist condensed index to (i, j) with i < j."""
    start = 0
    for i in range(n - 1):
        block_size = n - i - 1
        if k < start + block_size:
            j = i + 1 + (k - start)
            return i, j
        start += block_size
    return 0, 1  # Fallback


def _hexagonal_grid(n_circles):
    """
    Create a hexagonal packing arrangement for n_circles circles.
    Returns an array of shape (n_circles, 2).
    """
    best_centers = None
    best_r = 0.0

    # Try different numbers of rows
    for m in range(10, 25):
        base_per_row = N // m
        extra = N % m

        row_counts = []
        for i in range(m):
            if i < extra:
                row_counts.append(base_per_row + 1)
            else:
                row_counts.append(base_per_row)

        max_row_count = max(row_counts)

        r_width = 1.0 / (2.0 * max_row_count)
        r_height = 1.0 / ((m - 1) * np.sqrt(3) + 2)
        r_est = min(r_width, r_height)

        if r_est <= best_r and best_centers is not None:
            continue

        dx = 2.0 * r_est
        dy = np.sqrt(3) * r_est

        centers = []
        y_offset = -0.5 + r_est

        for row_idx, row_count in enumerate(row_counts):
            x_offset = 0.0
            if row_idx % 2 == 1:
                x_offset = dx / 2.0

            if row_count > 0:
                row_width = (row_count - 1) * dx
                start_x = -row_width / 2.0 + x_offset
            else:
                start_x = 0.0

            for col_idx in range(row_count):
                x = start_x + col_idx * dx
                y = y_offset + row_idx * dy
                centers.append([x, y])

        centers = np.array(centers)

        # Center the arrangement in the square
        centers[:, 0] -= np.mean(centers[:, 0])
        centers[:, 1] -= np.mean(centers[:, 1])

        # Clip to be within bounds
        centers = np.clip(centers, -0.5 + 1e-9, 0.5 - 1e-9)

        r = _compute_radius(centers)
        if r > best_r:
            best_r = r
            best_centers = centers.copy()

    return best_centers if best_centers is not None else np.random.uniform(-0.5, 0.5, (N, 2))


def _refine_lbfgs(centers, r_target, max_time=5.0):
    """
    Refine centers to fit radius r_target using L-BFGS-B.
    Minimizes the sum of squared constraint violations.
    """
    centers = centers.copy()
    x0 = centers.flatten()
    
    def objective_and_grad(x):
        if _remaining_time() < max_time:
            return 1e10, np.zeros_like(x)
        
        c = x.reshape(-1, 2)
        # Wall constraints
        # dist to walls: 0.5 - |x|, 0.5 - |y|
        # violation: max(0, r_target - dist)
        wx = 0.5 - np.abs(c[:, 0])
        wy = 0.5 - np.abs(c[:, 1])
        vw_x = np.maximum(0.0, r_target - wx)
        vw_y = np.maximum(0.0, r_target - wy)
        
        obj = np.sum(vw_x**2) + np.sum(vw_y**2)
        grad_x = np.zeros_like(x)
        grad_y = np.zeros_like(x)
        
        # Gradients for walls
        # d/dx max(0, r - (0.5 - |x|)) = -sign(x) * 2*max(0, r - (0.5 - |x|)) if x!=0
        # Note: if x=0, subgradient is in [-2v, 2v]. We can use 0 or sign(0)=0.
        sign_x = np.sign(c[:, 0])
        sign_y = np.sign(c[:, 1])
        # Handle x=0 carefully, but sign(0)=0 is a valid subgradient
        grad_x += -2.0 * vw_x * sign_x
        grad_y += -2.0 * vw_y * sign_y
        
        # Pair constraints
        # d = pdist(c)
        # This is expensive to compute gradient for every call.
        # We can use a simpler approximation or just compute pairwise for active constraints.
        # For L-BFGS-B, we need full gradient.
        # Let's compute all pair distances and gradients.
        # This is O(N^2). N=200, so 40000 pairs. Acceptable.
        
        d = pdist(c)
        v = np.maximum(0.0, 2.0 * r_target - d)
        obj += np.sum(v**2)
        
        # Gradient of pair term
        # d_ij = ||c_i - c_j||
        # d/d(c_i) max(0, 2r - d_ij)^2 = -2 * max(0, 2r - d_ij) * (c_i - c_j) / d_ij
        # We need to accumulate gradients for each c_i.
        # Since pdist doesn't give indices directly in a vectorized way for grad accumulation,
        # we loop over pairs or use a trick.
        # Given N=200, a loop over pairs is ~20k iterations, which is fast in Python if vectorized inside?
        # No, Python loop is slow. Let's try to vectorize or use a different approach.
        
        # Alternative: Use a sparse representation or just accept the cost if we reduce calls.
        # Actually, we can compute the gradient matrix.
        # Let D be the distance matrix.
        # G[i, j] = -2 * max(0, 2r - D[i,j]) * (c_i - c_j) / D[i,j]
        # grad[c_i] += sum_j G[i, j]
        
        # To avoid O(N^2) Python loop, we can use broadcasting.
        c_i = c[:, np.newaxis, :] # (N, 1, 2)
        c_j = c[np.newaxis, :, :] # (1, N, 2)
        diff = c_i - c_j # (N, N, 2)
        dist = np.linalg.norm(diff, axis=2) # (N, N)
        np.fill_diagonal(dist, 1e-10)
        
        v_matrix = np.maximum(0.0, 2.0 * r_target - dist)
        # Gradient contribution from pair (i, j) to c_i is -2 * v * (c_i - c_j) / dist
        # Note: v is symmetric.
        # grad_i = sum_j -2 * v_ij * diff_ij / dist_ij
        
        # We only care about i < j or i != j?
        # The objective is sum_{i<j} v_ij^2.
        # So we sum over i<j.
        # Using full matrix, we sum over all j != i, but v_ij = v_ji and diff_ij = -diff_ji.
        # So sum_{j != i} ... will double count if we are not careful?
        # No, the objective is sum_{i<j}.
        # If we compute grad for c_i as sum_{j != i} -2 v_ij (c_i - c_j)/dist_ij,
        # this is effectively summing over all j.
        # But the term for (i, j) in the objective is v_ij^2.
        # d/d(c_i) v_ij^2 = -2 v_ij (c_i - c_j)/dist_ij.
        # So yes, we sum over all j != i.
        
        # However, v_ij is 0 if dist > 2r.
        # So only active constraints contribute.
        
        # Vectorized gradient calculation:
        # factor = -2 * v_matrix / dist_matrix (shape N, N)
        # grad = sum(axis=1) of factor[:, :, np.newaxis] * diff
        # But we must be careful with the diagonal and the fact that we sum i<j.
        # If we sum over all j != i, we are effectively computing the gradient of sum_{i != j} 0.5 * v_ij^2 ?
        # No, sum_{i<j} v_ij^2 = 0.5 * sum_{i != j} v_ij^2.
        # So d/d(c_i) (0.5 * sum_{j != i} v_ij^2) = 0.5 * sum_{j != i} -2 v_ij (c_i - c_j)/dist_ij = - sum_{j != i} v_ij (c_i - c_j)/dist_ij.
        
        # Let's compute this.
        factor = np.zeros_like(dist)
        mask = dist < 2.0 * r_target
        factor[mask] = -v_matrix[mask] / dist[mask]
        np.fill_diagonal(factor, 0.0)
        
        # grad_c = sum_j factor[i, j] * diff[i, j]
        grad_c = np.sum(factor[:, :, np.newaxis] * diff, axis=1)
        
        grad_x += grad_c[:, 0]
        grad_y += grad_c[:, 1]
        
        grad = np.concatenate([grad_x, grad_y])
        return obj, grad

    bounds = [(-0.5, 0.5)] * (2 * N)
    
    try:
        res = minimize(objective_and_grad, x0, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
        if res.success or res.fun < 1e-6:
            new_centers = res.x.reshape(-1, 2)
            new_centers = np.clip(new_centers, -0.5, 0.5)
            return new_centers
    except Exception:
        pass
        
    return centers


def construct():
    """Construct optimal circle packing."""
    # Start with hexagonal grid
    centers = _hexagonal_grid(N)
    best_radius = _compute_radius(centers)
    best_centers = centers.copy()

    # Iteratively increase radius and refine
    # Start with current radius and try to increase it
    current_r = best_radius * 1.01
    step = 0.0005
    
    while _remaining_time() > 5.0:
        # Try to refine for current_r
        refined_centers = _refine_lbfgs(best_centers, current_r, max_time=3.0)
        
        # Check if the refined centers actually support current_r
        # We need to check if min_dist >= 2*current_r and wall_dist >= current_r
        # If not, we might have to reduce current_r or accept a smaller improvement
        r_achieved = _compute_radius(refined_centers)
        
        if r_achieved >= current_r - 1e-6:
            # Success, we can increase radius
            best_centers = refined_centers
            best_radius = r_achieved
            current_r += step
            step *= 1.0 # Keep step constant or increase slightly
            if step < 0.002:
                 step = 0.002
        else:
            # Failed to achieve current_r, try a smaller step or stop
            # If we can't even achieve the previous best, something is wrong
            if r_achieved > best_radius:
                best_centers = refined_centers
                best_radius = r_achieved
                current_r = best_radius + step * 0.5
            else:
                # No improvement, break
                break

    # Final safety clip
    best_centers = np.clip(best_centers, -0.5 + 1e-12, 0.5 - 1e-12)
    return best_centers
