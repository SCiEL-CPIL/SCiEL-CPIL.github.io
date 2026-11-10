import numpy as np
import time
from scipy.optimize import minimize

def _compute_area(positions, n=64, n_heights=20000):
    """
    Compute the union area of the triangles.
    positions: array of shape (n,)
    """
    a = (np.arange(n) + 1) / n
    y = (np.arange(n_heights) + 0.5) / n_heights
    w = (1 - y) / n
    
    # L[i, m] = x_i + a_i * y_m
    L = positions[:, np.newaxis] + a[:, np.newaxis] * y[np.newaxis, :]
    
    # Sort L along axis 0 for each height m
    L_sorted = np.sort(L, axis=0)
    
    # Differences between consecutive sorted L values
    diffs = np.diff(L_sorted, axis=0)
    
    # Overlap contribution: min(diff, w)
    overlaps = np.minimum(diffs, w[np.newaxis, :])
    
    # Union length at each height: w + sum(overlaps)
    union_lengths = w + np.sum(overlaps, axis=0)
    
    # Area is the mean of union lengths
    area = np.mean(union_lengths)
    
    return area

def _subgradient(positions, n=64, n_heights=20000):
    """
    Compute a subgradient of the area with respect to positions.
    The area is a sum of min terms, so we use the subgradient of min(a, b) = 1 if a < b else 0.
    """
    a = (np.arange(n) + 1) / n
    y = (np.arange(n_heights) + 0.5) / n_heights
    w = (1 - y) / n
    
    L = positions[:, np.newaxis] + a[:, np.newaxis] * y[np.newaxis, :]
    
    # Sort L and keep track of indices
    sorted_indices = np.argsort(L, axis=0)
    L_sorted = np.take_along_axis(L, sorted_indices, axis=0)
    
    diffs = np.diff(L_sorted, axis=0)
    
    # Indicator: 1 if diff < w, else 0 (subgradient of min(diff, w) w.r.t. diff)
    # Note: at diff == w, the subgradient is in [0, 1]. We use 0.5 for stability.
    indicator = np.where(diffs < w[np.newaxis, :], 1.0, 0.0)
    # Handle the case where diffs == w more carefully? 
    # Actually, if diff < w, the term is diff. d(diff)/d(x_i) depends on which triangle is i.
    # If diff >= w, the term is w. d(w)/d(x_i) = 0.
    
    # We need to map back to original indices.
    # For each height m, we have sorted_indices[:, m].
    # Let S be the sorted order. S[0], S[1], ..., S[n-1].
    # diffs[k, m] = L[S[k+1, m], m] - L[S[k, m], m]
    # If diffs[k, m] < w[m], then the contribution to the union length is diffs[k, m].
    # The derivative w.r.t. x_{S[k+1, m]} is +1 * a_{S[k+1, m]}
    # The derivative w.r.t. x_{S[k, m]} is -1 * a_{S[k, m]}
    # (since L_i = x_i + a_i y, dL_i/dx_i = 1, but wait... 
    # The objective is Area = mean( w + sum(min(diffs, w)) )
    # Let U_m = w[m] + sum_k min(diffs[k, m], w[m])
    # dU_m / dx_j = sum_k [ indicator(diffs[k,m] < w[m]) * (d(diffs[k,m])/dx_j) ]
    # d(diffs[k,m])/dx_j = (if S[k+1,m]==j then a_j else 0) - (if S[k,m]==j then a_j else 0)
    
    grad = np.zeros(n)
    
    # Vectorized approach:
    # For each k (0 to n-2) and m, if indicator[k, m] is 1:
    # grad[S[k+1, m]] += a[S[k+1, m]]
    # grad[S[k, m]] -= a[S[k, m]]
    
    # We can use np.add.at for accumulation
    # But we need to apply the indicator first.
    # Let's create arrays of indices and values.
    
    # This loop might be slow. Let's try to vectorize or use sparse logic.
    # Given n=64 and n_heights=20000, total pairs = 63 * 20000 = 1.26M.
    # A loop in Python is too slow. We need numpy.
    
    # Let's construct the updates.
    # idx_pos = sorted_indices[1:, :] (shape (n-1, n_heights))
    # idx_neg = sorted_indices[:-1, :] (shape (n-1, n_heights))
    
    idx_pos = sorted_indices[1:, :]
    idx_neg = sorted_indices[:-1, :]
    
    # Mask where the gradient is non-zero
    mask = indicator # shape (n-1, n_heights)
    
    # We want to add a[idx_pos] to grad[idx_pos] where mask is True
    # and subtract a[idx_neg] from grad[idx_neg] where mask is True.
    
    # Using np.bincount is efficient for this.
    # We need to flatten and weight.
    
    # Positive contributions
    # We want to sum a[idx_pos] for each index j, counting only where mask is True.
    # We can create a flat array of indices and weights.
    
    # However, np.add.at is simpler but potentially slower than bincount if not careful.
    # Let's try bincount.
    
    # Flatten indices
    flat_pos = idx_pos.flatten()
    flat_neg = idx_neg.flatten()
    
    # Weights are a[idx]
    weights_pos = a[flat_pos]
    weights_neg = a[flat_neg]
    
    # Apply mask
    flat_mask = mask.flatten()
    
    # Only count where mask is 1
    # We can use np.bincount with weights
    # But we need to zero out weights where mask is 0.
    
    weights_pos_masked = np.where(flat_mask, weights_pos, 0.0)
    weights_neg_masked = np.where(flat_mask, weights_neg, 0.0)
    
    # Sum contributions
    grad += np.bincount(flat_pos, weights=weights_pos_masked, minlength=n)
    grad -= np.bincount(flat_neg, weights=weights_neg_masked, minlength=n)
    
    # Normalize by n_heights (since area is mean)
    grad /= n_heights
    
    return grad

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    start_time = time.time()
    deadline = start_time + 590  # Safety margin before 600s limit

    a = (np.arange(n) + 1) / n
    
    # 1. Initialization
    # Based on findings, x_i = -a_i + 0.5 is a good starting point
    # Let's verify this logic. 
    # If x_i = -a_i + C, then L_i = -a_i + C + a_i y = C - a_i(1-y).
    # This aligns the triangles such that they fan out from a common point?
    # Let's just use the suggested initialization.
    
    best_positions = -a + 0.5
    best_area = _compute_area(best_positions, n, 20000)
    
    current_pos = best_positions.copy()
    current_area = best_area
    
    # 2. Subgradient Descent
    # Use a decreasing step size schedule.
    # L1 norm of gradient?
    
    step_size = 0.05
    min_step_size = 1e-5
    decay = 0.999
    
    # We can also try to use the norm of the gradient to adapt step size.
    
    while time.time() < deadline - 10:
        grad = _subgradient(current_pos, n, 20000)
        
        # Check if gradient is zero
        if np.allclose(grad, 0.0, atol=1e-8):
            break
            
        # Update
        trial_pos = current_pos - step_size * grad
        
        trial_area = _compute_area(trial_pos, n, 20000)
        
        if trial_area < current_area:
            current_pos = trial_pos
            current_area = trial_area
            if current_area < best_area:
                best_area = current_area
                best_positions = current_pos.copy()
            # Decay step size
            step_size = max(min_step_size, step_size * decay)
        else:
            # If not improving, reduce step size more aggressively
            step_size = max(min_step_size, step_size * 0.9)
            
            # If step size is very small, we might be stuck. 
            # But subgradient descent usually converges slowly.
            # Let's rely on the time budget.

    # 3. Fine Polish with Random Hill Climbing
    # Use the subgradient direction as a bias for the random search?
    # Or just simple random search with small steps.
    
    step_size = 0.001
    rng = np.random.RandomState(42)
    
    while time.time() < deadline - 1:
        idx = rng.randint(n)
        delta = rng.uniform(-step_size, step_size)
        
        trial_pos = current_pos.copy()
        trial_pos[idx] += delta
        
        trial_area = _compute_area(trial_pos, n, 20000)
        
        if trial_area < current_area:
            current_pos = trial_pos
            current_area = trial_area
            if current_area < best_area:
                best_area = current_area
                best_positions = current_pos.copy()
            step_size = max(1e-6, step_size * 0.99)
        else:
            step_size = max(1e-6, step_size * 0.999)

    # Ensure finite
    if not np.all(np.isfinite(best_positions)):
        return best_positions.astype(np.float64)
    
    return best_positions.astype(np.float64)

if __name__ == "__main__":
    # Quick test
    pos = kakeya_needle_2d_positions64()
    print(f"Shape: {pos.shape}")
    print(f"Area: {_compute_area(pos)}")
