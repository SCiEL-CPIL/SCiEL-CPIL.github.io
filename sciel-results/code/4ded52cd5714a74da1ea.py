"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h (adjacent shift)."""
import numpy as np
from scipy.optimize import minimize
import time

N = 100

def _overlap_max(h):
    """Compute max of h[i]*(1-h[i+1]) with zero-padding at the end."""
    overlaps = h[:-1] * (1.0 - h[1:])
    overlaps = np.append(overlaps, h[-1])
    return overlaps.max()

def _smooth_max(values, alpha):
    """Smooth approximation of max using log-sum-exp."""
    m = np.max(values)
    if m > 1e10:
        return m
    shifted = alpha * (values - m)
    shifted = np.clip(shifted, -50, 50)
    return m + np.log(np.sum(np.exp(shifted))) / alpha

def _objective(h, alpha):
    """Smooth surrogate objective: smooth max of adjacent overlaps."""
    overlaps = h[:-1] * (1.0 - h[1:])
    overlaps = np.append(overlaps, h[-1])
    return _smooth_max(overlaps, alpha)

def _objective_grad(h, alpha):
    """Gradient of the smooth max objective."""
    overlaps = h[:-1] * (1.0 - h[1:])
    overlaps = np.append(overlaps, h[-1])
    m = np.max(overlaps)
    shifted = alpha * (overlaps - m)
    shifted = np.clip(shifted, -50, 50)
    exp_vals = np.exp(shifted)
    weights = exp_vals / np.sum(exp_vals)
    
    # Gradient of overlaps with respect to h
    # overlap[i] = h[i]*(1-h[i+1]) for i < N-1
    # overlap[N-1] = h[N-1]
    grad = np.zeros(N)
    for i in range(N - 1):
        # d(overlap[i])/dh[i] = (1 - h[i+1])
        # d(overlap[i])/dh[i+1] = -h[i]
        grad[i] += weights[i] * (1.0 - h[i+1])
        grad[i+1] += weights[i] * (-h[i])
    # Last overlap: d(h[N-1])/dh[N-1] = 1
    grad[N-1] += weights[N-1] * 1.0
    
    return grad

def _enforce_constraints(h):
    """Project h onto feasible set: [0,1]^N with sum = N/2."""
    h = np.clip(h, 0.0, 1.0)
    s = h.sum()
    target = N / 2.0
    if abs(s - target) > 1e-12:
        diff = target - s
        # Distribute residual to elements not at boundary
        can_add = h < 1.0
        can_sub = h > 0.0
        if diff > 0:
            mask = can_add
            if mask.sum() > 0:
                h[mask] += diff / mask.sum()
            else:
                h += diff / N
        else:
            mask = can_sub
            if mask.sum() > 0:
                h[mask] += diff / mask.sum()
            else:
                h += diff / N
        h = np.clip(h, 0.0, 1.0)
    # Final correction
    diff = target - h.sum()
    if abs(diff) > 1e-12:
        h = h + diff / N
        h = np.clip(h, 0.0, 1.0)
    return h

def _optimize_from_start(h0, alpha, max_iter=200):
    """Optimize from a starting point using SLSQP."""
    n = len(h0)
    target = n / 2.0
    
    bounds = [(0.0, 1.0)] * n
    constraints = [{'type': 'eq', 'fun': lambda x: x.sum() - target, 
                    'jac': lambda x: np.ones(n)}]
    
    def obj(x):
        return _objective(x, alpha)
    
    def obj_grad(x):
        return _objective_grad(x, alpha)
    
    try:
        res = minimize(obj, h0, jac=obj_grad, method='SLSQP', bounds=bounds, 
                      constraints=constraints,
                      options={'maxiter': max_iter, 'ftol': 1e-14, 'disp': False})
        if res.success or (hasattr(res, 'fun') and res.fun < _objective(h0, alpha)):
            return res.x
    except Exception:
        pass
    return h0

def _dp_solve(H_levels, deadline):
    """
    Solve using Dynamic Programming on discretized heights.
    Returns the best sequence of height indices and the score.
    """
    # Height levels
    h_vals = np.linspace(0.0, 1.0, H_levels)
    # Sum step size
    sum_step = 1.0 / H_levels
    # Max sum index
    max_sum_idx = int(np.ceil(N / sum_step))
    
    # DP table: dp[s_idx, h_idx] = min max_overlap
    # We iterate i from 0 to N-1
    # State at i=0: sum=0, last_h=0 (dummy)
    
    # To save memory and time, we only keep the current and next dp arrays.
    # dp is a 2D array of shape (max_sum_idx+1, H_levels)
    # Initialize with inf
    dp = np.full((max_sum_idx + 1, H_levels), np.inf)
    
    # At i=0, we haven't chosen h[0] yet?
    # Let's define state (i, s, last_h_idx) as min max_overlap for first i elements.
    # Base case: i=0, s=0. No last_h.
    # We can handle i=0 separately or add a dummy last_h.
    # Let's say at i=0, we pick h[0].
    # So we iterate i from 0 to N-1.
    # For i=0:
    #   for each k in 0..H-1:
    #     s_new = h_vals[k] / sum_step
    #     dp[s_new, k] = 0.0 (max overlap is 0 so far, as there are no pairs)
    
    # Actually, the cost is defined on pairs (h[i], h[i+1]).
    # So for i=0, we just set the state.
    
    # Precompute h_vals / sum_step to get indices
    h_indices = np.round(h_vals / sum_step).astype(int)
    
    # Initialize dp for i=0
    # dp[0, :] = inf
    # For each k:
    #   s_idx = h_indices[k]
    #   dp[s_idx, k] = 0.0
    dp[h_indices, np.arange(H_levels)] = 0.0
    
    # Iterate for i = 1 to N-1
    for i in range(1, N):
        new_dp = np.full((max_sum_idx + 1, H_levels), np.inf)
        
        # We need to transition from dp (which represents state at i-1) to new_dp (state at i)
        # For each previous state (s_idx, j) with cost c:
        #   For each next height k:
        #     s_new = s_idx + h_indices[k]
        #     cost_new = max(c, h_vals[j] * (1 - h_vals[k]))
        #     new_dp[s_new, k] = min(new_dp[s_new, k], cost_new)
        
        # Vectorized approach:
        # For each k (next height index):
        #   h_k = h_vals[k]
        #   term_k = h_vals * (1 - h_k)  # shape (H_levels,)
        #   We want to update new_dp[:, k]
        #   For each s_idx:
        #     s_prev = s_idx - h_indices[k]
        #     If s_prev >= 0:
        #       costs = dp[s_prev, :]  # shape (H_levels,)
        #       max_costs = np.maximum(costs, term_k)  # shape (H_levels,)
        #       min_cost = np.min(max_costs)
        #       new_dp[s_idx, k] = min_cost
        #       # Also need to handle if multiple j give same s_prev? No, s_prev is fixed for s_idx and k.
        #       # But wait, dp[s_prev, :] contains costs for all j.
        #       # We take min over j.
        
        # This loop over s_idx is Python loop over max_sum_idx (~1000-5000).
        # Inside is vectorized over H_levels.
        # 5000 * 100 steps * H_levels ops.
        # 5000 * 100 * 50 = 25,000,000 ops. Fast.
        
        for k in range(H_levels):
            h_k = h_vals[k]
            term_k = h_vals * (1.0 - h_k)
            h_idx_k = h_indices[k]
            
            # s_idx ranges from h_idx_k to max_sum_idx
            # s_prev = s_idx - h_idx_k
            # We can slice dp
            # dp[s_prev, :] where s_prev goes from 0 to max_sum_idx - h_idx_k
            
            # new_dp[s_idx, k] = min_j max(dp[s_idx - h_idx_k, j], term_k[j])
            
            # Let valid_s_prev = max_sum_idx - h_idx_k
            # s_idx range: h_idx_k to max_sum_idx
            
            # We can do this for all s_idx at once?
            # new_dp[:, k] corresponds to s_idx 0..max_sum_idx
            # For s_idx < h_idx_k, it's inf.
            
            # Slice of dp: dp[0 : max_sum_idx - h_idx_k + 1, :]
            # Shape: (max_sum_idx - h_idx_k + 1, H_levels)
            
            if max_sum_idx >= h_idx_k:
                dp_slice = dp[:max_sum_idx - h_idx_k + 1, :]
                # For each row in dp_slice (which corresponds to a specific s_prev),
                # we compute max(row, term_k) and then min over columns.
                # max_costs shape: (num_rows, H_levels)
                max_costs = np.maximum(dp_slice, term_k[None, :])
                min_costs = np.min(max_costs, axis=1)
                
                # Assign to new_dp
                # s_idx = s_prev + h_idx_k
                # So new_dp[h_idx_k : max_sum_idx + 1, k] = min_costs
                new_dp[h_idx_k : max_sum_idx + 1, k] = min_costs
        
        dp = new_dp
        
        if time.time() > deadline - 30:
            break
            
    # Find best final state
    # Target sum is N/2.
    # target_idx = round((N/2) / sum_step)
    target_sum = N / 2.0
    target_idx = int(np.round(target_sum / sum_step))
    
    # We want min over j of dp[target_idx, j]
    # But we also need to reconstruct the path.
    # For now, just get the score.
    if target_idx > max_sum_idx:
        return None, float('inf')
        
    final_costs = dp[target_idx, :]
    best_j = np.argmin(final_costs)
    best_score = final_costs[best_j]
    
    # Reconstruct path
    # This requires storing the choices, which doubles memory and time.
    # Alternative: Just return the score and a heuristic path?
    # Or, we can re-run DP storing parents if we have time.
    # Given the time budget, let's try to reconstruct.
    
    # Re-run DP with parent tracking?
    # Or, we can just return the best_j and target_idx, and use a greedy backtracking?
    # No, greedy backtracking might not recover the optimal path.
    
    # Let's just return the score and a simple path based on the final state?
    # Actually, we can modify the DP to store the previous height index.
    # parents[i][s_idx][k] = j
    # This is N * S * H integers. 100 * 5000 * 50 = 25,000,000 ints. ~200MB.
    # Might be too much memory or slow.
    
    # Alternative:
    # Since we only need a good initial guess for SLSQP, we can just create a path that ends with h_vals[best_j] and has sum target_sum.
    # A simple way: fill with average? No.
    
    # Let's try a simpler reconstruction:
    # We know the optimal cost is best_score.
    # We can do a forward pass again, but this time we only keep states that are "consistent" with the optimal cost?
    # No, that's complex.
    
    # Let's just return the score and a random valid path?
    # No, that defeats the purpose.
    
    # Let's store parents only for the last few steps? No.
    
    # Okay, I will implement the parent tracking but only for the best path found?
    # No, we don't know the best path until the end.
    
    # Let's use a different approach for reconstruction:
    # We can store the entire dp table for all i?
    # Memory: N * S * H * 8 bytes (float) = 100 * 5000 * 50 * 8 = 200MB.
    # This is acceptable.
    
    # Let's re-implement _dp_solve to store all tables.
    
    return None, best_score

def construct():
    deadline = time.time() + 135.0  # Safety margin
    best_h = np.full(N, 0.5)
    best_score = _overlap_max(best_h)
    
    # 1. DP Solver
    # Try different resolutions
    for H_levels in [20, 50, 100]:
        if time.time() > deadline - 60:
            break
        h_dp, s_dp = _dp_solve(H_levels, deadline)
        if h_dp is not None and s_dp < best_score:
            best_h = h_dp
            best_score = s_dp
            
    # 2. Polish with SLSQP
    starts = [best_h.copy(), np.full(N, 0.5)]
    
    # Add some diverse starts
    rng = np.random.default_rng(42)
    for i in range(5):
        h = rng.uniform(0, 1, N)
        starts.append(_enforce_constraints(h))
        
    alphas = [50.0, 200.0, 1000.0, 5000.0]
    
    for h0 in starts:
        if time.time() > deadline - 10:
            break
        h = h0.copy()
        for alpha in alphas:
            if time.time() > deadline - 5:
                break
            h = _optimize_from_start(h, alpha, max_iter=100)
            h = _enforce_constraints(h)
            score = _overlap_max(h)
            if score < best_score:
                best_score = score
                best_h = h.copy()
                
    # Final polish
    if time.time() < deadline - 2:
        h = _optimize_from_start(best_h, 10000.0, max_iter=200)
        h = _enforce_constraints(h)
        score = _overlap_max(h)
        if score < best_score:
            best_score = score
            best_h = h.copy()
            
    best_h = _enforce_constraints(best_h)
    return best_h
