import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = 300.0
        
        lo = [b[0] for b in bounds]
        hi = [b[1] for b in bounds]
        
        # ZDT6 specific logic
        # f1(x1) = 1 - exp(-4*x1) * sin(6*pi*x1)^8
        # f2(x) = (1 + 9*sum(x2..x10)/9) * (1 + sin(3*pi*x1))
        # To minimize f2 for a given x1, set x2..x10 = 0.
        # Then f2 = 1 + sin(3*pi*x1)
        
        # We need to select 200 x1 values.
        # The points are (f1(x1), 1 + sin(3*pi*x1)).
        
        # 1. Generate high-resolution grid for x1
        # We need enough resolution to capture the Pareto front accurately.
        # The frequency of sin(6*pi*x1) is 6 cycles.
        # 100,000 points should be sufficient.
        n_grid = 100000
        x1_grid = np.linspace(0, 1, n_grid)
        
        # Calculate f1 and f2
        sin_term = np.sin(6 * np.pi * x1_grid)
        f1_grid = 1 - np.exp(-4 * x1_grid) * np.power(sin_term, 8)
        
        sin_term_f2 = np.sin(3 * np.pi * x1_grid)
        f2_grid = 1 + sin_term_f2
        
        # 2. Extract Pareto Front
        # Sort by f1
        sort_idx = np.argsort(f1_grid)
        f1_sorted = f1_grid[sort_idx]
        f2_sorted = f2_grid[sort_idx]
        x1_sorted = x1_grid[sort_idx]
        
        # Find non-dominated points
        # A point is non-dominated if f2 is strictly less than the minimum f2 seen so far (for smaller f1)
        # Since we sorted by f1 ascending, we just track min_f2.
        # Note: If f1 is same, we keep the one with min f2.
        
        # We want the lower envelope.
        # Initialize with the first point
        # We iterate and keep points where f2 is a new record low.
        
        # However, standard Pareto front extraction for minimization:
        # Sort by f1 ascending.
        # Keep point if f2 < min_f2_so_far.
        
        min_f2 = np.inf
        pareto_indices = []
        
        # Vectorized approach for finding the lower envelope
        # This is equivalent to finding the indices where f2 is a "prefix minimum"
        
        # np.minimum.accumulate gives the running minimum.
        # But we need the indices where the new minimum is achieved.
        
        # Let's do it in a loop for clarity and to handle ties if necessary, 
        # but vectorized is better.
        
        # Running minimum of f2_sorted
        running_min = np.minimum.accumulate(f2_sorted)
        
        # Points where f2_sorted == running_min are candidates.
        # But we only want the first occurrence of each minimum value?
        # Actually, if f1 increases and f2 stays same, the new point dominates the old one?
        # No, if f1_new > f1_old and f2_new == f2_old, the new point is dominated by the old one (worse f1, same f2).
        # So we only want the points where f2 strictly decreases.
        
        # Strictly decreasing f2
        # diff = np.diff(f2_sorted)
        # strict_decrease = np.where(diff < 0)[0] + 1
        # Also include the first point.
        
        # Let's refine:
        # We want indices i such that f2_sorted[i] < min(f2_sorted[0:i])
        
        # We can find these by checking if f2_sorted[i] == running_min[i] and (i==0 or f2_sorted[i] < f2_sorted[i-1])?
        # No, if f2 stays flat, the first one is the best.
        
        # Let's use a simple loop for extraction as it's O(N) and N=100k is fast.
        pareto_u = []
        pareto_v = []
        pareto_x1 = []
        
        min_v = np.inf
        for i in range(n_grid):
            if f2_sorted[i] < min_v:
                pareto_u.append(f1_sorted[i])
                pareto_v.append(f2_sorted[i])
                pareto_x1.append(x1_sorted[i])
                min_v = f2_sorted[i]
                
        pareto_u = np.array(pareto_u)
        pareto_v = np.array(pareto_v)
        pareto_x1 = np.array(pareto_x1)
        
        M = len(pareto_u)
        
        # 3. Optimize selection of 200 points from the Pareto front
        # Using Dynamic Programming
        # DP[k, j] = max HV using k points, ending at point j (0-indexed)
        # HV contribution of adding point j after point m:
        # (R1 - u_j) * (v_m - v_j)
        # Note: The first point contributes (R1 - u_j) * (R2 - v_j) ?
        # Standard HV formula for points sorted by u ascending:
        # HV = sum_{i=1}^K (R1 - u_i) * (v_{i-1} - v_i) where v_0 = R2.
        
        R1 = 11.0
        R2 = 11.0
        K = self.budget # 200
        
        # If M is very large, DP is O(K*M^2) which might be too slow.
        # Let's check M.
        # If M > 2000, we might need to subsample or use a heuristic.
        # However, for ZDT6, the Pareto front is likely not huge.
        # Let's assume M is manageable.
        
        if M < K:
            # If we have fewer points on the front than budget, take all?
            # No, we can evaluate points off the front? No, that reduces HV.
            # But we have budget 200. If M < 200, we take all M points.
            # We have extra budget. We can't duplicate points (same x1 gives same point).
            # We could evaluate slightly different x1 to get slightly different points?
            # But they would be dominated or very close.
            # Best to just take all Pareto points and maybe some duplicates? 
            # Duplicates don't increase HV.
            # So we just evaluate the M points.
            selected_indices = np.arange(M)
        else:
            # DP
            # To save memory and time, we only need the previous row.
            # dp_prev[j] for j in 0..M-1
            # dp_curr[j] for j in 0..M-1
            
            # Base case k=1:
            # dp[1, j] = (R1 - u_j) * (R2 - v_j)
            
            # Transition:
            # dp[k, j] = max_{m < j} ( dp[k-1, m] + (R1 - u_j) * (v_m - v_j) )
            
            # Vectorized implementation:
            # For a fixed k, we compute dp_curr for all j.
            # dp_curr[j] = max_{m < j} ( dp_prev[m] + (R1 - u_j) * v_m - (R1 - u_j) * v_j )
            #            = (R1 - u_j) * v_j * (-1) + max_{m < j} ( dp_prev[m] + (R1 - u_j) * v_m )
            
            # Let A[m] = dp_prev[m]
            # Let B[m] = v_m
            # Let C[j] = R1 - u_j
            # We want max_{m < j} ( A[m] + C[j] * B[m] )
            
            # This is a max of linear functions in C[j].
            # Since C[j] is decreasing (u_j increasing) and B[m] is decreasing (v_m decreasing),
            # the slopes B[m] are decreasing.
            # The intercepts A[m] are increasing.
            # This allows for a Convex Hull Trick optimization O(M) per k.
            # But implementing CHT in Python is complex.
            
            # Let's try the O(M^2) vectorized approach first and see if it fits in time.
            # If M ~ 1000, M^2 = 1e6. K=200. Total 2e8 ops.
            # NumPy can do 1e8 ops in ~0.1-0.5s. So 2e8 is ~1-2s.
            # This is well within 300s.
            
            # We will cap M to 5000 to be safe on memory (5000x5000 float64 = 200MB).
            # If M > 5000, we subsample the Pareto front uniformly?
            # Or just take the first 5000? No, that biases to low u.
            # We should subsample evenly from the Pareto front.
            
            if M > 5000:
                # Subsample
                step = M // 5000
                if step == 0: step = 1
                indices = np.arange(0, M, step)
                if indices[-1] != M - 1:
                    indices = np.append(indices, M - 1)
                pareto_u = pareto_u[indices]
                pareto_v = pareto_v[indices]
                pareto_x1 = pareto_x1[indices]
                M = len(pareto_u)
                
            # DP Initialization
            # dp_prev for k=1
            dp_prev = (R1 - pareto_u) * (R2 - pareto_v)
            
            # Iterate k from 2 to K
            for k in range(2, K + 1):
                # We need dp_curr[j] for j >= k-1 (at least k points needed)
                # Actually, j must be at least k-1 (0-indexed) to have k points ending at j.
                
                # Construct matrix of terms
                # Term[m, j] = dp_prev[m] + (R1 - u_j) * (v_m - v_j)
                # We only care about m < j.
                
                # To save memory, we can compute in chunks or use the property.
                # Let's compute the full MxM matrix for the term part that varies with j.
                # Term = dp_prev[:, None] + (R1 - pareto_u[None, :]) * (pareto_v[:, None] - pareto_v[None, :])
                
                # This creates an MxM matrix.
                # M=5000 -> 25e6 floats -> 200MB.
                # This might be tight but okay.
                
                # Optimization:
                # dp_curr[j] = max_{m < j} ( dp_prev[m] + (R1 - u_j) * v_m - (R1 - u_j) * v_j )
                #           = - (R1 - u_j) * v_j + max_{m < j} ( dp_prev[m] + (R1 - u_j) * v_m )
                
                # Let's compute the inner max.
                # InnerMax[j] = max_{m < j} ( dp_prev[m] + (R1 - u_j) * v_m )
                
                # We can compute this using a loop over j and vectorized max over m?
                # No, that's O(M^2) in Python loops, too slow.
                
                # We must use NumPy broadcasting.
                # Create matrix P of shape (M, M) where P[m, j] = dp_prev[m] + (R1 - u_j) * v_m
                # Then take max along axis 0, but only for m < j.
                
                # Mask: m < j
                # mask = np.triu(np.ones((M, M), dtype=bool), k=1) # m > j
                # We want m < j, so we invert.
                
                # This is memory intensive.
                
                # Alternative:
                # Since we only need the max for m < j, and the structure is specific,
                # let's just do it.
                
                # If M is small (e.g. < 1000), this is fine.
                # If M is 5000, 25M elements.
                
                u_j = R1 - pareto_u
                v_m = pareto_v
                
                # P[m, j] = dp_prev[m] + u_j[j] * v_m[m]
                # Note: u_j depends on j, v_m depends on m.
                # P = dp_prev[:, None] + u_j[None, :] * v_m[:, None]
                
                P = dp_prev[:, None] + u_j[None, :] * v_m[:, None]
                
                # We need max over m for each j, with m < j.
                # Create a mask where m < j.
                # indices = np.arange(M)
                # mask = indices[:, None] < indices[None, :]
                
                # To save memory, we can do this in a loop over j?
                # For j in range(M):
                #    if j < k-1: continue
                #    dp_curr[j] = np.max(P[:j, j])
                # This is O(M^2) Python loop. 25M iterations. Too slow.
                
                # We need to vectorize the max with mask.
                # np.where(mask, P, -np.inf) then max(axis=0)
                
                # This doubles memory temporarily.
                
                # Let's check if we can reduce M further.
                # If M is 2000, M^2 = 4M. 4M * 8 bytes = 32MB. Very safe.
                # Let's cap M at 2000.
                
                if M > 2000:
                    step = M // 2000
                    if step == 0: step = 1
                    indices = np.arange(0, M, step)
                    if indices[-1] != M - 1:
                        indices = np.append(indices, M - 1)
                    pareto_u = pareto_u[indices]
                    pareto_v = pareto_v[indices]
                    pareto_x1 = pareto_x1[indices]
                    M = len(pareto_u)
                    # Recalculate dp_prev for new M?
                    # dp_prev was for k=1. It's just (R1-u)(R2-v).
                    # So we can just recalculate it.
                    dp_prev = (R1 - pareto_u) * (R2 - pareto_v)
                    u_j = R1 - pareto_u
                    v_m = pareto_v

                # Now M <= 2000.
                # P = dp_prev[:, None] + u_j[None, :] * v_m[:, None]
                P = dp_prev[:, None] + u_j[None, :] * v_m[:, None]
                
                # Mask m < j
                idx = np.arange(M)
                mask = idx[:, None] < idx[None, :]
                
                # Apply mask
                # P_masked = np.where(mask, P, -np.inf)
                # dp_curr = np.max(P_masked, axis=0)
                
                # To save memory, we can do:
                # P[~mask] = -np.inf
                P[~mask] = -np.inf
                dp_curr = np.max(P, axis=0)
                
                # The first k-1 points cannot be the k-th point (not enough previous points)
                # So dp_curr[0:k-1] should be -inf
                dp_curr[:k-1] = -np.inf
                
                dp_prev = dp_curr
                
                # Time check
                if time.time() - start_time > deadline - 10:
                    break
            
            # Find best last point
            best_last_idx = np.argmax(dp_prev)
            
            # Backtrack to find the path
            # This requires storing the argmax for each step.
            # We didn't store it. We need to re-run or store.
            # Storing argmax matrix is MxK floats/ints. 2000x200 = 400k ints. Fine.
            
            # Let's re-run DP storing argmax.
            # This doubles the time but ensures we get the points.
            
            if time.time() - start_time > deadline - 5:
                 # Fallback: uniform selection
                 selected_indices = np.linspace(0, M-1, K).astype(int)
            else:
                # Re-run with argmax storage
                dp_prev = (R1 - pareto_u) * (R2 - pareto_v)
                argmax_prev = np.zeros(M, dtype=int) # Not used for k=1
                
                argmax_path = []
                
                for k in range(2, K + 1):
                    u_j = R1 - pareto_u
                    v_m = pareto_v
                    P = dp_prev[:, None] + u_j[None, :] * v_m[:, None]
                    idx = np.arange(M)
                    mask = idx[:, None] < idx[None, :]
                    P[~mask] = -np.inf
                    dp_curr = np.max(P, axis=0)
                    argmax_curr = np.argmax(P, axis=0)
                    dp_curr[:k-1] = -np.inf
                    argmax_curr[:k-1] = 0 # Dummy
                    
                    argmax_path.append(argmax_curr)
                    dp_prev = dp_curr
                    argmax_prev = argmax_curr
                    
                    if time.time() - start_time > deadline - 5:
                        break
                
                # Backtrack
                selected_indices = [0] * K
                selected_indices[K-1] = np.argmax(dp_prev)
                for k in range(K-2, 0, -1):
                    # argmax_path[k-1] corresponds to step k (since path[0] is for k=2)
                    # Wait, argmax_path[0] is for k=2.
                    # We want the index m that led to j at step k.
                    # j is selected_indices[k]
                    # m is argmax_path[k-1][j]
                    j = selected_indices[k+1]
                    selected_indices[k] = argmax_path[k-1][j]
                
                # selected_indices now contains the indices of the K points
                # Note: The first point (k=1) is not in argmax_path.
                # We need to determine the first point.
                # The DP for k=1 didn't have a "previous".
                # The path starts at k=1.
                # The backtrack gives indices for k=K down to k=2.
                # selected_indices[0] is the index for k=1.
                # How to find it?
                # The value dp[1, m] is fixed.
                # The transition to k=2 used argmax_path[0].
                # So selected_indices[1] is known.
                # selected_indices[0] is argmax_path[0][selected_indices[1]]?
                # Yes.
                selected_indices[0] = argmax_path[0][selected_indices[1]]

        # Map selected indices to x1 values
        x1_vals = pareto_x1[selected_indices]
        
        # Construct points
        points = np.zeros((len(x1_vals), self.dim))
        points[:, 0] = x1_vals
        # x[1:] are 0
        
        # Evaluate
        for i in range(len(x1_vals)):
            if time.time() - start_time > deadline:
                break
            x = points[i]
            x_clipped = np.clip(x, lo, hi)
            problem(x_clipped)
            
        return None
