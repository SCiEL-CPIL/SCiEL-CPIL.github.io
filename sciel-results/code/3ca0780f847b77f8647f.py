import time
import random
import numpy as np

N = 64
TIME_LIMIT = 300.0
TOTAL = N * N

def construct():
    start = time.time()
    deadline = start + TIME_LIMIT - 10.0  # Safety margin

    # Precompute coordinates
    xs = np.repeat(np.arange(N), N)
    ys = np.tile(np.arange(N), N)
    
    # Precompute squared distances matrix
    # dist[i, j] = (xi-xj)^2 + (yi-yj)^2
    dx = (xs[:, None] - xs[None, :]).astype(np.int32)
    dy = (ys[:, None] - ys[None, :]).astype(np.int32)
    dist = dx * dx + dy * dy

    def batch_check_validity(candidates, current_indices):
        """
        Checks which candidates in the 'candidates' array can be added to 'current_indices'
        without forming an isosceles triangle.
        
        Returns a boolean mask of the same shape as 'candidates'.
        
        Isosceles conditions for adding candidate c to set S:
        1. d(c, s1) == d(c, s2) for any distinct s1, s2 in S.
           (Apex at c)
        2. d(c, s1) == d(s1, s2) for any distinct s1, s2 in S.
           (Apex at s1, base c-s2)
        3. d(c, s1) == d(s2, s1) for any distinct s1, s2 in S.
           (Apex at s2, base c-s1) -- Symmetric to 2 if we check all pairs.
           
        Actually, the condition is:
        For any pair (s1, s2) in S, and candidate c:
        The triangle (c, s1, s2) is isosceles if:
        d(c, s1) == d(c, s2)  OR
        d(c, s1) == d(s1, s2)  OR
        d(c, s2) == d(s1, s2)
        """
        K = len(current_indices)
        if K < 2:
            return np.ones(len(candidates), dtype=bool)
            
        # D_c: (M, K) where M is number of candidates
        # D_c[m, k] = dist(candidates[m], current_indices[k])
        D_c = dist[np.ix_(candidates, current_indices)]
        
        # Condition 1: d(c, s1) == d(c, s2)
        # For each candidate m, check if all values in D_c[m, :] are unique.
        # We can do this by sorting and checking for duplicates, or using unique.
        # Vectorized unique check: 
        # If we sort D_c[m], duplicates are adjacent.
        # But sorting M*K elements might be slow if M is large.
        # Alternative: Check if any pair in D_c[m] is equal.
        # This is O(K^2) per candidate.
        # Let's use a trick: 
        # For each row, compute the number of unique elements.
        # np.unique is slow in loops.
        
        # Better approach for Condition 1:
        # Create a matrix of differences? No.
        # Let's stick to the efficient vectorized comparison for Condition 2 & 3 first, 
        # which is often the bottleneck for dense sets.
        
        # D_sub: (K, K) distances between points in current set
        D_sub = dist[np.ix_(current_indices, current_indices)]
        
        # Condition 2: d(c, s1) == d(s1, s2)
        # We need to check if D_c[m, k1] == D_sub[k1, k2] for any k1 != k2.
        # This is equivalent to: 
        # For each candidate m, is there any k1, k2 such that D_c[m, k1] == D_sub[k1, k2]?
        
        # Let's construct a set of "forbidden distances" for each candidate.
        # This is tricky to vectorize directly without loops over candidates or sets.
        
        # Let's try a different vectorization strategy for the whole check.
        # We can check validity by iterating through the K points in current_indices?
        # No, we want to check M candidates at once.
        
        # Let's use the following logic:
        # A candidate c is INVALID if:
        # A) It has duplicate distances to any pair in S.
        # B) For any s1 in S, the distance d(c, s1) is equal to the distance d(s1, s2) for some s2 in S (s2 != s1).
        
        # Check A:
        # Sort D_c along axis 1.
        D_c_sorted = np.sort(D_c, axis=1)
        # Check for adjacent duplicates
        has_dup_in_row = np.any(D_c_sorted[:, 1:] == D_c_sorted[:, :-1], axis=1)
        
        # Check B:
        # We need to check if D_c[m, k1] is in the set { D_sub[k1, k2] for k2 != k1 }.
        # Let's precompute for each s1 in S, the set of distances to other points in S.
        # This is hard to vectorize across M candidates directly without creating huge arrays.
        
        # Alternative:
        # For each s1 (k1), we have a vector of distances to all other s2 (k2 != k1).
        # Let's call this row D_sub[k1, :].
        # We want to know if D_c[m, k1] equals ANY value in D_sub[k1, :] (excluding D_sub[k1, k1]=0, but distances are >0 for distinct points).
        # Actually, D_sub[k1, k1] is 0. d(c, s1) is > 0. So we can just check against the whole row.
        
        # Vectorized Check B:
        # For each k1 (0 to K-1):
        #   target_dist = D_c[:, k1]  (shape M)
        #   forbidden_dists = D_sub[k1, :] (shape K)
        #   We want to know if any element of target_dist is in forbidden_dists.
        #   This is a membership test.
        #   We can do: 
        #   match = (target_dist[:, None] == forbidden_dists[None, :]).any(axis=1)
        #   This creates an (M, K) boolean matrix for each k1.
        #   Total memory/time: K * (M * K) operations.
        #   If M=1000, K=100, this is 100 * 1000 * 100 = 10^7 operations. Fast in NumPy.
        
        invalid_b = np.zeros(len(candidates), dtype=bool)
        for k1 in range(K):
            # d(c, s_k1)
            d_cs = D_c[:, k1]
            # d(s_k1, s_k2) for all k2
            d_ss = D_sub[k1, :]
            
            # Check if any d_cs is in d_ss
            # d_cs shape (M,), d_ss shape (K,)
            # Broadcast to (M, K)
            # Note: d_ss contains 0 at index k1. d_cs is > 0. So no false positive from self-distance.
            # But wait, if d_cs == 0? No, c is not in S, so d(c, s) > 0.
            
            # Efficient check:
            # Use np.isin? 
            # np.isin(d_cs, d_ss) is vectorized and likely optimized.
            matches = np.isin(d_cs, d_ss)
            invalid_b |= matches
            
        # Combine conditions
        valid = ~(has_dup_in_row | invalid_b)
        
        return valid

    # --- Simulated Annealing Setup ---
    
    # Initialize with a greedy random set to get a good starting point
    # This ensures we start in a valid state (energy = 0)
    
    def initial_greedy():
        current_indices = []
        current_mask = np.zeros(TOTAL, dtype=bool)
        remaining = np.arange(TOTAL)
        random.shuffle(remaining)
        
        # We can use batch_check_validity here too to speed up initialization
        # But for simplicity and safety, let's do a standard greedy first, 
        # or use the batch checker in chunks.
        
        # Let's use the batch checker for initialization as well.
        # We process candidates in batches of 1000.
        for i in range(0, TOTAL, 1000):
            if time.time() > deadline - 60: break
            batch = remaining[i:i+1000]
            # Filter out already added
            # Since we shuffle once, and only add if valid, we need to track mask.
            # It's easier to just iterate if K is small, but let's stick to simple greedy for init
            # to ensure we have a valid set.
            pass
            
        # Simple greedy for init is safer and fast enough for ~50-100 points
        for idx in remaining:
            if time.time() > deadline - 60: break
            if len(current_indices) < 2:
                current_indices.append(idx)
                current_mask[idx] = True
                continue
                
            # Check single candidate
            # Reuse logic from batch_check_validity for single element?
            # Or just call batch_check_validity with [idx]
            valid_mask = batch_check_validity(np.array([idx]), current_indices)
            if valid_mask[0]:
                current_indices.append(idx)
                current_mask[idx] = True
                
        return current_indices, current_mask

    current_indices, current_mask = initial_greedy()
    best_indices = list(current_indices)
    best_size = len(current_indices)
    
    # SA Parameters
    T = 100.0
    T_min = 0.1
    cooling_rate = 0.995
    batch_size = 2000
    
    # Precompute all indices for sampling
    all_indices = np.arange(TOTAL)
    
    while time.time() < deadline - 10:
        if T < T_min:
            # Restart with lower temperature or break
            # Let's try to keep going with minimum temperature (greedy)
            T = T_min
            
        # Sample candidates
        # We want to sample from points NOT in current set.
        # Sampling without replacement from a large set is O(N).
        # We can just sample random indices and filter out current ones.
        # If density is low, this is efficient.
        # If density is high, we might need more samples.
        
        # To be robust, let's sample a large batch and filter.
        # If current_size is S, probability of hitting an empty spot is (TOTAL-S)/TOTAL.
        # If S is small, this is fine.
        
        raw_candidates = np.random.choice(TOTAL, size=batch_size * 2, replace=False)
        
        # Filter out current points
        # current_mask is boolean array of size TOTAL
        # valid_candidates = raw_candidates[~current_mask[raw_candidates]]
        
        # This creates a copy, which is fine.
        valid_candidates = raw_candidates[~current_mask[raw_candidates]]
        
        if len(valid_candidates) == 0:
            # No empty spots? Unlikely.
            break
            
        # Limit to batch_size
        if len(valid_candidates) > batch_size:
            valid_candidates = valid_candidates[:batch_size]
            
        # Check validity of all these candidates against current set
        valid_mask = batch_check_validity(valid_candidates, current_indices)
        
        # Get indices of valid candidates
        valid_cands = valid_candidates[valid_mask]
        
        if len(valid_cands) == 0:
            # No moves possible. Cool down.
            T *= cooling_rate
            continue
            
        # Choose a random valid candidate to add
        # This is a "Hill Climbing" step if we only add.
        # For SA, we should also allow removals to escape local optima.
        # However, the objective is to MAXIMIZE size.
        # Removing a point decreases size (energy increase in minimization, or penalty).
        # Since we want to maximize points, "Energy" can be defined as -size + penalty.
        # But our state is always valid (penalty=0).
        # So we are maximizing size.
        # Standard SA for maximization: accept if new_size > old_size, or with prob exp((new-old)/T).
        # Since we only add, new_size > old_size always. So we always accept.
        # This is just Greedy Hill Climbing.
        
        # To make it SA, we need to allow swaps or removals.
        # Let's implement a simple swap mechanism:
        # 1. Try to add a point.
        # 2. If no point can be added, try to remove a random point and see if we can add a different one.
        
        # For now, let's just add. If we get stuck, we will implement swaps later.
        # Actually, the current program uses local search swaps.
        # Let's integrate swaps into the SA loop.
        
        # Strategy:
        # 1. Try to add a valid candidate.
        # 2. If successful, accept (size increases).
        # 3. If no valid candidate found, try a swap:
        #    - Pick a random point p from current set to remove.
        #    - Check if any candidate from valid_candidates (or new sample) can be added to (current - {p}).
        
        added = False
        if len(valid_cands) > 0:
            # Pick one randomly
            chosen_idx = valid_cands[random.randrange(len(valid_cands))]
            current_indices.append(chosen_idx)
            current_mask[chosen_idx] = True
            added = True
            if len(current_indices) > best_size:
                best_size = len(current_indices)
                best_indices = list(current_indices)
        else:
            # Try Swap
            # Remove a random point
            if len(current_indices) > 0:
                remove_pos = random.randrange(len(current_indices))
                remove_idx = current_indices[remove_pos]
                
                # Temporarily remove
                temp_indices = [x for x in current_indices if x != remove_idx]
                
                # Sample new candidates
                # We can reuse valid_candidates if they are still "valid" against temp_indices?
                # No, removing a point might make them valid.
                # So we need to re-check.
                
                # Sample a fresh batch of candidates
                raw_swap_cands = np.random.choice(TOTAL, size=batch_size, replace=False)
                valid_swap_cands = raw_swap_cands[~current_mask[raw_swap_cands]]
                # Also exclude remove_idx? It's in current_mask, so it's excluded.
                
                if len(valid_swap_cands) > 0:
                    valid_mask_swap = batch_check_validity(valid_swap_cands, temp_indices)
                    valid_swaps = valid_swap_cands[valid_mask_swap]
                    
                    if len(valid_swaps) > 0:
                        # Accept swap with some probability?
                        # Since size stays same, we might accept to explore.
                        # But we prefer larger sets.
                        # Let's accept if we find one.
                        chosen_swap = valid_swaps[random.randrange(len(valid_swaps))]
                        
                        # Perform swap
                        current_indices[remove_pos] = chosen_swap
                        current_mask[remove_idx] = False
                        current_mask[chosen_swap] = True
                        
                        # Note: Size is same, so best_size doesn't update.
                        # But we might have escaped a local optimum.
                        
        T *= cooling_rate

    # Final greedy pass
    if time.time() < deadline - 5:
        # Try to add any remaining points
        remaining_pts = [j for j in range(TOTAL) if not current_mask[j]]
        random.shuffle(remaining_pts)
        # Process in batches
        for i in range(0, len(remaining_pts), 1000):
            if time.time() > deadline - 5: break
            batch = np.array(remaining_pts[i:i+1000])
            valid_mask = batch_check_validity(batch, current_indices)
            valids = batch[valid_mask]
            for v in valids:
                current_indices.append(v)
                current_mask[v] = True

    # Return best found
    result_indices = best_indices if len(best_indices) > len(current_indices) else current_indices
    result = [(int(xs[idx]), int(ys[idx])) for idx in result_indices]
    return result

if __name__ == "__main__":
    pts = construct()
    print(f"Found {len(pts)} points")
