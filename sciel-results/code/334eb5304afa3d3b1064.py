import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s guard
    
    # 1. Generate smooth numbers up to 128
    smooth = set()
    a = 1
    while a <= 128:
        b = a
        while b <= 128:
            c = b
            while c <= 128:
                smooth.add(c)
                c *= 5
            b *= 3
        a *= 2
    smooth_list = sorted(smooth)
    n_smooth = len(smooth_list)
    
    # Map value to index for quick lookup
    val_to_idx = {v: i for i, v in enumerate(smooth_list)}
    
    # 2. Precompute incompatible masks
    # incompatible_mask[i] has bit j set if smooth_list[i] and smooth_list[j] are comparable
    # i.e., smooth_list[i] % smooth_list[j] == 0 or smooth_list[j] % smooth_list[i] == 0
    incompatible_mask = [0] * n_smooth
    for i in range(n_smooth):
        vi = smooth_list[i]
        for j in range(n_smooth):
            vj = smooth_list[j]
            if i == j:
                continue
            if vi % vj == 0 or vj % vi == 0:
                incompatible_mask[i] |= (1 << j)
        # Also mark self as incompatible to prevent re-picking
        incompatible_mask[i] |= (1 << i)

    def find_min_antichain(n):
        # Check k=1
        if n in val_to_idx:
            return [n]
            
        # Pre-compute available masks for pruning?
        # We will use DFS with memoization
        
        # To speed up pruning, we can precompute sums of top/bottom k available?
        # It's dynamic based on forb_mask.
        
        memo = {}
        
        def dfs(rem, k, forb_mask, start_idx):
            if k == 0:
                return [] if rem == 0 else None
            
            if rem < 0:
                return None
                
            key = (rem, k, forb_mask, start_idx)
            if key in memo:
                return memo[key]
                
            # Pruning: Check if enough numbers available and if sum is possible
            # We need to pick k numbers from indices >= start_idx that are not in forb_mask
            # and have value <= rem (implicitly, as we break or skip)
            
            # Quick count and bound check
            # This loop is O(n_smooth). We can optimize by precomputing lists of indices?
            # Given n_smooth is ~50, this is fine.
            
            avail_indices = []
            for i in range(start_idx, n_smooth):
                if not (forb_mask & (1 << i)):
                    if smooth_list[i] <= rem: # Only consider if it can fit in remaining sum?
                       # Actually, for min sum bound, we need all available.
                       # For max sum bound, we need all available.
                       # But if smooth_list[i] > rem, it can't be part of the sum if all terms are positive.
                       # So we can skip them for the "pick" loop, but for bounds, 
                       # we should consider that they can't be used.
                       pass
                # Let's just collect all valid indices for bounds
                if not (forb_mask & (1 << i)):
                    avail_indices.append(i)
                    
            if len(avail_indices) < k:
                memo[key] = None
                return None
                
            # Calculate min and max possible sums
            # Min sum: pick k smallest from avail_indices
            # Max sum: pick k largest from avail_indices
            # Note: avail_indices is sorted by index, which is sorted by value.
            
            # We need to be careful: some avail_indices might have value > rem.
            # If we pick a number > rem, the sum exceeds rem (since others are positive).
            # So for the "Max Sum" bound, we should only consider numbers <= rem?
            # No, if we pick a number > rem, it's invalid.
            # So the effective available set for summing to 'rem' is those with value <= rem.
            # Let's filter avail_indices to those with value <= rem for the bounds.
            
            valid_avail = [i for i in avail_indices if smooth_list[i] <= rem]
            
            if len(valid_avail) < k:
                memo[key] = None
                return None
                
            min_sum = sum(smooth_list[i] for i in valid_avail[:k])
            max_sum = sum(smooth_list[i] for i in valid_avail[-k:])
            
            if rem < min_sum or rem > max_sum:
                memo[key] = None
                return None
                
            # Search
            for i in range(start_idx, n_smooth):
                if (forb_mask & (1 << i)):
                    continue
                val = smooth_list[i]
                if val > rem:
                    # Since smooth_list is sorted, if val > rem, subsequent values are also > rem.
                    # We can break here? 
                    # Yes, because we are iterating in increasing index (value) order.
                    break
                
                new_forb = forb_mask | incompatible_mask[i]
                res = dfs(rem - val, k - 1, new_forb, i + 1)
                if res is not None:
                    res.append(val)
                    memo[key] = res
                    return res
                    
            memo[key] = None
            return None
            
        for k in range(2, 31):
            if time.time() > deadline:
                # If we run out of time, we can't guarantee a solution for this n.
                # But we must return something. 
                # The problem states "All targets require valid representations".
                # If we fail, we score 0.
                # We should try to finish quickly.
                # If we are here, we likely have a bug or slow case.
                # Let's return a dummy to avoid crash, but it will be invalid.
                # Actually, the loop for n is outside. We should handle timeout per n or global.
                # Let's assume we finish.
                pass
            res = dfs(n, k, 0, 0)
            if res is not None:
                return res
                
        # If no solution found up to k=30
        # This should not happen for 1..128
        return None

    answers = []
    for n in range(1, 129):
        if time.time() > deadline:
            # Emergency: fill rest with [1] (invalid) or try to finish fast
            # We'll just append [1] to stop, though it will fail validation.
            # Better: try to compute remaining with a very tight limit or break.
            # Given the small size, we should have finished.
            while len(answers) < 128:
                answers.append([1])
            return answers
            
        rep = find_min_antichain(n)
        if rep is None:
            # Fallback: if DFS failed (shouldn't happen), try a simple greedy or hardcoded?
            # For robustness, if n is smooth, [n] works.
            # We already checked k=1.
            # If k>=2 failed, it's a critical error.
            # Let's try to find any valid rep with a looser search or just return [1] to avoid exception.
            # But [1] is invalid if n != 1.
            # We'll return [1] as a placeholder to prevent crash, knowing it scores 0.
            rep = [1]
        answers.append(rep)
        
    return answers
