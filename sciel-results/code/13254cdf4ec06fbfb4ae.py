import time
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit
    
    # Generate smooth numbers (2^a * 3^b * 5^c) <= 128
    smooth = set()
    for a in range(8):
        for b in range(5):
            for c in range(4):
                val = (2 ** a) * (3 ** b) * (5 ** c)
                if val <= 128:
                    smooth.add(val)
    
    smooth_list = sorted(smooth)
    n_smooth = len(smooth_list)
    smooth_set = set(smooth_list)
    val_to_idx = {v: i for i, v in enumerate(smooth_list)}
    
    # Precompute divisibility: div[i][j] = True if smooth_list[i] divides smooth_list[j]
    # Antichain constraint: no two chosen terms where one divides the other
    # Build compatibility: compat[i] = bitmask of indices j where i and j can coexist
    # i and j can coexist if smooth_list[i] does NOT divide smooth_list[j] AND vice versa
    compat = [0] * n_smooth
    for i in range(n_smooth):
        for j in range(n_smooth):
            if i == j:
                continue
            vi = smooth_list[i]
            vj = smooth_list[j]
            # They can coexist if neither divides the other
            if vi % vj != 0 and vj % vi != 0:
                compat[i] |= (1 << j)
    
    # For a set represented by bitmask, check if it's an antichain
    # We can precompute for small masks, but let's use a different approach
    # Actually, let's just check incrementally during DFS
    
    # Precompute for each smooth number, which other smooth numbers it's compatible with
    # compat_mask[i] = bitmask of j (j != i) such that i and j form an antichain pair
    
    # For DFS, we maintain a bitmask of chosen indices
    # When adding index i, we need: (chosen_mask & compat[i]) == chosen_mask? No.
    # Actually: for all j in chosen_mask, i must be compatible with j
    # i.e., for all j in chosen_mask: (compat[i] >> j) & 1 == 1
    # Equivalently: chosen_mask & ~compat[i] == 0
    
    # Let's precompute: for each i, incompat[i] = bitmask of j that are incompatible with i
    # Then when adding i, we need: chosen_mask & incompat[i] == 0
    incompat = [0] * n_smooth
    for i in range(n_smooth):
        for j in range(n_smooth):
            if i == j:
                continue
            vi = smooth_list[i]
            vj = smooth_list[j]
            if vi % vj == 0 or vj % vi == 0:
                incompat[i] |= (1 << j)
    
    # Precompute suffix sums for pruning
    # suffix_sum[i] = sum of smooth_list[i:]
    suffix_sum = [0] * (n_smooth + 1)
    for i in range(n_smooth - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + smooth_list[i]
    
    reps = [None] * 129
    
    def dfs(rem, start_idx, current_mask, k_remaining):
        if k_remaining == 0:
            return current_mask if rem == 0 else None
        if rem <= 0:
            return None
        
        # Pruning: if remaining sum of all available numbers < rem, fail
        if suffix_sum[start_idx] < rem:
            return None
        
        # Pruning: if we need k_remaining numbers but there aren't enough available
        if n_smooth - start_idx < k_remaining:
            return None
        
        # Pruning: the minimum possible sum of k_remaining numbers from start_idx
        # is the sum of the k_remaining smallest available numbers
        if k_remaining <= n_smooth - start_idx:
            min_possible_sum = sum(smooth_list[start_idx:start_idx + k_remaining])
            if min_possible_sum > rem:
                return None
        
        # Pruning: the maximum possible sum of k_remaining numbers from start_idx
        # that are <= rem... this is complex. Use a simpler bound.
        # If even the largest k_remaining numbers can't sum to rem, fail
        # Actually, we need sum == rem, so if max possible < rem, fail
        # Max possible sum of k_remaining numbers from start_idx is sum of largest k_remaining
        # But they must all be <= rem (since they're positive)
        # Let's compute sum of largest k_remaining numbers from start_idx that are <= rem
        # This is expensive, so use a weaker bound: if sum of largest k_remaining overall < rem, fail
        if k_remaining <= n_smooth - start_idx:
            max_possible_sum = sum(smooth_list[n_smooth - k_remaining:n_smooth])
            if max_possible_sum < rem:
                return None
        
        for idx in range(start_idx, n_smooth):
            val = smooth_list[idx]
            if val > rem:
                break
            
            # Check compatibility with all currently chosen
            if current_mask & incompat[idx]:
                continue
            
            new_mask = current_mask | (1 << idx)
            res = dfs(rem - val, idx + 1, new_mask, k_remaining - 1)
            if res is not None:
                return res
        return None
    
    def mask_to_list(mask):
        result = []
        for i in range(n_smooth):
            if mask & (1 << i):
                result.append(smooth_list[i])
        return result
    
    for t in range(1, 129):
        if time.time() > deadline:
            break
        
        found = False
        
        # Try k=1
        if t in smooth_set:
            reps[t] = tuple([t])
            found = True
        else:
            # Try k=2
            for i in range(n_smooth):
                a = smooth_list[i]
                if a >= t:
                    break
                b = t - a
                if b < a:
                    continue
                if b in smooth_set:
                    j_idx = val_to_idx.get(b)
                    if j_idx is not None and j_idx != i and (incompat[i] & (1 << j_idx)) == 0:
                        reps[t] = tuple(sorted([a, b]))
                        found = True
                        break
                if time.time() > deadline:
                    break
            if found:
                continue
            
            # Try k=3
            for i in range(n_smooth):
                a = smooth_list[i]
                if a * 3 >= t:  # a + b + c >= 3a, so if 3a >= t, a is too big
                    break
                for j in range(i + 1, n_smooth):
                    b = smooth_list[j]
                    if a + b >= t:
                        break
                    c = t - a - b
                    if c <= b:
                        continue
                    if c in smooth_set:
                        k_idx = val_to_idx.get(c)
                        if k_idx is not None and k_idx != i and k_idx != j:
                            # Check all pairs compatible
                            if (incompat[i] & (1 << j)) == 0 and \
                               (incompat[i] & (1 << k_idx)) == 0 and \
                               (incompat[j] & (1 << k_idx)) == 0:
                                reps[t] = tuple(sorted([a, b, c]))
                                found = True
                                break
                    if time.time() > deadline:
                        break
                if found or time.time() > deadline:
                    break
            if found:
                continue
            
            # Try k>=4 using DFS
            for k in range(4, 31):
                if time.time() > deadline:
                    break
                res_mask = dfs(t, 0, 0, k)
                if res_mask is not None:
                    reps[t] = tuple(mask_to_list(res_mask))
                    found = True
                    break
            
            if not found:
                # Should not happen for t <= 128
                pass
    
    # Build result
    result = []
    for t in range(1, 129):
        if reps[t] is None:
            # Fallback: this should not happen
            # Try a greedy approach as last resort
            result.append([])
        else:
            result.append(list(reps[t]))
    
    return result
