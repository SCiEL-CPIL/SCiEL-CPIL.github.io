import time

def construct():
    n = 5040
    target = 4979
    
    # Generate all divisors of 5040
    # 5040 = 2^4 * 3^2 * 5 * 7
    # Number of divisors = (4+1)*(2+1)*(1+1)*(1+1) = 5*3*2*2 = 60
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    
    divisors.sort(reverse=True)
    m = len(divisors)
    
    # Precompute prefix sums for pruning
    # prefix_sums[i] = sum of divisors[i:]
    prefix_sums = [0] * (m + 1)
    for i in range(m - 1, -1, -1):
        prefix_sums[i] = prefix_sums[i + 1] + divisors[i]
    
    start_time = time.time()
    time_limit = 55.0  # Leave margin for 60s limit
    
    # Helper to check if a combination is valid and sum matches
    # We use iterative deepening DFS with strong pruning
    
    def find_combination(k, target_sum):
        # If k=1, check if target_sum is a divisor
        if k == 1:
            if target_sum in div_set:
                return [target_sum]
            return None
        
        # Recursive helper
        def dfs(start_idx, remaining_k, remaining_sum, current_comb):
            # Time check
            if time.time() - start_time > time_limit:
                return None
            
            if remaining_k == 0:
                if remaining_sum == 0:
                    return current_comb
                return None
            
            # Pruning 1: Not enough divisors left
            if m - start_idx < remaining_k:
                return None
            
            # Pruning 2: Even taking the largest remaining_k divisors from start_idx, 
            # we can't reach remaining_sum
            # The max sum we can get is sum of the next remaining_k largest divisors
            # But they must be distinct and >= divisors[start_idx + remaining_k - 1]
            # Actually, the max sum is sum of divisors[start_idx : start_idx + remaining_k]
            max_possible_sum = prefix_sums[start_idx] - prefix_sums[start_idx + remaining_k]
            if max_possible_sum < remaining_sum:
                return None
            
            # Pruning 3: The smallest possible sum is sum of the last remaining_k divisors
            # But we are picking from start_idx onwards, so the smallest possible sum
            # is sum of divisors[m - remaining_k : m]
            # However, we must pick from start_idx onwards. 
            # The absolute minimum sum for remaining_k distinct divisors from start_idx onwards
            # is sum of the smallest remaining_k divisors in divisors[start_idx:]
            # Which is divisors[m - remaining_k : m] if start_idx <= m - remaining_k
            min_possible_sum = prefix_sums[m - remaining_k] - prefix_sums[m]
            # Wait, this assumes we can pick any. But we are constrained to start_idx.
            # If start_idx > m - remaining_k, we can't pick that many. Already checked above.
            # The smallest sum is indeed the sum of the smallest remaining_k available divisors.
            # Since divisors is sorted desc, the smallest are at the end.
            # But we must pick from start_idx onwards.
            # So the smallest possible sum is sum(divisors[m-remaining_k:m]) only if start_idx <= m-remaining_k.
            # Actually, the smallest possible sum from start_idx onwards is sum of the last remaining_k elements
            # IF start_idx <= m - remaining_k.
            # Let's just use: if remaining_sum < min_possible_sum: return None
            # But min_possible_sum depends on start_idx.
            # The smallest remaining_k divisors in divisors[start_idx:] are divisors[m-remaining_k:m]
            # provided start_idx <= m - remaining_k.
            # Since we already checked m - start_idx >= remaining_k, this holds.
            if remaining_sum < min_possible_sum:
                return None
            
            # Iterate through possible divisors
            # We need to pick remaining_k divisors from indices start_idx to m-1
            # The largest index we can pick for the first element is m - remaining_k
            for i in range(start_idx, m - remaining_k + 1):
                d = divisors[i]
                
                # Pruning: if d > remaining_sum, we can skip? 
                # Since divisors are sorted desc, if d > remaining_sum, 
                # any subsequent d' is smaller, so we should continue.
                # Wait, if d > remaining_sum, we can't pick d. 
                # But we might pick a smaller one. So we continue.
                if d > remaining_sum:
                    continue
                
                # Pruning: even if we pick d, can we reach the remaining sum with remaining_k-1 more?
                new_remaining_sum = remaining_sum - d
                new_remaining_k = remaining_k - 1
                
                if new_remaining_k == 0:
                    if new_remaining_sum == 0:
                        return current_comb + [d]
                    else:
                        continue
                
                # Check if new_remaining_sum is achievable with new_remaining_k divisors from i+1 onwards
                # Max sum from i+1 onwards for new_remaining_k divisors:
                if m - (i + 1) < new_remaining_k:
                    break  # Not enough divisors left, and since we're going to smaller i? No, i is increasing.
                           # Wait, i is increasing, so start_idx for next is i+1.
                           # If m - (i+1) < new_remaining_k, then for any j > i, m - (j+1) will be even smaller.
                           # So we can break.
                
                max_possible_next = prefix_sums[i + 1] - prefix_sums[i + 1 + new_remaining_k]
                if max_possible_next < new_remaining_sum:
                    continue  # Try smaller d? No, d is decreasing. 
                               # If max possible sum with next divisors is too small, 
                               # it means d was too big? 
                               # Wait, if we pick a smaller d, new_remaining_sum becomes larger.
                               # And the available divisors for the rest are even smaller (since start index increases).
                               # So if max_possible_next < new_remaining_sum, 
                               # then for any j > i, new_remaining_sum' = remaining_sum - divisors[j] > new_remaining_sum
                               # and max_possible_next' < max_possible_next.
                               # So the gap widens. We can break?
                               # Let's check: 
                               # We want sum of new_remaining_k divisors from i+1 onwards to be new_remaining_sum.
                               # If the max possible sum is less than new_remaining_sum, it's impossible.
                               # If we pick a smaller d (larger i), new_remaining_sum increases.
                               # The pool of divisors for the rest shrinks (starts later), so max possible sum decreases.
                               # So the deficit increases. Thus, we can break.
                    # Actually, let's be careful. 
                    # Condition: prefix_sums[i+1] - prefix_sums[i+1+k'] < remaining_sum - d_i
                    # For j > i: d_j < d_i, so remaining_sum - d_j > remaining_sum - d_i.
                    # Max sum from j+1 is prefix_sums[j+1] - prefix_sums[j+1+k'] which is < prefix_sums[i+1] - prefix_sums[i+1+k'].
                    # So LHS decreases, RHS increases. Inequality becomes stricter.
                    # So yes, break.
                    break
                
                # Min sum check for next
                min_possible_next = prefix_sums[m - new_remaining_k] - prefix_sums[m]
                # This min sum is constant regardless of start_idx (as long as start_idx <= m - new_remaining_k)
                # We already checked m - (i+1) >= new_remaining_k.
                if new_remaining_sum < min_possible_next:
                    continue  # d is too small? 
                               # If d is too small, new_remaining_sum is too large.
                               # If we pick a larger d (smaller i), new_remaining_sum decreases.
                               # But we are iterating i increasing (d decreasing).
                               # So we already passed the larger d's.
                               # So if new_remaining_sum is too large, we should have picked a larger d.
                               # Since we are moving to smaller d, new_remaining_sum will only get larger.
                               # So we can break?
                               # Wait. 
                               # If new_remaining_sum > max_possible_next, we broke.
                               # If new_remaining_sum < min_possible_next, it means we need a larger sum for the rest.
                               # This happens if d is too small.
                               # Since we are iterating to smaller d, the required sum for the rest increases.
                               # So it will never satisfy the min condition.
                               # So we can break.
                    break
                
                new_comb = current_comb + [d]
                result = dfs(i + 1, new_remaining_k, new_remaining_sum, new_comb)
                if result:
                    return result
            
            return None
        
        return dfs(0, k, target_sum, [])
    
    div_set = set(divisors)
    
    # Try k=1
    res = find_combination(1, target)
    if res:
        return res
    
    # Try k=2
    res = find_combination(2, target)
    if res:
        return res
    
    # Try k=3
    res = find_combination(3, target)
    if res:
        return res
    
    # Try k=4
    res = find_combination(4, target)
    if res:
        return res
    
    # Try k=5
    res = find_combination(5, target)
    if res:
        return res
    
    # Try k=6
    res = find_combination(6, target)
    if res:
        return res
    
    # Fallback: Known solution or return empty (should not happen)
    # The problem guarantees a solution.
    # Let's return a valid set if we found one in previous steps, else empty.
    # But we should return a list.
    return []
