import time

def construct():
    start_time = time.time()
    DEADLINE = 85  # safety margin before 90s kill
    
    n = 5040
    target = 4979
    
    # Get all divisors of 5040
    divisors = []
    for d in range(1, n + 1):
        if n % d == 0:
            divisors.append(d)
    
    # Sort divisors descending to prioritize larger values (helps minimize count)
    divisors.sort(reverse=True)
    nd = len(divisors)
    
    # Precompute prefix sums: prefix_sum[i] = sum(divisors[0:i])
    prefix_sum = [0] * (nd + 1)
    for i in range(nd):
        prefix_sum[i + 1] = prefix_sum[i] + divisors[i]
    
    # Precompute suffix sums: suffix_sum[i] = sum(divisors[i:nd])
    suffix_sum = [0] * (nd + 1)
    for i in range(nd - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + divisors[i]
    
    # For fast max/min sum of k elements from index i onwards:
    # max_sum(i, k) = sum of divisors[i : i+k] (largest k from position i)
    # min_sum(i, k) = sum of divisors[nd-k : nd] is NOT local. 
    # Local min: sum of the last k elements in divisors[i:] = sum(divisors[nd-k : nd]) if nd-k >= i, 
    # but that's not right either. The smallest k elements from divisors[i:] are divisors[nd-k : nd] 
    # only if nd-k >= i. Actually, since sorted descending, the smallest k elements in divisors[i:] 
    # are divisors[nd-k : nd] if nd-k >= i, otherwise it's the last k of divisors[i:].
    # More precisely: smallest k elements in divisors[i:] are divisors[nd-k : nd] if (nd-k) >= i,
    # which means k <= nd - i. Since we check len(divisors[i:]) >= k, this is always true when valid.
    # So min_sum(i, k) = sum(divisors[nd-k : nd]) when k <= nd - i.
    # But wait, that's the global smallest k, not necessarily from position i onwards.
    # Actually since the array is sorted descending, divisors[nd-k : nd] are the k smallest overall.
    # If nd-k >= i, then these k smallest are all in divisors[i:], so min_sum(i,k) = sum(divisors[nd-k:nd]).
    # If nd-k < i, then we need the k smallest from divisors[i:], which would be divisors[nd-k:nd] 
    # but some of those might be before i. In that case, the k smallest from divisors[i:] are 
    # divisors[nd-k:nd] intersected with indices >= i, which is just divisors[max(i, nd-k) : nd] 
    # but we need exactly k elements. This is getting complex.
    
    # Simpler approach: precompute a 2D table or use the fact that nd is small (60).
    # For each i and k, max_sum(i,k) = prefix_sum[i+k] - prefix_sum[i]
    # min_sum(i,k): the k smallest in divisors[i:] are the last k elements of divisors[i:],
    # i.e., divisors[nd-k : nd] if nd-k >= i, else we need to think again.
    # Since divisors is sorted descending, divisors[i:] has elements from large to small.
    # The k smallest in divisors[i:] are divisors[nd-k : nd] provided that nd-k >= i.
    # If nd-k < i, then the k smallest in divisors[i:] would be divisors[i : i + (nd-i)] which is all of them,
    # but we only have nd-i elements and we need k > nd-i, which we already check for.
    # So when len(divisors[i:]) >= k, i.e., nd - i >= k, i.e., nd-k >= i.
    # So min_sum(i,k) = sum(divisors[nd-k : nd]) = prefix_sum[nd] - prefix_sum[nd-k]
    
    def get_max_sum(i, k):
        """Sum of k largest elements from divisors[i:]"""
        if i + k > nd:
            return 0
        return prefix_sum[i + k] - prefix_sum[i]
    
    def get_min_sum(i, k):
        """Sum of k smallest elements from divisors[i:]"""
        if i + k > nd:
            return 0
        # k smallest in divisors[i:] are the last k elements overall (since sorted desc)
        # and they are all >= i because nd-k >= i (from the check above)
        return prefix_sum[nd] - prefix_sum[nd - k]
    
    best_result = None
    
    def dfs(start_idx, remaining_sum, count_needed, current_path):
        nonlocal best_result
        
        if count_needed == 0:
            if remaining_sum == 0:
                best_result = current_path[:]
            return best_result is not None
        
        if remaining_sum < 0:
            return False
            
        # Pruning: if not enough divisors left
        if start_idx + count_needed > nd:
            return False
            
        # Pruning: if remaining_sum > max possible sum
        max_possible = get_max_sum(start_idx, count_needed)
        if remaining_sum > max_possible:
            return False
            
        # Pruning: if remaining_sum < min possible sum
        min_possible = get_min_sum(start_idx, count_needed)
        if remaining_sum < min_possible:
            return False
            
        # Check time
        if time.time() - start_time > DEADLINE:
            return False

        for i in range(start_idx, nd):
            d = divisors[i]
            if d > remaining_sum:
                continue
            
            new_remaining = remaining_sum - d
            new_count = count_needed - 1
            
            if new_count > 0:
                # Additional pruning for the rest
                max_rest = get_max_sum(i + 1, new_count)
                if new_remaining > max_rest:
                    continue
                min_rest = get_min_sum(i + 1, new_count)
                if new_remaining < min_rest:
                    continue
            
            current_path.append(d)
            result = dfs(i + 1, new_remaining, new_count, current_path)
            if result:
                return True
            current_path.pop()
                
        return False

    # Try increasing k to find the minimum number of divisors
    for k in range(1, nd + 1):
        # Prune k if sum of k largest < target
        if get_max_sum(0, k) < target:
            continue
            
        # Also prune if target < min sum of k divisors (but min sum is the k smallest)
        if target < get_min_sum(0, k):
            # Since k is increasing, and min_sum increases with k, we can break
            # Actually no, min_sum(0,k) is the sum of k smallest divisors, which increases with k
            # But we want target >= min_sum, so if target < min_sum, this k is too large? 
            # No, min_sum is the minimum possible sum with k divisors. If target < min_sum, 
            # it's impossible. But as k increases, min_sum increases, so once target < min_sum,
            # all larger k will also have target < min_sum. So we can break.
            break
            
        best_result = None
        res = dfs(0, target, k, [])
        if res:
            return best_result
            
    # Fallback: greedy (should not be reached for valid instances)
    remainder = target
    chosen = []
    for d in divisors:
        if d <= remainder:
            chosen.append(d)
            remainder -= d
    return chosen
