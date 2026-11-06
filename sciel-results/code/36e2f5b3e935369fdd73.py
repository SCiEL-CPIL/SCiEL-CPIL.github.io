from fractions import Fraction

def construct():
    # Search for the minimum number of terms
    # We know k=1..5 are impossible for min denom 5.
    # Try k=6, 7, 8...
    
    def find_solution(k):
        # Backtracking search
        # State: current index, current sum, last denominator
        # To avoid deep recursion and slow execution, we limit the denominator range.
        # For small k, denominators won't be astronomically large.
        
        def dfs(idx, current_sum, last_denom):
            if idx == k:
                if current_sum == 1:
                    return True
                return False
            
            # Heuristic: The next denominator must be at least last_denom + 1
            # Also, the maximum possible sum with remaining terms (k - idx)
            # using denominators starting from last_denom + 1 is (k-idx)/(last_denom+1)
            # If current_sum + max_possible < 1, prune.
            # Also, if current_sum > 1, prune.
            
            # Determine range for next denominator
            # 1/d <= 1 - current_sum
            # d >= 1 / (1 - current_sum)
            
            remaining_terms = k - idx
            if remaining_terms == 0:
                return current_sum == 1
            
            # Lower bound for d
            # We need sum of 1/d_i = 1 - current_sum
            # Max sum with remaining_terms terms starting from d_min is remaining_terms / d_min
            # So we need remaining_terms / d_min >= 1 - current_sum
            # d_min <= remaining_terms / (1 - current_sum)
            
            # Also d > last_denom
            start_d = max(last_denom + 1, 5)
            
            # If even the smallest possible denominators cannot sum to the remainder, fail
            # Max sum = sum_{i=0}^{remaining_terms-1} 1/(start_d + i)
            # Approximate with integral or just check first term
            if 1 / start_d > 1 - current_sum:
                # Next term alone is too big, so all subsequent are too big
                return False
            
            # Iterate d
            # Upper bound for d:
            # 1/d <= 1 - current_sum
            # d >= 1 / (1 - current_sum)
            # Also, to leave enough room for remaining terms:
            # 1/d + (remaining_terms - 1) / (d+1) >= 1 - current_sum
            # This is complex. Let's just iterate up to a limit.
            
            # Limit d to prevent infinite loop in search
            # If d is very large, 1/d is tiny.
            # For k=6, d should be < 1000 usually.
            
            # Optimization: 
            # The remainder R = 1 - current_sum
            # We need to represent R as sum of (remaining_terms) distinct unit fractions > last_denom.
            
            # Try d from start_d upwards
            # Stop if 1/d < R / remaining_terms (too small to contribute enough on average? No, just too small to help reach R if others are also small? Actually if 1/d is very small, we can still reach R with larger denominators? No, larger denominators are even smaller. So if 1/d < R / remaining_terms, then even if we pick the smallest possible for all, we might not reach R? 
            # Wait. If we pick d, the remaining sum is R - 1/d.
            # The max sum of the remaining (remaining_terms - 1) terms is approx (remaining_terms - 1) / (d+1).
            # So we need R - 1/d <= (remaining_terms - 1) / (d+1).
            
            max_d = 10000 # Safe upper bound for small k
            for d in range(start_d, max_d + 1):
                if 1/d > 1 - current_sum:
                    continue
                
                new_sum = current_sum + 1/d
                if new_sum > 1:
                    continue
                
                # Pruning: Check if remaining sum can be achieved
                rem_sum = 1 - new_sum
                rem_terms = remaining_terms - 1
                if rem_terms == 0:
                    if rem_sum == 0:
                        return True
                    continue
                
                # Max possible sum with rem_terms starting from d+1
                # Approx: rem_terms / (d+1)
                if rem_terms / (d+1) < rem_sum:
                    # Even if we pick the smallest possible denominators, we can't reach the remainder
                    # So larger d will only make 1/d smaller, leaving larger rem_sum, and smaller max capacity
                    # Thus, break
                    break
                
                if dfs(idx + 1, new_sum, d):
                    return True
            
            return False
        
        # Start search
        # We need to store the path
        path = []
        
        def dfs_path(idx, current_sum, last_denom):
            if idx == k:
                if current_sum == 1:
                    return True
                return False
            
            start_d = max(last_denom + 1, 5)
            
            rem_terms = k - idx
            if rem_terms == 0:
                return current_sum == 1
            
            max_d = 10000
            for d in range(start_d, max_d + 1):
                if 1/d > 1 - current_sum:
                    continue
                
                new_sum = current_sum + 1/d
                if new_sum > 1:
                    continue
                
                rem_sum = 1 - new_sum
                rem_terms_next = rem_terms - 1
                
                if rem_terms_next == 0:
                    if rem_sum == 0:
                        path.append(d)
                        return True
                    continue
                
                if rem_terms_next / (d+1) < rem_sum:
                    break
                
                path.append(d)
                if dfs_path(idx + 1, new_sum, d):
                    return True
                path.pop()
            
            return False
        
        if dfs_path(0, 0, 4):
            return path
        
        return None

    for k in range(6, 15):
        sol = find_solution(k)
        if sol:
            return sol
    
    # Fallback: Greedy
    rem = 1
    denoms = []
    last = 4
    while rem > 0:
        # Find smallest d > last such that 1/d <= rem
        d = last + 1
        while 1/d > rem:
            d += 1
        denoms.append(d)
        rem -= 1/d
        last = d
        if len(denoms) > 256:
            break
    return denoms
