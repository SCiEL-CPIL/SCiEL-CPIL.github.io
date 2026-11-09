import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    # Generate all 2,3,5-smooth numbers <= 128
    smooth = []
    for x in range(1, 129):
        y = x
        for p in (2, 3, 5):
            while y % p == 0:
                y //= p
        if y == 1:
            smooth.append(x)
    
    n = len(smooth)
    
    # Precompute conflict bitsets: conflict[i] has bit j set if smooth[i] and smooth[j] conflict
    conflict = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            a, b = smooth[i], smooth[j]
            if a % b == 0 or b % a == 0:
                conflict[i] |= (1 << j)
                conflict[j] |= (1 << i)
    
    # Precompute prefix sums for pruning
    prefix_sum = [0] * (n + 1)
    for i in range(n):
        prefix_sum[i + 1] = prefix_sum[i] + smooth[i]
    
    def find_min_antichain(target):
        """Find minimum-cardinality antichain summing to target."""
        for k in range(1, 31):
            if time.time() > deadline:
                return None
            
            # DFS to find subset of size k
            def dfs(start_idx, remaining_sum, remaining_k, current_mask, path):
                if time.time() > deadline:
                    return None
                
                if remaining_k == 0:
                    return path[:] if remaining_sum == 0 else None
                
                if start_idx >= n:
                    return None
                
                # Not enough elements remaining
                if n - start_idx < remaining_k:
                    return None
                
                # Pruning: minimum possible sum (smallest remaining_k from start_idx)
                min_sum = prefix_sum[start_idx + remaining_k] - prefix_sum[start_idx]
                if min_sum > remaining_sum:
                    return None
                
                # Pruning: maximum possible sum (largest remaining_k from start_idx to end)
                # We can pick any remaining_k from [start_idx, n)
                # The max is the last remaining_k elements in the available range
                avail_count = n - start_idx
                if avail_count >= remaining_k:
                    max_start = n - remaining_k
                    max_sum = prefix_sum[n] - prefix_sum[max_start]
                else:
                    return None  # Already handled above
                
                if max_sum < remaining_sum:
                    return None
                
                # Iterate candidates
                for i in range(start_idx, n):
                    val = smooth[i]
                    
                    # If val > remaining_sum, no point continuing (sorted)
                    if val > remaining_sum:
                        break
                    
                    # Check antichain constraint
                    if conflict[i] & current_mask:
                        continue
                    
                    # Additional pruning: can we still achieve the sum after picking i?
                    new_remaining = remaining_sum - val
                    new_k = remaining_k - 1
                    new_start = i + 1
                    
                    if new_k > 0:
                        if n - new_start < new_k:
                            continue
                        min_possible = prefix_sum[new_start + new_k] - prefix_sum[new_start]
                        if min_possible > new_remaining:
                            continue
                        max_possible = prefix_sum[n] - prefix_sum[n - new_k]
                        if max_possible < new_remaining:
                            continue
                    
                    # Pick i
                    new_mask = current_mask | (1 << i)
                    path.append(i)
                    
                    res = dfs(new_start, new_remaining, new_k, new_mask, path)
                    if res is not None:
                        return res
                    
                    path.pop()
                
                return None
            
            result = dfs(0, target, k, 0, [])
            if result is not None:
                return [smooth[i] for i in result]
        
        return None
    
    solutions = [None] * 129
    
    for t in range(1, 129):
        if time.time() > deadline:
            # Fallback: use [t] if t is smooth
            if t in smooth:
                solutions[t] = [t]
            else:
                solutions[t] = None
            continue
        
        sol = find_min_antichain(t)
        if sol is not None:
            solutions[t] = sol
        else:
            # Fallback
            if t in smooth:
                solutions[t] = [t]
            else:
                solutions[t] = None
    
    return solutions[1:129]
