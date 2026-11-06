import math

def construct():
    n = 5040
    target = 4979
    
    # Generate all divisors of n
    divisors = []
    for i in range(1, int(math.isqrt(n)) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    
    # Sort divisors in descending order to prioritize larger values in search
    divisors.sort(reverse=True)
    
    # Precompute suffix sums for pruning
    # suffix_sum[i] is the sum of divisors[i:]
    suffix_sum = [0] * (len(divisors) + 1)
    for i in range(len(divisors) - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + divisors[i]
        
    # IDDFS: Try solutions of size 1, 2, 3, ...
    # Max possible size is len(divisors), but likely much smaller.
    # We limit to len(divisors) to be safe.
    
    def dfs(start_idx, remaining_sum, depth_limit, path):
        if remaining_sum == 0:
            return len(path) == depth_limit # We want exactly depth_limit items? 
            # Actually, IDDFS finds the first solution at depth k. 
            # If we find a solution at depth k, it is minimal.
            # So if remaining_sum == 0, we check if we used exactly depth_limit items?
            # No, if we are in the k-th iteration, we are looking for a solution of size k.
            # If we hit 0 sum before using k items, that's a solution of size < k, 
            # which would have been found in a previous iteration.
            # So we can just return True if remaining_sum == 0 and len(path) == depth_limit.
            # Or simply, if remaining_sum == 0, we have found a valid sum. 
            # Since we iterate k=1,2,..., the first one found is minimal.
            return True
            
        if depth_limit == 0:
            return False
            
        if start_idx >= len(divisors):
            return False
            
        # Pruning: If the sum of all remaining divisors is less than remaining_sum, impossible
        if suffix_sum[start_idx] < remaining_sum:
            return False
            
        # Pruning: If the smallest divisor is greater than remaining_sum (and remaining_sum != 0), impossible
        # Since sorted desc, the smallest is at the end.
        # But we can just rely on the loop.
        
        for i in range(start_idx, len(divisors)):
            d = divisors[i]
            if d > remaining_sum:
                continue
            # Additional pruning: 
            # If we pick d, we have depth_limit-1 items left to sum to remaining_sum - d.
            # The max sum we can get with depth_limit-1 items from i+1 onwards is sum of largest depth_limit-1 items from i+1.
            # This is hard to precompute for arbitrary depth_limit.
            # Simple suffix_sum check is sufficient for now.
            
            path.append(d)
            if dfs(i + 1, remaining_sum - d, depth_limit - 1, path):
                return True
            path.pop()
            
        return False

    # Try increasing depth limits
    for k in range(1, len(divisors) + 1):
        path = []
        if dfs(0, target, k, path):
            return path
            
    # Fallback if no solution found (should not happen)
    return []
