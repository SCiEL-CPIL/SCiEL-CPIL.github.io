import time

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s limit
    
    n = 5040
    target = 4979
    
    # 1. Generate all divisors of n
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    
    # 2. Sort divisors in descending order
    divisors.sort(reverse=True)
    num_divs = len(divisors)
    
    # Precompute prefix sums for O(1) range sum queries
    # prefix[i] = sum(divisors[0:i])
    prefix = [0] * (num_divs + 1)
    for i in range(num_divs):
        prefix[i+1] = prefix[i] + divisors[i]
        
    def range_sum(a, b):
        # Sum of divisors[a:b]
        if a < 0: a = 0
        if b > num_divs: b = num_divs
        if a >= b: return 0
        return prefix[b] - prefix[a]

    # Precompute min_sum_for_k[k] = sum of k smallest divisors
    # k smallest are divisors[num_divs-k : num_divs]
    min_sum_for_k = [0] * (num_divs + 1)
    # We can use range_sum(num_divs-k, num_divs)
    for k in range(1, num_divs + 1):
        min_sum_for_k[k] = range_sum(num_divs - k, num_divs)

    # DFS function
    def dfs(start_idx, remaining_sum, k, path):
        if time.time() > deadline:
            return False, None
            
        if k == 0:
            if remaining_sum == 0:
                return True, path[:]
            return False, None
            
        if remaining_sum < 0:
            return False, None
            
        # Not enough divisors left
        if start_idx + k > num_divs:
            return False, None
            
        # Pruning 1: Min sum
        if min_sum_for_k[k] > remaining_sum:
            return False, None
            
        # Pruning 2: Max sum
        # Max sum of k elements from divisors[start_idx:] is sum(divisors[start_idx : start_idx+k])
        max_sum = range_sum(start_idx, start_idx + k)
        if max_sum < remaining_sum:
            return False, None
            
        # Try each divisor from start_idx
        for i in range(start_idx, num_divs):
            d = divisors[i]
            if d > remaining_sum:
                continue
                
            # If we pick d, we need k-1 more from divisors[i+1:]
            
            # Check min sum for next step
            if k > 1:
                if min_sum_for_k[k-1] > remaining_sum - d:
                    continue
                # Check max sum for next step
                # Max sum of k-1 from i+1 is range_sum(i+1, i+1+k-1)
                next_max_sum = range_sum(i + 1, i + 1 + k - 1)
                if next_max_sum < remaining_sum - d:
                    continue
            
            path.append(d)
            found, result = dfs(i + 1, remaining_sum - d, k - 1, path)
            if found:
                return True, result
            path.pop()
            
        return False, None

    # Iterate k from 1 to num_divs
    for k in range(1, num_divs + 1):
        if time.time() > deadline:
            break
            
        # Quick check: if min_sum_for_k[k] > target, no solution for this k
        if min_sum_for_k[k] > target:
            continue
            
        # Quick check: if max_sum_for_k (sum of largest k divisors) < target, no solution
        max_possible = range_sum(0, k)
        if max_possible < target:
            continue
            
        found, result = dfs(0, target, k, [])
        if found:
            return result
            
    return []
