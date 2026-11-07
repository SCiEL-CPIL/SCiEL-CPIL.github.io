# EVOLVE-BLOCK-START
import math
import time

def construct():
    n = 256
    # Precompute prime factors and masks
    prime_factors = [[] for _ in range(n + 1)]
    for i in range(2, n + 1):
        if not prime_factors[i]:  # i is prime
            for j in range(i, n + 1, i):
                prime_factors[j].append(i)
    
    # Map primes to bit positions
    primes = [i for i in range(2, n + 1) if prime_factors[i]]
    prime_index = {p: i for i, p in enumerate(primes)}
    
    # Create masks for each number
    masks = [0] * (n + 1)
    for i in range(2, n + 1):
        mask = 0
        for p in prime_factors[i]:
            mask |= (1 << prime_index[p])
        masks[i] = mask
    
    # List of numbers to consider, sorted descending
    nums = list(range(n, 1, -1))
    num_masks = [masks[x] for x in nums]
    
    # Suffix sums for loose upper bound
    suffix_sum = [0] * (len(nums) + 1)
    for i in range(len(nums) - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + nums[i]
    
    # Greedy initialization to get a good lower bound
    def get_greedy_solution():
        used_mask = 0
        result = [1]
        result_set = {1}
        for i in range(len(nums)):
            x = nums[i]
            if (used_mask & num_masks[i]) == 0:
                result.append(x)
                result_set.add(x)
                used_mask |= num_masks[i]
        return result, result_set

    best_result, best_result_set = get_greedy_solution()
    best_sum = sum(best_result)
    
    start_time = time.time()
    TIME_LIMIT = 55.0
    
    # DFS Branch and Bound
    def dfs(i, mask, curr_sum, curr_set):
        nonlocal best_sum, best_result_set
        
        # Time check
        if time.time() - start_time > TIME_LIMIT:
            return
        
        # Loose bound pruning
        if curr_sum + suffix_sum[i] <= best_sum:
            return
        
        # Tighter bound: sum of remaining numbers compatible with current mask
        # Only do this for early stages or if loose bound is close to be efficient?
        # Actually, let's do it for all i < 200 to prune deeper branches
        if i < 200:
            compat_sum = 0
            for j in range(i, len(nums)):
                if (mask & num_masks[j]) == 0:
                    compat_sum += nums[j]
            if curr_sum + compat_sum <= best_sum:
                return
        
        if i == len(nums):
            if curr_sum > best_sum:
                best_sum = curr_sum
                best_result_set = set(curr_set)
            return
        
        x = nums[i]
        m = num_masks[i]
        
        # Option 1: Include x
        if (mask & m) == 0:
            curr_set.add(x)
            dfs(i + 1, mask | m, curr_sum + x, curr_set)
            curr_set.remove(x)
        
        # Option 2: Exclude x
        dfs(i + 1, mask, curr_sum, curr_set)
    
    dfs(0, 0, 0, set())
    
    return list(best_result_set)
# EVOLVE-BLOCK-END
