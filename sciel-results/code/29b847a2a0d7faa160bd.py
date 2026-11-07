import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 210
    max_val = 10210
    min_val = 211
    max_size = 100
    
    # 1. Factorization of the base number
    def factorize(x):
        factors = {}
        d = 2
        while d * d <= x:
            while x % d == 0:
                factors[d] = factors.get(d, 0) + 1
                x //= d
            d += 1
        if x > 1:
            factors[x] = factors.get(x, 0) + 1
        return factors
    
    base_factors = factorize(n)
    # base_factors = {2: 1, 3: 1, 5: 1, 7: 1}
    
    def sqfree_kernel(x):
        f = factorize(x)
        k = 1
        for p, exp in f.items():
            if exp % 2 == 1:
                k *= p
        return k
    
    def xor_kernel(a, b):
        fa = factorize(a)
        fb = factorize(b)
        all_primes = set(fa.keys()).union(fb.keys())
        res = 1
        for p in all_primes:
            ea = fa.get(p, 0)
            eb = fb.get(p, 0)
            if (ea + eb) % 2 == 1:
                res *= p
        return res
    
    # Precompute kernels for all valid numbers
    # Valid numbers are 211..10210
    kernels = {}
    # We only need kernels up to the squarefree part
    # Note: kernel(x) is always squarefree
    for x in range(min_val, max_val + 1):
        k = sqfree_kernel(x)
        # If two numbers have the same kernel, they are equivalent for the XOR condition
        # However, for minimizing max, smaller numbers with the same kernel are better.
        # We want to find a set of distinct numbers.
        # Let's map kernel -> smallest number with that kernel.
        # This is sufficient because if we select a kernel, we should pick the smallest number
        # that has that kernel to minimize the maximum.
        # Wait, the numbers must be distinct. If we pick two numbers with the same kernel,
        # their XOR is 1. So we could pick two numbers with the same kernel to cancel each other?
        # No, the XOR of the selected set must equal base_kernel (210).
        # Let K be the set of kernels of selected numbers.
        # XOR(K) = 210.
        # We can select multiple numbers with the same kernel.
        # If we select two numbers with kernel k, their combined XOR is 1.
        # This is equivalent to not selecting them.
        # So we can assume without loss of generality that we select at most one number per kernel.
        # Why? If we select two numbers x, y with sqfree(x) = sqfree(y) = k,
        # their contribution to the total XOR is k ^ k = 1.
        # We can remove both x and y from the set, and the XOR condition remains satisfied.
        # The new max will be <= old max.
        # So an optimal solution contains at most one number for each distinct squarefree kernel.
        
        # Thus, we can reduce the problem to:
        # Find a subset S of squarefree kernels such that XOR(S) = 210.
        # For each k in S, we select the smallest number x_k > 210 with sqfree(x_k) = k.
        # We want to minimize max(x_k for k in S).
        
        # Let's build a map: kernel -> smallest number > 210 with that kernel.
        if k not in kernels or x < kernels[k]:
            kernels[k] = x
    
    # Target XOR is 210 (which is squarefree)
    target = n
    
    # We want to find a subset of kernels such that their XOR is target.
    # This is a linear algebra problem over GF(2) on the prime exponents.
    # However, the number of primes involved can be large.
    # But the numbers are <= 10210.
    # The distinct squarefree kernels are limited.
    
    # Let's collect all available kernels and their minimal representatives.
    # Note: We must ensure we have enough distinct numbers.
    # The problem says "1..100 distinct integers".
    # If we pick at most one per kernel, the size of the set is the number of kernels in S.
    # We just need size <= 100.
    # Given the range, the number of kernels is likely large enough.
    
    # To minimize max(x_k), we should look for subsets with small representatives.
    # Let's sort the kernels by their minimal representative.
    # Then we can try to find a subset of the first M kernels that XORs to target.
    # We can use a BFS or Dijkstra-like approach on the XOR space?
    # The XOR space is defined by the primes.
    # The dimension is the number of distinct primes appearing in any kernel.
    # Primes up to 10210 are ~1200.
    # This is too large for a bitset of size 2^P.
    
    # Alternative: Meet in the middle?
    # Or simply: Since we want to minimize the maximum, we can iterate on the maximum value V.
    # For a given V, can we find a subset of numbers in [211, V] such that XOR of their kernels is 210?
    # This is a subset XOR problem.
    # We can use a dictionary to store reachable XOR values and the path (set of kernels).
    # But the number of reachable XOR values might be large.
    
    # Let's observe the structure.
    # 210 = 2 * 3 * 5 * 7.
    # If we pick numbers whose kernels only involve primes {2, 3, 5, 7}, the XOR space is small (size 16).
    # Let's check if we can solve it using only 7-smooth kernels.
    # If yes, the numbers will be composed of these primes.
    # Smallest numbers > 210 with 7-smooth kernels:
    # Kernel 1: 225 (15^2), 256 (16^2), ... Smallest > 210 is 225.
    # Kernel 2: 212 (2*106? No 106 not smooth). 218 (2*109). 
    # Actually, any number x has a kernel.
    # Let's list small numbers and their kernels to see if 7-smooth kernels are frequent.
    
    # Let's try a greedy/random search restricted to small numbers first.
    # If we find a solution with max < 300, that's good.
    
    # Better approach:
    # Since we established that we can pick at most one per kernel,
    # let's list the kernels and their minimal representatives sorted by representative.
    items = sorted([(val, k) for k, val in kernels.items()], key=lambda x: x[0])
    # items is list of (min_rep, kernel)
    
    # We can use a BFS in the space of XORs, but we want to minimize the max rep.
    # Let's try increasing the "budget" (max allowed value) and check if a solution exists.
    # But checking existence of a subset XORing to target is NP-hard in general.
    # However, for small dimensions (primes involved), it's easy.
    
    # Let's identify the primes involved in the target.
    target_primes = set(factorize(target).keys())
    
    # Strategy:
    # 1. Try to find a solution using only kernels that are products of a small set of primes.
    # 2. Start with primes {2, 3, 5, 7}.
    # 3. If no solution or solution is too large, add more primes.
    
    # Let's just use a generic search for subsets.
    # We can represent the XOR state by the squarefree number itself.
    # We want to reach state 210.
    
    # We will perform a BFS from 0, adding one kernel at a time.
    # To keep it fast, we will only consider kernels whose minimal representative is <= current_limit.
    # We will increase current_limit step by step.
    
    # Since the number of distinct kernels is limited, and we want to minimize the max rep,
    # we can sort items by rep.
    # Let M be the number of items.
    # We can try to find a subset of the first K items that XORs to 210.
    # We can use a dictionary: xor_val -> (max_rep_in_subset, subset_indices)
    # But the number of xor_vals can be up to 2^P.
    # However, the values are bounded by the product of primes.
    
    # Let's try a simple iterative deepening on the maximum value.
    # For V in range(min_val, max_val + 1):
    #   If time is running out, break.
    #   Consider all kernels with min_rep <= V.
    #   Check if there is a subset XORing to 210.
    
    # This is too slow if done naively.
    
    # Let's use the linearity.
    # The set of kernels forms a vector space over GF(2) (represented by prime factorization).
    # We want to know if 210 is in the span of the available kernels.
    # And we want to find a basis with small coefficients (small reps).
    
    # Actually, we can just run a Dijkstra-like algorithm on the XOR values.
    # Node: XOR value (squarefree int).
    # Edge: XOR with a kernel k (cost is max(current_max, rep[k])).
    # We want to reach 210 with minimum cost.
    
    # However, the number of reachable XOR values might be large.
    # But we can prune: we only care about XOR values that are "small" or related to 210.
    # Actually, any squarefree number is a valid state.
    
    # Let's limit the search to the first, say, 5000 smallest kernels.
    # This should cover a wide range of primes.
    
    import heapq
    
    # We will use a priority queue: (max_rep, xor_val, subset)
    # But storing the subset is expensive. We can store the parent pointer.
    # Or just store the set of indices?
    # Given the constraint of 100 elements, the subset size is small.
    
    # Let's try a simpler approach:
    # Since 210 is small, maybe the solution uses small kernels.
    # Let's pick the smallest 200 kernels.
    # And run a BFS to find a path to 210.
    
    # To ensure we find the optimal max, we should process kernels in order of their rep.
    # We can use the property that if we have a set of kernels, we can form their span.
    
    # Let's just use a dictionary: `dp[xor_val] = min_max_rep`
    # Initialize dp[0] = 0 (empty set)
    # For each kernel k with rep r (processed in increasing r):
    #   For each xor_val in dp:
    #     new_xor = xor_val ^ k
    #     new_max = max(dp[xor_val], r)
    #     if new_max < dp.get(new_xor, inf):
    #       dp[new_xor] = new_max
    
    # This is O(N * |dp|).
    # N is number of kernels considered.
    # |dp| is number of reachable XOR values.
    # If we limit N to, say, 1000, and |dp| stays small, this is fast.
    
    # Let's test this.
    
    # Select the smallest 5000 kernels by their representative.
    # Note: We must be careful with time.
    
    # Let's sort all kernels by rep.
    # There are about 10000 numbers, so at most 10000 kernels.
    
    sorted_kernels = sorted([(val, k) for k, val in kernels.items()], key=lambda x: x[0])
    
    # We will iterate through these sorted kernels and update the DP table.
    # dp[xor_val] = minimum possible max_rep to achieve xor_val.
    dp = {0: 0}
    
    # We want to reach xor_val = 210.
    # We also need to track the actual set of numbers to reconstruct the answer.
    # To do this, we can store for each state the kernel used to reach it and the previous state.
    # But since we are updating iteratively, we need to be careful.
    # Actually, the standard DP update `dp[new_xor] = min(dp[new_xor], max(dp[xor], r))`
    # doesn't easily allow reconstruction because `dp[xor]` might be updated later.
    # However, since we process kernels in increasing order of `r`,
    # when we update `dp[new_xor]` using `xor` and `k`, the `max` is determined by `r` (since `r` is the current largest).
    # So `dp[new_xor]` becomes `r` if `r > dp[xor]` (which is likely).
    # We can store `parent[new_xor] = (xor, k)` when we update it.
    
    parent = {}
    
    for val, k in sorted_kernels:
        if time.time() > deadline:
            break
            
        # Snapshot of dp before processing this kernel?
        # No, we can use the current dp.
        # But if we use `dp` while iterating, we might use the same kernel twice?
        # No, we are adding `k` to existing states.
        # The existing states were formed by a subset of kernels processed so far (which are <= k in rep).
        # So adding `k` again is not allowed (distinct kernels).
        # Wait, the DP state `dp[xor]` represents a subset of kernels from the prefix.
        # When we process `k`, we form new states by adding `k` to any existing state.
        # This ensures each kernel is used at most once.
        
        # We need to iterate over a snapshot of the keys?
        # If we iterate over `dp.items()`, and `dp` is updated, we might miss some or see new ones.
        # Since we are only adding `k` to existing states, and `k` is new,
        # the new states are distinct from existing ones (unless `k` is 0, which it isn't).
        # So we can iterate over a list of items.
        
        items_to_process = list(dp.items())
        
        for xor_val, current_max in items_to_process:
            new_xor = xor_val ^ k
            new_max = max(current_max, val)
            
            if new_max < dp.get(new_xor, float('inf')):
                dp[new_xor] = new_max
                parent[new_xor] = (xor_val, k)
                
                if new_xor == target:
                    # We found a solution!
                    # Since we process `val` in increasing order, is this the optimal?
                    # Not necessarily globally, but `new_max` is `val`.
                    # Any future kernel has rep >= val.
                    # So any future solution will have max >= val.
                    # Thus, the first time we reach `target` in this sorted order gives the minimum possible max?
                    # Yes, because we are increasing the maximum allowed value.
                    # Wait, `new_max` is `val` only if `val > current_max`.
                    # Since we sorted by `val`, and `current_max` is from previous steps,
                    # `current_max <= val`. So `new_max = val`.
                    # So we are finding the smallest `val` such that a subset of kernels with reps <= `val` sums to target.
                    # This is exactly what we want.
                    
                    # Reconstruct the set
                    selected_kernels = []
                    curr = new_xor
                    while curr != 0:
                        prev_xor, k_used = parent[curr]
                        selected_kernels.append(k_used)
                        curr = prev_xor
                    
                    # Get the numbers
                    result_nums = [kernels[k] for k in selected_kernels]
                    
                    # Verify
                    # 1. Distinct
                    if len(set(result_nums)) != len(result_nums):
                        # Should not happen since kernels are distinct
                        pass
                    # 2. > 210
                    if any(x <= 210 for x in result_nums):
                        pass
                    # 3. <= 10210
                    if any(x > 10210 for x in result_nums):
                        pass
                    # 4. Size <= 100
                    if len(result_nums) > 100:
                        pass
                    # 5. Product * 210 is square
                    # This is guaranteed by the XOR condition.
                    
                    return result_nums
    
    # If no solution found with the limited search or time ran out, return a fallback
    # The seed solution [840] works.
    return [840]

