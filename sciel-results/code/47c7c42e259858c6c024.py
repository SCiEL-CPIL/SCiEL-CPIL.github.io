# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    DEADLINE = 85  # safety margin
    
    n = 5040
    target = 4979
    
    # Get all divisors of 5040
    divisors = []
    for d in range(1, n + 1):
        if n % d == 0:
            if d <= target: # Only keep divisors that can possibly be in the sum
                divisors.append(d)
    
    # Sort divisors descending for better pruning in backtracking
    divisors.sort(reverse=True)
    num_divs = len(divisors)
    
    # Step 1: Find minimum K using Bitset DP
    # dp[k] is a bitset where bit s is 1 if sum s is achievable with k divisors
    # We only need to go up to K such that we find the target.
    # Max K is num_divs.
    
    # Initialize dp
    # dp[0] = 1 (sum 0)
    # dp[k] = 0 for k > 0
    dp = [0] * (num_divs + 1)
    dp[0] = 1
    
    mask = (1 << (target + 1)) - 1
    
    for d in divisors:
        # Update from high k to low k to ensure each divisor is used at most once
        # We can limit k to num_divs
        for k in range(num_divs, 0, -1):
            if dp[k-1]:
                dp[k] |= (dp[k-1] << d)
                dp[k] &= mask
    
    # Find minimum k
    min_k = -1
    for k in range(1, num_divs + 1):
        if (dp[k] >> target) & 1:
            min_k = k
            break
    
    if min_k == -1:
        # Fallback if not found (should not happen)
        return []
        
    # Step 2: Backtrack to find the specific divisors
    # Use recursion with pruning
    import sys
    sys.setrecursionlimit(10000)
    
    def backtrack(index, k, remaining_sum):
        """
        Returns a list of divisors or None.
        index: current index in divisors
        k: number of divisors to pick
        remaining_sum: sum to achieve
        """
        if k == 0:
            if remaining_sum == 0:
                return []
            else:
                return None
        
        if index >= num_divs:
            return None
            
        # Pruning: Not enough divisors left
        if num_divs - index < k:
            return None
            
        # Pruning: Max possible sum with k divisors from index onwards
        # Since divisors are sorted desc, the max sum is sum of first k available
        # We can precompute suffix sums or just sum the slice (k is small)
        if k > 0:
             # Sum of largest k divisors from index
             max_possible = sum(divisors[index:index+k])
             if max_possible < remaining_sum:
                 return None
             
             # Sum of smallest k divisors from index
             min_possible = sum(divisors[num_divs-k:num_divs]) # Wait, this is global min.
             # Actually, min possible from index is sum(divisors[num_divs-k:num_divs]) is wrong if index > 0.
             # Min possible is sum of last k divisors in the subarray divisors[index:]
             # Which is divisors[num_divs-k:num_divs] only if index == 0.
             # Generally: sum(divisors[num_divs-k:num_divs]) is not correct for subarray.
             # Min sum is sum(divisors[index + (num_divs-index - k) : num_divs])
             # i.e. the last k elements of divisors[index:]
             min_possible = sum(divisors[num_divs-k:num_divs]) # This is constant, but is it valid?
             # If index > 0, we can't pick divisors before index.
             # The smallest k divisors available are the last k in the list divisors[index:].
             # Which are divisors[num_divs-k : num_divs] IF index == 0.
             # If index > 0, the available divisors are divisors[index:].
             # The smallest k are divisors[num_divs-k : num_divs] ONLY IF index <= num_divs-k.
             # Actually, just sum the last k of divisors[index:].
             # len(divisors[index:]) = num_divs - index.
             # Last k are divisors[num_divs-k : num_divs].
             # This is only true if the subarray ends at num_divs.
             # Yes, divisors[index:] ends at num_divs.
             # So min_possible = sum(divisors[num_divs-k : num_divs]) is INCORRECT if index > 0?
             # No, divisors is sorted. The smallest elements are at the end.
             # The smallest k elements in divisors[index:] are indeed the last k elements of the whole list,
             # PROVIDED that index <= num_divs - k.
             # If index > num_divs - k, we don't have k elements? No, we checked num_divs - index >= k.
             # So index <= num_divs - k.
             # So the last k elements of divisors[index:] are divisors[num_divs-k : num_divs].
             # Wait. divisors[index:] is a slice.
             # Example: divs = [10, 8, 5, 2]. index=1. k=2.
             # divs[1:] = [8, 5, 2]. Smallest 2 are 5, 2. Sum=7.
             # divs[num_divs-k:] = divs[2:] = [5, 2]. Sum=7. Correct.
             # Example: index=0. k=2. divs[0:] = [10, 8, 5, 2]. Smallest 2 are 5, 2.
             # divs[2:] = [5, 2]. Correct.
             
             if min_possible > remaining_sum:
                 return None
        
        # Option 1: Include divisors[index]
        if divisors[index] <= remaining_sum:
            res = backtrack(index + 1, k - 1, remaining_sum - divisors[index])
            if res is not None:
                return [divisors[index]] + res
        
        # Option 2: Exclude divisors[index]
        res = backtrack(index + 1, k, remaining_sum)
        if res is not None:
            return res
            
        return None

    solution = backtrack(0, min_k, target)
    if solution:
        return solution
    else:
        # Fallback to simple greedy or return empty
        return []

# EVOLVE-BLOCK-END
