import time
import sys

def construct():
    n = 5040
    target = 4979
    
    # Generate divisors
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    
    divisors.sort(reverse=True)
    divisor_set = set(divisors)
    
    # We search for k=1, 2, ...
    # k=1
    if target in divisor_set:
        return [target]
        
    # k=2
    # Check pairs
    # To avoid duplicates in search logic, we can just iterate
    for i in range(len(divisors)):
        d1 = divisors[i]
        if d1 >= target: 
            continue
        rem = target - d1
        if rem in divisor_set and rem != d1:
            return [d1, rem]
            
    # k=3
    # We can use a recursive helper or nested loops
    # Given small N, nested loops are fine for small k
    # But let's use a generic DFS for k >= 3 to be safe and clean
    
    def dfs(start_idx, remaining, k, current):
        if k == 0:
            if remaining == 0:
                return current
            return None
        
        # Pruning:
        # Max possible sum with k items from start_idx onwards
        # Min possible sum with k items from start_idx onwards
        # Since divisors are sorted descending:
        # Max sum is sum of divisors[start_idx : start_idx+k]
        # Min sum is sum of the k smallest available divisors? 
        # Actually, since we just need *a* solution, and we are minimizing k,
        # any valid set of size k works.
        
        if start_idx + k > len(divisors):
            return None
            
        # Simple bound: if the largest k divisors from start_idx sum to less than remaining, impossible.
        max_sum = sum(divisors[start_idx : start_idx+k])
        if max_sum < remaining:
            return None
            
        # If the smallest possible sum (using the k smallest divisors overall? No, we are constrained by index)
        # Actually, since we are picking distinct divisors, and they are positive,
        # if the sum of the k smallest divisors in the entire set is > remaining, impossible.
        # But we are iterating in order.
        # A tighter bound: The minimum sum we can form with k distinct divisors from index start_idx onwards
        # is sum(divisors[-k:]) if start_idx is small enough? No.
        # Let's just rely on max_sum pruning and the fact that k is small.
        
        for i in range(start_idx, len(divisors) - k + 1):
            d = divisors[i]
            if d > remaining:
                continue
            # Optional: if d + sum of k-1 smallest available < remaining, continue?
            # Let's just recurse
            res = dfs(i + 1, remaining - d, k - 1, current + [d])
            if res is not None:
                return res
        return None

    # Search k from 3 upwards
    for k in range(3, len(divisors) + 1):
        res = dfs(0, target, k, [])
        if res is not None:
            return res
            
    return [] # Should not reach here
