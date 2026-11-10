"""Optimized program for the suite task; uses IDDFS to minimize number of divisors."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    n = 5040
    target = 4979
    
    # Generate all divisors of 5040
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    
    # Sort descending for better pruning (largest first)
    divisors.sort(reverse=True)
    num_divs = len(divisors)
    divisor_set = set(divisors)
    
    # Precompute cumulative sums for O(1) range sum queries
    # cumsum[i] = sum of divisors[0..i-1]
    cumsum = [0] * (num_divs + 1)
    for i in range(num_divs):
        cumsum[i+1] = cumsum[i] + divisors[i]
    
    # Helper to check if a sum is achievable with exactly k distinct divisors
    # using DFS with pruning
    def solve_with_k(k):
        if time.time() > deadline:
            return None
        
        if k > num_divs:
            return None
        
        # Quick bound checks
        # Max sum is sum of largest k divisors
        max_possible = cumsum[k]
        if max_possible < target:
            return None
        
        # Min sum is sum of smallest k divisors
        # Smallest k divisors are at the end of the sorted list
        min_possible = cumsum[num_divs] - cumsum[num_divs - k]
        if min_possible > target:
            return None
        
        path = []
        
        def dfs(idx, remaining_sum, slots_left):
            if time.time() > deadline:
                return False
            
            if slots_left == 0:
                return remaining_sum == 0
            
            if idx >= num_divs:
                return False
            
            # Not enough divisors left to fill slots
            if num_divs - idx < slots_left:
                return False
            
            # Pruning: Max sum we can get with slots_left divisors from idx onwards
            # These are divisors[idx], divisors[idx+1], ..., divisors[idx+slots_left-1]
            max_sum = cumsum[idx + slots_left] - cumsum[idx]
            if remaining_sum > max_sum:
                return False
            
            # Pruning: Min sum we can get with slots_left divisors from idx onwards
            # These are the smallest slots_left divisors available in the whole list
            # Since we are at idx, we can pick any from idx..end, but to get min sum
            # we pick the smallest ones available. However, in standard subset sum DFS
            # where order matters (we pick increasing indices), the min sum from idx
            # is sum of divisors[idx..idx+slots_left-1] if we MUST take consecutive?
            # No, we can skip. So the absolute minimum sum we can form with slots_left
            # items from indices idx..num_divs-1 is the sum of the smallest slots_left
            # items in that range. Since list is sorted desc, the smallest are at the end.
            # So min_sum = sum of divisors[num_divs-slots_left : num_divs]
            # Wait, if we are at idx, we can only pick from idx onwards.
            # The smallest available are at the end of the array.
            min_sum = cumsum[num_divs] - cumsum[num_divs - slots_left]
            if remaining_sum < min_sum:
                return False
            
            # Option 1: Include divisors[idx]
            if divisors[idx] <= remaining_sum:
                path.append(divisors[idx])
                if dfs(idx + 1, remaining_sum - divisors[idx], slots_left - 1):
                    return True
                path.pop()
            
            # Option 2: Exclude divisors[idx]
            # We need to check if we can still fill slots_left from idx+1 onwards
            if num_divs - (idx + 1) >= slots_left:
                if dfs(idx + 1, remaining_sum, slots_left):
                    return True
            
            return False
        
        if dfs(0, target, k):
            return path[:]
        return None
    
    # Check k=1
    if target in divisor_set:
        return [target]
    
    # Check k=2
    # Efficient pair check
    for d in divisors:
        if d > target:
            continue
        rem = target - d
        if rem in divisor_set and rem != d:
            return [d, rem]
    
    # Check k=3
    # Efficient triple check
    for i in range(num_divs):
        if time.time() > deadline:
            break
        d1 = divisors[i]
        if d1 > target:
            continue
        rem1 = target - d1
        if rem1 <= 0:
            continue
        # Find two distinct divisors d2, d3 from divisors[i+1:] such that d2+d3=rem1
        # Since divisors are sorted desc, we can iterate
        for j in range(i + 1, num_divs):
            if time.time() > deadline:
                break
            d2 = divisors[j]
            if d2 >= rem1:
                continue
            rem2 = rem1 - d2
            if rem2 in divisor_set and rem2 != d1 and rem2 != d2:
                return [d1, d2, rem2]
    
    # Check k >= 4 using IDDFS
    # We start from k=4 because 1,2,3 are handled above
    # Upper bound for k: The greedy solution count gives an upper bound
    # Let's compute a greedy solution to set an upper bound and fallback
    greedy_rem = target
    greedy_chosen = []
    for d in divisors:
        if d <= greedy_rem:
            greedy_chosen.append(d)
            greedy_rem -= d
            if greedy_rem == 0:
                break
    
    best_solution = greedy_chosen if greedy_rem == 0 else None
    max_k_to_check = len(greedy_chosen) if best_solution else num_divs
    
    for k in range(4, min(max_k_to_check + 1, num_divs + 1)):
        if time.time() > deadline:
            break
        result = solve_with_k(k)
        if result is not None:
            return result
    
    # Fallback
    if best_solution:
        return best_solution
    
    # If no solution found (should not happen for practical numbers like 5040),
    # return a valid subset if possible, or empty list
    return []

# EVOLVE-BLOCK-END
