"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s limit
    
    n = 5040
    target = 4979
    
    # Generate all divisors of 5040
    divisors = [d for d in range(1, n + 1) if n % d == 0]
    
    # Filter to divisors <= target (can't use anything larger)
    divisors = [d for d in divisors if d <= target]
    
    # Split into odd and even divisors for structural search
    odd_divs = sorted([d for d in divisors if d % 2 != 0], reverse=True)
    even_divs = sorted([d for d in divisors if d % 2 == 0])
    
    # First, try to find a solution with k=4 using the odd+even structure
    # Since 4979 is odd, we need an odd number of odd divisors.
    # With k=4, we need 1 or 3 odd divisors.
    # Case 1: 1 odd + 3 even
    # Case 2: 3 odd + 1 even
    
    # Try Case 1: 1 odd + 3 even
    # For each odd divisor d1, find 3 distinct even divisors summing to target - d1
    found = False
    best_solution = None
    
    # Precompute set of even divisors for O(1) lookup
    even_set = set(even_divs)
    
    # Sort even divisors for binary search
    even_sorted = sorted(even_divs)
    
    for d1 in odd_divs:
        if time.time() > deadline:
            break
        rem_target = target - d1
        if rem_target < 0:
            continue
        
        # Find 3 distinct even divisors e1, e2, e3 such that e1+e2+e3 = rem_target
        # We can iterate over pairs (e1, e2) and check if rem_target - e1 - e2 is in even_set
        # To ensure distinctness and avoid duplicates, we can enforce e1 < e2 < e3
        # Or just check set membership and distinctness
        
        # Iterate over e1 in even_sorted
        # For efficiency, e1 should be small enough that there's room for e2, e3
        # Max even divisor is 2520.
        # If e1 > rem_target - 2 * min_even, it's impossible? No, min_even is 2.
        # If e1 > rem_target - 2 - 4, it's impossible?
        
        # Let's just iterate. There are ~30 even divisors. 30^2 = 900 pairs.
        # 30 * 900 = 27000 checks per odd divisor.
        # ~15 odd divisors. 15 * 27000 = 405,000 checks. Very fast.
        
        for i, e1 in enumerate(even_sorted):
            if time.time() > deadline:
                break
            if e1 >= rem_target:
                break
            rem2 = rem_target - e1
            # e2 must be > e1 to ensure distinctness (since we iterate sorted)
            # and e2 < rem2 - min_even (since e3 >= min_even=2)
            # Actually e3 > e2, so e3 >= e2 + 2 (since even)
            # So e2 + (e2+2) <= rem2 => 2*e2 <= rem2 - 2 => e2 <= (rem2-2)/2
            
            # Iterate e2 from i+1
            for j in range(i + 1, len(even_sorted)):
                e2 = even_sorted[j]
                if e2 >= rem2:
                    break
                e3 = rem2 - e2
                # Check if e3 is in even_set and e3 > e2
                if e3 > e2 and e3 in even_set:
                    best_solution = [d1, e1, e2, e3]
                    found = True
                    break
            if found:
                break
        if found:
            break
    
    if not found:
        # Try Case 2: 3 odd + 1 even
        # For each even divisor e1, find 3 distinct odd divisors summing to target - e1
        for e1 in even_sorted:
            if time.time() > deadline:
                break
            rem_target = target - e1
            if rem_target < 0:
                continue
            
            # Find 3 distinct odd divisors o1, o2, o3 such that o1+o2+o3 = rem_target
            # o1 < o2 < o3
            # Iterate o1, o2
            for i, o1 in enumerate(odd_divs):
                if time.time() > deadline:
                    break
                if o1 >= rem_target:
                    break
                rem2 = rem_target - o1
                # o2 > o1
                for j in range(i + 1, len(odd_divs)):
                    o2 = odd_divs[j]
                    if o2 >= rem2:
                        break
                    o3 = rem2 - o2
                    # o3 > o2 and o3 is odd and in divisors
                    if o3 > o2 and o3 in divisors and o3 % 2 != 0:
                        best_solution = [e1, o1, o2, o3]
                        found = True
                        break
                if found:
                    break
            if found:
                break
    
    if found:
        # Verify
        assert len(set(best_solution)) == len(best_solution), "Duplicate divisors"
        assert all(n % d == 0 for d in best_solution), "Not all are divisors of 5040"
        assert sum(best_solution) == target, f"Sum is {sum(best_solution)}, not {target}"
        return best_solution
    
    # If k=4 not found, try k=5 with DP or similar
    # But k=4 is very likely to exist. Let's add a DP fallback for k=5 just in case.
    
    # DP for minimum count
    INF = float('inf')
    dp = [INF] * (target + 1)
    parent_div = [-1] * (target + 1)
    parent_sum = [-1] * (target + 1)
    
    dp[0] = 0
    
    for d in divisors:
        for s in range(target, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                parent_div[s] = d
                parent_sum[s] = s - d
    
    if dp[target] != INF:
        chosen = []
        s = target
        while s > 0:
            d = parent_div[s]
            chosen.append(d)
            s = parent_sum[s]
        chosen.reverse()
        assert len(set(chosen)) == len(chosen), "Duplicate divisors"
        assert all(n % d == 0 for d in chosen), "Not all are divisors of 5040"
        assert sum(chosen) == target, f"Sum is {sum(chosen)}, not {target}"
        return chosen
    
    # Final fallback: Greedy
    remainder = target
    chosen = []
    for d in sorted(divisors, reverse=True):
        if d <= remainder:
            chosen.append(d)
            remainder -= d
            if remainder == 0:
                break
    
    assert remainder == 0, "Greedy failed"
    return chosen

# EVOLVE-BLOCK-END
