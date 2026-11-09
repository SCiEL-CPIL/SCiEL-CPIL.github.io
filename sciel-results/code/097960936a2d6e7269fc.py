import time
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill
    
    # Precompute all divisors of 5040
    n = 5040
    target = 4979
    divisors = sorted([d for d in range(1, n + 1) if n % d == 0], reverse=True)
    # Filter to only divisors <= target (can't use anything larger)
    divisors = [d for d in divisors if d <= target]
    
    best_solution = None
    
    # Try k = 1, 2, 3, ... until we find a solution
    for k in range(1, len(divisors) + 1):
        if time.time() > deadline:
            break
        
        found = False
        
        if k <= 3:
            # Direct enumeration for small k
            for combo in itertools.combinations(divisors, k):
                if sum(combo) == target:
                    best_solution = list(combo)
                    found = True
                    break
                if time.time() > deadline:
                    break
        else:
            # Use meet-in-the-middle or DFS with strong pruning
            # For k >= 4, use a smarter approach
            
            # First, try a greedy + local search approach
            # Start with greedy: pick largest possible divisors
            chosen = []
            rem = target
            for d in divisors:
                if len(chosen) >= k:
                    break
                if d <= rem:
                    chosen.append(d)
                    rem -= d
            if rem == 0 and len(chosen) == k:
                best_solution = chosen
                found = True
            else:
                # Use DFS with strong pruning
                divs = divisors
                # Precompute suffix sums for pruning
                # suffix_sum[i] = sum of divs[i:]
                suffix_sum = [0] * (len(divs) + 1)
                for i in range(len(divs) - 1, -1, -1):
                    suffix_sum[i] = suffix_sum[i + 1] + divs[i]
                
                # Also precompute: for each position i and number of items needed,
                # the max sum we can get by picking 'needed' items from divs[i:]
                # This is sum of the largest 'needed' items from divs[i:]
                # Since divs is sorted descending, it's sum(divs[i:i+needed])
                
                def dfs(idx, remaining, needed, chosen_list):
                    if time.time() > deadline:
                        return None
                    if needed == 0:
                        if remaining == 0:
                            return list(chosen_list)
                        return None
                    if idx >= len(divs):
                        return None
                    
                    # Pruning 1: not enough items left
                    if needed > len(divs) - idx:
                        return None
                    
                    # Pruning 2: max possible sum from idx onward with 'needed' items
                    # Since divs is sorted descending, the max sum is sum(divs[idx:idx+needed])
                    max_possible = sum(divs[idx:idx + needed])
                    if max_possible < remaining:
                        return None
                    
                    # Pruning 3: min possible sum - we need at least 'remaining'
                    # The min sum of 'needed' items from idx onward is sum of the smallest 'needed'
                    # which would be from the end, but since we can skip, the actual min is more complex
                    # A simpler bound: each item is at least 1, so min possible is 'needed'
                    if remaining < needed:
                        return None
                    
                    # Pruning 4: if remaining is too large even taking all remaining items
                    if remaining > suffix_sum[idx]:
                        return None
                    
                    # Option 1: take divs[idx]
                    if divs[idx] <= remaining:
                        chosen_list.append(divs[idx])
                        result = dfs(idx + 1, remaining - divs[idx], needed - 1, chosen_list)
                        if result is not None:
                            return result
                        chosen_list.pop()
                    
                    # Option 2: skip divs[idx]
                    result = dfs(idx + 1, remaining, needed, chosen_list)
                    return result
                
                result = dfs(0, target, k, [])
                if result is not None:
                    best_solution = result
                    found = True
        
        if found:
            break
    
    if best_solution is None:
        # Fallback: use the original greedy approach
        remainder = target
        chosen = []
        for d in divisors:
            if d <= remainder:
                chosen.append(d)
                remainder -= d
        best_solution = chosen
    
    # Verify
    assert len(set(best_solution)) == len(best_solution), "Duplicate divisors"
    assert all(n % d == 0 for d in best_solution), "Not all are divisors of 5040"
    assert sum(best_solution) == target, f"Sum is {sum(best_solution)}, not {target}"
    
    return best_solution
