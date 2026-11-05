"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from functools import lru_cache

def construct():
    start_time = time.time()
    DEADLINE = 85  # seconds, safety margin before 90s kill
    
    n = 5040
    target = 4979
    
    # Generate all divisors of 5040
    divisors = []
    for d in range(1, n + 1):
        if n % d == 0:
            divisors.append(d)
    
    # Sort in descending order to prefer larger divisors (minimizes count)
    divisors.sort(reverse=True)
    
    # Remove divisors > target since we can't use them
    divisors = [d for d in divisors if d <= target]
    
    # Iterative deepening: try to find solution with k terms
    # Start from a lower bound: at least ceil(target / max_divisor)
    max_d = divisors[0] if divisors else 1
    min_k = max(1, (target + max_d - 1) // max_d)
    
    best_solution = None
    
    def dfs(k, start_idx, remaining, current):
        """Try to find k divisors starting from index start_idx that sum to remaining."""
        if time.time() - start_time > DEADLINE:
            return None
        
        if k == 0:
            if remaining == 0:
                return list(current)
            return None
        
        if remaining < 0:
            return None
        
        # Pruning: if even taking the largest k available can't reach remaining, fail
        # The largest k divisors from start_idx
        if start_idx + k > len(divisors):
            return None
        
        # Upper bound on what we can achieve with k divisors from start_idx
        max_possible = sum(divisors[start_idx:start_idx + k])
        if max_possible < remaining:
            return None
        
        # Try each divisor from start_idx
        for i in range(start_idx, len(divisors)):
            d = divisors[i]
            if d > remaining:
                continue
            
            current.append(d)
            result = dfs(k - 1, i + 1, remaining - d, current)
            if result is not None:
                return result
            current.pop()
        
        return None
    
    # Try increasing k from min_k
    for k in range(min_k, len(divisors) + 1):
        if time.time() - start_time > DEADLINE:
            break
        
        result = dfs(k, 0, target, [])
        if result is not None:
            best_solution = result
            break
    
    # Fallback: if no solution found within deadline, use greedy
    if best_solution is None:
        remainder = target
        chosen = []
        for d in divisors:
            if d <= remainder:
                chosen.append(d)
                remainder -= d
            if remainder == 0:
                break
        if remainder == 0:
            best_solution = chosen
        else:
            # Should not happen for practical numbers, but safety
            raise ValueError("No solution found")
    
    # Verify
    assert len(set(best_solution)) == len(best_solution), "Divisors must be distinct"
    assert all(n % d == 0 for d in best_solution), "All must be divisors of 5040"
    assert sum(best_solution) == target, f"Sum must be {target}, got {sum(best_solution)}"
    assert len(best_solution) <= 256, "At most 256 divisors"
    
    return best_solution
# EVOLVE-BLOCK-END
