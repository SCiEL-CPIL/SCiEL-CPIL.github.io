import time
from fractions import Fraction

def construct():
    """
    Finds 1..256 distinct integer denominators in [5, 10000000] whose reciprocal sum is exactly 1,
    minimizing their count.
    """
    start_time = time.time()
    deadline = 80.0  # Leave 10s margin for the 90s limit
    
    # Theoretical lower bound on count is 8
    # We search for increasing numbers of terms
    for target_count in range(8, 257):
        if time.time() > deadline:
            break
            
        # Try to find a solution with exactly target_count terms
        solution = search_for_solution(target_count, deadline)
        
        if solution is not None:
            # Since we search increasing counts, the first valid solution is the minimal count
            return solution
            
    # If we reach here, we failed to find a solution within the time limit or bounds
    # This should not happen for the given problem constraints
    raise RuntimeError("No solution found within time limit")

def search_for_solution(target_count, deadline):
    """
    DFS to find target_count distinct denominators >= 5 summing to 1.
    """
    remaining = Fraction(1, 1)
    last_denom = 4
    path = []
    
    def dfs(depth, remaining, last_denom):
        if time.time() > deadline:
            return None
            
        if remaining == 0:
            if depth == target_count:
                return list(path)
            else:
                return None
                
        if depth == target_count:
            return None
            
        slots = target_count - depth
        
        # Pruning: Max possible sum with 'slots' terms starting from last_denom+1
        # Sum_{i=0}^{slots-1} 1/(last_denom + 1 + i)
        max_possible_sum = Fraction(0, 1)
        for i in range(slots):
            max_possible_sum += Fraction(1, last_denom + 1 + i)
            
        if max_possible_sum < remaining:
            return None
            
        # Determine min_d
        # Need 1/d <= remaining => d >= 1/remaining
        ratio = Fraction(1, 1) / remaining
        # ceil(ratio)
        d_min_exact = (ratio.numerator + ratio.denominator - 1) // ratio.denominator
        
        min_d = max(last_denom + 1, d_min_exact)
        
        d = min_d
        
        while True:
            if time.time() > deadline:
                return None
                
            # Check if current d allows a solution
            # Sum of slots terms starting at d must be >= remaining
            current_max_sum = Fraction(0, 1)
            for i in range(slots):
                current_max_sum += Fraction(1, d + i)
                
            if current_max_sum < remaining:
                break # d is too large
                
            # Check if 1/d <= remaining (should be guaranteed by min_d, but double check)
            if Fraction(1, d) > remaining:
                d += 1
                continue
                
            # Try picking d
            path.append(d)
            new_remaining = remaining - Fraction(1, d)
            
            res = dfs(depth + 1, new_remaining, d)
            if res is not None:
                return res
                
            path.pop()
            d += 1
            
        return None

    return dfs(0, Fraction(1, 1), 4)
