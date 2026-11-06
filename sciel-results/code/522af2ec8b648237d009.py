"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin

    # Known solutions with small denominators - verify and return immediately
    known_solutions = [
        [5, 6, 7, 42, 140, 210, 420, 840, 1260, 2520],
        [5, 7, 10, 35, 70, 140, 210, 420, 840, 1680],
        [6, 7, 10, 15, 21, 35, 42, 70, 105, 210],
    ]
    
    for sol in known_solutions:
        s = sum(Fraction(1, d) for d in sol)
        if s == 1 and len(set(sol)) == len(sol) and all(5 <= d <= 10000000 for d in sol):
            return sol

    # Greedy Egyptian fraction approach: take largest possible fraction first
    def greedy_egyptian(target, min_d, max_d, max_terms):
        """Find fewest terms to sum to target using denominators in [min_d, max_d]."""
        if target == 0:
            return []
        if max_terms == 0:
            return None
        
        # d must satisfy: 1/d <= target => d >= 1/target
        # and we want to minimize count, so try smallest d first (largest fraction)
        d_min = max(min_d, -(-target.denominator // target.numerator))
        # d must also satisfy: remaining (max_terms-1) terms can sum to target - 1/d
        # i.e., target - 1/d >= (max_terms-1)/max_d
        # => 1/d <= target - (max_terms-1)/max_d
        # => d >= 1/(target - (max_terms-1)/max_d) if target > (max_terms-1)/max_d
        
        for d in range(d_min, max_d + 1):
            if time.time() > deadline:
                return None
            frac = Fraction(1, d)
            if frac > target:
                continue
            new_target = target - frac
            if new_target == 0:
                return [d]
            if new_target < 0:
                break
            
            # Check if remaining can be achieved with (max_terms-1) terms
            # Max possible sum with (max_terms-1) terms starting from d+1
            # is at most (max_terms-1)/(d+1) if d+1 < max_d
            # Min possible is (max_terms-1)/max_d
            
            # Simple pruning: can (max_terms-1) terms starting from d+1 sum to new_target?
            # Upper bound: (max_terms-1)/(d+1) >= new_target
            # Lower bound: (max_terms-1)/max_d <= new_target
            if (max_terms - 1) / max_d > new_target:
                break
            # If even the max possible sum is less than new_target, we need smaller d
            # But we're iterating d upward, so this check helps us continue
            
            result = greedy_egyptian(new_target, d + 1, max_d, max_terms - 1)
            if result is not None:
                return [d] + result
        
        return None

    # Try increasing number of terms starting from a reasonable minimum
    # We know 10 terms work from known solutions, try to find fewer
    for k in range(3, 11):
        if time.time() > deadline:
            break
        result = greedy_egyptian(Fraction(1, 1), 5, 10000000, k)
        if result is not None:
            # Verify
            s = sum(Fraction(1, d) for d in result)
            if s == 1 and len(set(result)) == len(result) and all(5 <= d <= 10000000 for d in result):
                return result

    # Fallback: use a known representation
    for sol in known_solutions:
        s = sum(Fraction(1, d) for d in sol)
        if s == 1 and len(set(sol)) == len(sol) and all(5 <= d <= 10000000 for d in sol):
            return sol

    # Final fallback: DP approach
    target = 2520
    reached = {0: []}
    for d in range(5, target + 1):
        if target % d:
            continue
        w = target // d
        for total, values in list(reached.items()):
            if total + w <= target and total + w not in reached:
                reached[total + w] = values + [d]
        if target in reached:
            return reached[target]
    
    raise RuntimeError("no solution found")
# EVOLVE-BLOCK-END
