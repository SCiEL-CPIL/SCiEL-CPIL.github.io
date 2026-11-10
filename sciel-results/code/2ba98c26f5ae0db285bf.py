"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from fractions import Fraction
    from itertools import combinations, combinations_with_replacement
    
    target = Fraction(5, 31)
    
    # We want to find distinct denominators d1 < d2 < ... < dk such that
    # sum(1/di) = 5/31, minimizing max(di) = dk.
    
    # Strategy: Search for the minimum possible largest denominator D.
    # For a given D, we check if there exists a set of distinct denominators
    # all <= D that sum reciprocally to 5/31.
    #
    # A more efficient approach:
    # 1. Use a greedy-like search but with backtracking, trying to keep denominators small.
    # 2. Or, use the fact that for Egyptian fractions, we can bound the number of terms.
    #
    # Let's try a systematic search for the optimal solution.
    # Key insight: The largest denominator in an optimal solution is likely not too large.
    # Let's try to find the minimum D such that 5/31 can be represented as a sum of
    # distinct unit fractions with all denominators <= D.
    
    # First, let's get an upper bound using the current greedy method
    def greedy_fraction(r):
        values = []
        while r:
            d = (r.denominator + r.numerator - 1) // r.numerator
            values.append(d)
            r -= Fraction(1, d)
        return values
    
    greedy_sol = greedy_fraction(target)
    upper_bound = max(greedy_sol)
    
    # Now try to find a better solution by searching for smaller maximum denominators.
    # We'll use a recursive search with pruning.
    
    def search(remaining, min_d, max_d, current_set):
        """
        Try to represent 'remaining' as sum of distinct unit fractions
        with denominators in [min_d, max_d].
        Returns a list of denominators if found, else None.
        """
        if remaining == 0:
            return []
        if remaining < 0:
            return None
        
        # Number of terms needed: at least ceil(remaining * min_d)
        # Since each term is at most 1/min_d
        if remaining * min_d < 1:
            # Even with one term 1/min_d, it's too small? No, 1/min_d could be > remaining
            # Actually, we need sum >= remaining, and each term <= 1/min_d
            # So we need at least ceil(remaining * min_d) terms
            pass
        
        # Upper bound on number of terms: each term >= 1/max_d
        # So we need at most floor(remaining * max_d) terms... wait
        # sum <= k / min_d, so k >= remaining * min_d
        # sum >= k / max_d, so k <= remaining * max_d... no
        # If we use k terms, max sum is k/min_d, min sum is k/max_d
        # We need k/max_d <= remaining <= k/min_d
        # So k >= remaining * min_d and k <= remaining * max_d
        
        # For the next denominator d:
        # remaining - 1/d must be representable with denominators > d and <= max_d
        # Also, we need enough room: remaining - 1/d >= 0
        # And remaining - 1/d <= (max_terms-1) / (d+1) approximately
        
        for d in range(min_d, max_d + 1):
            if 1 / d > remaining + 1e-15:
                continue
            new_remaining = remaining - Fraction(1, d)
            if new_remaining < 0:
                continue
            if new_remaining == 0:
                return [d]
            
            # Pruning: can we represent new_remaining with denominators in [d+1, max_d]?
            # The maximum sum with denominators d+1, ..., max_d is bounded.
            # Simple bound: if new_remaining > sum of 1/i for i in d+1..max_d, skip
            # But computing that is expensive. Use a simpler bound.
            
            # Minimum possible sum with any number of terms from d+1..max_d:
            # We can use as many terms as we want, but they must be distinct.
            # The maximum number of terms is max_d - d.
            # The minimum sum for k terms is k/max_d... no, the smallest unit fractions are 1/max_d, 1/(max_d-1), etc.
            # Actually, to get a small sum, we use large denominators.
            # To check feasibility, we need: new_remaining <= sum_{i=d+1}^{max_d} 1/i
            # And new_remaining >= 1/max_d (if at least one term)
            
            # For efficiency, just try the next level
            result = search(new_remaining, d + 1, max_d, current_set + [d])
            if result is not None:
                return [d] + result
        
        return None
    
    # The search above might be too slow. Let's use a more targeted approach.
    # Since we want to minimize the largest denominator, let's try D from a lower bound upward.
    
    # Lower bound: The largest denominator must be at least ceil(31/5) = 7? No.
    # If we use k terms, the largest denominator D satisfies:
    # 5/31 <= k/7 (if min denom is 7)... this is getting complex.
    
    # Let's try a different approach: use the known structure of Egyptian fractions.
    # For 5/31, let's try small number of terms first.
    
    # Try 2 terms: 1/a + 1/b = 5/31 => 31(a+b) = 5ab => 5ab - 31a - 31b = 0
    # (5a-31)(5b-31) = 31^2 = 961
    # 961 = 1 * 961 = 31 * 31
    # 5a-31 = 1, 5b-31 = 961 => a = 32/5 (not int)
    # 5a-31 = 31, 5b-31 = 31 => a = 62/5 (not int)
    # So no 2-term solution.
    
    # Try 3 terms: 1/a + 1/b + 1/c = 5/31
    # This is more complex. Let's use a systematic search.
    
    # Actually, let's just use a well-optimized greedy with local search.
    # Start with the greedy solution and try to reduce the largest denominator.
    
    # Alternative: Use the approach from the literature on Egyptian fractions.
    # For a fraction p/q, an upper bound on the largest denominator in a k-term expansion
    # is known. For 5/31, let's try to find the optimal.
    
    # Let's do a BFS/DFS search for the minimum maximum denominator.
    # State: (remaining_fraction, min_denominator, current_max_denominator)
    # We want to minimize the final max_denominator.
    
    # Given time constraints, let's use a heuristic:
    # 1. Generate the greedy solution.
    # 2. Try to improve it by replacing the largest term with a combination of smaller terms.
    
    # Actually, let's just implement a clean backtracking search with good pruning.
    
    def solve(remaining, min_d, max_d_allowed):
        """
        Find distinct denominators d1 < d2 < ... < dk with d1 >= min_d, dk <= max_d_allowed
        such that sum(1/di) = remaining.
        Returns list of denominators or None.
        """
        if remaining == 0:
            return []
        if remaining < 0:
            return None
        
        # Pruning: if remaining is too large to be represented with denominators >= min_d
        # Max sum with denominators min_d, min_d+1, ... is bounded by harmonic series tail
        # Simple bound: if remaining > 1/min_d + 1/(min_d+1) + ... but we don't know how many terms
        # Use: remaining * min_d < 1 is not a problem, we can have many terms
        
        # Key pruning: the next denominator d must satisfy 1/d <= remaining
        # So d >= ceil(1/remaining)
        start_d = max(min_d, (1 // remaining.numerator) * remaining.denominator + (1 if 1 % remaining.numerator else 0))
        # More carefully: d >= ceil(remaining.denominator / remaining.numerator)
        if remaining.numerator == 0:
            return []
        start_d = max(min_d, -(-remaining.denominator // remaining.numerator))  # ceil division
        
        if start_d > max_d_allowed:
            return None
        
        for d in range(start_d, max_d_allowed + 1):
            new_rem = remaining - Fraction(1, d)
            if new_rem < 0:
                break  # Since d is increasing, 1/d is decreasing, so new_rem will only get larger... wait
                # Actually as d increases, 1/d decreases, so new_rem = remaining - 1/d increases
                # So if new_rem < 0, it will stay < 0 for larger d? No:
                # remaining - 1/d: as d increases, 1/d decreases, so remaining - 1/d increases
                # So if new_rem < 0 for some d, for larger d, new_rem will be less negative or positive
                # So we should continue, not break
                continue
            
            if new_rem == 0:
                return [d]
            
            # Pruning: new_rem must be representable with denominators > d and <= max_d_allowed
            # Minimum possible value with one term: 1/max_d_allowed
            # If new_rem < 1/max_d_allowed, we need at least two terms
            # Maximum possible value with any number of terms from d+1 to max_d_allowed:
            # This is bounded by the harmonic sum, but for simplicity:
            # If new_rem > sum of 1/i for i=d+1 to max_d_allowed, skip
            # Approximate bound: new_rem > (max_d_allowed - d) / (d + 1) is a loose upper bound
            # Actually, the max sum is sum_{i=d+1}^{max_d_allowed} 1/i < integral from d to max_d_allowed of 1/x dx = ln(max_d_allowed/d)
            
            # For efficiency, let's just recurse
            result = solve(new_rem, d + 1, max_d_allowed)
            if result is not None:
                return [d] + result
        
        return None
    
    # Try increasing max_d_allowed until we find a solution
    # Start from a reasonable lower bound
    # The greedy solution has max denominator:
    greedy_max = max(greedy_sol)
    
    # Let's try from a small value up to greedy_max
    # But we need to be careful about time.
    
    # Actually, let's try a fixed set of candidate denominators and use a smarter search.
    
    # For 5/31, let's try to find the optimal solution by searching up to a reasonable bound.
    # Based on known results, the optimal solution for 5/31 should have a relatively small max denominator.
    
    # Let's try max_d_allowed from 100 up to greedy_max
    for max_d in range(100, greedy_max + 1):
        result = solve(target, 1, max_d)
        if result is not None:
            return result
    
    # If no solution found (shouldn't happen), return the greedy solution
    return greedy_sol

# EVOLVE-BLOCK-END
