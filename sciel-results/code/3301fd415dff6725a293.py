"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from fractions import Fraction
import time
import heapq

def construct():
    deadline = time.time() + 80  # 80s wall-clock guard (90s limit, 10s margin)
    
    def greedy_egyptian(target_frac, min_denom, max_denom=10000000):
        """Greedy Egyptian fraction with strictly increasing denominators >= min_denom."""
        result = []
        remaining = target_frac
        last_denom = min_denom - 1
        
        while remaining > 0:
            if time.time() > deadline:
                return None
            
            if remaining.numerator == 0:
                break
                
            # d >= ceil(1/remaining) = ceil(remaining.denominator / remaining.numerator)
            lower = (remaining.denominator + remaining.numerator - 1) // remaining.numerator
            min_d = max(last_denom + 1, min_denom)
            d = max(lower, min_d)
            
            if d > max_denom:
                return None
            
            result.append(d)
            remaining = remaining - Fraction(1, d)
            last_denom = d
            
            if len(result) > 256:
                return None
        
        if remaining == 0:
            return result
        return None
    
    def beam_search(max_terms=12, beam_width=200, max_denom=10000000):
        """Beam search to find a short Egyptian fraction representation of 1."""
        # State: (remaining_fraction, last_denom, list_of_denominators)
        # We want to find a path where remaining becomes 0
        
        # Start: remaining = 1, last_denom = 4, denoms = []
        # At each step, try denominators d >= last_denom+1, d >= 5, d <= max_denom
        # such that 1/d <= remaining
        
        # Use a priority queue: prioritize states with fewer terms used
        # and smaller remaining (closer to 0)
        
        best_solution = None
        best_count = max_terms + 1
        
        # Initial state
        # remaining = 1, last_denom = 4
        # We need d >= 5 and 1/d <= 1, so d >= 1, always true
        # But we also need remaining - 1/d >= 0, i.e., 1/d <= remaining
        
        # For efficiency, limit the range of d we try at each step
        # d must satisfy: 1/d <= remaining => d >= ceil(1/remaining)
        # Also d >= last_denom + 1 and d >= 5
        
        # Additionally, to keep the search tractable, we can limit d to a reasonable range
        # At each step, trying d from lower_bound to some upper_bound
        
        # Key insight: if we have k terms left to use, and remaining = r,
        # then we need k * (1/(last_denom+1)) >= r (roughly)
        # More precisely, the maximum sum of k unit fractions with denominators > last_denom
        # is approximately k/(last_denom+1) (if all were 1/(last_denom+1))
        # But they must be distinct and increasing, so the max is sum of 1/(last_denom+i) for i=1..k
        
        # Let's use a simpler pruning: at each step, try d in [lower, upper] where:
        # lower = max(last_denom + 1, 5, ceil(1/remaining))
        # upper = some reasonable bound, e.g., last_denom + 1000 or based on remaining
        
        # Actually, for a beam search, we should try a range of d values and keep
        # the best beam_width states
        
        # Let's try a different approach: iterative deepening with pruning
        # Try to find a solution with exactly k terms, for k = 5, 6, 7, ...
        
        for target_count in range(5, max_terms + 1):
            if time.time() > deadline:
                break
            
            if best_count <= target_count:
                break
            
            # Try to find a representation with exactly target_count terms
            # Using DFS with pruning
            found = _dfs_exactly(Fraction(1, 1), 4, target_count, 0, [], max_denom, deadline)
            if found is not None:
                best_solution = found
                best_count = target_count
                break
        
        return best_solution
    
    def _dfs_exactly(remaining, last_denom, terms_left, depth, current, max_denom, deadline):
        """DFS to find exactly terms_left more unit fractions summing to remaining."""
        if time.time() > deadline:
            return None
        
        if remaining.numerator == 0:
            if terms_left == 0:
                return current[:]
            return None
        
        if terms_left == 0:
            return None
        
        # Pruning: the maximum possible sum with terms_left terms, each >= 1/(last_denom+1)
        # Actually, the denominators must be > last_denom, so the largest unit fraction is 1/(last_denom+1)
        # The maximum sum is terms_left / (last_denom + 1) (upper bound, not tight)
        # More tightly: sum of 1/(last_denom+1) + 1/(last_denom+2) + ... + 1/(last_denom+terms_left)
        # But computing this exactly is expensive. Use a loose bound:
        # max_sum = terms_left / (last_denom + 1)
        # If remaining > max_sum, prune
        max_sum = Fraction(terms_left, last_denom + 1)
        if remaining > max_sum:
            return None
        
        # The minimum possible sum with terms_left terms:
        # Each denominator >= last_denom + 1, and they must be distinct and increasing
        # The smallest sum is 1/(last_denom+1) + 1/(last_denom+2) + ... 
        # But we can use a looser bound: min_sum = terms_left / (last_denom + terms_left) 
        # (if all denominators were last_denom + terms_left)
        # Actually, the denominators are at least last_denom+1, last_denom+2, ..., last_denom+terms_left
        # So min_sum = sum_{i=1}^{terms_left} 1/(last_denom + i)
        # This is a lower bound on what we can achieve
        # If remaining < min_sum, prune
        # But computing this sum exactly each time is expensive. Use an approximation:
        # min_sum >= terms_left / (last_denom + terms_left)  (harmonic lower bound)
        # Actually 1/(last_denom+i) >= 1/(last_denom+terms_left) for i <= terms_left
        # So min_sum >= terms_left / (last_denom + terms_left)
        min_sum = Fraction(terms_left, last_denom + terms_left)
        if remaining < min_sum:
            return None
        
        # Determine range of d to try
        # d >= last_denom + 1, d >= 5
        # 1/d <= remaining => d >= ceil(1/remaining)
        lower_d = max(last_denom + 1, 5)
        if remaining.numerator > 0:
            ceil_inv = (remaining.denominator + remaining.numerator - 1) // remaining.numerator
            lower_d = max(lower_d, ceil_inv)
        
        # Upper bound on d:
        # We need to be able to represent remaining - 1/d with (terms_left - 1) terms
        # The maximum sum of (terms_left-1) terms with denominators > d is:
        # (terms_left - 1) / (d + 1)
        # So we need: remaining - 1/d <= (terms_left - 1) / (d + 1)
        # remaining <= 1/d + (terms_left - 1)/(d+1)
        # This gives an upper bound on d
        
        # Also, 1/d must be positive, so d > 0 (always true)
        # And we need remaining - 1/d > 0 (since we need more terms), so 1/d < remaining
        # => d > 1/remaining
        # This is already captured by lower_d
        
        # For the upper bound, let's use:
        # If terms_left == 1, then d must satisfy 1/d == remaining, so d = 1/remaining
        # If terms_left > 1, we need d such that remaining - 1/d can be represented
        # with (terms_left - 1) terms, each with denominator > d
        # The maximum such sum is (terms_left - 1) / (d + 1)
        # So: remaining - 1/d <= (terms_left - 1) / (d + 1)
        # remaining <= 1/d + (terms_left - 1)/(d + 1)
        # Let's solve for d: this is a constraint that d cannot be too large
        
        # For simplicity, let's set an upper bound:
        # If remaining is small, d can be large. But we also have the constraint d <= max_denom
        # Let's try d from lower_d to min(lower_d + 5000, max_denom) initially
        # and expand if needed
        
        # Actually, for the first few terms, d won't be that large. Let's use a reasonable range.
        # The key constraint is that after choosing d, the remaining must be representable.
        
        # Let's try a limited range first
        upper_d = min(lower_d + 2000, max_denom)
        
        # Special case: if terms_left == 1, d must be exactly 1/remaining
        if terms_left == 1:
            if remaining.denominator % remaining.numerator == 0:
                d = remaining.denominator // remaining.numerator
                if d > last_denom and d >= 5 and d <= max_denom:
                    return current + [d]
            return None
        
        # Try d values
        # To keep the search tractable, we'll try d in a range
        # and use the pruning conditions above
        
        # For efficiency, let's try d in steps. Start with a moderate range.
        # We can also try some heuristic choices first.
        
        # Heuristic: try d = lower_d first (greedy choice), then nearby values
        d_start = lower_d
        d_end = upper_d
        
        # Limit the number of d values we try to keep it tractable
        # If the range is too large, sample it
        max_d_tries = 200
        if d_end - d_start + 1 > max_d_tries:
            # Sample d values
            step = (d_end - d_start + 1) // max_d_tries
            d_values = list(range(d_start, d_end + 1, step))
            if d_values[-1] != d_end:
                d_values.append(d_end)
        else:
            d_values = range(d_start, d_end + 1)
        
        for d in d_values:
            if time.time() > deadline:
                return None
            
            if d <= last_denom or d < 5 or d > max_denom:
                continue
            
            # Check 1/d <= remaining
            if remaining.numerator * d < remaining.denominator:
                continue
            
            new_remaining = remaining - Fraction(1, d)
            
            if new_remaining < 0:
                continue
            
            if new_remaining == 0:
                if terms_left == 1:
                    return current + [d]
                # Can't have more terms left if remaining is 0
                continue
            
            result = _dfs_exactly(new_remaining, d, terms_left - 1, depth + 1, current + [d], max_denom, deadline)
            if result is not None:
                return result
        
        return None
    
    # First, try the greedy approach as a baseline
    solution = greedy_egyptian(Fraction(1, 1), 5)
    
    if solution is not None:
        greedy_len = len(solution)
        # Try to find a shorter solution
        better = beam_search(max_terms=min(greedy_len - 1, 12), beam_width=200)
        if better is not None:
            return better
        return solution
    
    # Fallback: try the DP approach
    target, reached = 2520, {0: []}
    for d in range(5, target + 1):
        if target % d:
            continue
        w = target // d
        for total, values in list(reached.items()):
            if total + w <= target and total + w not in reached:
                reached[total + w] = values + [d]
        if target in reached:
            return reached[target]
    
    raise RuntimeError("no baseline representation")
# EVOLVE-BLOCK-END
