"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import factorial

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill
    
    N = factorial(12)
    MIN_FACTOR = 13  # all factors must be > 12
    MAX_FACTORS = 12
    
    # Get all divisors of N that are >= MIN_FACTOR
    divisors = []
    i = 1
    while i * i <= N:
        if N % i == 0:
            if i >= MIN_FACTOR:
                divisors.append(i)
            other = N // i
            if other != i and other >= MIN_FACTOR:
                divisors.append(other)
        i += 1
    
    divisors.sort()
    divisor_set = set(divisors)
    
    best = [N]  # trivial solution: single factor
    best_max = N
    
    # DFS with pruning
    def dfs(remaining, current_factors, min_allowed):
        nonlocal best, best_max
        
        if time.time() > deadline:
            return
        
        if remaining == 1:
            if current_factors:
                cur_max = max(current_factors)
                if cur_max < best_max:
                    best_max = cur_max
                    best = list(current_factors)
            return
        
        # Lower bound: if we use just one more factor, it must be >= min_allowed
        # and must equal remaining. If remaining < min_allowed, we can't use it.
        # Actually, the next factor must be >= min_allowed and divide remaining.
        
        # Pruning: the largest factor in any solution with current_factors + future
        # is at least max(max(current_factors), remaining) if we use one factor,
        # or could be smaller if we split. But a key bound: 
        # If we have k more factors to place, each >= min_allowed, and they multiply to remaining,
        # the largest is at least remaining^(1/k). But more usefully:
        # The next factor f must satisfy f >= min_allowed and f <= remaining // (min_allowed ** (k-1))
        # for some k >= 1. Since we don't know k, we just iterate possible f.
        
        # Upper bound on the next factor: it must be at most remaining (if it's the last factor)
        # but also, if we want to improve best_max, we need the final max to be < best_max.
        # So the next factor must be < best_max, and all subsequent factors must be < best_max.
        # Also, the product of the remaining factors is 'remaining', and each is < best_max.
        
        # Key pruning: if remaining >= best_max, we could still split it, but the smallest
        # factor we can pick is min_allowed. The largest factor in the final solution will be
        # at least remaining / (product of other factors). 
        # Simpler bound: the next factor f must satisfy f < best_max (otherwise no improvement).
        # Also f must be a divisor of remaining, f >= min_allowed.
        # Additionally, after picking f, the new remaining is remaining // f, and we need to
        # factor it into factors each >= f (to keep distinct and sorted) and each < best_max.
        
        # If remaining // f < min_allowed and f < remaining, that's impossible.
        # More importantly, if we pick f, the max of the final solution is at least max(f, ...).
        # To ensure potential improvement, we need the eventual max < best_max.
        
        # Iterate over possible next factors
        for f in divisors:
            if f < min_allowed:
                continue
            if f > remaining:
                break
            if remaining % f != 0:
                continue
            if f >= best_max:
                # Can't improve since this factor alone is >= best_max
                break
            
            new_remaining = remaining // f
            
            # After picking f, the remaining factors must be >= f (distinct, sorted)
            # and their product is new_remaining.
            # If new_remaining == 1, we're done (handled above).
            # The next minimum allowed is f + 1 (strictly greater for distinctness).
            
            # Pruning: the largest factor in the final solution will be at least
            # new_remaining (if one factor) or could be less. But at minimum,
            # the next factor must be >= f+1, and the product of all remaining
            # factors is new_remaining. 
            # A tight bound: if we use exactly one more factor, it's new_remaining.
            # If new_remaining >= best_max, we can't improve with one factor.
            # With multiple factors, the largest is at least new_remaining^(1/k) but
            # also at least f+1. 
            # Simple bound: the max of the final solution >= max(f, new_remaining^(1/k))
            # but we don't know k. Let's use: the next factor g >= f+1, and g divides
            # new_remaining. If new_remaining // (f+1) < f+1 and new_remaining != 1,
            # we can't place two more factors. But this is complex.
            
            # Key insight: the largest factor in the final solution is at least
            # new_remaining / (f+1)^(k-1) for k factors... too complex.
            
            # Simpler: just recurse and let the recursion handle pruning.
            # But add: if new_remaining >= best_max and new_remaining < (f+1), 
            # that means we'd need a factor >= new_remaining which is >= best_max. 
            # Actually, if new_remaining >= best_max, we can still split it into
            # smaller factors. So we can't prune just on that.
            
            # Prune if the minimum possible max factor from here is >= best_max.
            # The minimum possible max factor when factoring new_remaining into
            # factors >= f+1 is achieved by making them as equal as possible.
            # But a lower bound: the max factor is at least new_remaining / ((f+1) * (f+2) * ...)
            # which is hard. Let's use: the max factor is at least new_remaining^(1/k)
            # for any k, and at least f+1. So max >= max(f+1, new_remaining^(1/k)).
            # For k=1: max >= new_remaining.
            # For k=2: max >= sqrt(new_remaining).
            # So if sqrt(new_remaining) >= best_max, we can prune (since even the best
            # 2-factor split has max >= sqrt(new_remaining) >= best_max).
            # More generally, if new_remaining^(1/k) >= best_max for all k <= MAX_FACTORS - len(current_factors) - 1,
            # we can prune. But this is expensive.
            
            # Simple useful prune: if new_remaining >= best_max * best_max, then even
            # splitting into two factors of ~sqrt(new_remaining) gives max >= sqrt(new_remaining) >= best_max.
            # Actually sqrt(new_remaining) >= best_max iff new_remaining >= best_max^2.
            if new_remaining >= best_max * best_max:
                continue
            
            current_factors.append(f)
            dfs(new_remaining, current_factors, f + 1)
            current_factors.pop()
    
    dfs(N, [], MIN_FACTOR)
    
    return best
# EVOLVE-BLOCK-END
