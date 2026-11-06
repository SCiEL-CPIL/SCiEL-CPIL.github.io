"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import factorial, isqrt
from functools import lru_cache

def construct():
    deadline = time.time() + 80  # safety margin before 90s kill
    target = factorial(12)  # 479001600
    
    # Precompute prime factorization: 12! = 2^10 * 3^5 * 5^2 * 7
    # We need distinct factors > 12, product = target, minimize max factor
    
    best_result = None
    best_max = target  # single factor [target]
    
    def check_valid(factors):
        if not factors:
            return False
        if len(factors) > 12:
            return False
        if any(f <= 12 for f in factors):
            return False
        if len(set(factors)) != len(factors):
            return False
        prod = 1
        for f in factors:
            prod *= f
        return prod == target
    
    def search(remaining, min_factor, max_factors, used, current):
        nonlocal best_result, best_max
        if time.time() > deadline:
            return
        
        if remaining == 1:
            if current:
                if max(current) < best_max:
                    best_max = max(current)
                    best_result = list(current)
            return
        
        # We need at least 1 more factor
        # remaining must be composed of factors >= min_factor, all distinct, all > 12
        # The smallest possible next factor is max(min_factor, 13)
        
        if max_factors <= 0:
            return
        
        # If remaining itself is a valid single factor
        if remaining >= min_factor and remaining > 12 and remaining not in used:
            candidate = current + [remaining]
            if max(candidate) < best_max:
                best_max = max(candidate)
                best_result = list(candidate)
            return
        
        # Try to split remaining into two or more factors
        # The next factor must be >= min_factor and <= sqrt(remaining) (if we need at least 2 more)
        # Actually, if we can use 1 factor, we already handled that above.
        # So here we need at least 2 more factors.
        
        if max_factors < 2:
            return
        
        # Next factor f must satisfy:
        # - f >= min_factor
        # - f > 12
        # - f not in used
        # - f <= remaining // (min_factor * ... ) but at minimum f <= sqrt(remaining) if we need 2 factors
        # Actually for 2 factors: f * g = remaining, f <= g, so f <= sqrt(remaining)
        # For more factors, f can be even smaller, but let's bound reasonably
        
        # Upper bound for f: if we use exactly 2 more factors, f <= sqrt(remaining)
        # If we use more, f could be smaller, but let's start with a reasonable range
        # f must divide remaining
        
        # Find divisors of remaining that are >= min_factor
        # To be efficient, iterate f from min_factor to sqrt(remaining)
        
        sqrt_rem = isqrt(remaining)
        # But f could be > sqrt(remaining) if we use more than 2 factors? No, if f > sqrt(remaining),
        # then remaining/f < sqrt(remaining) < f, contradicting that f is the smallest remaining factor.
        # So f <= sqrt(remaining) when we need at least 2 more factors.
        
        start = max(min_factor, 13)
        end = sqrt_rem
        
        for f in range(start, end + 1):
            if time.time() > deadline:
                return
            if remaining % f != 0:
                continue
            if f in used:
                continue
            if f <= 12:
                continue
            
            g = remaining // f
            # g must be >= f (since f is the smallest next factor)
            if g < f:
                continue
            
            # Recurse: now we need to factor g with factors >= f+1 (to keep distinct and sorted)
            # Wait, we need distinct factors, and we've used f, so next must be > f? 
            # Not necessarily > f, just != f and distinct from all used.
            # But to avoid duplicates and maintain order, let's require next factor > f.
            # Actually, the factors don't need to be sorted, but for search efficiency, let's enforce
            # that we pick factors in increasing order. So next factor must be > f.
            
            new_used = used | {f}
            search(g, f + 1, max_factors - 1, new_used, current + [f])
    
    # Try different numbers of factors from 2 to 12
    # For each, do a search
    # Start with more factors (more balanced) to find good solutions early
    
    for num_factors in range(12, 1, -1):
        if time.time() > deadline:
            break
        # Initial: no factors used, min_factor = 13, need to factor target into num_factors distinct parts
        # But we don't know the exact count in advance in the recursive search above.
        # Let me restructure: the search function should try to use exactly some number of factors.
        
        # Actually, let me use a different approach: recursive backtracking that builds up factors
        # in increasing order, and at each step decides whether to stop or continue.
        
        pass
    
    # Better approach: recursive function that builds factors in sorted order
    # State: (remaining_product, min_next_factor, factors_chosen_so_far)
    # At each step, try all valid next factors f >= min_next_factor, f divides remaining,
    # then recurse on remaining/f with min_next_factor = f+1
    
    best_result = None
    best_max = target
    
    def backtrack(remaining, min_f, chosen):
        nonlocal best_result, best_max
        if time.time() > deadline:
            return
        
        if remaining == 1:
            if chosen and len(chosen) >= 1:
                mx = max(chosen)
                if mx < best_max:
                    best_max = mx
                    best_result = list(chosen)
            return
        
        if len(chosen) >= 12:
            return
        
        # Try stopping here: use 'remaining' as the last factor
        if remaining >= min_f and remaining > 12 and remaining not in chosen:
            candidate = chosen + [remaining]
            mx = max(candidate)
            if mx < best_max:
                best_max = mx
                best_result = list(candidate)
        
        # Try continuing: pick a factor f
        # f must be >= min_f, f > 12, f divides remaining, f not in chosen
        # f <= sqrt(remaining) if we need at least 2 more factors
        # If we only need 1 more, that's handled by the "stopping" case above
        
        sqrt_rem = isqrt(remaining)
        start = max(min_f, 13)
        end = sqrt_rem
        
        for f in range(start, end + 1):
            if time.time() > deadline:
                return
            if f > 12 and remaining % f == 0 and f not in chosen:
                backtrack(remaining // f, f + 1, chosen + [f])
    
    backtrack(target, 13, [])
    
    if best_result is None:
        return [target]
    return best_result
# EVOLVE-BLOCK-END
