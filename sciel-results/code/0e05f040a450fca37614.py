"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import factorial, isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin
    
    target = factorial(12)  # 479001600 = 2^10 * 3^5 * 5^2 * 7
    
    best_max = target
    best_factors = [target]
    
    # Quick greedy to get initial solution
    # Try to split into small factors
    def greedy_initial():
        """Try to find a good initial decomposition."""
        # Try using as many factors as possible, each just above 12
        # 13, 14, 15, 16, ... but they must divide 12!
        # 13 doesn't divide 12! (13 is prime > 12)
        # So factors must be composed of primes 2,3,5,7
        # Smallest valid factors: 16=2^4, 18=2*3^2, 20=2^2*5, 21=3*7, 24=2^3*3, 25=5^2, 27=3^3, 28=2^2*7, 30=2*3*5, 32=2^5, 35=5*7, 36=2^2*3^2, 40=2^3*5, 42=2*3*7, 45=3^2*5, 48=2^4*3, 49=7^2, 50=2*5^2, 54=2*3^3, 56=2^3*7, 60=2^2*3*5, 63=3^2*7, 64=2^6, 70=2*5*7, 72=2^3*3^2, 75=3*5^2, 80=2^4*5, 81=3^4, 84=2^2*3*7, 90=2*3^2*5, 96=2^5*3, 98=2*7^2, 100=2^2*5^2, 105=3*5*7, 108=2^2*3^3, 112=2^4*7, 120=2^3*3*5, 126=2*3^2*7, 128=2^7, 135=3^3*5, 140=2^2*5*7, 144=2^4*3^2, 150=2*3*5^2, 160=2^5*5, 162=2*3^4, 168=2^3*3*7, 175=5^2*7, 180=2^2*3^2*5, 192=2^6*3, 196=2^2*7^2, 200=2^3*5^2, 210=2*3*5*7, 216=2^3*3^3, 224=2^5*7, 225=3^2*5^2, 240=2^4*3*5, 252=2^2*3^2*7, 270=2*3^3*5, 280=2^3*5*7, 288=2^5*3^2, 300=2^2*3*5^2, 315=3^2*5*7, 320=2^6*5, 324=2^2*3^4, 336=2^4*3*7, 350=2*5^2*7, 360=2^3*3^2*5, 375=3*5^3 -- no, only 5^2
        # Let me just try a simple approach: repeatedly divide by small valid factors
        
        remaining = target
        factors = []
        # Try to use factors in increasing order
        # Valid small factors (divisors of 12! that are > 12):
        # Let's just try: 16, 18, 20, 21, 24, 25, 27, 28, 30, 32, 35, 36, ...
        
        # Greedy: try to divide by smallest possible factor > 12 that divides remaining
        # and hasn't been used
        used = set()
        while remaining > 1 and len(factors) < 12:
            found = False
            # Try factors from 13 upward
            for d in range(13, min(remaining, 200) + 1):  # limit search
                if remaining % d == 0 and d not in used:
                    # Check d divides 12! (it will if it divides remaining which divides 12!)
                    factors.append(d)
                    used.add(d)
                    remaining //= d
                    found = True
                    break
            if not found:
                break
        
        if remaining == 1:
            return factors
        else:
            # Append remaining
            if remaining not in used:
                factors.append(remaining)
                return factors
            else:
                return None
    
    # Try greedy
    gf = greedy_initial()
    if gf:
        best_max = max(gf)
        best_factors = gf
    
    # Now DFS to improve
    # Precompute: for each possible remaining value, its divisors > 12
    # But remaining can be many values, so compute on the fly
    
    # Better DFS with strong pruning
    def dfs(remaining, min_factor, factors_chosen, count):
        nonlocal best_max, best_factors, deadline
        
        if time.time() > deadline:
            return
        
        if remaining == 1:
            max_f = max(factors_chosen)
            if max_f < best_max:
                best_max = max_f
                best_factors = factors_chosen[:]
            return
        
        if count >= 12:
            return
        
        # Pruning: if remaining >= min_factor and remaining > 12 and remaining not in factors_chosen
        if remaining >= min_factor and remaining > 12:
            if remaining < best_max:
                if remaining not in set(factors_chosen):
                    best_max = remaining
                    best_factors = factors_chosen + [remaining]
                    return  # Can't do better with this branch
        
        # Lower bound on max: if we need to split remaining into (count+1) to 12 factors
        # The max must be at least ceil(remaining^(1/num_factors))
        # But this is a weak bound. Better: if remaining > best_max, we can't use it directly
        # and any factor we pick must be <= best_max
        
        # Find divisors of remaining that are >= min_factor, > 12, <= best_max
        rem_divs = []
        r = remaining
        for i in range(1, isqrt(r) + 1):
            if r % i == 0:
                if i >= min_factor and i > 12 and i <= best_max:
                    rem_divs.append(i)
                j = r // i
                if j != i and j >= min_factor and j > 12 and j <= best_max:
                    rem_divs.append(j)
        
        rem_divs.sort()
        
        # Additional pruning: if no valid divisors, return
        if not rem_divs:
            return
        
        for d in rem_divs:
            if time.time() > deadline:
                return
            
            if d in factors_chosen:
                continue
            
            # Pruning: if d >= best_max, skip (we want to improve)
            # Actually d <= best_max already, but if d == best_max, the max won't improve
            # unless remaining//d < d, which means max is d = best_max, no improvement
            if d >= best_max:
                continue
            
            factors_chosen.append(d)
            dfs(remaining // d, d, factors_chosen, count + 1)
            factors_chosen.pop()
    
    dfs(target, 13, [], 0)
    
    return best_factors

# EVOLVE-BLOCK-END
