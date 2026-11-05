import time
from math import factorial
from itertools import combinations_with_replacement

_START = time.time()
_LIMIT = 55.0
_N = 12
_TARGET = factorial(_N)  # 479001600 = 2^10 * 3^5 * 5^2 * 7


def _remaining_time():
    return _LIMIT - (time.time() - _START)


def _is_valid(sol):
    if not (1 <= len(sol) <= _N):
        return False
    if len(set(sol)) != len(sol):
        return False
    if any(x <= _N for x in sol):
        return False
    prod = 1
    for x in sol:
        prod *= x
        if prod > _TARGET:
            return False
    return prod == _TARGET


def construct():
    best_solution = None
    best_largest = float('inf')
    
    def update(parts):
        nonlocal best_solution, best_largest
        if not _is_valid(parts):
            return
        mx = max(parts)
        if mx < best_largest:
            best_largest = mx
            best_solution = sorted(parts)
    
    # 12! = 2^10 * 3^5 * 5^2 * 7
    # The factor containing 7 must be > 12, so 7 * (product of some 2s,3s,5s)
    # 7 alone = 7 <= 12, invalid
    # 7*2=14, 7*3=21, 7*5=35, 7*2*2=28, etc.
    
    # Strategy: enumerate the 7-factor's composition, then solve for remaining factors
    # For remaining: partition 2^a * 3^b * 5^c into distinct factors > 12
    
    def solve_remaining(a, b, c, existing_factors):
        """Partition 2^a * 3^b * 5^c into distinct factors > 12,
        each not in existing_factors, minimizing max factor."""
        if a == 0 and b == 0 and c == 0:
            # No more primes to distribute
            if len(existing_factors) >= 1:
                update(list(existing_factors))
            return
        if _remaining_time() <= 0:
            return
        
        # Try forming a new factor from remaining primes
        # Factor = 2^i * 3^j * 5^k where i<=a, j<=b, k<=c
        # Must be > 12 and not in existing_factors
        # Also must be distinct from all other factors we'll create
        
        # To keep search tractable, we build factors in increasing order
        # and ensure each new factor > previous ones (to avoid permutations)
        # Actually, let's just ensure distinctness and > 12
        
        # Lower bound on remaining: we need at least one more factor > 12
        # The smallest possible new factor using remaining primes
        
        # Enumerate possible (i,j,k) for the next factor
        # To reduce symmetry, require this factor >= min possible
        # Let's enumerate i from 0..a, j from 0..b, k from 0..c
        # But skip (0,0,0)
        
        # Pruning: if we have remaining primes, the new factor must use at least one
        # Also, the remaining after this factor must still be partitionable into factors > 12
        
        for i in range(a + 1):
            for j in range(b + 1):
                for k in range(c + 1):
                    if i == 0 and j == 0 and k == 0:
                        continue
                    val = (1 << i) * (3 ** j) * (5 ** k)
                    if val <= _N:
                        continue
                    if val in existing_factors:
                        continue
                    # Check if remaining can form valid factors
                    rem_a = a - i
                    rem_b = b - j
                    rem_c = c - k
                    # If remaining is 0, we're done with this factor
                    if rem_a == 0 and rem_b == 0 and rem_c == 0:
                        update(list(existing_factors) + [val])
                    else:
                        # Recurse: remaining primes must form at least one factor > 12
                        # Quick check: the product of remaining must be > 12
                        rem_prod = (1 << rem_a) * (3 ** rem_b) * (5 ** rem_c)
                        if rem_prod <= _N:
                            continue
                        # Also need rem_prod to be partitionable into distinct factors > 12
                        # Each such factor > 12, so if we need m factors, rem_prod >= 13^m
                        # But m is variable (1 to 12 - len(existing_factors) - 1)
                        # Simple check: rem_prod > 12
                        solve_remaining(rem_a, rem_b, rem_c, existing_factors + [val])
    
    # Try different compositions for the 7-factor
    # 7 * 2^i * 3^j * 5^k, where the factor > 12
    # i from 0..10, j from 0..5, k from 0..2
    # But 7 * 2^0 * 3^0 * 5^0 = 7 <= 12, skip
    # We want to try balanced solutions
    
    # To find good solutions, let's try several 7-factor options
    # and for each, solve the remaining
    
    # Key insight: to minimize max factor, we want factors to be as equal as possible
    # 12! = 479001600
    # Let's try to get 4-6 factors of similar magnitude
    
    # Enumerate 7-factor compositions
    for i in range(11):  # 2^i
        for j in range(6):  # 3^j
            for k in range(3):  # 5^k
                if i == 0 and j == 0 and k == 0:
                    continue
                f7 = 7 * (1 << i) * (3 ** j) * (5 ** k)
                if f7 <= _N:
                    continue
                # Remaining exponents
                rem_a = 10 - i
                rem_b = 5 - j
                rem_c = 2 - k
                # The 7-factor uses i twos, j threes, k fives
                # Solve for remaining
                solve_remaining(rem_a, rem_b, rem_c, [f7])
                if _remaining_time() <= 0:
                    break
            if _remaining_time() <= 0:
                break
        if _remaining_time() <= 0:
            break
    
    if best_solution is not None:
        return best_solution
    
    # Fallback
    return [factorial(_N)]


if __name__ == "__main__":
    r = construct()
    print(len(r), r, max(r) if r else None)
    print("valid:", _is_valid(r))
