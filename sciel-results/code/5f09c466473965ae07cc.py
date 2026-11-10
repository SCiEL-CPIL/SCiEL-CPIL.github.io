"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def _get_divisors(n):
    divs = set()
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divs.add(i)
            divs.add(n // i)
    return sorted(divs, reverse=True)

def construct():
    start_time = time.time()
    deadline = start_time + 50  # safety margin before 90s kill
    
    n = 5040
    target = 4979
    divisors = _get_divisors(n)
    # Filter: only divisors <= target (can't use 5040 since 5040 > 4979)
    divisors = [d for d in divisors if d <= target]
    
    # Try increasing k from minimum possible
    # Minimum k: at least 1, but let's start from 2 (1 divisor can't work since no divisor equals 4979)
    # Actually check k=1 first
    for d in divisors:
        if d == target:
            return [d]
    
    # Try k=2
    for a, b in combinations(divisors, 2):
        if a + b == target:
            return [a, b]
        if time.time() > deadline:
            break
    else:
        # Try k=3
        for a, b, c in combinations(divisors, 3):
            if a + b + c == target:
                return [a, b, c]
            if time.time() > deadline:
                break
        else:
            # Try k=4
            for a, b, c, d in combinations(divisors, 4):
                if a + b + c + d == target:
                    return [a, b, c, d]
                if time.time() > deadline:
                    break
            else:
                # Try k=5 (should find quickly)
                for a, b, c, d, e in combinations(divisors, 5):
                    if a + b + c + d + e == target:
                        return [a, b, c, d, e]
                    if time.time() > deadline:
                        break
                else:
                    # Fallback: greedy
                    remainder = target
                    chosen = []
                    for d in divisors:
                        if d <= remainder:
                            chosen.append(d)
                            remainder -= d
                            if remainder == 0:
                                break
                    if remainder == 0:
                        return chosen
                    # Should not reach here
                    return chosen
    return []
# EVOLVE-BLOCK-END
