"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    
    # Candidate pool: numbers 1 to 512
    # We'll use a greedy approach to build a large compatible set
    
    # Strategy: Start with all numbers, iteratively remove conflicts
    # A number n is "bad" if there exists m in set with lcm(n,m) > 512
    
    # Better: use a priority-based greedy construction
    # Sort numbers by how "restrictive" they are (numbers with large prime factors are more restrictive)
    
    # Key insight: If we only include numbers that divide some "smooth" base, 
    # or numbers whose prime factors are small, we get good compatibility.
    
    # Let's try: include numbers n where for every prime p dividing n, 
    # the power of p in n is limited such that lcm stays bounded.
    
    # Simpler approach: greedy set construction
    # Sort candidates by value (smaller numbers tend to be more compatible)
    # For each candidate in order, add it if it's compatible with all current members
    
    candidates = list(range(1, 513))
    
    # Sort by a heuristic: prefer numbers with smaller max prime factor, 
    # then smaller value. Numbers that are "smoother" are more compatible.
    def max_prime_factor(n):
        if n <= 1:
            return 1
        f = 2
        while f * f <= n:
            while n % f == 0:
                n //= f
            f += 1
        if n > 1:
            return max(f, n) if f * f > n else max(f, n)
        return f
    
    # Actually, let's just try a simple greedy: sort by value, add if compatible
    # But this might not be optimal. Let's try multiple orderings and pick the best.
    
    best_set = set()
    best_size = 0
    
    # Ordering 1: ascending
    for order_candidates in [
        sorted(candidates),  # ascending
        sorted(candidates, key=lambda x: (max_prime_factor(x), x)),  # by max prime factor
    ]:
        if time.time() > deadline:
            break
        current = []
        for n in order_candidates:
            # Check if n is compatible with all in current
            compatible = True
            for m in current:
                if _lcm(n, m) > 512:
                    compatible = False
                    break
            if compatible:
                current.append(n)
        
        if len(current) > best_size:
            best_size = len(current)
            best_set = set(current)
    
    # Now try to improve: try adding back numbers we didn't include
    # and removing conflicting ones
    remaining = [n for n in candidates if n not in best_set]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_list = list(best_set)
        for n in remaining:
            if time.time() > deadline:
                break
            # Check if adding n requires removing fewer elements than we gain
            conflicts = [m for m in current_list if _lcm(n, m) > 512]
            if len(conflicts) == 0:
                # Can add n directly
                best_set.add(n)
                remaining.remove(n)
                improved = True
            elif len(conflicts) < 1:
                # Would need to remove at least 1 to add 1, not worth it unless we can chain
                pass
    
    # Final attempt: try all remaining numbers again after improvements
    remaining = [n for n in candidates if n not in best_set]
    for n in remaining:
        if time.time() > deadline:
            break
        compatible = True
        for m in best_set:
            if _lcm(n, m) > 512:
                compatible = False
                break
        if compatible:
            best_set.add(n)
    
    # Return as a list
    result = sorted(best_set)
    if not result:
        result = [1]
    return result
# EVOLVE-BLOCK-END
