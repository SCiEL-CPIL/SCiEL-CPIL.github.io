"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
from fractions import Fraction
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    # Generate primes up to 1000
    def gen_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = False
        return [i for i, is_p in enumerate(sieve) if is_p]
    
    primes = gen_primes(1000)
    MAX_L = 100000
    
    # Use NumPy for fast coverage updates and checks
    covered = np.zeros(MAX_L + 1, dtype=bool)
    used_primes = set()
    classes = []
    budget = Fraction(1, 1)
    
    # Helper to find the first uncovered index >= 1
    def get_first_uncovered():
        if np.all(covered[1:]):
            return MAX_L + 1
        return np.argmax(~covered[1:]) + 1

    while True:
        u = get_first_uncovered()
        if u > MAX_L:
            break
            
        best_new_prefix = u - 1
        best_p = None
        best_r = None
        
        # Candidates: only residues that cover 'u'
        for p in primes:
            if p in used_primes:
                continue
            
            cost = Fraction(1, p)
            if budget - cost < 0:
                continue
                
            r = u % p
            
            # Simulate adding (p, r) to find new prefix
            v = u
            while v <= MAX_L:
                # Find first uncovered >= v in current 'covered'
                sub_arr = ~covered[v:]
                if not np.any(sub_arr):
                    v = MAX_L + 1
                    break
                offset = np.argmax(sub_arr)
                v = v + offset
                
                # v is an uncovered number in current state
                # Check if it is covered by new class (p, r)
                if v % p == r:
                    # It is covered by new class, continue to next uncovered
                    v += 1
                else:
                    # It is NOT covered by new class, so prefix ends at v-1
                    break
            
            new_prefix = v - 1
            
            if new_prefix > best_new_prefix:
                best_new_prefix = new_prefix
                best_p = p
                best_r = r
            # Tie-breaker: if same prefix, prefer smaller p (cheaper)
            elif new_prefix == best_new_prefix and best_p is not None and p < best_p:
                best_p = p
                best_r = r

        if best_p is None:
            break
            
        # Apply best choice
        p = best_p
        r = best_r
        used_primes.add(p)
        classes.append([p, r])
        budget -= Fraction(1, p)
        
        # Update coverage
        start = r if r > 0 else p
        covered[start:MAX_L+1:p] = True

    return classes
# EVOLVE-BLOCK-END
