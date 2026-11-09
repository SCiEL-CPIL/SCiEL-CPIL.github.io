"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def _get_primes(limit=1000):
    """Return list of primes up to limit."""
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [i for i in range(2, limit + 1) if sieve[i]]

def _compute_coverage(classes, cap=100000):
    """Compute the largest L such that every integer 1..L is covered."""
    if not classes:
        return 0
    # For each integer n, check if it's in any class
    # Efficient: for each class [p, r], mark numbers r, r+p, r+2p, ...
    # But we only need consecutive coverage from 1
    covered = [False] * (cap + 2)
    for p, r in classes:
        # r is the residue, so numbers n where n % p == r are covered
        # First number >= 1 with n % p == r:
        # If r == 0, first is p (since 0 is not in 1..cap, next is p)
        # If r >= 1, first is r
        start = r if r >= 1 else p
        for n in range(start, cap + 1, p):
            covered[n] = True
    # Find largest L such that 1..L all covered
    L = 0
    for n in range(1, cap + 1):
        if covered[n]:
            L = n
        else:
            break
    return L

def _sum_reciprocals(classes):
    """Compute sum of 1/p as Fraction."""
    s = Fraction(0)
    for p, r in classes:
        s += Fraction(1, p)
    return s

def construct():
    deadline = time.time() + 115  # 115s safety margin for 120s limit
    primes = _get_primes(1000)
    prime_set = set(primes)
    
    # Initial solution: use primes from 13 upward, cover consecutive integers
    init_primes = [p for p in primes if p >= 13]
    classes = [[p, (i + 1) % p] for i, p in enumerate(init_primes)]
    
    # Compute initial coverage
    best_L = _compute_coverage(classes)
    best_classes = classes[:]
    best_sum = _sum_reciprocals(classes)
    
    # Local search: try to improve by swapping or adding classes
    # Strategy: for each position in the coverage, try to find a better assignment
    # We'll do iterative improvement
    
    for iteration in range(200):
        if time.time() > deadline:
            break
        
        # Try to extend coverage: find the first uncovered integer after best_L
        # and see if we can cover it by modifying classes
        # Simpler: try random swaps and additions
        
        # Try adding a new class if budget allows
        current_sum = best_sum
        if current_sum < Fraction(1):
            # Try adding a prime not in current classes
            used_primes = set(p for p, r in best_classes)
            candidate_primes = [p for p in primes if p not in used_primes]
            if candidate_primes:
                # Pick a random unused prime and try to cover the gap
                gap = best_L + 1
                # Find best residue for this prime to cover gap and beyond
                for p in candidate_primes[:20]:  # limit candidates
                    r = gap % p
                    new_sum = current_sum + Fraction(1, p)
                    if new_sum <= Fraction(1):
                        new_classes = best_classes[:] + [[p, r]]
                        new_L = _compute_coverage(new_classes)
                        if new_L > best_L:
                            best_L = new_L
                            best_classes = new_classes
                            best_sum = new_sum
                            break
        
        # Try swapping: replace one class with another
        if len(best_classes) > 0:
            # Try replacing a class with a different prime/residue
            for idx in range(min(len(best_classes), 30)):
                if time.time() > deadline:
                    break
                old_p, old_r = best_classes[idx]
                # Try different residues for the same prime
                for r in range(old_p):
                    if r == old_r:
                        continue
                    new_classes = best_classes[:]
                    new_classes[idx] = [old_p, r]
                    new_L = _compute_coverage(new_classes)
                    if new_L > best_L:
                        best_L = new_L
                        best_classes = new_classes
                        best_sum = _sum_reciprocals(best_classes)
                        break
                # Try replacing with a different prime
                used_primes = set(p for p, r in best_classes)
                for p in primes:
                    if p in used_primes:
                        continue
                    new_sum = best_sum - Fraction(1, old_p) + Fraction(1, p)
                    if new_sum > Fraction(1):
                        continue
                    # Choose residue to cover best_L + 1
                    gap = best_L + 1
                    r = gap % p
                    new_classes = best_classes[:]
                    new_classes[idx] = [p, r]
                    new_L = _compute_coverage(new_classes)
                    if new_L > best_L:
                        best_L = new_L
                        best_classes = new_classes
                        best_sum = new_sum
                        break
    
    return best_classes
# EVOLVE-BLOCK-END
