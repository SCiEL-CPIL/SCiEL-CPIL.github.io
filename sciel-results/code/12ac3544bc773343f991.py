"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from math import gcd
from itertools import combinations

def _pairwise_coprime(vals):
    """Check if all distinct values in vals are pairwise coprime."""
    distinct = list(set(vals))
    n = len(distinct)
    for i in range(n):
        for j in range(i + 1, n):
            if gcd(distinct[i], distinct[j]) != 1:
                return False
    return True

def _evaluate(A, B):
    """Return (num_distinct_sums, is_valid) for sets A, B."""
    sums = set()
    for a in A:
        for b in B:
            sums.add(a + b)
    distinct = list(sums)
    # Check pairwise coprime
    n = len(distinct)
    for i in range(n):
        for j in range(i + 1, n):
            if gcd(distinct[i], distinct[j]) != 1:
                return 0, False
    return len(distinct), True

def _search(deadline):
    """Search for best (A, B) pair maximizing distinct coprime sums."""
    best_count = 5
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    
    # Precompute coprime pairs for speed
    # We'll do a structured search
    
    # Strategy: iterate over candidate sets and check
    # Use a smarter approach: pick elements to maximize distinct sums while keeping them coprime
    
    # First, let's try a more systematic search
    # The key insight: we want sums that are all coprime to each other
    # So we want sums that are prime or products of distinct primes not shared
    
    # Let's try to find 6 distinct sums that are pairwise coprime
    # The sums range from 2 to 256 (1+1 to 128+128)
    
    # Generate candidate sums that are pairwise coprime
    # For 6 sums, we need 6 numbers in [2, 256] that are pairwise coprime
    # Easy: 6 distinct primes
    
    # Let's try a direct search with pruning
    # For each triple A, find best triple B
    
    range_ = range(1, 129)
    
    # To keep search tractable, use a deadline
    start_time = time.time()
    
    # Try all combinations of 3 from a reduced set first
    # Then expand
    
    # Actually, let's be smarter. The sums a_i + b_j must be pairwise coprime.
    # This is a strong constraint. Let's enumerate possible sum sets.
    
    # For 6 distinct pairwise-coprime sums in [2, 256]:
    # We can use primes. Let's find 6 primes in [2,256] that can be expressed as a+b
    # where a in A (3 distinct), b in B (3 distinct)
    
    # Let's do a structured search:
    # Pick A = {a1, a2, a3}, then for each b in B, the three sums a1+b, a2+b, a3+b
    # must be coprime to all other sums.
    
    # Let's try a randomized + local search approach with deadline
    
    import random
    random.seed(42)
    
    best_count = 5
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    
    # Try systematic search over smaller ranges first
    # Then expand if time permits
    
    # Phase 1: Try all A from [1..60], B from [1..60]
    # That's C(60,3)^2 which is too large. Let's be smarter.
    
    # Phase 1: Try A from [1..40], B from [1..40]
    # C(40,3) = 9880, so 9880^2 ~ 97M - too large
    
    # Let's do a smarter search:
    # For each A, find best B
    
    # Actually, let's just do a targeted search for 6+ sums
    # Key: all 9 sums (or fewer if some coincide) must be pairwise coprime
    # For 6 distinct sums, we need at least 6 of the 9 sums to be distinct and pairwise coprime
    
    # Let's try: pick 6 pairwise coprime target sums, then see if they can be
    # formed as a_i + b_j for some 3x3 grid
    
    # Simpler: just do a bounded exhaustive search with early termination
    
    # Let's search A in [1..50], B in [1..50] but with smart pruning
    
    # For a given A, the valid B elements must make all sums coprime to each other
    # and to the existing sums
    
    # Let's just do a randomized search with local improvement
    
    def try_random():
        A = random.sample(range(1, 129), 3)
        B = random.sample(range(1, 129), 3)
        count, valid = _evaluate(A, B)
        return count, valid, A, B
    
    # Try many random samples
    iterations = 0
    while time.time() < deadline - 5:
        iterations += 1
        A = random.sample(range(1, 129), 3)
        B = random.sample(range(1, 129), 3)
        count, valid = _evaluate(A, B)
        if valid and count > best_count:
            best_count = count
            best_A = A[:]
            best_B = B[:]
            print(f"Found {count} sums: A={A}, B={B}")
            if best_count >= 9:
                break
        
        # Local search: perturb one element
        if iterations % 10 == 0 and best_count >= 5:
            for _ in range(50):
                A2 = best_A[:]
                B2 = best_B[:]
                # Perturb one element in A or B
                if random.random() < 0.5:
                    idx = random.randint(0, 2)
                    A2[idx] = random.randint(1, 128)
                    if len(set(A2)) < 3:
                        continue
                else:
                    idx = random.randint(0, 2)
                    B2[idx] = random.randint(1, 128)
                    if len(set(B2)) < 3:
                        continue
                count2, valid2 = _evaluate(A2, B2)
                if valid2 and count2 > best_count:
                    best_count = count2
                    best_A = A2[:]
                    best_B = B2[:]
                    print(f"Local found {count2} sums: A={A2}, B={B2}")
                    if best_count >= 9:
                        return best_A, best_B
    
    return best_A, best_B

def construct():
    deadline = time.time() + 80  # 80s deadline, leaving 10s margin
    A, B = _search(deadline)
    # Final validation
    count, valid = _evaluate(A, B)
    if not valid:
        # Fallback to seed
        return [1, 7, 13], [4, 10, 16]
    return A, B
