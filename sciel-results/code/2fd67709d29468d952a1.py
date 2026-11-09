"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
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

def _compute_coverage_fast(classes, cap=100000):
    """Compute the largest L such that every integer 1..L is covered using bitsets."""
    if not classes:
        return 0
    
    # Use a bytearray for coverage - faster than list of bools for large caps
    # Actually, for cap=100000, let's use a different approach
    # Mark coverage by iterating through each class
    
    # For efficiency, we'll use a set of covered numbers or a boolean array
    # But the key optimization: we only need to find the FIRST gap
    # So we can early-terminate
    
    # Strategy: for each position n from 1 upward, check if n is covered
    # n is covered if exists (p,r) in classes such that n % p == r
    
    # Precompute: for each n, which primes could cover it
    # This is expensive. Alternative: iterate through classes and mark
    
    # Fast approach: create a boolean array, mark all covered positions
    covered = bytearray(cap + 2)  # 0 = uncovered, 1 = covered
    
    for p, r in classes:
        # Numbers n where n % p == r, starting from max(1, r if r>=1 else p)
        if r == 0:
            start = p
        else:
            start = r
        if start < 1:
            start = 1
        for n in range(start, cap + 1, p):
            covered[n] = 1
    
    # Find first gap
    L = 0
    for n in range(1, cap + 1):
        if covered[n]:
            L = n
        else:
            break
    return L

def _compute_coverage_bitset(classes, cap=100000):
    """Compute coverage using bitset for speed. Returns largest consecutive L from 1."""
    if not classes:
        return 0
    
    # For cap up to 100000, use a bytearray approach but optimized
    # Key insight: we can stop early when we find the first gap
    
    # Build coverage mask using integer bit operations for small L
    # For large L, use bytearray
    
    # Hybrid: first check if L is small enough for bitset
    # For L <= 2000, use integer bitset (very fast)
    # For larger, use bytearray
    
    if cap <= 2000:
        # Integer bitset: bit i represents number (i+1)
        mask = 0
        for p, r in classes:
            if r == 0:
                start = p
            else:
                start = r
            if start < 1:
                start = 1
            for n in range(start, cap + 1, p):
                mask |= (1 << (n - 1))
        
        # Find first zero bit
        L = 0
        for n in range(1, cap + 1):
            if mask & (1 << (n - 1)):
                L = n
            else:
                break
        return L
    else:
        # Use bytearray for larger caps
        covered = bytearray(cap + 2)
        for p, r in classes:
            if r == 0:
                start = p
            else:
                start = r
            if start < 1:
                start = 1
            for n in range(start, cap + 1, p):
                covered[n] = 1
        
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
    
    # Strategy: Start with primes >= 13, then add small primes to fill gaps
    # Then do local search on residues
    
    # Phase 1: Initial construction
    # Use primes from 13 upward as the base
    init_primes = [p for p in primes if p >= 13]
    
    # Assign residues greedily: for each prime, pick residue that covers the most new numbers
    classes = []
    used_primes = set()
    
    # Greedy assignment: for each prime, choose residue to maximize coverage of 1..L
    # Start with a reasonable L estimate
    # Sum of reciprocals of primes >= 13 up to 1000
    base_sum = sum(Fraction(1, p) for p in init_primes)
    
    # For each prime, pick the residue that best covers consecutive integers from 1
    # Simple heuristic: residue = 1 for all (covers 1, 1+p, 1+2p, ...)
    # Better: try to cover gaps
    
    # Initialize with residue 1 for all
    for p in init_primes:
        classes.append([p, 1])
    
    # Compute initial coverage
    best_L = _compute_coverage_bitset(classes, 100000)
    best_classes = classes[:]
    best_sum = _sum_reciprocals(classes)
    
    # Phase 2: Add small primes to fill gaps
    small_primes = [p for p in primes if p < 13]  # [2, 3, 5, 7, 11]
    small_primes.sort()  # Add in order: 2, 3, 5, 7, 11
    
    for sp in small_primes:
        if time.time() > deadline - 5:
            break
        
        # Try to add this small prime with optimal residue
        # The optimal residue should cover the first gap
        gap = best_L + 1
        if gap > 100000:
            break
        
        # Try all residues for this small prime
        best_r = None
        best_new_L = best_L
        
        for r in range(sp):
            new_classes = best_classes + [[sp, r]]
            new_sum = best_sum + Fraction(1, sp)
            if new_sum > Fraction(1):
                continue
            new_L = _compute_coverage_bitset(new_classes, 100000)
            if new_L > best_new_L:
                best_new_L = new_L
                best_r = r
        
        if best_r is not None and best_new_L > best_L:
            best_classes.append([sp, best_r])
            best_L = best_new_L
            best_sum = best_sum + Fraction(1, sp)
    
    # Phase 3: Local search - optimize residues of all primes
    # For each prime, try all residues and keep the best
    improved = True
    max_iterations = 50
    iteration = 0
    
    while improved and iteration < max_iterations and time.time() < deadline - 5:
        improved = False
        iteration += 1
        
        for idx in range(len(best_classes)):
            if time.time() > deadline - 5:
                break
            
            p, old_r = best_classes[idx]
            
            # Try all residues for this prime
            best_r = old_r
            best_new_L = best_L
            
            for r in range(p):
                if r == old_r:
                    continue
                new_classes = best_classes[:]
                new_classes[idx] = [p, r]
                new_L = _compute_coverage_bitset(new_classes, 100000)
                if new_L > best_new_L:
                    best_new_L = new_L
                    best_r = r
            
            if best_r != old_r and best_new_L > best_L:
                best_classes[idx] = [p, best_r]
                best_L = best_new_L
                improved = True
    
    # Phase 4: Simulated annealing for further improvement
    if time.time() < deadline - 10:
        current_L = best_L
        current_classes = best_classes[:]
        current_sum = best_sum
        
        temp = 50.0
        cooling = 0.95
        
        while time.time() < deadline - 5 and temp > 1.0:
            # Random move: change one residue
            idx = random.randint(0, len(current_classes) - 1)
            p, old_r = current_classes[idx]
            new_r = random.randint(0, p - 1)
            
            if new_r == old_r:
                continue
            
            new_classes = current_classes[:]
            new_classes[idx] = [p, new_r]
            new_L = _compute_coverage_bitset(new_classes, 100000)
            
            delta = new_L - current_L
            
            if delta > 0 or random.random() < pow(2.718281828, delta / temp):
                current_classes = new_classes
                current_L = new_L
                if current_L > best_L:
                    best_L = current_L
                    best_classes = current_classes[:]
                    best_sum = current_sum
            
            temp *= cooling
    
    return best_classes
# EVOLVE-BLOCK-END
