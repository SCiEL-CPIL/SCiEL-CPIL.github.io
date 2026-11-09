"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from fractions import Fraction

# EVOLVE-BLOCK-START
def _primes_upto(n):
    """Sieve of Eratosthenes for primes up to n."""
    if n < 2:
        return []
    sieve = bytearray(b'\x01') * (n + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(2, n + 1) if sieve[i]]

def _compute_L(classes, cap=100000):
    """Compute the largest L such that 1..L are all covered using bitset."""
    if not classes:
        return 0
    pr_pairs = [(p, r) for p, r in classes]
    
    # Build coverage array
    covered = bytearray(cap + 1)
    for p, r in pr_pairs:
        start = r if r >= 1 else p
        if start > cap:
            continue
        for n in range(start, cap + 1, p):
            covered[n] = 1
    
    L = 0
    for n in range(1, cap + 1):
        if covered[n]:
            L = n
        else:
            break
    return L

def _compute_L_fast(classes, cap=100000):
    """Faster L computation using bitset."""
    return _compute_L(classes, cap)

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s kill
    
    all_primes = _primes_upto(1000)
    primes_desc = sorted(all_primes, reverse=True)
    
    # Start with a better baseline: use small primes with optimized residues
    # Try multiple starting strategies and pick the best
    all_strategies = []
    
    # Strategy 1: Primes from 2 upward
    primes_from_2 = [p for p in range(2, 1001) if p in set(all_primes)]
    classes_s1 = [[p, 0] for p in primes_from_2]
    # Check budget
    cost_s1 = sum(Fraction(1, p) for p in primes_from_2)
    if cost_s1 <= Fraction(1, 1):
        all_strategies.append(classes_s1)
    
    # Strategy 2: Primes from 13 upward (original)
    baseline_primes = [p for p in range(13, 1001) if p in set(all_primes)]
    baseline_classes = [[p, (i + 1) % p] for i, p in enumerate(baseline_primes)]
    cost_s2 = sum(Fraction(1, p) for p in baseline_primes)
    if cost_s2 <= Fraction(1, 1):
        all_strategies.append(baseline_classes)
    
    # Strategy 3: Mix of small and medium primes
    # Use primes: 2,3,5,7,11,13,17,19,23,29,31,37,41,43,47
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    classes_s3 = [[p, 0] for p in small_primes]
    cost_s3 = sum(Fraction(1, p) for p in small_primes)
    if cost_s3 <= Fraction(1, 1):
        all_strategies.append(classes_s3)
    
    # Pick the best starting strategy
    best_classes = []
    best_L = 0
    best_cost = Fraction(0)
    
    for strategy in all_strategies:
        L_test = _compute_L(strategy)
        cost_test = sum(Fraction(1, p) for p, r in strategy)
        if L_test > best_L and cost_test <= Fraction(1, 1):
            best_L = L_test
            best_classes = strategy
            best_cost = cost_test
    
    used_primes = set(p for p, r in best_classes)
    
    # Greedy: try to extend coverage by adding classes
    # For each unused prime, try to find a residue that extends L the most
    for p in primes_desc:
        if time.time() > deadline:
            break
        if p in used_primes:
            continue
        
        current_L = best_L
        best_r = None
        best_new_L = current_L
        
        # Try residues that cover current_L + 1 first (most likely to help)
        # The residue that covers current_L+1 is r = (current_L+1) % p
        r_target = (current_L + 1) % p
        
        # Try r_target and more residues for better coverage
        residues_to_try = [r_target]
        for dr in range(1, min(20, p)):
            residues_to_try.append((r_target + dr) % p)
            residues_to_try.append((r_target - dr) % p)
        # Also try some "round" residues
        for r_extra in [0, 1, 2, 3, 4, 5]:
            if r_extra < p:
                residues_to_try.append(r_extra)
        # Remove duplicates while preserving order
        seen = set()
        unique_residues = []
        for r in residues_to_try:
            if r not in seen:
                seen.add(r)
                unique_residues.append(r)
        
        for r in unique_residues:
            if time.time() > deadline:
                break
            
            # Simulate: with this class, how far can we go?
            test_classes = best_classes + [[p, r]]
            test_L = _compute_L_fast(test_classes, cap=min(current_L + 2000, 100000))
            
            if test_L > best_new_L:
                best_new_L = test_L
                best_r = r
        
        if best_r is not None and best_new_L > best_L:
            # Check budget
            new_cost = best_cost + Fraction(1, p)
            if new_cost <= Fraction(1, 1):
                best_classes = best_classes + [[p, best_r]]
                best_L = best_new_L
                best_cost = new_cost
                used_primes.add(p)
    
    # Try to improve residues of existing classes
    # Multiple passes for better convergence
    for _pass in range(3):
        if time.time() > deadline:
            break
        improved = False
        for idx in range(len(best_classes)):
            if time.time() > deadline:
                break
            p, r = best_classes[idx]
            old_r = r
            best_r_for_p = r
            best_L_for_p = best_L
            
            # Try more alternative residues
            for dr in range(-10, 11):
                if dr == 0:
                    continue
                new_r = (r + dr) % p
                if new_r == r:
                    continue
                # Test this alternative
                test_classes = best_classes.copy()
                test_classes[idx] = [p, new_r]
                test_L = _compute_L_fast(test_classes, cap=min(best_L + 500, 100000))
                if test_L > best_L_for_p:
                    best_L_for_p = test_L
                    best_r_for_p = new_r
            
            if best_r_for_p != old_r:
                best_classes[idx] = [p, best_r_for_p]
                best_L = best_L_for_p
                improved = True
        if not improved:
            break
    
    # Final verification
    final_L = _compute_L(best_classes)
    if final_L > best_L:
        best_L = final_L
    
    return best_classes
# EVOLVE-BLOCK-END
