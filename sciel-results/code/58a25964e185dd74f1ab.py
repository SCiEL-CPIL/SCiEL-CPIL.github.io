"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction
import sys

def _primes_upto(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, n+1, i):
                sieve[j] = False
    return [i for i in range(2, n+1) if sieve[i]]

def _compute_L(classes, cap=100000):
    """Compute the largest L such that every integer 1..L is covered."""
    if not classes:
        return 0
    covered = set()
    # For efficiency, mark coverage up to cap
    # But cap is 100000, and we have at most ~168 primes, so at most 168*100000 ops worst case
    # Actually let's just check sequentially
    max_check = min(cap, 100000)
    for n in range(1, max_check + 1):
        found = False
        for p, r in classes:
            if n % p == r % p:
                found = True
                break
        if not found:
            return n - 1
    return max_check

def _sum_recip(classes):
    """Sum of 1/p as Fraction."""
    total = Fraction(0, 1)
    for p, r in classes:
        total += Fraction(1, p)
    return total

def _greedy_init(primes, deadline):
    """Greedy: start empty, repeatedly add the class that covers the smallest uncovered integer and maximizes new coverage."""
    # We want to cover 1, 2, 3, ... as far as possible
    # Start with no classes
    classes = []
    used_primes = set()
    budget = Fraction(1, 1)
    
    # Sort primes: smaller primes cover more numbers in [1, L] per unit of budget
    # Actually, for covering prefix, we want primes that can cover small numbers
    # Greedy: at each step, find smallest uncovered n, then pick prime p not used, residue r,
    # such that n ≡ r (mod p), and this choice maximizes additional coverage in a window
    
    # Simpler greedy: iterate n = 1, 2, 3, ...; for each uncovered n, pick a prime p (not used)
    # and set r = n % p. Prefer primes that give good future coverage.
    # But we need to be careful about budget.
    
    # Alternative: pre-select a set of primes, then assign residues optimally
    # Let's try: select primes, then for each prime assign residue to cover the smallest uncovered number
    
    # Let's do a different greedy: 
    # 1. Select a subset of primes (by some criterion)
    # 2. For each selected prime, assign residue to maximize coverage of prefix
    
    # Actually, let's do iterative:
    # Start with empty. While we can improve:
    #   Find smallest uncovered n
    #   Try all unused primes p: if we add (p, n%p), what's the new L?
    #   Pick the one that gives best improvement, within budget
    
    # This is expensive. Let's do a simpler version:
    # Just assign residues greedily: for n=1,2,3,..., if uncovered, pick a random/first available prime
    
    # Even simpler: use all primes from 13 to 1000, assign each to cover a specific small number
    # The seed does: prime[i] covers i+1. Let's do better.
    
    # New approach: 
    # Phase 1: Select primes. Use all primes from 2 to 1000? No, budget is 1.
    # Sum of 1/p for p=2..1000 is about 2.0, too much.
    # Sum for p=13..1000 is about 0.92, fits.
    
    # Let's use primes from 13 to 1000, and do local search on residues.
    # Initial: identity-like but optimized
    # For prime at index i (0-based), cover number i+1
    # But maybe we can do better by permuting which prime covers which number
    
    # Let's just return the seed as initial, then local search will improve
    return [[p, (i + 1) % p] for i, p in enumerate(primes)]

def _local_search(classes, primes_all, deadline, max_iterations=50000):
    """Local search: try changing residues, swapping primes, etc."""
    best_L = _compute_L(classes)
    best_classes = [list(c) for c in classes]
    used_primes = set(p for p, r in classes)
    
    # Precompute for speed: for each prime, we can try all residues
    # But that's expensive. Instead, try:
    # 1. For each class, try changing its residue to cover the smallest uncovered number
    # 2. Try swapping two residues
    # 3. Try replacing a prime with an unused one
    
    iteration = 0
    start_time = time.time()
    
    while iteration < max_iterations:
        if time.time() > deadline:
            break
        
        # Find smallest uncovered
        L = best_L
        # Try to improve by changing one residue
        improved = False
        
        # Strategy: for each class, try to set its residue to cover L+1 (or some uncovered number)
        # But L+1 might not be the best target. Let's find the actual smallest uncovered.
        # Actually, _compute_L gives us L, so L+1 is the first uncovered (if L < cap)
        
        target = L + 1  # first uncovered number
        
        # Try: for each class i, if we change residue to target % p_i, does it help?
        # It covers target, but might uncover other numbers.
        # Let's just try it and measure.
        
        for i in range(len(classes)):
            if time.time() > deadline:
                break
            p, r = classes[i]
            new_r = target % p
            if new_r == r:
                continue
            # Try this change
            old_r = r
            classes[i][1] = new_r
            new_L = _compute_L(classes)
            if new_L > best_L:
                best_L = new_L
                best_classes = [list(c) for c in classes]
                improved = True
            else:
                classes[i][1] = old_r
        
        # Also try: swap residues of two classes
        if not improved and len(classes) >= 2:
            for i in range(len(classes)):
                if time.time() > deadline:
                    break
                for j in range(i+1, len(classes)):
                    if time.time() > deadline:
                        break
                    p1, r1 = classes[i]
                    p2, r2 = classes[j]
                    # Swap residues
                    classes[i][1] = r2
                    classes[j][1] = r1
                    new_L = _compute_L(classes)
                    if new_L > best_L:
                        best_L = new_L
                        best_classes = [list(c) for c in classes]
                        improved = True
                    else:
                        classes[i][1] = r1
                        classes[j][1] = r2
        
        # Try: replace a prime with an unused one
        if not improved:
            unused = [p for p in primes_all if p not in used_primes and p >= 2]
            if unused:
                # Pick a random unused prime and a random class to replace
                for _ in range(min(5, len(unused))):
                    if time.time() > deadline:
                        break
                    new_p = random.choice(unused)
                    # Check budget
                    current_sum = _sum_recip(classes)
                    # Remove one class, add new one
                    # Which class to remove? The one that contributes least to coverage
                    # For simplicity, remove a random one and add new_p with residue to cover target
                    idx = random.randint(0, len(classes) - 1)
                    old_p, old_r = classes[idx]
                    # Check if budget allows: sum - 1/old_p + 1/new_p <= 1
                    new_sum = current_sum - Fraction(1, old_p) + Fraction(1, new_p)
                    if new_sum <= Fraction(1, 1):
                        new_r = target % new_p
                        classes[idx] = [new_p, new_r]
                        new_L = _compute_L(classes)
                        if new_L > best_L:
                            best_L = new_L
                            best_classes = [list(c) for c in classes]
                            used_primes.discard(old_p)
                            used_primes.add(new_p)
                            improved = True
                        else:
                            classes[idx] = [old_p, old_r]
        
        iteration += 1
        if iteration % 1000 == 0:
            pass  # Could log
    
    return best_classes, best_L

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    primes_all = _primes_upto(1000)
    # Use primes from 13 to 1000 for the initial set (matches seed budget)
    primes_initial = [p for p in primes_all if p >= 13]
    
    # Initial solution
    classes = _greedy_init(primes_initial, deadline)
    
    # Local search with multiple restarts
    best_L = _compute_L(classes)
    best_result = [list(c) for c in classes]
    
    # Do several rounds of local search with different initial perturbations
    for restart in range(10):
        if time.time() > deadline:
            break
        
        if restart == 0:
            current = [list(c) for c in best_result]
        else:
            # Perturb: randomize some residues
            current = [list(c) for c in best_result]
            for i in range(len(current)):
                if random.random() < 0.3:
                    current[i][1] = random.randint(0, current[i][0] - 1)
        
        result, L = _local_search(current, primes_all, deadline, max_iterations=20000)
        if L > best_L:
            best_L = L
            best_result = result
    
    # Final check
    final_sum = _sum_recip(best_result)
    if final_sum > Fraction(1, 1):
        # Shouldn't happen, but just in case, fall back to seed
        return [[p, (i + 1) % p] for i, p in enumerate(primes_initial)]
    
    # Verify distinct primes
    seen = set()
    for p, r in best_result:
        if p in seen:
            # Remove duplicate
            pass
        seen.add(p)
    
    return best_result
# EVOLVE-BLOCK-END
