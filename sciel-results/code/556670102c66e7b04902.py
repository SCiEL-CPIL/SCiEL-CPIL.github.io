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

def _compute_coverage(classes, cap=100000):
    """Compute the largest L such that every integer 1..L is covered."""
    if not classes:
        return 0
    covered = [False] * (cap + 2)
    for p, r in classes:
        start = r if r >= 1 else p
        for n in range(start, cap + 1, p):
            covered[n] = True
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
    
    # Randomized local search with multiple strategies
    random.seed(42)
    
    for iteration in range(5000):
        if time.time() > deadline:
            break
        
        current_classes = best_classes[:]
        current_sum = best_sum
        current_L = best_L
        
        # Strategy 1: Random residue permutation
        if random.random() < 0.3:
            # Pick a random class and try a new residue
            if current_classes:
                idx = random.randint(0, len(current_classes) - 1)
                old_p, old_r = current_classes[idx]
                new_r = random.randint(0, old_p - 1)
                if new_r != old_r:
                    current_classes[idx] = [old_p, new_r]
                    new_L = _compute_coverage(current_classes)
                    if new_L > current_L:
                        best_L = new_L
                        best_classes = current_classes
                        best_sum = current_sum
        
        # Strategy 2: Swap two classes' residues
        elif random.random() < 0.3:
            if len(current_classes) >= 2:
                idx1, idx2 = random.sample(range(len(current_classes)), 2)
                p1, r1 = current_classes[idx1]
                p2, r2 = current_classes[idx2]
                current_classes[idx1] = [p1, r2 % p1]
                current_classes[idx2] = [p2, r1 % p2]
                new_L = _compute_coverage(current_classes)
                if new_L > current_L:
                    best_L = new_L
                    best_classes = current_classes
                    best_sum = current_sum
        
        # Strategy 3: Replace a class with a different prime
        elif random.random() < 0.3:
            if current_classes:
                idx = random.randint(0, len(current_classes) - 1)
                old_p, old_r = current_classes[idx]
                used_primes = set(p for p, r in current_classes)
                candidate_primes = [p for p in primes if p not in used_primes]
                if candidate_primes:
                    new_p = random.choice(candidate_primes)
                    new_sum = current_sum - Fraction(1, old_p) + Fraction(1, new_p)
                    if new_sum <= Fraction(1):
                        # Choose residue to cover the first gap
                        gap = current_L + 1
                        new_r = gap % new_p
                        current_classes[idx] = [new_p, new_r]
                        new_L = _compute_coverage(current_classes)
                        if new_L > current_L:
                            best_L = new_L
                            best_classes = current_classes
                            best_sum = new_sum
        
        # Strategy 4: Add a new class if budget allows
        elif current_sum < Fraction(1):
            used_primes = set(p for p, r in current_classes)
            candidate_primes = [p for p in primes if p not in used_primes]
            if candidate_primes:
                new_p = random.choice(candidate_primes)
                new_sum = current_sum + Fraction(1, new_p)
                if new_sum <= Fraction(1):
                    gap = current_L + 1
                    new_r = gap % new_p
                    current_classes.append([new_p, new_r])
                    new_L = _compute_coverage(current_classes)
                    if new_L > current_L:
                        best_L = new_L
                        best_classes = current_classes
                        best_sum = new_sum
        
        # Strategy 5: Remove a class that doesn't help much
        elif len(current_classes) > 5 and random.random() < 0.1:
            idx = random.randint(0, len(current_classes) - 1)
            old_p, old_r = current_classes.pop(idx)
            new_L = _compute_coverage(current_classes)
            if new_L >= current_L:
                best_L = new_L
                best_classes = current_classes
                best_sum = current_sum
            else:
                current_classes.insert(idx, [old_p, old_r])
    
    return best_classes
# EVOLVE-BLOCK-END
