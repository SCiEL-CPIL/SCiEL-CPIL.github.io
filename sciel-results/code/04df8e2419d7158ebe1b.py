import math
import time
import random
from fractions import Fraction

# Precompute primes up to 1000
def _sieve(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(n)) + 1):
        if is_prime[i]:
            for j in range(i * i, n + 1, i):
                is_prime[j] = False
    return [i for i in range(2, n + 1) if is_prime[i]]

PRIMES = _sieve(1000)
PRIME_SET = set(PRIMES)

def _compute_coverage(classes, cap=100000):
    """Compute the largest L such that every integer 1..L is covered."""
    if not classes:
        return 0
    # Use a bytearray for efficiency
    covered = bytearray(cap + 1)  # indices 1..cap
    for p, r in classes:
        # Mark all numbers n in [1, cap] with n % p == r
        # Find smallest n >= 1 with n % p == r
        if r >= 1:
            start = r
        else:
            start = r + p
        for n in range(start, cap + 1, p):
            covered[n] = 1
    # Find largest L such that all 1..L are covered
    L = 0
    for i in range(1, cap + 1):
        if covered[i]:
            L = i
        else:
            break
    return L

def _sum_reciprocals(classes):
    """Compute sum of 1/p as Fraction."""
    s = Fraction(0, 1)
    for p, r in classes:
        s += Fraction(1, p)
    return s

def _is_valid(classes):
    """Check distinct primes, 0 <= r < p, sum 1/p <= 1."""
    seen = set()
    s = Fraction(0, 1)
    for p, r in classes:
        if p in seen:
            return False
        if p < 2 or p > 1000 or not PRIME_SET.__contains__(p):
            return False
        if r < 0 or r >= p:
            return False
        seen.add(p)
        s += Fraction(1, p)
        if s > 1:
            return False
    return True

def _greedy_init(cap=2000):
    """Greedy initialization: pick primes that maximize coverage per unit cost."""
    classes = []
    used_primes = set()
    budget = Fraction(1, 1)
    covered = bytearray(cap + 1)
    
    # Sort primes by size? Smaller primes are cheaper.
    # Let's try adding primes one by one, choosing the best residue for each.
    # Actually, let's iterate:
    # 1. Find the best (p, r) that adds the most new coverage.
    # 2. Add it.
    # 3. Repeat until no more improvement or budget exhausted.
    
    # To speed up, we can precompute which numbers are covered.
    # But checking all (p, r) pairs is expensive: sum(p) ~ 76k primes sum? No, sum of primes up to 1000 is ~ 76127.
    # For each step, we check ~76k residues. If we do 20 steps, that's 1.5M checks.
    # Each check involves iterating n = r, r+p, ... up to cap.
    # Average p is ~ 500. So 2000/500 = 4 numbers per residue check.
    # 1.5M * 4 = 6M operations. This is acceptable.
    
    available_primes = [p for p in PRIMES]
    
    while True:
        best_p = -1
        best_r = -1
        best_gain = 0
        best_cost = Fraction(0, 1)
        
        for p in available_primes:
            if p in used_primes:
                continue
            cost = Fraction(1, p)
            if budget - cost < 0:
                continue
            
            # Try all residues for this p
            # Optimization: We only care about residues that cover at least one uncovered number.
            # But we must check all to be sure.
            for r in range(p):
                # Calculate gain
                gain = 0
                if r >= 1:
                    start = r
                else:
                    start = r + p
                
                # Count new covers
                # We can break early if gain + (remaining possible) < best_gain?
                # Simpler: just count.
                for n in range(start, cap + 1, p):
                    if not covered[n]:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_p = p
                    best_r = r
                    best_cost = cost
        
        if best_p == -1 or best_gain == 0:
            break
        
        # Add the best class
        classes.append([best_p, best_r])
        used_primes.add(best_p)
        budget -= best_cost
        
        # Update covered array
        p = best_p
        r = best_r
        if r >= 1:
            start = r
        else:
            start = r + p
        for n in range(start, cap + 1, p):
            covered[n] = 1
            
    return classes

def _local_search(classes, max_time=100.0, cap=2000):
    """Local search to improve coverage using fast cap."""
    start_time = time.time()
    best_classes = [list(c) for c in classes]
    best_coverage = _compute_coverage(best_classes, cap)
    best_sum = _sum_reciprocals(best_classes)
    
    # Precompute used primes for quick lookup
    used_primes = set(c[0] for c in best_classes)
    
    for iteration in range(1000000): # High limit, time guarded
        if time.time() - start_time > max_time:
            break
        
        move_type = random.random()
        
        if move_type < 0.5:
            # Change residue of an existing class
            if not best_classes:
                continue
            idx = random.randint(0, len(best_classes) - 1)
            p, r = best_classes[idx]
            new_r = random.randint(0, p - 1)
            if new_r == r:
                continue
            new_classes = [list(c) for c in best_classes]
            new_classes[idx] = [p, new_r]
            cov = _compute_coverage(new_classes, cap)
            if cov > best_coverage:
                best_classes = new_classes
                best_coverage = cov
                # No change to sum
        elif move_type < 0.75:
            # Add a new prime class
            available = [p for p in PRIMES if p not in used_primes]
            if not available:
                continue
            p = random.choice(available)
            if best_sum + Fraction(1, p) > 1:
                continue
            r = random.randint(0, p - 1)
            new_classes = best_classes + [[p, r]]
            cov = _compute_coverage(new_classes, cap)
            if cov > best_coverage:
                best_classes = new_classes
                best_coverage = cov
                best_sum = _sum_reciprocals(best_classes)
                used_primes.add(p)
        elif move_type < 0.85:
            # Remove a prime class
            if len(best_classes) <= 1:
                continue
            idx = random.randint(0, len(best_classes) - 1)
            removed_p = best_classes[idx][0]
            new_classes = [c for i, c in enumerate(best_classes) if i != idx]
            cov = _compute_coverage(new_classes, cap)
            if cov > best_coverage:
                best_classes = new_classes
                best_coverage = cov
                best_sum = _sum_reciprocals(best_classes)
                used_primes.discard(removed_p)
        else:
            # Swap: remove one, add another
            if not best_classes:
                continue
            remove_idx = random.randint(0, len(best_classes) - 1)
            removed_p = best_classes[remove_idx][0]
            available = [p for p in PRIMES if p not in used_primes]
            if not available:
                continue
            add_p = random.choice(available)
            if best_sum - Fraction(1, removed_p) + Fraction(1, add_p) > 1:
                continue
            add_r = random.randint(0, add_p - 1)
            new_classes = [c for i, c in enumerate(best_classes) if i != remove_idx]
            new_classes.append([add_p, add_r])
            cov = _compute_coverage(new_classes, cap)
            if cov > best_coverage:
                best_classes = new_classes
                best_coverage = cov
                best_sum = _sum_reciprocals(best_classes)
                used_primes.discard(removed_p)
                used_primes.add(add_p)
    
    return best_classes

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # 1. Greedy initialization
    classes = _greedy_init(cap=2000)
    
    # 2. Local search
    classes = _local_search(classes, max_time=100.0, cap=2000)
    
    # 3. Final verification and return
    # The score is based on coverage up to 100,000.
    # We don't need to compute it here, the evaluator does.
    # But we must ensure validity.
    if not _is_valid(classes):
        # Fallback to a valid simple solution if something went wrong
        classes = []
    
    return classes
