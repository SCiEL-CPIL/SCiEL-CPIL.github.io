"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
import math

def _primes_up_to(n):
    """Return list of primes up to n."""
    if n < 2:
        return []
    sieve = bytearray(b'\x01') * (n + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(2, n + 1) if sieve[i]]

def _covered_length(classes, cap=100000):
    """
    Given a list of [p, r] classes, return the largest L such that every
    integer 1..L is covered by at least one class.
    An integer n is covered by class [p, r] if n % p == r.
    """
    if not classes:
        return 0
    # Build a boolean array for coverage up to cap
    # For efficiency, we'll mark coverage
    covered = bytearray(cap + 1)  # 0-indexed, we care about 1..cap
    for p, r in classes:
        # Mark all n in 1..cap such that n % p == r
        # The first such n >= 1 is: if r == 0, then p, 2p, ...; else r, r+p, ...
        # Actually n % p == r means n = k*p + r for some k >= 0, and n >= 1
        # So the smallest n >= 1 with n % p == r:
        #   if r == 0: n = p (since 0 is not in 1..cap, next is p)
        #   if r >= 1: n = r (if r >= 1)
        start = r if r >= 1 else p
        # But wait: if r == 0, n % p == 0 means n is multiple of p. Smallest >= 1 is p.
        # If r >= 1, smallest n >= 1 with n % p == r is r itself (since r < p and r >= 1).
        # Let me re-check: for r in 0..p-1, the numbers n with n % p == r are ..., r-p, r, r+p, r+2p, ...
        # We need n >= 1. So:
        #   if r == 0: the sequence is ..., -p, 0, p, 2p, ... -> smallest >= 1 is p
        #   if r >= 1: the sequence is ..., r-p, r, r+p, ... -> smallest >= 1 is r (since r >= 1)
        # So start = r if r >= 1 else p
        if start <= cap:
            for n in range(start, cap + 1, p):
                covered[n] = 1
    # Now find the largest L such that 1..L are all covered
    L = 0
    for n in range(1, cap + 1):
        if covered[n]:
            L = n
        else:
            break
    return L

def _sum_recip(classes):
    """Return the sum of 1/p for each class as a Fraction."""
    total = Fraction(0, 1)
    for p, r in classes:
        total += Fraction(1, p)
    return total

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    primes = _primes_up_to(1000)
    
    # We want to select a subset of primes and assign residues to maximize the
    # covered length L, subject to sum(1/p) <= 1.
    
    # Strategy: Greedy approach
    # 1. Start with no classes.
    # 2. At each step, try adding a prime p with the largest 1/p (i.e., smallest p)
    #    that fits in the budget, and choose the residue r that maximizes the
    #    marginal gain in covered length.
    # 3. Continue until no more primes can be added or budget is exhausted.
    
    # Actually, a better approach: 
    # - We have primes 2, 3, 5, ..., 997.
    # - We need to choose a subset S and assign residues r_p for each p in S
    #   such that sum_{p in S} 1/p <= 1.
    # - Goal: maximize L where 1..L are all covered.
    
    # Let's think about this differently. The constraint is on the sum of reciprocals.
    # Small primes have large 1/p, so they are "expensive" but also cover more densely.
    
    # A simple greedy: 
    # - Sort primes by 1/p descending (i.e., by p ascending).
    # - For each prime, try to add it with the best residue.
    # - But we need to check if adding it keeps the sum <= 1.
    
    # Let me implement a greedy that:
    # 1. Maintains current classes and current sum.
    # 2. For each candidate prime (in order of increasing p, i.e., decreasing 1/p),
    #    if 1/p can be added (sum + 1/p <= 1), try all residues r and pick the one
    #    that gives the best improvement in covered length.
    
    # But trying all residues for each prime is expensive. Let's think about which
    # residue is best.
    
    # Actually, for a given set of classes, adding a new class [p, r] covers
    # n ≡ r (mod p). The marginal gain is the number of new n in 1..L' that are
    # covered, where L' is the new covered length.
    
    # This is complex. Let me try a simpler approach first:
    # Use the same strategy as the seed but try to optimize the residues.
    
    # Seed strategy: use primes from 13 upward, assign r = (i+1) % p where i is the index.
    # This covers each integer i+1 exactly once (by construction, since p > i+1 for small i).
    
    # Let me try a different approach: 
    # - Use a greedy algorithm that builds the covering set.
    # - At each step, find the uncovered integer n (starting from 1) and find
    #   the cheapest prime p (smallest 1/p, i.e., largest p) that can cover n
    #   and fits in the budget.
    
    # Wait, that's not quite right either. Let me think more carefully.
    
    # The key insight: we want to cover 1, 2, 3, ... as far as possible.
    # For each uncovered integer n, we need to add a class [p, r] such that n % p == r.
    # We want to minimize the cost (1/p) of the classes we add.
    
    # Algorithm:
    # 1. Start with empty set of classes, sum = 0.
    # 2. Find the smallest uncovered integer n (starting from 1).
    # 3. For each prime p (sorted by 1/p ascending, i.e., p descending), check if
    #    we can add a class [p, n % p] that fits in the budget.
    # 4. Among all such primes, pick the one with the smallest 1/p (largest p) to
    #    minimize cost.
    # 5. Add that class and repeat.
    
    # But this might not be optimal because adding a class for n might also cover
    # other integers, which is good.
    
    # Let me implement this greedy:
    
    def solve_greedy():
        classes = []
        current_sum = Fraction(0, 1)
        # Track which integers are covered
        # We'll use a set or a boolean array
        # For efficiency, let's use a bytearray for coverage up to some limit
        # But we don't know the limit in advance. Let's use a set.
        covered_set = set()
        
        # We need to find the smallest uncovered integer
        # Start from 1
        n = 1
        while True:
            if n in covered_set:
                n += 1
                continue
            
            # n is uncovered. Find the best prime to cover it.
            best_p = None
            best_cost = None
            
            for p in reversed(primes):  # Try largest p first (smallest 1/p)
                cost = Fraction(1, p)
                if current_sum + cost > Fraction(1, 1):
                    continue
                r = n % p
                # Check if adding [p, r] is valid (it always is, since 0 <= r < p)
                # We prefer the smallest cost, which is largest p
                best_p = p
                best_cost = cost
                break  # Since we're going from largest to smallest p, first valid is best
            
            if best_p is None:
                break  # Can't cover n with any remaining budget
            
            # Add the class
            r = n % best_p
            classes.append([best_p, r])
            current_sum += best_cost
            
            # Update covered_set
            # Add all numbers covered by [best_p, r] that are <= some reasonable limit
            # But we don't know the limit. Let's add up to, say, 200000.
            # Actually, we only need to track up to the point where we can't extend further.
            # Let's add up to a reasonable cap.
            cap = 200000
            for k in range(0, cap // best_p + 1):
                val = k * best_p + r
                if val >= 1 and val <= cap:
                    covered_set.add(val)
            
            n += 1
            # Safety check for time
            if time.time() > deadline - 10:
                break
        
        return classes
    
    # Try the greedy approach
    best_classes = None
    best_L = 0
    
    # First, try the seed approach as a baseline
    seed_primes = [p for p in primes if p >= 13]
    seed_classes = [[p, (i + 1) % p] for i, p in enumerate(seed_primes)]
    seed_sum = _sum_recip(seed_classes)
    if seed_sum <= Fraction(1, 1):
        seed_L = _covered_length(seed_classes)
        best_classes = seed_classes
        best_L = seed_L
    
    # Now try the greedy approach
    try:
        greedy_classes = solve_greedy()
        greedy_sum = _sum_recip(greedy_classes)
        if greedy_sum <= Fraction(1, 1):
            greedy_L = _covered_length(greedy_classes)
            if greedy_L > best_L:
                best_classes = greedy_classes
                best_L = greedy_L
    except Exception:
        pass
    
    # If we still have time, try to improve by local search
    if time.time() < deadline - 20 and best_classes is not None:
        # Try swapping residues or adding/removing classes
        # This is complex, so let's just return what we have
        pass
    
    return best_classes if best_classes is not None else []
# EVOLVE-BLOCK-END
