"""Covering 1..L by prime residue classes with reciprocal budget 1."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    t0 = time.time()
    DEADLINE = 140  # seconds, safety margin before 150s kill

    # Generate primes up to 1000
    primes = []
    for p in range(2, 1001):
        is_prime = True
        for q in range(2, int(p**0.5) + 1):
            if p % q == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(p)

    n_primes = len(primes)
    # For each prime p, possible residues are 0..p-1
    # We need to select at most one residue per prime (or none)
    # such that sum(1/p for selected p) <= 1
    # and coverage of 1..L is maximized.

    # Greedy approach:
    # 1. Start with empty set of classes.
    # 2. Repeatedly add the class (p, r) that gives the best marginal coverage
    #    per unit cost (1/p), while keeping total cost <= 1.
    #    To maximize L, we want to cover 1, 2, 3, ... in order.
    #    So we track the smallest uncovered integer.

    # Better greedy:
    # Maintain a coverage array for 1..MAX_L (say 10^5)
    # But that's expensive. Instead, track the set of covered integers up to some limit.

    # Alternative: use the fact that we want to cover 1..L contiguously.
    # Greedy: at each step, find the (p, r) that covers the most new integers
    # in the range [current_max_covered+1, some_bound] per unit cost.

    # Let's use a smarter greedy:
    # We want to maximize L. The constraint is sum(1/p) <= 1.
    # Start with no classes. L = 0.
    # At each step, for each prime p not yet used, and each residue r in 0..p-1,
    # compute the marginal gain in L if we add (p, r).
    # Add the one with best gain/cost ratio, update L.
    # Stop when no more can be added within budget or time runs out.

    # This is O(n_primes * avg_p * L) per step, which is too slow for large L.

    # Let's use a different approach:
    # The Erdos-Ruzsa conjecture suggests that for C=1, we can cover up to ~1000.
    # The seed program achieves L=999 (score 1.0).
    # Let's try to improve by using a local search / greedy with better ordering.

    # Key insight: the seed uses primes starting from 13, assigning residue (i+1) % p
    # to the i-th prime. This is a very specific construction.
    # Let's try a greedy that builds up the coverage from 1 upward.

    MAX_L = 100000
    # covered[i] = True if integer i is covered
    covered = [False] * (MAX_L + 1)
    used_primes = set()
    total_cost = Fraction(0, 1)
    L = 0

    def update_L():
        nonlocal L
        while L + 1 <= MAX_L and covered[L + 1]:
            L += 1

    # Greedy: repeatedly add the class that extends L the most per unit cost.
    # To make this efficient, we only consider residues that cover L+1 or nearby.
    # For a given prime p, the residue r that covers L+1 is r = (L+1) % p.
    # Adding (p, (L+1)%p) will cover L+1 and potentially many more.

    # Let's try: at each step, for each unused prime p, consider r = (L+1) % p.
    # Compute how far this extends coverage. Pick the best.

    # But this might get stuck. Let's also consider other residues.

    # Time-budgeted greedy:
    import numpy as np

    # Precompute for speed
    # For each prime p, precompute the coverage pattern for each residue? No, too big.

    # Let's use a simpler but effective greedy:
    # 1. Start with L=0.
    # 2. For each step, find the best (p, r) to add.
    #    For each unused prime p, try r = (L+1) % p first (most promising).
    #    Also try a few other residues.
    #    Evaluate marginal L gain.
    #    Pick the best, add it.
    # 3. Repeat until no improvement or time runs out.

    # To speed up evaluation, we can use a bitset for coverage.
    # Python ints can serve as arbitrary-precision bitsets.

    # Let's use a list of booleans for simplicity, but limit the search range.
    # Actually, let's just use the seed approach and try to improve it.

    # The seed: primes from 13 upward, residue (i+1) % p for i-th prime.
    # This gives L=999. Let's verify and then try local search.

    # Build the seed solution:
    seed_primes = [p for p in primes if p >= 13]
    seed_classes = []
    seed_cost = Fraction(0, 1)
    for i, p in enumerate(seed_primes):
        r = (i + 1) % p
        seed_classes.append([p, r])
        seed_cost += Fraction(1, p)

    # Compute L for seed:
    covered_arr = [False] * (MAX_L + 1)
    for p, r in seed_classes:
        for x in range(r if r > 0 else p, MAX_L + 1, p):
            covered_arr[x] = True

    seed_L = 0
    while seed_L + 1 <= MAX_L and covered_arr[seed_L + 1]:
        seed_L += 1

    best_L = seed_L
    best_classes = seed_classes[:]
    best_cost = seed_cost

    # Now try to improve with local search:
    # 1. Try swapping residues.
    # 2. Try replacing a prime with a smaller/larger one.
    # 3. Try adding a new prime with a good residue.

    # Given time constraints, let's do a limited local search.
    def compute_L(classes):
        cov = [False] * (min(MAX_L, 20000) + 1)  # Limit for speed
        for p, r in classes:
            start = r if r > 0 else p
            for x in range(start, min(MAX_L, 20000) + 1, p):
                cov[x] = True
        L = 0
        while L + 1 < len(cov) and cov[L + 1]:
            L += 1
        return L

    def try_improve(classes, cost):
        nonlocal best_L, best_classes, best_cost
        improved = True
        while improved and time.time() - t0 < DEADLINE:
            improved = False
            n = len(classes)
            # Try changing residue of each class
            for i in range(n):
                if time.time() - t0 > DEADLINE - 5:
                    break
                p, r = classes[i]
                old_r = r
                # Try a few alternative residues
                for new_r in range(p):
                    if new_r == old_r:
                        continue
                    classes[i] = [p, new_r]
                    L = compute_L(classes)
                    if L > best_L:
                        best_L = L
                        best_classes = [c[:] for c in classes]
                        best_cost = cost
                        improved = True
                    classes[i] = [p, old_r]
                if time.time() - t0 > DEADLINE - 5:
                    break
            if time.time() - t0 > DEADLINE - 5:
                break
            # Try adding a new prime
            for p in primes:
                if p in [c[0] for c in classes]:
                    continue
                new_cost = cost + Fraction(1, p)
                if new_cost > 1:
                    continue
                # Find best residue for this prime
                best_r_for_p = -1
                best_L_for_p = best_L
                for r in range(p):
                    test_classes = classes + [[p, r]]
                    L = compute_L(test_classes)
                    if L > best_L_for_p:
                        best_L_for_p = L
                        best_r_for_p = r
                if best_r_for_p >= 0 and best_L_for_p > best_L:
                    classes.append([p, best_r_for_p])
                    cost = new_cost
                    best_L = best_L_for_p
                    best_classes = [c[:] for c in classes]
                    best_cost = cost
                    improved = True
                    break  # Restart search after adding
                if time.time() - t0 > DEADLINE - 5:
                    break

    try_improve(best_classes[:], best_cost)

    if time.time() - t0 > DEADLINE - 3:
        return best_classes

    return best_classes
# EVOLVE-BLOCK-END
