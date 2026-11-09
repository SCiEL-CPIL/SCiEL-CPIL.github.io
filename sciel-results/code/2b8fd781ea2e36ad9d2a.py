"""Rewritten program using a CSP / gap-filling approach with incremental state."""
import time
from fractions import Fraction
import random
import sys

# EVOLVE-BLOCK-START

def _primes_upto(n):
    if n < 2:
        return []
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    return [i for i, is_p in enumerate(sieve) if is_p]


def _compute_L_from_covered(covered, max_check=200000):
    """Given a boolean covered array (index 0 unused), return largest L s.t. 1..L all covered."""
    L = 0
    for n in range(1, min(len(covered), max_check + 1)):
        if covered[n]:
            L = n
        else:
            break
    return L


def _build_covered(classes, size):
    """Build a covered boolean array of given size from a list of [p, r] classes."""
    covered = [False] * (size + 1)
    for p, r in classes:
        if r == 0:
            first = p
        else:
            first = r
        if first < 1:
            first += p
        for n in range(first, size + 1, p):
            covered[n] = True
    return covered


def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    random.seed(42)

    primes = _primes_upto(1000)
    all_primes = primes  # 2,3,5,...,997

    # We'll work with a working size for the covered array.
    # Start with a size that's a bit above what we expect to cover.
    WORK_SIZE = 50000

    # Initial solution: use the seed-like construction but optimized.
    # Seed: primes >= 13, residue = (index+1) % p
    seed_primes = [p for p in all_primes if p >= 13]
    seed_classes = [[p, (i + 1) % p] for i, p in enumerate(seed_primes)]
    seed_budget = sum(Fraction(1, p) for p in seed_primes)

    # Build initial covered state
    covered = _build_covered(seed_classes, WORK_SIZE)
    current_L = _compute_L_from_covered(covered, WORK_SIZE)

    best_L = current_L
    best_classes = list(seed_classes)
    best_budget = seed_budget

    # Track which primes are used
    used_primes = set(p for p, r in seed_classes)
    remaining_primes = [p for p in all_primes if p not in used_primes]
    # Sort remaining by cost (ascending 1/p means descending p) -- we want cheap primes first for gap filling
    remaining_primes.sort(key=lambda p: -p)  # largest first (cheapest)

    # Also track count of how many classes cover each integer (for incremental updates)
    # count[n] = number of classes covering n
    count = [0] * (WORK_SIZE + 1)
    for p, r in seed_classes:
        if r == 0:
            first = p
        else:
            first = r
        if first < 1:
            first += p
        for n in range(first, WORK_SIZE + 1, p):
            count[n] += 1

    # Recompute L from count (covered if count[n] >= 1)
    # Actually let's just use covered array for L computation.

    # Phase 1: Greedy gap-filling with remaining primes
    # For each uncovered integer (starting from current_L+1), try to find a prime+residue that covers it
    # and maximizes additional coverage.

    # Actually, let's do a smarter greedy: repeatedly find the prime+residue that covers the most
    # currently-uncovered integers in 1..WORK_SIZE, subject to budget.

    # But that's expensive. Let's do: for the first uncovered integer n after current_L,
    # try all remaining primes and all residues, pick the one that covers the most new integers.

    # More efficient: for a given uncovered n, the residue for prime p is determined: r = n % p.
    # So for each remaining prime p, there's exactly one residue that covers n: r = n % p.
    # We can evaluate how many NEW integers each such (p, r=n%p) would cover.

    # Let's do iterative improvement:
    # 1. Find first gap: n = current_L + 1
    # 2. For each remaining prime p, compute r = n % p, evaluate new coverage
    # 3. Pick best, add it, update state
    # 4. Repeat until no improvement or budget exhausted

    # To evaluate "new coverage" for (p, r), we need to count how many n' in 1..WORK_SIZE
    # are NOT currently covered but WOULD be covered by (p, r).
    # This is expensive. Let's approximate: count how many n' ≡ r (mod p) have count[n'] == 0.

    # Precompute for speed: for each prime p, we can quickly check residues.
    # But with 163 primes and 50000 size, this is still manageable if done carefully.

    # Let's use a different approach: numpy-like operations with lists.

    # Actually, let's just do the greedy but with a limited window.
    # For each gap position, try each remaining prime, compute the coverage gain.

    # To speed up: precompute for each prime p, an array gain_p[r] = number of n in 1..WORK_SIZE
    # with n % p == r and count[n] == 0.
    # When we add a class, we update count and thus gain arrays.

    # This is getting complex. Let's do a simpler but effective approach:

    # Phase 1: Try to fill gaps by adding primes greedily.
    # For the first uncovered n, try all remaining primes with r = n % p.
    # Score each by how many currently-uncovered integers it covers (approximate by checking
    # a sample or just the immediate neighborhood).

    # Let's just do full evaluation for the first few gaps, then switch to local search.

    max_gaps_to_fill = 20
    for gap_iter in range(max_gaps_to_fill):
        if time.time() > deadline - 20:
            break

        # Find first uncovered integer
        if current_L >= WORK_SIZE:
            break
        n_gap = current_L + 1
        if count[n_gap] > 0:
            # Shouldn't happen, but skip
            continue

        # Try each remaining prime
        best_p = None
        best_r = None
        best_gain = 0
        best_cost = None

        for p in remaining_primes:
            cost = Fraction(1, p)
            if best_budget + cost > Fraction(1):
                continue

            r = n_gap % p

            # Count how many currently-uncovered integers this would cover
            # n' ≡ r (mod p), count[n'] == 0
            gain = 0
            if r == 0:
                first = p
            else:
                first = r
            if first < 1:
                first += p
            for n_prime in range(first, WORK_SIZE + 1, p):
                if count[n_prime] == 0:
                    gain += 1

            if gain > best_gain:
                best_gain = gain
                best_p = p
                best_r = r
                best_cost = cost

        if best_p is None or best_gain == 0:
            break

        # Add the prime
        seed_classes.append([best_p, best_r])
        used_primes.add(best_p)
        remaining_primes.remove(best_p)
        best_budget += best_cost

        # Update count and covered
        p, r = best_p, best_r
        if r == 0:
            first = p
        else:
            first = r
        if first < 1:
            first += p
        for n_prime in range(first, WORK_SIZE + 1, p):
            if count[n_prime] == 0:
                covered[n_prime] = True
            count[n_prime] += 1

        # Recompute L
        current_L = _compute_L_from_covered(covered, WORK_SIZE)

        if current_L > best_L:
            best_L = current_L
            best_classes = list(seed_classes)

        if time.time() > deadline - 15:
            break

    # Phase 2: Local search / Simulated Annealing on residues of existing primes
    # and swapping primes in/out.

    if time.time() < deadline - 10:
        # SA on residues
        T = 5.0
        T_min = 0.01
        cooling = 0.995

        current_classes_sa = list(best_classes)
        current_budget_sa = best_budget
        current_covered = _build_covered(current_classes_sa, WORK_SIZE)
        current_count = [0] * (WORK_SIZE + 1)
        for p, r in current_classes_sa:
            if r == 0:
                first = p
            else:
                first = r
            if first < 1:
                first += p
            for n in range(first, WORK_SIZE + 1, p):
                current_count[n] += 1
        current_L_sa = _compute_L_from_covered(current_covered, WORK_SIZE)

        best_L_sa = current_L_sa
        best_classes_sa = list(current_classes_sa)
        best_budget_sa = current_budget_sa

        no_improve_count = 0
        max_no_improve = 500

        while T > T_min and time.time() < deadline - 5:
            if no_improve_count > max_no_improve:
                T *= 0.9
                no_improve_count = 0
                if T < T_min:
                    break

            # Pick a random prime from current set
            idx = random.randint(0, len(current_classes_sa) - 1)
            p, old_r = current_classes_sa[idx]
            new_r = random.randint(0, p - 1)
            if new_r == old_r:
                continue

            # Evaluate change: remove old_r, add new_r for prime p
            # Decrement count for old_r
            old_first = old_r if old_r > 0 else p
            if old_first < 1:
                old_first += p
            old_newly_uncovered = 0
            for n in range(old_first, WORK_SIZE + 1, p):
                current_count[n] -= 1
                if current_count[n] == 0:
                    old_newly_uncovered += 1
                    current_covered[n] = False

            # Increment count for new_r
            new_first = new_r if new_r > 0 else p
            if new_first < 1:
                new_first += p
            new_newly_covered = 0
            for n in range(new_first, WORK_SIZE + 1, p):
                if current_count[n] == 0:
                    new_newly_covered += 1
                    current_covered[n] = True
                current_count[n] += 1

            # Compute new L
            new_L = _compute_L_from_covered(current_covered, WORK_SIZE)
            delta_L = new_L - current_L_sa

            if delta_L >= 0 or random.random() < math.exp(delta_L / T):
                # Accept
                current_classes_sa[idx] = [p, new_r]
                current_L_sa = new_L
                if new_L > best_L_sa:
                    best_L_sa = new_L
                    best_classes_sa = list(current_classes_sa)
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                # Reject: revert
                # Re-increment for new_r (undo)
                for n in range(new_first, WORK_SIZE + 1, p):
                    current_count[n] -= 1
                    if current_count[n] == 0:
                        current_covered[n] = False
                # Re-decrement for old_r (undo)
                for n in range(old_first, WORK_SIZE + 1, p):
                    if current_count[n] == 0:
                        current_covered[n] = True
                    current_count[n] += 1

            T *= cooling

        if best_L_sa > best_L:
            best_L = best_L_sa
            best_classes = list(best_classes_sa)
            best_budget = best_budget_sa

    # Phase 3: Try to add small primes (2,3,5,7,11) if budget allows
    if time.time() < deadline - 5:
        small_primes = [p for p in all_primes if p < 13 and p not in used_primes]
        for p in small_primes:
            cost = Fraction(1, p)
            if best_budget + cost > Fraction(1):
                continue
            # Try all residues
            best_r_for_p = None
            best_L_for_p = best_L
            test_classes = list(best_classes)
            test_covered = _build_covered(test_classes, WORK_SIZE)
            test_count = [0] * (WORK_SIZE + 1)
            for pp, rr in test_classes:
                ff = rr if rr > 0 else pp
                if ff < 1:
                    ff += pp
                for nn in range(ff, WORK_SIZE + 1, pp):
                    test_count[nn] += 1

            for r in range(p):
                # Add (p, r)
                ff = r if r > 0 else p
                if ff < 1:
                    ff += p
                for nn in range(ff, WORK_SIZE + 1, p):
                    if test_count[nn] == 0:
                        test_covered[nn] = True
                    test_count[nn] += 1

                L_test = _compute_L_from_covered(test_covered, WORK_SIZE)
                if L_test > best_L_for_p:
                    best_L_for_p = L_test
                    best_r_for_p = r

                # Undo
                for nn in range(ff, WORK_SIZE + 1, p):
                    test_count[nn] -= 1
                    if test_count[nn] == 0:
                        test_covered[nn] = False

            if best_r_for_p is not None and best_L_for_p > best_L:
                best_classes.append([p, best_r_for_p])
                best_budget += cost
                best_L = best_L_for_p
                used_primes.add(p)

    # Final verification and return
    # Ensure budget constraint is satisfied
    final_budget = sum(Fraction(1, p) for p, r in best_classes)
    if final_budget > Fraction(1):
        # This shouldn't happen, but fallback to seed
        best_classes = [[p, (i + 1) % p] for i, p in enumerate(seed_primes)]

    return best_classes


import math

# EVOLVE-BLOCK-END
