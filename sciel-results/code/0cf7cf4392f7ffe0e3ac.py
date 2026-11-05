"""Improved: diverse prime-set candidates + vectorized greedy + local refinement with better heuristics."""
import time
from fractions import Fraction
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill

    def _time_ok():
        return time.time() < deadline

    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n ** 0.5) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, n + 1) if is_prime[i]]

    primes = sieve(1000)
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    large_primes = [p for p in primes if p > 23]

    MAX_L = 100000

    # ---- Build candidate prime sets ----
    seen_sets = set()
    candidate_sets = []

    def add_set(P):
        P = sorted(P)
        if not P:
            return
        key = tuple(P)
        if key in seen_sets:
            return
        # budget check
        s = sum(Fraction(1, p) for p in P)
        if s > 1:
            return
        seen_sets.add(key)
        candidate_sets.append(P)

    # Seed: primes >= 13
    seed_set = [p for p in primes if p >= 13]
    add_set(seed_set)

    # Subsets of small primes, fill rest with smallest large primes under budget
    import itertools
    for r in range(len(small_primes) + 1):
        if not _time_ok():
            break
        for subset in itertools.combinations(small_primes, r):
            current_sum = sum(Fraction(1, p) for p in subset)
            if current_sum > 1:
                continue
            current_set = list(subset)
            for p in large_primes:
                if current_sum + Fraction(1, p) <= 1:
                    current_set.append(p)
                    current_sum += Fraction(1, p)
                else:
                    break
            add_set(current_set)

    # Targeted swaps: remove some large primes, add a small prime
    for small in [11, 7, 5, 3, 2]:
        if not _time_ok():
            break
        for remove_list in [[], [13], [17], [19], [13, 17], [13, 19], [17, 19],
                            [13, 17, 19], [13, 17, 19, 23], [23, 29], [13, 17, 23]]:
            s = [p for p in seed_set if p not in remove_list]
            s_sum = sum(Fraction(1, p) for p in s)
            if s_sum + Fraction(1, small) <= 1:
                s.append(small)
                add_set(s)

    # ---- Multi-strategy residue assignment ----
    def get_L(covered, max_check):
        """Find the largest L such that 1..L are all covered."""
        if max_check < 1:
            return 0
        seg = covered[1:max_check + 1]
        z = np.nonzero(seg == 0)[0]
        if z.size:
            return int(z[0])
        return max_check

    def evaluate_set_multi(P, max_check=MAX_L):
        """Try multiple greedy strategies and return the best (L, residues)."""
        if not P:
            return 0, {}
        P_sorted = sorted(P)
        n_primes = len(P_sorted)

        def greedy_strategy(strategy):
            """Run a greedy strategy and return (L, residues)."""
            covered = np.zeros(max_check + 1, dtype=np.uint8)
            residues = {}
            assigned_primes = set()

            for n in range(1, max_check + 1):
                if not _time_ok():
                    break
                if covered[n]:
                    continue

                # Pick prime based on strategy
                best_p = None
                if strategy == 0:
                    # Smallest unused prime
                    for p in P_sorted:
                        if p not in assigned_primes:
                            best_p = p
                            break
                elif strategy == 1:
                    # Largest unused prime
                    for p in reversed(P_sorted):
                        if p not in assigned_primes:
                            best_p = p
                            break
                elif strategy == 2:
                    # Prime that covers most new numbers in [n, n+1000]
                    best_count = -1
                    for p in P_sorted:
                        if p not in assigned_primes:
                            r = n % p
                            # Count how many new numbers this would cover in a window
                            start = r
                            count = 0
                            for k in range(max(0, n - start), min(1000, max_check - n + 1 - start)):
                                if k >= 0 and n + k <= max_check and not covered[n + k]:
                                    count += 1
                            if count > best_count:
                                best_count = count
                                best_p = p
                elif strategy == 3:
                    # Smallest prime > n/2 (if any unused), else smallest unused
                    for p in P_sorted:
                        if p not in assigned_primes and p > n // 2:
                            best_p = p
                            break
                    if best_p is None:
                        for p in P_sorted:
                            if p not in assigned_primes:
                                best_p = p
                                break

                if best_p is None:
                    break

                r = n % best_p
                residues[best_p] = r
                assigned_primes.add(best_p)

                # Update coverage
                idx = np.arange(r, max_check + 1, best_p, dtype=np.int64)
                if idx.size:
                    covered[idx] = 1

            L = get_L(covered, max_check)
            return L, residues

        # Try multiple strategies
        best_L = 0
        best_residues = {}
        for strat in [0, 1, 2, 3]:
            if not _time_ok():
                break
            L, residues = greedy_strategy(strat)
            if L > best_L:
                best_L = L
                best_residues = residues

        return best_L, best_residues

    # Prioritize candidate sets: more small primes first, then more total primes
    def small_prime_count(P):
        return sum(1 for p in P if p <= 23)

    candidate_sets.sort(key=lambda P: (-small_prime_count(P), -len(P)))

    best_L = 0
    best_classes = [[p, (i + 1) % p] for i, p in enumerate(seed_set)]

    for P in candidate_sets:
        if not _time_ok():
            break
        L, residues = evaluate_set_multi(P)
        if L > best_L:
            best_L = L
            best_classes = [[p, residues[p]] for p in sorted(P) if p in residues]

    # ---- Local refinement: try to extend coverage by reassigning residues ----
    if best_classes and _time_ok():
        best_P = sorted(p for p, r in best_classes)
        
        # Try to extend by trying different residue assignments for existing primes
        # Start from the best solution and try swapping residues
        current_L = best_L
        current_residues = {p: r for p, r in best_classes}
        
        # Try local search: for each prime, try all possible residues and keep the best
        improved = True
        max_iterations = 50
        iteration = 0
        while improved and iteration < max_iterations and _time_ok():
            improved = False
            iteration += 1
            for p in best_P:
                if not _time_ok():
                    break
                old_r = current_residues[p]
                # Try all residues for this prime
                for new_r in range(p):
                    if new_r == old_r:
                        continue
                    # Temporarily change residue and check if L improves
                    current_residues[p] = new_r
                    
                    # Recompute coverage
                    covered = np.zeros(current_L + 10000, dtype=np.uint8)
                    for pp, rr in current_residues.items():
                        idx = np.arange(rr, current_L + 10000, pp, dtype=np.int64)
                        if idx.size:
                            covered[idx] = 1
                    
                    new_L = get_L(covered, current_L + 10000)
                    if new_L > current_L:
                        current_L = new_L
                        improved = True
                        break
                    else:
                        current_residues[p] = old_r
        
        if current_L > best_L:
            best_L = current_L
            best_classes = [[p, current_residues[p]] for p in sorted(best_P)]

    # Final verification
    if best_classes:
        final_sum = sum(Fraction(1, p) for p, r in best_classes)
        if final_sum > 1:
            best_classes = [[p, (i + 1) % p] for i, p in enumerate(seed_set)]
        prime_list = [p for p, r in best_classes]
        if len(prime_list) != len(set(prime_list)):
            best_classes = [[p, (i + 1) % p] for i, p in enumerate(seed_set)]

    return best_classes
# EVOLVE-BLOCK-END
