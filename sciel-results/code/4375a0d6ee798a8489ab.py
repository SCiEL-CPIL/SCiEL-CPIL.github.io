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

    # ---- Vectorized greedy residue assignment with better heuristics ----
    def evaluate_set(P, max_check=MAX_L):
        """Assign residues to cover the longest prefix 1..L using vectorized greedy."""
        if not P:
            return 0, {}
        P_sorted = sorted(P)
        
        covered = np.zeros(max_check + 1, dtype=np.uint8)
        residues = {}
        assigned_primes = set()

        # 1. Initial Greedy Pass - pick smallest unused prime for each uncovered n
        for n in range(1, max_check + 1):
            if not _time_ok():
                break
            if covered[n]:
                continue
            
            # Heuristic: Pick the smallest unused prime
            best_p = None
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

        # 2. Calculate L
        def get_L():
            if max_check < 1:
                return 0
            seg = covered[1:max_check + 1]
            z = np.nonzero(seg == 0)[0]
            if z.size:
                return int(z[0])
            return max_check

        L = get_L()
        return L, residues

    # Prioritize candidate sets: more small primes first, then more total primes
    def small_prime_count(P):
        return sum(1 for p in P if p <= 23)

    candidate_sets.sort(key=lambda P: (-small_prime_count(P), -len(P)))

    best_L = 0
    best_classes = [[p, (i + 1) % p] for i, p in enumerate(seed_set)]

    for P in candidate_sets:
        if not _time_ok():
            break
        L, residues = evaluate_set(P)
        if L > best_L:
            best_L = L
            best_classes = [[p, residues[p]] for p in sorted(P) if p in residues]

    # ---- Local refinement: try to extend the best prefix by re-optimizing residues ----
    if best_classes and _time_ok():
        best_P = sorted(p for p, r in best_classes)
        # Re-run with a slightly larger check range to see if we can extend
        L2, residues2 = evaluate_set(best_P, max_check=min(MAX_L, best_L + 50000))
        if L2 > best_L:
            best_L = L2
            best_classes = [[p, residues2[p]] for p in sorted(best_P) if p in residues2]

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
