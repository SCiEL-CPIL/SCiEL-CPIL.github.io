"""Combined: Diverse prime sets + Optimized Max-Impact Greedy + Two-Pass Refinement."""
import time
from fractions import Fraction
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s kill

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
    small_primes = [2, 3, 5, 7, 11]
    large_primes = [p for p in primes if p > 11]

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

    # ---- Precompute index arrays for each prime ----
    prime_indices = {}
    for p in primes:
        arrays = []
        for r in range(p):
            if r == 0:
                idx = np.arange(p, MAX_L + 1, p, dtype=np.int64)
            else:
                idx = np.arange(r, MAX_L + 1, p, dtype=np.int64)
            arrays.append(idx)
        prime_indices[p] = arrays

    # ---- Fast L computation from coverage array ----
    def compute_L(covered, max_check=MAX_L):
        seg = covered[1:max_check + 1]
        z = np.nonzero(seg == 0)[0]
        if z.size:
            return int(z[0])
        return max_check

    # ---- Evaluation with optimized heuristics ----
    def evaluate_set(P, max_check=MAX_L, heuristic="impact"):
        """
        Assign residues to cover the longest prefix 1..L.
        """
        if not P:
            return 0, {}
        P_sorted = sorted(P)

        covered = np.zeros(max_check + 1, dtype=np.uint8)
        residues = {}
        unused_primes = set(P)

        # For 'impact' heuristic, pre-sort primes by size for quick candidate filtering
        if heuristic == "impact":
            # Small primes cover more numbers per residue, so try them first
            candidates_sorted = sorted(P, key=lambda p: p)

        for n in range(1, max_check + 1):
            if not _time_ok():
                break
            if covered[n]:
                continue

            if not unused_primes:
                break

            best_p = None
            best_score = -1
            best_r = 0

            if heuristic == "impact":
                # Try small primes first (they cover more numbers)
                for p in candidates_sorted:
                    if p not in unused_primes:
                        continue
                    r = n % p
                    idx = prime_indices[p][r]
                    if idx.size == 0:
                        continue
                    # Only count uncovered in the prefix up to max_check
                    score = np.sum(covered[idx] == 0)
                    if score > best_score:
                        best_score = score
                        best_p = p
                        best_r = r
                        # Early termination: if this prime covers n and is small,
                        # it's likely optimal
                        if p <= 13 and score > best_score * 0.8:
                            break
            elif heuristic == "small":
                for p in P_sorted:
                    if p in unused_primes:
                        best_p = p
                        break
                if best_p is None:
                    break
                best_r = n % best_p
            elif heuristic == "large":
                for p in reversed(P_sorted):
                    if p in unused_primes:
                        best_p = p
                        break
                if best_p is None:
                    break
                best_r = n % best_p
            else:
                candidates = list(unused_primes)
                for p in candidates:
                    r = n % p
                    idx = prime_indices[p][r]
                    if idx.size == 0:
                        continue
                    score = np.sum(covered[idx] == 0)
                    if score > best_score:
                        best_score = score
                        best_p = p
                        best_r = r

            if best_p is None:
                break

            residues[best_p] = best_r
            unused_primes.discard(best_p)

            idx = prime_indices[best_p][best_r]
            if idx.size:
                covered[idx] = 1

        L = compute_L(covered, max_check)
        return L, residues

    # ---- Prioritize candidate sets ----
    def small_prime_count(P):
        return sum(1 for p in P if p <= 11)

    candidate_sets.sort(key=lambda P: (-small_prime_count(P), -len(P)))

    best_L = 0
    best_classes = [[p, (i + 1) % p] for i, p in enumerate(seed_set)]

    # Evaluate each set with multiple heuristics
    for P in candidate_sets:
        if not _time_ok():
            break

        # Try 'impact' heuristic first
        L, residues = evaluate_set(P, max_check=MAX_L, heuristic="impact")
        if L > best_L:
            best_L = L
            best_classes = [[p, residues[p]] for p in sorted(P) if p in residues]

        if not _time_ok():
            break

        # Try 'small' heuristic if time permits
        L_small, residues_small = evaluate_set(P, max_check=MAX_L, heuristic="small")
        if L_small > best_L:
            best_L = L_small
            best_classes = [[p, residues_small[p]] for p in sorted(P) if p in residues_small]

        if not _time_ok():
            break

        # Try 'large' heuristic if time permits
        L_large, residues_large = evaluate_set(P, max_check=MAX_L, heuristic="large")
        if L_large > best_L:
            best_L = L_large
            best_classes = [[p, residues_large[p]] for p in sorted(P) if p in residues_large]

    # ---- Local Refinement: iterative gap-fixing with multiple attempts ----
    if best_classes and _time_ok():
        best_P = sorted(p for p, r in best_classes)

        def compute_L_from_residues(residues_dict):
            cov = np.zeros(MAX_L + 1, dtype=np.uint8)
            for p, r in residues_dict.items():
                idx = prime_indices[p][r]
                if idx.size:
                    cov[idx] = 1
            return compute_L(cov, MAX_L)

        # Iteratively try to fix gaps
        for polish_iter in range(40):
            if not _time_ok():
                break

            current_residues = {p: r for p, r in best_classes}
            current_L = compute_L_from_residues(current_residues)

            if current_L >= MAX_L:
                break

            gap_number = current_L + 1  # actual integer that's uncovered

            improved = False
            # Try all primes in the set, sorted by likelihood of success
            # Small primes are more likely to help since they cover more numbers
            for p in sorted(best_P, key=lambda x: x):
                if not _time_ok():
                    break

                old_r = current_residues.get(p, None)
                if old_r is None:
                    continue

                new_r = gap_number % p

                if new_r == old_r:
                    continue

                # Quick check: does this change uncover anything in 1..current_L?
                # We'll do a full verification only if it looks promising
                trial_residues = dict(current_residues)
                trial_residues[p] = new_r

                trial_L = compute_L_from_residues(trial_residues)

                if trial_L > current_L:
                    best_classes = [[p, r] for p, r in sorted(trial_residues.items())]
                    current_residues = trial_residues
                    current_L = trial_L
                    # Rebuild coverage
                    current_cov = np.zeros(MAX_L + 1, dtype=np.uint8)
                    for pp, rr in current_residues.items():
                        idx = prime_indices[pp][rr]
                        if idx.size:
                            current_cov[idx] = 1
                    improved = True
                    break

            if not improved:
                # Try pairs of primes
                sorted_best_P = sorted(best_P, key=lambda x: x)
                for i, p1 in enumerate(sorted_best_P):
                    if not _time_ok():
                        break
                    for p2 in sorted_best_P[i+1:]:
                        if not _time_ok():
                            break

                        old_r1 = current_residues.get(p1, None)
                        old_r2 = current_residues.get(p2, None)
                        if old_r1 is None or old_r2 is None:
                            continue

                        new_r1 = gap_number % p1
                        new_r2 = gap_number % p2

                        if new_r1 == old_r1 and new_r2 == old_r2:
                            continue

                        trial_residues = dict(current_residues)
                        trial_residues[p1] = new_r1
                        trial_residues[p2] = new_r2

                        trial_L = compute_L_from_residues(trial_residues)

                        if trial_L > current_L:
                            best_classes = [[p, r] for p, r in sorted(trial_residues.items())]
                            current_residues = trial_residues
                            current_L = trial_L
                            # Rebuild coverage
                            current_cov = np.zeros(MAX_L + 1, dtype=np.uint8)
                            for pp, rr in current_residues.items():
                                idx = prime_indices[pp][rr]
                                if idx.size:
                                    current_cov[idx] = 1
                            improved = True
                            break
                    if improved:
                        break

            if not improved:
                break

    # Final verification
    if best_classes:
        final_sum = sum(Fraction(1, p) for p, r in best_classes)
        if final_sum > 1:
            best_classes = [[p, (i + 1) % p] for i, p in enumerate(seed_set)]
        prime_list = [p for p, r in best_classes]
        if len(prime_list) != len(set(prime_list)):
            best_classes = [[p, (i + 1) % p] for i, p in enumerate(seed_set)]

    # Safety: ensure we return within time
    if time.time() > deadline - 5:
        if best_classes:
            final_sum = sum(Fraction(1, p) for p, r in best_classes)
            if final_sum > 1:
                best_classes = [[p, (i + 1) % p] for i, p in enumerate(seed_set)]

    return best_classes
# EVOLVE-BLOCK-END
