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

    # ---- Fast evaluation using vectorized gap-filling ----
    def evaluate_set_fast(P, max_check=MAX_L):
        """
        Vectorized greedy: find first gap, assign best prime/residue, repeat.
        """
        if not P:
            return 0, {}

        covered = np.zeros(max_check + 1, dtype=np.uint8)
        residues = {}
        unused_primes = set(P)
        P_sorted = sorted(P)

        # Precompute for each prime and residue: the indices
        # We'll track which primes are used

        while unused_primes:
            if not _time_ok():
                break

            # Find first uncovered number
            seg = covered[1:max_check + 1]
            z = np.nonzero(seg == 0)[0]
            if z.size == 0:
                break
            gap = int(z[0]) + 1

            # Find best (p, r) that covers gap and maximizes new coverage
            best_p = None
            best_r = 0
            best_new = -1

            for p in P_sorted:
                if p not in unused_primes:
                    continue
                r = gap % p
                idx = prime_indices[p][r]
                if idx.size == 0:
                    continue
                # Count new (uncovered) numbers
                new_count = np.sum(covered[idx] == 0)
                if new_count > best_new:
                    best_new = new_count
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

    # Evaluate each set
    for P in candidate_sets:
        if not _time_ok():
            break

        L, residues = evaluate_set_fast(P, max_check=MAX_L)
        if L > best_L:
            best_L = L
            best_classes = [[p, residues[p]] for p in sorted(P) if p in residues]

    # ---- Local Refinement: iterative gap-fixing with incremental updates ----
    if best_classes and _time_ok():
        best_P = sorted(p for p, r in best_classes)

        def build_coverage(residues_dict):
            cov = np.zeros(MAX_L + 1, dtype=np.uint8)
            for p, r in residues_dict.items():
                idx = prime_indices[p][r]
                if idx.size:
                    cov[idx] = 1
            return cov

        def compute_L_from_cov(cov):
            return compute_L(cov, MAX_L)

        current_residues = {p: r for p, r in best_classes}
        current_cov = build_coverage(current_residues)
        current_L = compute_L_from_cov(current_cov)

        for polish_iter in range(50):
            if not _time_ok():
                break

            if current_L >= MAX_L:
                break

            gap_number = current_L + 1

            improved = False
            # Try single prime changes
            for p in sorted(best_P):
                if not _time_ok():
                    break

                old_r = current_residues.get(p, None)
                if old_r is None:
                    continue

                new_r = gap_number % p
                if new_r == old_r:
                    continue

                # Incremental update: remove old coverage, add new
                old_idx = prime_indices[p][old_r]
                new_idx = prime_indices[p][new_r]

                # Temporarily remove old
                if old_idx.size:
                    current_cov[old_idx] = 0
                # Add new
                if new_idx.size:
                    current_cov[new_idx] = 1

                trial_L = compute_L_from_cov(current_cov)

                if trial_L > current_L:
                    current_residues[p] = new_r
                    current_L = trial_L
                    improved = True
                    best_classes = [[p, r] for p, r in sorted(current_residues.items())]
                    break
                else:
                    # Revert
                    if new_idx.size:
                        current_cov[new_idx] = 0
                    if old_idx.size:
                        current_cov[old_idx] = 1

            if improved:
                continue

            # Try pairs of primes
            for i, p1 in enumerate(sorted(best_P)):
                if not _time_ok():
                    break
                for p2 in sorted(best_P)[i+1:]:
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

                    # Remove old coverages
                    old_idx1 = prime_indices[p1][old_r1]
                    old_idx2 = prime_indices[p2][old_r2]
                    new_idx1 = prime_indices[p1][new_r1]
                    new_idx2 = prime_indices[p2][new_r2]

                    if old_idx1.size:
                        current_cov[old_idx1] = 0
                    if old_idx2.size:
                        current_cov[old_idx2] = 0
                    if new_idx1.size:
                        current_cov[new_idx1] = 1
                    if new_idx2.size:
                        current_cov[new_idx2] = 1

                    trial_L = compute_L_from_cov(current_cov)

                    if trial_L > current_L:
                        current_residues[p1] = new_r1
                        current_residues[p2] = new_r2
                        current_L = trial_L
                        improved = True
                        best_classes = [[p, r] for p, r in sorted(current_residues.items())]
                        break
                    else:
                        # Revert
                        if new_idx1.size:
                            current_cov[new_idx1] = 0
                        if new_idx2.size:
                            current_cov[new_idx2] = 0
                        if old_idx1.size:
                            current_cov[old_idx1] = 1
                        if old_idx2.size:
                            current_cov[old_idx2] = 1

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
