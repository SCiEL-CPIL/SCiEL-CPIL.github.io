import time
from fractions import Fraction
import itertools
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill

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

    # Generate candidate prime sets
    candidate_sets = []
    seen_sets = set()

    seed_set = [p for p in primes if p >= 13]
    candidate_sets.append(seed_set)
    seen_sets.add(tuple(sorted(seed_set)))

    # Systematically try subsets of small primes
    for r in range(len(small_primes) + 1):
        for subset in itertools.combinations(small_primes, r):
            current_sum = sum(Fraction(1, p) for p in subset)
            if current_sum > 1:
                continue
            current_set = list(subset)
            for p in large_primes:
                if current_sum + Fraction(1, p) <= 1:
                    current_set.append(p)
                    current_sum += Fraction(1, p)
            key = tuple(sorted(current_set))
            if key not in seen_sets:
                seen_sets.add(key)
                candidate_sets.append(current_set)

    # Targeted swaps
    extra_candidates = []
    for remove_list in [[53], [59], [53, 59], [61], [67]]:
        s = [p for p in seed_set if p not in remove_list]
        s_sum = sum(Fraction(1, p) for p in s)
        if s_sum + Fraction(1, 11) <= 1:
            s.append(11)
            key = tuple(sorted(s))
            if key not in seen_sets:
                seen_sets.add(key)
                extra_candidates.append(s)
    for remove_list in [[13], [13, 17], [17], [19]]:
        s = [p for p in seed_set if p not in remove_list]
        s_sum = sum(Fraction(1, p) for p in s)
        if s_sum + Fraction(1, 7) <= 1:
            s.append(7)
            key = tuple(sorted(s))
            if key not in seen_sets:
                seen_sets.add(key)
                extra_candidates.append(s)
    for remove_list in [[13, 17], [17, 19], [13, 19], [13, 17, 19]]:
        s = [p for p in seed_set if p not in remove_list]
        s_sum = sum(Fraction(1, p) for p in s)
        if s_sum + Fraction(1, 5) <= 1:
            s.append(5)
            key = tuple(sorted(s))
            if key not in seen_sets:
                seen_sets.add(key)
                extra_candidates.append(s)
    for remove_list in [[13, 17, 19, 23], [17, 19, 23, 29], [13, 17, 19, 23, 29]]:
        s = [p for p in seed_set if p not in remove_list]
        s_sum = sum(Fraction(1, p) for p in s)
        if s_sum + Fraction(1, 3) <= 1:
            s.append(3)
            key = tuple(sorted(s))
            if key not in seen_sets:
                seen_sets.add(key)
                extra_candidates.append(s)

    candidate_sets.extend(extra_candidates)

    MAX_L = 100000
    best_L = 0
    best_classes = [[p, (i + 1) % p] for i, p in enumerate(seed_set)]

    def compute_L(covered):
        """Compute the longest prefix 1..L where all are covered."""
        L = 0
        for i in range(1, MAX_L + 1):
            if covered[i]:
                L = i
            else:
                break
        return L

    def max_impact_greedy(P):
        """Assign residues greedily: each prime covers the most uncovered numbers."""
        if time.time() > deadline - 5:
            return 0, {}

        P_sorted = sorted(P)
        n_primes = len(P_sorted)
        if n_primes == 0:
            return 0, {}

        covered = bytearray(MAX_L + 1)
        residues = {}
        unassigned = list(P_sorted)

        # For each prime, find the residue that maximizes newly covered numbers
        # Process primes in order (smallest first for better coverage density)
        for p in P_sorted:
            if time.time() > deadline - 3:
                break

            # Count uncovered numbers for each residue class
            best_r = 0
            best_count = -1
            for r in range(p):
                count = 0
                start_k = r if r > 0 else p
                for k in range(start_k, MAX_L + 1, p):
                    if not covered[k]:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_r = r

            residues[p] = best_r

            # Mark covered
            start_k = best_r if best_r > 0 else p
            for k in range(start_k, MAX_L + 1, p):
                covered[k] = 1

        L = compute_L(covered)
        return L, residues

    def local_search(P, residues, covered, L, max_iters=50):
        """Try shifting residues of individual primes to extend L."""
        if time.time() > deadline - 5:
            return L, residues, covered

        current_L = L
        current_residues = dict(residues)
        current_covered = bytearray(covered)

        for iteration in range(max_iters):
            if time.time() > deadline - 3:
                break

            improved = False
            # Find the first gap
            gap_pos = current_L + 1
            if gap_pos > MAX_L:
                break

            # Try shifting each prime's residue to cover gap_pos
            for p in sorted(P):
                if time.time() > deadline - 3:
                    break

                old_r = current_residues[p]
                new_r = gap_pos % p

                if new_r == old_r:
                    continue

                # Simulate: remove old residue class, add new one
                # Remove old
                old_start = old_r if old_r > 0 else p
                for k in range(old_start, MAX_L + 1, p):
                    current_covered[k] = 0
                # Add new
                new_start = new_r if new_r > 0 else p
                for k in range(new_start, MAX_L + 1, p):
                    current_covered[k] = 1

                new_L = compute_L(current_covered)

                if new_L > current_L:
                    current_residues[p] = new_r
                    current_L = new_L
                    improved = True
                    break
                else:
                    # Revert
                    for k in range(new_start, MAX_L + 1, p):
                        current_covered[k] = 0
                    for k in range(old_start, MAX_L + 1, p):
                        current_covered[k] = 1

            if not improved:
                break

        return current_L, current_residues, current_covered

    # Evaluate candidates with max-impact greedy + local search
    def small_prime_count(P):
        return sum(1 for p in P if p <= 11)

    candidate_sets.sort(key=lambda P: (-small_prime_count(P), len(P)))

    for P in candidate_sets:
        if time.time() > deadline - 10:
            break

        s = sum(Fraction(1, p) for p in P)
        if s > 1:
            continue

        L, residues = max_impact_greedy(P)
        if L == 0:
            continue

        # Get covered array for local search
        covered = bytearray(MAX_L + 1)
        for p, r in residues.items():
            start_k = r if r > 0 else p
            for k in range(start_k, MAX_L + 1, p):
                covered[k] = 1

        L2, residues2, covered2 = local_search(P, residues, covered, L)

        if L2 > best_L:
            best_L = L2
            best_classes = [[p, residues2[p]] for p in sorted(P) if p in residues2]

    # Fallback
    if best_L == 0:
        best_classes = [[p, (i + 1) % p] for i, p in enumerate(seed_set)]

    # Final verification
    final_sum = sum(Fraction(1, p) for p, r in best_classes)
    if final_sum > 1:
        best_classes = [[p, (i + 1) % p] for i, p in enumerate(seed_set)]

    prime_list = [p for p, r in best_classes]
    if len(prime_list) != len(set(prime_list)):
        best_classes = [[p, (i + 1) % p] for i, p in enumerate(seed_set)]

    return best_classes
