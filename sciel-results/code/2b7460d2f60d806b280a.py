import time
from fractions import Fraction
import itertools

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

    # Generate candidate prime sets:
    # For each subset of small primes (that fits in budget), fill with smallest large primes
    candidate_sets = []
    seen_sets = set()

    # Include the original seed: primes >= 13
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
            # Fill remaining budget with smallest large primes
            for p in large_primes:
                if current_sum + Fraction(1, p) <= 1:
                    current_set.append(p)
                    current_sum += Fraction(1, p)

            key = tuple(sorted(current_set))
            if key not in seen_sets:
                seen_sets.add(key)
                candidate_sets.append(current_set)

    # Also try some targeted swaps: replace some medium primes with smaller ones
    extra_candidates = []

    # Try: seed minus some primes plus 11
    for remove_list in [[53], [59], [53, 59], [61], [67]]:
        s = [p for p in seed_set if p not in remove_list]
        s_sum = sum(Fraction(1, p) for p in s)
        if s_sum + Fraction(1, 11) <= 1:
            s.append(11)
            key = tuple(sorted(s))
            if key not in seen_sets:
                seen_sets.add(key)
                extra_candidates.append(s)

    # Try: seed minus some primes plus 7
    for remove_list in [[13], [13, 17], [17], [19]]:
        s = [p for p in seed_set if p not in remove_list]
        s_sum = sum(Fraction(1, p) for p in s)
        if s_sum + Fraction(1, 7) <= 1:
            s.append(7)
            key = tuple(sorted(s))
            if key not in seen_sets:
                seen_sets.add(key)
                extra_candidates.append(s)

    # Try: seed minus some primes plus 5
    for remove_list in [[13, 17], [17, 19], [13, 19], [13, 17, 19]]:
        s = [p for p in seed_set if p not in remove_list]
        s_sum = sum(Fraction(1, p) for p in s)
        if s_sum + Fraction(1, 5) <= 1:
            s.append(5)
            key = tuple(sorted(s))
            if key not in seen_sets:
                seen_sets.add(key)
                extra_candidates.append(s)

    # Try: seed minus some primes plus 3
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

    def evaluate_set(P):
        """Optimized greedy residue assignment using efficient coverage marking."""
        if time.time() > deadline - 5:
            return 0, {}

        P_sorted = sorted(P)
        n_primes = len(P_sorted)
        if n_primes == 0:
            return 0, {}

        covered = bytearray(MAX_L + 1)
        residues = {}
        used_primes = set()

        # Process in batches to reduce time check overhead
        batch_size = 1000
        for batch_start in range(1, MAX_L + 1, batch_size):
            if time.time() > deadline - 3:
                break
            batch_end = min(batch_start + batch_size - 1, MAX_L)

            for n in range(batch_start, batch_end + 1):
                if covered[n]:
                    continue

                # Find smallest unused prime to assign
                assigned = False
                for p in P_sorted:
                    if p in used_primes:
                        continue
                    r = n % p
                    residues[p] = r
                    used_primes.add(p)

                    # Mark all k in [1, MAX_L] with k % p == r
                    start_k = r if r > 0 else p
                    covered[start_k::p] = b'\x01' * len(covered[start_k::p])

                    assigned = True
                    break

                if not assigned:
                    return 0, residues

        # Compute L efficiently
        L = 0
        for i in range(1, MAX_L + 1):
            if covered[i]:
                L = i
            else:
                break
        return L, residues

    # Local search: try to improve residue assignments for the best set
    def local_search(P, initial_residues):
        """Try to improve L by adjusting residues of some primes."""
        if time.time() > deadline - 15:
            return initial_residues, 0

        P_sorted = sorted(P)
        current_residues = dict(initial_residues)

        def compute_L(residues):
            """Compute L for given residues."""
            covered = bytearray(MAX_L + 1)
            for p, r in residues.items():
                start_k = r if r > 0 else p
                covered[start_k::p] = b'\x01' * len(covered[start_k::p])

            L = 0
            for i in range(1, MAX_L + 1):
                if covered[i]:
                    L = i
                else:
                    break
            return L

        best_L_ls = compute_L(current_residues)

        # Try adjusting residues of primes with large gaps
        for p in P_sorted:
            if time.time() > deadline - 10:
                break

            current_r = current_residues.get(p, 0)

            # Try a few alternative residues
            for alt_r in [0, 1, (p - 1) % p, (p // 2) % p]:
                if alt_r == current_r:
                    continue

                test_residues = dict(current_residues)
                test_residues[p] = alt_r
                test_L = compute_L(test_residues)

                if test_L > best_L_ls:
                    best_L_ls = test_L
                    current_residues = test_residues

        return current_residues, best_L_ls

    # Evaluate candidates, prioritizing those with smaller primes first
    def small_prime_count(P):
        return sum(1 for p in P if p <= 11)

    candidate_sets.sort(key=lambda P: (-small_prime_count(P), len(P)))

    for P in candidate_sets:
        if time.time() > deadline - 10:
            break

        # Verify budget
        s = sum(Fraction(1, p) for p in P)
        if s > 1:
            continue

        L, residues = evaluate_set(P)
        if L > best_L:
            # Try local search to improve
            if time.time() < deadline - 15:
                improved_residues, improved_L = local_search(P, residues)
                if improved_L > L:
                    L = improved_L
                    residues = improved_residues

            if L > best_L:
                best_L = L
                best_classes = [[p, residues[p]] for p in sorted(P) if p in residues]

    # Fallback: if best_L is 0, use seed
    if best_L == 0:
        best_classes = [[p, (i + 1) % p] for i, p in enumerate(seed_set)]

    # Final verification
    final_sum = sum(Fraction(1, p) for p, r in best_classes)
    if final_sum > 1:
        best_classes = [[p, (i + 1) % p] for i, p in enumerate(seed_set)]

    # Verify distinct primes
    prime_list = [p for p, r in best_classes]
    if len(prime_list) != len(set(prime_list)):
        best_classes = [[p, (i + 1) % p] for i, p in enumerate(seed_set)]

    return best_classes
