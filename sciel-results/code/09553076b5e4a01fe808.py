import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    random.seed(42)

    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n ** 0.5) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, n + 1) if is_prime[i]]

    primes = sieve(1000)

    def build_set(base_primes):
        s = sum(Fraction(1, p) for p in base_primes)
        if s > 1:
            return []
        final_set = list(base_primes)
        for p in primes:
            if p in base_primes:
                continue
            if s + Fraction(1, p) <= 1:
                final_set.append(p)
                s += Fraction(1, p)
        return final_set

    # Generate candidate prime sets
    candidates = []
    candidates.append(build_set([2, 5, 11]))
    candidates.append(build_set([2, 3, 11]))
    candidates.append(build_set([3, 5, 7]))
    candidates.append(build_set([2, 7, 13]))
    candidates.append(build_set([5, 7, 11]))
    candidates.append(build_set([2, 3, 7]))
    candidates.append(build_set([2, 5, 7]))
    candidates.append(build_set([3, 5, 11]))

    # Fast coverage check using NumPy
    import numpy as np

    def compute_L(primes_list, residues, max_check=100000):
        """Compute the largest L such that all 1..L are covered."""
        if not primes_list:
            return 0
        covered = np.zeros(max_check + 1, dtype=bool)
        for p, r in zip(primes_list, residues):
            if r == 0:
                start = p
            else:
                start = r
            if start <= max_check:
                covered[start::p] = True
        # Find first gap
        gaps = np.where(~covered[1:])[0]
        if len(gaps) == 0:
            return max_check
        return gaps[0]

    def greedy_assign(primes_list):
        """Greedy assignment: for each number 1, 2, 3..., assign the first unused prime that covers it."""
        P_sorted = sorted(primes_list)
        residues = {}
        used = set()
        max_check = 100000
        covered = np.zeros(max_check + 1, dtype=bool)

        for n in range(1, max_check + 1):
            if time.time() > deadline - 5:
                break
            if covered[n]:
                continue
            assigned = False
            for p in P_sorted:
                if p in used:
                    continue
                r = n % p
                residues[p] = r
                used.add(p)
                start = r if r > 0 else p
                if start <= max_check:
                    covered[start::p] = True
                assigned = True
                break
            if not assigned:
                break

        L = compute_L(P_sorted, [residues.get(p, 0) for p in P_sorted])
        return L, [[p, residues.get(p, 0)] for p in P_sorted]

    def local_search(primes_list, init_residues, max_iter=500):
        """Local search: try to improve by changing one residue at a time."""
        P_sorted = sorted(primes_list)
        residues = list(init_residues)
        current_L = compute_L(P_sorted, residues)

        for _ in range(max_iter):
            if time.time() > deadline - 10:
                break
            improved = False
            # Try changing each residue
            for i in range(len(P_sorted)):
                p = P_sorted[i]
                old_r = residues[i]
                # Try a few random residues
                for _ in range(3):
                    new_r = random.randint(0, p - 1)
                    if new_r == old_r:
                        continue
                    residues[i] = new_r
                    new_L = compute_L(P_sorted, residues)
                    if new_L > current_L:
                        current_L = new_L
                        improved = True
                        break
                    residues[i] = old_r
                if improved:
                    break
            if not improved:
                break

        return current_L, [[p, r] for p, r in zip(P_sorted, residues)]

    best_L = 0
    best_classes = []

    for P in candidates:
        if time.time() > deadline - 15:
            break
        if not P:
            continue

        # First, try greedy
        L_greedy, classes_greedy = greedy_assign(P)
        if L_greedy > best_L:
            best_L = L_greedy
            best_classes = classes_greedy

        # Then, try local search starting from greedy
        if time.time() > deadline - 10:
            continue
        init_residues = [c[1] for c in classes_greedy]
        L_ls, classes_ls = local_search(P, init_residues, max_iter=200)
        if L_ls > best_L:
            best_L = L_ls
            best_classes = classes_ls

        # Also try a few random initializations
        for _ in range(3):
            if time.time() > deadline - 10:
                break
            random_residues = [random.randint(0, p - 1) for p in P]
            L_rand, classes_rand = local_search(P, random_residues, max_iter=100)
            if L_rand > best_L:
                best_L = L_rand
                best_classes = classes_rand

    return best_classes
