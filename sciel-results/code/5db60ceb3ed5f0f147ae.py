"""Combined: Diverse prime sets + Vectorized Greedy + Fast Local Search + Random Restarts."""
import time
import random
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

    # ---- Fast vectorized evaluate: assign residues to cover longest prefix ----
    def evaluate_vectorized(P, max_check=MAX_L):
        """
        Vectorized greedy: repeatedly find the smallest uncovered number,
        assign the prime that covers the most uncovered numbers in its class.
        """
        if not P:
            return 0, {}
        
        covered = np.zeros(max_check + 1, dtype=np.uint8)
        residues = {}
        unused = list(P)
        
        # Sort by prime size (smaller primes cover more numbers per residue)
        unused.sort()
        
        # Work in chunks to avoid processing all 100k numbers one by one
        n = 1
        while n <= max_check and unused and _time_ok():
            # Find next uncovered number
            while n <= max_check and covered[n]:
                n += 1
            if n > max_check:
                break
            
            # Find best prime to assign
            best_p = None
            best_score = -1
            best_r = 0
            
            for p in unused:
                r = n % p
                idx = prime_indices[p][r]
                if idx.size == 0:
                    continue
                # Count how many new numbers this would cover
                score = np.sum(covered[idx] == 0)
                if score > best_score:
                    best_score = score
                    best_p = p
                    best_r = r
            
            if best_p is None:
                break
            
            residues[best_p] = best_r
            unused.remove(best_p)
            
            idx = prime_indices[best_p][best_r]
            if idx.size:
                covered[idx] = 1
        
        L = compute_L(covered, max_check)
        return L, residues

    # ---- Fast L computation from coverage array ----
    def compute_L(covered, max_check=MAX_L):
        seg = covered[1:max_check + 1]
        z = np.nonzero(seg == 0)[0]
        if z.size:
            return int(z[0])
        return max_check

    # ---- Fast local search: try changing one or two residues at a time ----
    def fast_local_search(residues_dict, max_check=MAX_L, max_iters=50):
        """
        Iteratively try to improve L by changing residues of individual primes
        or pairs of primes. Uses incremental coverage updates.
        """
        current = dict(residues_dict)
        primes_list = sorted(current.keys())
        current_L = compute_L_from_residues(current)
        
        for it in range(max_iters):
            if not _time_ok():
                break
            if current_L >= max_check:
                break
            
            gap = current_L + 1
            improved = False
            
            # Try single-prime changes
            for p in primes_list:
                if not _time_ok():
                    break
                new_r = gap % p
                if new_r == current[p]:
                    continue
                
                # Quick check: does this help?
                trial = dict(current)
                trial[p] = new_r
                trial_L = compute_L_from_residues(trial)
                if trial_L > current_L:
                    current = trial
                    current_L = trial_L
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try pair changes
            for i, p1 in enumerate(primes_list):
                if not _time_ok():
                    break
                for p2 in primes_list[i+1:]:
                    if not _time_ok():
                        break
                    new_r1 = gap % p1
                    new_r2 = gap % p2
                    if new_r1 == current[p1] and new_r2 == current[p2]:
                        continue
                    
                    trial = dict(current)
                    trial[p1] = new_r1
                    trial[p2] = new_r2
                    trial_L = compute_L_from_residues(trial)
                    if trial_L > current_L:
                        current = trial
                        current_L = trial_L
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                break
        
        return current_L, current

    # ---- Prioritize candidate sets ----
    def small_prime_count(P):
        return sum(1 for p in P if p <= 11)

    candidate_sets.sort(key=lambda P: (-small_prime_count(P), -len(P)))

    best_L = 0
    best_classes = [[p, (i + 1) % p] for i, p in enumerate(seed_set)]

    # Evaluate each set with vectorized greedy
    for P in candidate_sets:
        if not _time_ok():
            break

        L, residues = evaluate_vectorized(P, max_check=MAX_L)
        if L > best_L:
            best_L = L
            best_classes = [[p, residues[p]] for p in sorted(P) if p in residues]

    # ---- Local Refinement using fast local search ----
    if best_classes and _time_ok():
        def compute_L_from_residues(residues_dict):
            cov = np.zeros(MAX_L + 1, dtype=np.uint8)
            for p, r in residues_dict.items():
                idx = prime_indices[p][r]
                if idx.size:
                    cov[idx] = 1
            return compute_L(cov, MAX_L)

        current_residues = {p: r for p, r in best_classes}
        L_final, refined_residues = fast_local_search(current_residues, max_check=MAX_L, max_iters=100)
        if L_final > best_L:
            best_L = L_final
            best_classes = [[p, r] for p, r in sorted(refined_residues.items())]

    # ---- Random restarts with different prime set compositions ----
    if _time_ok():
        random.seed(42)
        restart_count = 0
        max_restarts = 30

        while restart_count < max_restarts and _time_ok():
            restart_count += 1
            
            # Generate a random valid prime set
            P = []
            budget = Fraction(1, 1)
            
            # Randomly decide which small primes to include
            for sp in small_primes:
                if random.random() < 0.5:
                    if Fraction(1, sp) <= budget:
                        P.append(sp)
                        budget -= Fraction(1, sp)
            
            # Fill remaining budget with random large primes
            remaining_large = [p for p in large_primes if p not in P]
            random.shuffle(remaining_large)
            for p in remaining_large:
                if Fraction(1, p) <= budget:
                    P.append(p)
                    budget -= Fraction(1, p)
            
            if not P:
                continue
            
            # Verify budget
            total = sum(Fraction(1, p) for p in P)
            if total > 1:
                continue
            
            # Evaluate with vectorized greedy
            L, residues = evaluate_vectorized(P, max_check=MAX_L)
            if L > best_L:
                best_L = L
                best_classes = [[p, residues[p]] for p in sorted(P) if p in residues]
                
                # Apply local search on the new best
                if _time_ok():
                    current_residues = {p: r for p, r in best_classes}
                    L_final, refined_residues = fast_local_search(current_residues, max_check=MAX_L, max_iters=50)
                    if L_final > best_L:
                        best_L = L_final
                        best_classes = [[p, r] for p, r in sorted(refined_residues.items())]

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
