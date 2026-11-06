"""Combined program: CSP-style local search with cost-aware moves and random restarts."""
import time
import math
import random
from fractions import Fraction

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    DEADLINE = 110  # safety margin before 120s/150s limit

    def get_primes(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(math.sqrt(n)) + 1):
            if sieve[i]:
                for j in range(i * i, n + 1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]

    all_primes = get_primes(1000)

    # Build prime set: start with primes >= 11, remove largest until sum(1/p) <= 1
    candidates = [p for p in all_primes if p >= 11]
    candidates_desc = sorted(candidates, reverse=True)
    current_sum = sum(Fraction(1, p) for p in candidates)
    primes_set = list(candidates)
    while current_sum > 1:
        if not candidates_desc:
            break
        p_remove = candidates_desc.pop()
        primes_set.remove(p_remove)
        current_sum -= Fraction(1, p_remove)

    primes_set = sorted(primes_set)
    n_primes = len(primes_set)

    # Verify budget
    exact_sum = sum(Fraction(1, p) for p in primes_set)
    if exact_sum > 1:
        primes_set = [p for p in all_primes if p >= 13]
        exact_sum = sum(Fraction(1, p) for p in primes_set)
        n_primes = len(primes_set)

    L_CHECK_LIMIT = 5000
    L_FINAL_LIMIT = 100000

    def evaluate_coverage(residues, L_limit):
        cover_count = [0] * (L_limit + 1)
        for i in range(n_primes):
            p = primes_set[i]
            r = residues[i]
            start = r
            if start == 0:
                start = p
            if start <= L_limit:
                for n in range(start, L_limit + 1, p):
                    cover_count[n] += 1
        L = 0
        for n in range(1, L_limit + 1):
            if cover_count[n] == 0:
                break
            L = n
        return L, cover_count

    def update_counts(cover_count, p, r_old, r_new, L_limit):
        if r_old == r_new:
            return
        start_old = r_old
        if start_old == 0:
            start_old = p
        if start_old <= L_limit:
            for n in range(start_old, L_limit + 1, p):
                cover_count[n] -= 1
        start_new = r_new
        if start_new == 0:
            start_new = p
        if start_new <= L_limit:
            for n in range(start_new, L_limit + 1, p):
                cover_count[n] += 1

    def calc_L(cover_count, L_limit):
        L = 0
        for n in range(1, L_limit + 1):
            if cover_count[n] == 0:
                break
            L = n
        return L

    def init_residues_diagonal():
        res = [0] * n_primes
        for i in range(n_primes):
            p = primes_set[i]
            res[i] = (i + 1) % p
        return res

    def init_residues_random():
        res = [0] * n_primes
        for i in range(n_primes):
            res[i] = random.randint(0, primes_set[i] - 1)
        return res

    def init_residues_greedy():
        covered = [0] * (L_CHECK_LIMIT + 1)
        res = [0] * n_primes
        for i in range(n_primes):
            p = primes_set[i]
            best_r = 0
            best_count = -1
            for r in range(p):
                start = r
                if start == 0:
                    start = p
                count = 0
                for n in range(start, L_CHECK_LIMIT + 1, p):
                    if covered[n] == 0:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_r = r
            res[i] = best_r
            start = best_r
            if start == 0:
                start = p
            for n in range(start, L_CHECK_LIMIT + 1, p):
                covered[n] += 1
        return res

    best_L = 0
    best_classes = None
    best_residues = None

    # Try diagonal initialization first
    residues = init_residues_diagonal()
    L, cover_count = evaluate_coverage(residues, L_CHECK_LIMIT)
    if L > best_L:
        best_L = L
        best_classes = [[primes_set[i], residues[i]] for i in range(n_primes)]
        best_residues = residues[:]

    # Try greedy initialization
    residues_g = init_residues_greedy()
    L_g, cover_count_g = evaluate_coverage(residues_g, L_CHECK_LIMIT)
    if L_g > best_L:
        best_L = L_g
        best_classes = [[primes_set[i], residues_g[i]] for i in range(n_primes)]
        best_residues = residues_g[:]

    # Main search: local search with cost-aware moves and random restarts
    while time.time() - start_time < DEADLINE - 5:
        # Random restart with perturbation from best or random
        if random.random() < 0.3 or best_residues is None:
            residues = init_residues_random()
        else:
            residues = best_residues[:]
            # Perturb a few residues
            for _ in range(random.randint(1, 5)):
                idx = random.randint(0, n_primes - 1)
                p = primes_set[idx]
                residues[idx] = random.randint(0, p - 1)

        L, cover_count = evaluate_coverage(residues, L_CHECK_LIMIT)
        if L > best_L:
            best_L = L
            best_classes = [[primes_set[i], residues[i]] for i in range(n_primes)]
            best_residues = residues[:]

        # Local search: greedy prefix-extension with cost awareness
        stuck_counter = 0
        max_stuck = 200

        while time.time() - start_time < DEADLINE - 5 and stuck_counter < max_stuck:
            if L >= L_CHECK_LIMIT:
                break

            hole = L + 1

            best_new_L = L
            best_idx = -1
            best_r_new = -1
            best_r_old = -1
            best_cost = float('inf')

            # Try each prime to cover the hole, preferring low-cost moves
            indices = list(range(n_primes))
            random.shuffle(indices)

            for idx in indices:
                p = primes_set[idx]
                r_old = residues[idx]
                r_new = hole % p

                if r_old == r_new:
                    continue

                # Calculate cost: number of k in [1, hole-1] that become uncovered
                cost = 0
                start = r_old
                if start == 0:
                    start = p

                if start < hole:
                    for k in range(start, hole, p):
                        if cover_count[k] == 1:
                            cost += 1

                # If cost is 0, this is a free move - take it immediately
                if cost == 0:
                    best_cost = 0
                    best_idx = idx
                    best_r_new = r_new
                    best_r_old = r_old
                    break

                # Otherwise, track the lowest cost move
                if cost < best_cost:
                    best_cost = cost
                    best_idx = idx
                    best_r_new = r_new
                    best_r_old = r_old

            if best_idx != -1 and (best_cost == 0 or best_cost <= 2):
                p = primes_set[best_idx]
                update_counts(cover_count, p, best_r_old, best_r_new, L_CHECK_LIMIT)
                residues[best_idx] = best_r_new
                new_L = calc_L(cover_count, L_CHECK_LIMIT)

                if new_L > L:
                    L = new_L
                    stuck_counter = 0
                    if L > best_L:
                        best_L = L
                        best_classes = [[primes_set[i], residues[i]] for i in range(n_primes)]
                        best_residues = residues[:]
                else:
                    # Revert if no improvement
                    update_counts(cover_count, p, best_r_new, best_r_old, L_CHECK_LIMIT)
                    residues[best_idx] = best_r_old
                    stuck_counter += 1
            else:
                # No good move found; try random perturbation
                stuck_counter += 1
                if stuck_counter > max_stuck // 2:
                    for _ in range(10):
                        idx = random.randint(0, n_primes - 1)
                        p = primes_set[idx]
                        old_r = residues[idx]
                        new_r = random.randint(0, p - 1)
                        if new_r == old_r:
                            continue
                        update_counts(cover_count, p, old_r, new_r, L_CHECK_LIMIT)
                        residues[idx] = new_r
                        new_L = calc_L(cover_count, L_CHECK_LIMIT)
                        if new_L >= L:
                            L = new_L
                            break
                        else:
                            update_counts(cover_count, p, new_r, old_r, L_CHECK_LIMIT)
                            residues[idx] = old_r
                    stuck_counter = 0

    # Final evaluation up to L_FINAL_LIMIT
    if best_classes:
        final_L = 0
        covered = [False] * (L_FINAL_LIMIT + 1)
        for p, r in best_classes:
            start = r
            if start == 0:
                start = p
            if start <= L_FINAL_LIMIT:
                for n in range(start, L_FINAL_LIMIT + 1, p):
                    covered[n] = True
        for n in range(1, L_FINAL_LIMIT + 1):
            if covered[n]:
                final_L = n
            else:
                break
        if final_L > best_L:
            best_L = final_L

    # Safety: verify budget
    if best_classes:
        total = sum(Fraction(1, p) for p, r in best_classes)
        if total > 1:
            primes_set_fb = [p for p in all_primes if p >= 13]
            n_primes_fb = len(primes_set_fb)
            residues_fb = [(i + 1) % p for i, p in enumerate(primes_set_fb)]
            best_classes = [[primes_set_fb[i], residues_fb[i]] for i in range(n_primes_fb)]

    if not best_classes:
        primes_set_fb = [p for p in all_primes if p >= 13]
        n_primes_fb = len(primes_set_fb)
        residues_fb = [(i + 1) % p for i, p in enumerate(primes_set_fb)]
        best_classes = [[primes_set_fb[i], residues_fb[i]] for i in range(n_primes_fb)]

    return best_classes

# EVOLVE-BLOCK-END
