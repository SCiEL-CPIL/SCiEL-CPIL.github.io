"""Combined program: multi-prime-set structural search + gap-targeted extension + SA polish."""
import time
import math
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    DEADLINE = 140  # safety margin before 150s limit

    def get_primes(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(math.sqrt(n)) + 1):
            if sieve[i]:
                for j in range(i * i, n + 1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]

    all_primes = get_primes(1000)

    # Define multiple candidate prime sets with sum(1/p) <= 1
    # These represent different structural approaches to the problem
    candidate_sets = []
    
    # Set 1: primes >= 13 (baseline)
    set1 = [p for p in all_primes if p >= 13]
    candidate_sets.append(set1)
    
    # Set 2: {3, 5} + primes >= 13
    set2 = [3, 5] + [p for p in all_primes if p >= 13]
    candidate_sets.append(set2)
    
    # Set 3: {2} + primes >= 67
    set3 = [2] + [p for p in all_primes if p >= 67]
    candidate_sets.append(set3)
    
    # Set 4: {3, 5, 7} + primes >= 150
    set4 = [3, 5, 7] + [p for p in all_primes if p >= 150]
    candidate_sets.append(set4)
    
    # Set 5: {2, 3} + primes >= 43
    set5 = [2, 3] + [p for p in all_primes if p >= 43]
    candidate_sets.append(set5)
    
    # Validate sets with exact rationals
    valid_sets = []
    for P in candidate_sets:
        s = sum(Fraction(1, p) for p in P)
        if s <= 1:
            valid_sets.append(sorted(P))
    
    if not valid_sets:
        # Fallback
        primes_seed = [p for p in all_primes if p >= 13]
        return [[p, (i + 1) % p] for i, p in enumerate(primes_seed)]

    L_CHECK_LIMIT = 20000
    L_FINAL_LIMIT = 100000

    def evaluate_coverage(primes_list, residues, L_limit):
        """Return L where every 1..L is covered."""
        cover_count = [0] * (L_limit + 1)
        for i in range(len(primes_list)):
            p = primes_list[i]
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

    def init_residues_diagonal(primes_list):
        """Diagonal init: residue i covers number i+1."""
        res = [0] * len(primes_list)
        for i in range(len(primes_list)):
            p = primes_list[i]
            res[i] = (i + 1) % p
        return res

    def init_residues_greedy(primes_list, L_limit):
        """Greedy init: for each prime, pick residue covering most uncovered numbers."""
        covered = [0] * (L_limit + 1)
        res = [0] * len(primes_list)
        for i in range(len(primes_list)):
            p = primes_list[i]
            best_r = 0
            best_count = -1
            for r in range(p):
                start = r
                if start == 0:
                    start = p
                count = 0
                for n in range(start, L_limit + 1, p):
                    if covered[n] == 0:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_r = r
            res[i] = best_r
            start = best_r
            if start == 0:
                start = p
            for n in range(start, L_limit + 1, p):
                covered[n] += 1
        return res

    def gap_targeted_extension(primes_list, residues, cover_count, L, L_limit, deadline):
        """Extend coverage by targeting the first gap. Returns (L, residues, cover_count)."""
        n_primes = len(primes_list)
        stuck_counter = 0
        max_stuck = 100
        
        while time.time() < deadline:
            if L >= L_limit:
                break
            
            hole = L + 1
            best_new_L = L
            best_idx = -1
            best_r_new = -1
            best_r_old = -1

            # Try each prime to cover the hole
            for idx in range(n_primes):
                p = primes_list[idx]
                r_old = residues[idx]
                r_new = hole % p

                if r_old == r_new:
                    continue

                update_counts(cover_count, p, r_old, r_new, L_limit)
                residues[idx] = r_new
                new_L = calc_L(cover_count, L_limit)

                if new_L > best_new_L:
                    best_new_L = new_L
                    best_idx = idx
                    best_r_new = r_new
                    best_r_old = r_old
                    # Revert to test other primes fairly
                    update_counts(cover_count, p, r_new, r_old, L_limit)
                    residues[idx] = r_old
                else:
                    update_counts(cover_count, p, r_new, r_old, L_limit)
                    residues[idx] = r_old

            if best_idx != -1:
                p = primes_list[best_idx]
                update_counts(cover_count, p, best_r_old, best_r_new, L_limit)
                residues[best_idx] = best_r_new
                L = best_new_L
                stuck_counter = 0
            else:
                stuck_counter += 1
                if stuck_counter > max_stuck:
                    # Use SA to escape local minima
                    remaining = deadline - time.time()
                    if remaining > 2:
                        L, residues, cover_count = simulated_annealing(
                            primes_list, cover_count, residues, L, L_limit,
                            min(3000, int(remaining * 300)), time.time() + remaining * 0.8
                        )
                    stuck_counter = 0

        return L, residues, cover_count

    def simulated_annealing(primes_list, cover_count, residues, L, L_limit, max_iters, deadline):
        """SA to escape local minima. Returns (L, residues, cover_count)."""
        n_primes = len(primes_list)
        T = 30.0
        cooling = 0.999
        best_L = L
        best_res = list(residues)
        best_cc = list(cover_count)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            if it % 300 == 0:
                T *= cooling
                if T < 0.05:
                    T = 0.05
            
            idx = random.randint(0, n_primes - 1)
            p = primes_list[idx]
            old_r = residues[idx]
            new_r = random.randint(0, p - 1)
            if new_r == old_r:
                continue
            
            update_counts(cover_count, p, old_r, new_r, L_limit)
            residues[idx] = new_r
            new_L = calc_L(cover_count, L_limit)
            
            if new_L >= L:
                L = new_L
                if L > best_L:
                    best_L = L
                    best_res = list(residues)
                    best_cc = list(cover_count)
            else:
                # Accept with probability exp((new_L - L) / T)
                delta = new_L - L
                prob = math.exp(delta / T)
                if random.random() < prob:
                    # Accept worse move
                    L = new_L
                else:
                    # Reject
                    update_counts(cover_count, p, new_r, old_r, L_limit)
                    residues[idx] = old_r
        
        # Restore best
        for i in range(n_primes):
            if residues[i] != best_res[i]:
                p = primes_list[i]
                update_counts(cover_count, p, residues[i], best_res[i], L_limit)
                residues[i] = best_res[i]
        return best_L, best_res, best_cc

    best_L_global = 0
    best_classes_global = None

    # Allocate time to each candidate set
    n_sets = len(valid_sets)
    time_per_set = (DEADLINE - 15) / n_sets  # Reserve 15s for final evaluation

    for set_idx, P in enumerate(valid_sets):
        if time.time() > start_time + DEADLINE - 10:
            break
        
        set_deadline = min(start_time + DEADLINE - 10, start_time + time_per_set * (set_idx + 1))
        
        # Try multiple initialization strategies for this prime set
        inits = [
            ("diagonal", init_residues_diagonal(P)),
            ("greedy", init_residues_greedy(P, L_CHECK_LIMIT))
        ]
        
        for init_name, residues in inits:
            if time.time() > set_deadline:
                break
            
            L, cover_count = evaluate_coverage(P, residues, L_CHECK_LIMIT)
            
            if L > best_L_global:
                best_L_global = L
                best_classes_global = [[P[i], residues[i]] for i in range(len(P))]
            
            # Gap-targeted extension
            ext_deadline = min(set_deadline, time.time() + time_per_set * 0.4)
            L, residues, cover_count = gap_targeted_extension(
                P, residues, cover_count, L, L_CHECK_LIMIT, ext_deadline
            )
            
            if L > best_L_global:
                best_L_global = L
                best_classes_global = [[P[i], residues[i]] for i in range(len(P))]
            
            # SA polish if time remains
            if time.time() < set_deadline - 2 and L < 999:
                sa_deadline = set_deadline - 2
                sa_iters = min(5000, int((sa_deadline - time.time()) * 400))
                L, residues, cover_count = simulated_annealing(
                    P, cover_count, residues, L, L_CHECK_LIMIT, sa_iters, sa_deadline
                )
                
                if L > best_L_global:
                    best_L_global = L
                    best_classes_global = [[P[i], residues[i]] for i in range(len(P))]

    # Final evaluation up to L_FINAL_LIMIT
    if best_classes_global:
        final_L = 0
        covered = [False] * (L_FINAL_LIMIT + 1)
        for p, r in best_classes_global:
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
        # Use the larger of the two
        if final_L > best_L_global:
            best_L_global = final_L

    # Safety: verify budget
    if best_classes_global:
        total = sum(Fraction(1, p) for p, r in best_classes_global)
        if total > 1:
            # Fallback to primes >= 13 with diagonal residues
            primes_set = [p for p in all_primes if p >= 13]
            residues = init_residues_diagonal(primes_set)
            best_classes_global = [[primes_set[i], residues[i]] for i in range(len(primes_set))]

    if not best_classes_global:
        # Last resort fallback
        primes_set = [p for p in all_primes if p >= 13]
        residues = [(i + 1) % p for i, p in enumerate(primes_set)]
        best_classes_global = [[primes_set[i], residues[i]] for i in range(len(primes_set))]

    return best_classes_global
