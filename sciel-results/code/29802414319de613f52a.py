import time
import random
import math
from fractions import Fraction

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    DEADLINE = 115  # Safety margin for 120s limit

    def get_primes(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(math.sqrt(n)) + 1):
            if sieve[i]:
                sieve[i*i:n+1:i] = [False] * len(sieve[i*i:n+1:i])
        return [i for i, is_prime in enumerate(sieve) if is_prime]
    
    all_primes = get_primes(1000)
    
    # Select primes: include 11, remove largest until sum <= 1
    candidates = [p for p in all_primes if p >= 11]
    candidates_sorted_desc = sorted(candidates, reverse=True)
    
    current_sum = sum(Fraction(1, p) for p in candidates)
    primes_set = list(candidates)
    
    while current_sum > 1:
        if not candidates_sorted_desc:
            break
        p_to_remove = candidates_sorted_desc.pop()
        primes_set.remove(p_to_remove)
        current_sum -= Fraction(1, p_to_remove)
        
    primes_set = sorted(primes_set)
    n_primes = len(primes_set)
    
    L_CHECK_LIMIT = 5000
    L_FINAL_LIMIT = 100000
    
    def evaluate_coverage(residues, L_limit):
        cover_count = [0] * (L_limit + 1)
        for i, p in enumerate(primes_set):
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

    def find_first_hole(cover_count, L_limit):
        for n in range(1, L_limit + 1):
            if cover_count[n] == 0:
                return n
        return L_limit + 1

    def update_counts(cover_count, p, r_old, r_new, L_limit):
        if r_old == r_new:
            return
        # Decrement
        start = r_old
        if start == 0:
            start = p
        if start <= L_limit:
            for n in range(start, L_limit + 1, p):
                cover_count[n] -= 1
        
        # Increment
        start = r_new
        if start == 0:
            start = p
        if start <= L_limit:
            for n in range(start, L_limit + 1, p):
                cover_count[n] += 1

    def init_residues():
        # Greedy init: try to cover 1..n with i-th prime
        res = [0] * n_primes
        # Random init is often better for escaping local minima
        for i in range(n_primes):
            res[i] = random.randint(0, primes_set[i] - 1)
        return res

    best_L = 0
    best_classes = None
    best_residues = None

    # Main Search Loop
    while time.time() - start_time < DEADLINE:
        residues = init_residues()
        L, cover_count = evaluate_coverage(residues, L_CHECK_LIMIT)
        
        if L > best_L:
            best_L = L
            best_residues = residues[:]
            best_classes = [[primes_set[i], residues[i]] for i in range(n_primes)]
            
        stuck_counter = 0
        max_stuck = 500
        
        while time.time() - start_time < DEADLINE and stuck_counter < max_stuck:
            hole = find_first_hole(cover_count, L_CHECK_LIMIT)
            
            if hole > L:
                L = hole
                if L > best_L:
                    best_L = L
                    best_residues = residues[:]
                    best_classes = [[primes_set[i], residues[i]] for i in range(n_primes)]
                
                if L >= L_CHECK_LIMIT:
                    break

            # Try to cover 'hole'
            # We look for a prime p and new residue r_new = hole % p
            # such that the number of uncovered numbers in [1, hole-1] is minimized.
            
            best_cost = float('inf')
            best_idx = -1
            best_r_new = -1
            best_r_old = -1
            best_new_L = L # If no improvement, we stay
            
            # To speed up, we can randomize the order of primes we check
            indices = list(range(n_primes))
            random.shuffle(indices)
            
            for idx in indices:
                p = primes_set[idx]
                r_old = residues[idx]
                r_new = hole % p
                
                if r_old == r_new:
                    continue
                
                # Calculate cost: number of k in [1, hole-1] such that
                # k % p == r_old and cover_count[k] == 1
                # If we change r_old to r_new, these k's become uncovered
                # unless they are covered by another prime.
                
                cost = 0
                start = r_old
                if start == 0:
                    start = p
                
                # Only check up to hole-1
                # Optimization: if we find one, cost > 0. 
                # We want to find a move with cost 0 if possible.
                # If we find a cost 0 move, we can break immediately?
                # Yes, cost 0 is the best possible.
                
                found_cost_zero = False
                if start < hole:
                    for k in range(start, hole, p):
                        if cover_count[k] == 1:
                            cost = 1
                            break
                    if cost == 0:
                        found_cost_zero = True
                else:
                    found_cost_zero = True # No k in range, so cost is 0

                if found_cost_zero:
                    best_cost = 0
                    best_idx = idx
                    best_r_new = r_new
                    best_r_old = r_old
                    # Apply and break? 
                    # Yes, this is a "free" move. It doesn't uncover anything before hole.
                    # It might uncover things after hole, but we only care about prefix.
                    # So this is always a good move.
                    break
                else:
                    # Cost is at least 1. We keep the minimum cost found so far.
                    # But calculating exact cost is expensive. 
                    # We can just store this candidate and evaluate later if needed?
                    # Or just pick the first one with cost 1?
                    # Let's try to find a move that results in a higher L.
                    # This requires applying the move and finding the new hole.
                    # That is expensive.
                    
                    # Heuristic: If we don't find a cost 0 move, 
                    # we pick a random candidate from the first few we checked?
                    # Or we pick the one with the smallest prime?
                    # Small primes have larger steps, so fewer k's to check? No, larger steps means fewer k's in range.
                    # Large primes have smaller steps? No, p is the step.
                    # Small p -> many k's. Large p -> few k's.
                    # So checking large primes for cost is faster.
                    
                    if best_cost > 1: # We haven't found a cost 1 candidate yet?
                         # We know cost is >= 1.
                         # Let's just tentatively pick this one if we don't find a better one.
                         # But we don't know the exact cost.
                         # Let's assume cost 1 is acceptable if we can't find 0.
                         best_cost = 1
                         best_idx = idx
                         best_r_new = r_new
                         best_r_old = r_old
            
            if best_idx != -1:
                p = primes_set[best_idx]
                update_counts(cover_count, p, best_r_old, best_r_new, L_CHECK_LIMIT)
                residues[best_idx] = best_r_new
                stuck_counter = 0
            else:
                # No move found? This shouldn't happen unless all r_old == r_new
                # which means hole is already covered? Contradiction.
                # Or maybe we didn't check any primes?
                stuck_counter += 1
                if stuck_counter > max_stuck:
                    break

    # Final Evaluation
    if best_classes is None:
        best_classes = [[primes_set[i], residues[i]] for i in range(n_primes)]
    
    cover_count_final = [0] * (L_FINAL_LIMIT + 1)
    for p, r in best_classes:
        start = r
        if start == 0:
            start = p
        if start <= L_FINAL_LIMIT:
            for n in range(start, L_FINAL_LIMIT + 1, p):
                cover_count_final[n] += 1
                
    final_L = 0
    for n in range(1, L_FINAL_LIMIT + 1):
        if cover_count_final[n] == 0:
            break
        final_L = n
        
    return best_classes

# EVOLVE-BLOCK-END
