import time
import random
import math
from fractions import Fraction

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    DEADLINE = 110  # seconds, safety margin for 120s limit (or 150s if evaluator kills at 150)
    # The prompt says "evaluator KILLS ... after 150s".
    # "Keep an EXPLICIT deadline guard ... safety margin before the limit."
    # 120s is a good margin. Let's use 115s to be safe.
    
    def get_primes(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(math.sqrt(n)) + 1):
            if sieve[i]:
                sieve[i*i:n+1:i] = [False] * len(sieve[i*i:n+1:i])
        return [i for i, is_prime in enumerate(sieve) if is_prime]
    
    all_primes = get_primes(1000)
    
    # Strategy for Prime Set:
    # We want to include small primes (11, 13, ...) as they are more "efficient" for covering small integers.
    # We want to exclude large primes if they don't contribute much to the prefix 1..L.
    # Let's try to include 11 and 13.
    # Sum of 1/p for p >= 13 is ~0.92.
    # Sum of 1/11 is ~0.09. Total ~1.01.
    # We need to remove ~0.01 worth of reciprocals.
    # Removing largest primes (smallest 1/p) is the best way to preserve "coverage density" of the remaining set?
    # Actually, for covering 1..L, a prime p covers L/p numbers.
    # If we remove a large prime p, we lose L/p coverage.
    # If we add a small prime 11, we gain L/11 coverage.
    # Since 11 is much smaller than the largest primes (e.g., 997), L/11 >> L/997.
    # So adding 11 and removing several large primes should net a gain in coverage density for small L.
    
    # Let's build the set:
    # 1. Start with all primes >= 11.
    # 2. While sum(1/p) > 1, remove the largest prime.
    
    candidates = [p for p in all_primes if p >= 11]
    # Sort by prime value descending to remove largest first
    candidates_sorted_desc = sorted(candidates, reverse=True)
    
    current_sum = sum(Fraction(1, p) for p in candidates)
    primes_set = list(candidates)
    
    # Remove largest primes until sum <= 1
    # We need to be careful with Fraction performance in a loop, but N is small (~140).
    while current_sum > 1:
        if not candidates_sorted_desc:
            break
        p_to_remove = candidates_sorted_desc.pop()
        primes_set.remove(p_to_remove)
        current_sum -= Fraction(1, p_to_remove)
        
    primes_set = sorted(primes_set)
    n_primes = len(primes_set)
    
    # Verify sum
    # exact_sum = sum(Fraction(1, p) for p in primes_set)
    # assert exact_sum <= 1, f"Sum {exact_sum} > 1"
    
    L_CHECK_LIMIT = 2000
    L_FINAL_LIMIT = 100000
    
    def evaluate_coverage_exact(residues, L_limit):
        """
        Returns (L, cover_count) for the given residues up to L_limit.
        """
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

    def calculate_L_from_counts(cover_count, L_limit):
        L = 0
        # Optimization: check from current best L? 
        # But we don't pass current L. 
        # However, if we only change one residue, L usually stays same or increases by a bit, or decreases.
        # Scanning from 1 is O(L). L is small (2000).
        for n in range(1, L_limit + 1):
            if cover_count[n] == 0:
                break
            L = n
        return L

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
        # Heuristic: Cover i+1 with i-th prime? 
        # This works well for the first few primes.
        res = [0] * n_primes
        for i, p in enumerate(primes_set):
            target = i + 1
            res[i] = target % p
        return res

    best_L = 0
    best_classes = None
    
    # Main Search
    residues = init_residues()
    L, cover_count = evaluate_coverage_exact(residues, L_CHECK_LIMIT)
    
    if L > best_L:
        best_L = L
        best_classes = [[primes_set[i], residues[i]] for i in range(n_primes)]
    
    stuck_counter = 0
    max_stuck = 1000 # Reduced to allow more restarts if needed
    
    # We will run until deadline
    while time.time() - start_time < DEADLINE:
        if L >= L_CHECK_LIMIT:
            # We covered the check limit. 
            # Try to extend? 
            # For now, break and do final eval. 
            # If we have time, we could increase L_CHECK_LIMIT and continue.
            # But increasing L_CHECK_LIMIT requires re-initializing cover_count which is O(L * sum(1/p)).
            # Let's just break to be safe.
            break
        
        # Find the first hole
        hole = L + 1
        
        # Try to cover this hole.
        # We will try a random subset of primes or all primes?
        # n_primes is ~130. 
        # Let's try all primes, but break early if we find a good one?
        # Or just pick the best one.
        
        best_new_L = L
        best_idx = -1
        best_r_new = -1
        best_r_old = -1
        
        # To speed up, we can shuffle the order of primes we try?
        # Or just iterate.
        for idx in range(n_primes):
            p = primes_set[idx]
            r_old = residues[idx]
            r_new = hole % p
            
            if r_old == r_new:
                continue
            
            # Apply change
            update_counts(cover_count, p, r_old, r_new, L_CHECK_LIMIT)
            residues[idx] = r_new
            
            new_L = calculate_L_from_counts(cover_count, L_CHECK_LIMIT)
            
            if new_L > best_new_L:
                best_new_L = new_L
                best_idx = idx
                best_r_new = r_new
                best_r_old = r_old
                # Keep this state as the current "best candidate" state?
                # If we revert, we lose it.
                # But we want to test other primes against the *original* state to find the *single best* move.
                # So we must revert to test the next prime fairly.
                update_counts(cover_count, p, r_new, r_old, L_CHECK_LIMIT)
                residues[idx] = r_old
            else:
                # Revert
                update_counts(cover_count, p, r_new, r_old, L_CHECK_LIMIT)
                residues[idx] = r_old
        
        if best_idx != -1:
            # Apply the best change found
            p = primes_set[best_idx]
            update_counts(cover_count, p, best_r_old, best_r_new, L_CHECK_LIMIT)
            residues[best_idx] = best_r_new
            L = best_new_L
            stuck_counter = 0
            
            if L > best_L:
                best_L = L
                best_classes = [[primes_set[i], residues[i]] for i in range(n_primes)]
        else:
            stuck_counter += 1
            if stuck_counter > max_stuck:
                # Random restart
                residues = init_residues()
                # Add some randomness to init?
                # Let's try a random init
                for i in range(n_primes):
                    residues[i] = random.randint(0, primes_set[i] - 1)
                
                L, cover_count = evaluate_coverage_exact(residues, L_CHECK_LIMIT)
                if L > best_L:
                    best_L = L
                    best_classes = [[primes_set[i], residues[i]] for i in range(n_primes)]
                stuck_counter = 0
                max_stuck = 1000

    # Final Evaluation
    if best_classes is None:
        best_classes = [[primes_set[i], residues[i]] for i in range(n_primes)]
    
    # Calculate exact L up to L_FINAL_LIMIT
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
