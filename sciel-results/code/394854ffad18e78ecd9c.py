import time
import math
import random
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit

    # Generate primes up to 1000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(math.sqrt(n)) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]

    primes = sieve(1000)

    # Define candidate sets of primes
    # We want sets where sum(1/p) <= 1.
    candidate_sets = [
        [3, 5] + [p for p in primes if p >= 13],
        [2] + [p for p in primes if p >= 67],
        [3, 5, 7] + [p for p in primes if p >= 150],
        [2, 3] + [p for p in primes if p >= 330]
    ]
    
    # Validate sets with exact rationals
    valid_sets = []
    for P in candidate_sets:
        s = sum(Fraction(1, p) for p in P)
        if s <= 1:
            valid_sets.append(P)
            
    if not valid_sets:
        # Fallback to seed-like if no valid sets found
        primes_seed = [p for p in primes if p >= 13]
        return [[p, (i + 1) % p] for i, p in enumerate(primes_seed)]

    best_L_global = 0
    best_result_global = []
    
    # Allocate time per set
    time_per_set = 30
    
    L_MAX = 10000 # We aim for high L, but cap search space for speed
    # Using NumPy for fast coverage checks
    
    for P in valid_sets:
        if time.time() > deadline - 5:
            break
            
        set_start = time.time()
        set_deadline = set_start + time_per_set
        
        # Sort primes for deterministic greedy init
        P_sorted = sorted(P)
        n_primes = len(P_sorted)
        
        # Helper to update coverage
        def update_cover(cover_count, p, r, delta):
            if r == 0:
                start = p
            else:
                start = r
            if start > L_MAX:
                return
            cover_count[start:L_MAX+1:p] += delta

        def get_L(cover_count):
            zeros = np.nonzero(cover_count[1:L_MAX+1] == 0)[0]
            if len(zeros) == 0:
                return L_MAX
            else:
                return zeros[0]

        # 1. Greedy Initialization: "Smallest Uncovered" heuristic
        # We assign residues to primes one by one.
        # Order: Smallest primes first? Or largest?
        # Smallest primes have higher density. Let's process them first.
        # For each prime p, find the smallest n in [1, L_MAX] not covered by previously assigned primes.
        # Set r = n % p.
        
        def greedy_init(P_sorted):
            cover_count = np.zeros(L_MAX + 1, dtype=np.int32)
            residues = {p: 0 for p in P_sorted}
            
            # We will assign a residue to each prime in P_sorted
            # To make it smarter, we can re-evaluate? No, simple greedy.
            for p in P_sorted:
                # Find smallest uncovered n
                # This is O(L_MAX) per prime, so O(N * L_MAX). 
                # N~100, L_MAX=10000 -> 10^6 ops. Very fast.
                # We can optimize by maintaining a pointer to the first uncovered?
                # But coverage changes.
                
                # Efficiently find first uncovered:
                # cover_count is updated. We can scan from 1.
                # Since we want the *smallest* uncovered, we scan 1..L_MAX.
                # However, if we just scan, it's O(L_MAX).
                
                # Optimization: Keep a list of uncovered? No, just scan.
                # Actually, we can just check cover_count[n] == 0.
                
                # To speed up, we can stop scanning if we find one.
                # But we need to be careful: if we pick p=3, r=1, we cover 1, 4, 7...
                # Then for p=5, we find smallest uncovered.
                
                # Let's just find the first n where cover_count[n] == 0.
                # We can use np.nonzero(cover_count[1:] == 0)[0]
                # If empty, all covered? Unlikely for initial steps.
                
                uncovered_idx = np.nonzero(cover_count[1:L_MAX+1] == 0)[0]
                if len(uncovered_idx) == 0:
                    # All covered, pick random residue?
                    r = random.randint(0, p-1)
                else:
                    n = uncovered_idx[0] + 1 # 1-based
                    r = n % p
                
                residues[p] = r
                update_cover(cover_count, p, r, 1)
                
            return residues, cover_count

        # 2. Iterated Local Search (ILS)
        # Local Search: Hill Climbing (accept only improvements)
        # Perturbation: Randomly change K residues.
        
        best_set_L = -1
        best_set_residues = None
        best_set_cover = None
        
        # Initial solution
        residues, cover_count = greedy_init(P_sorted)
        current_L = get_L(cover_count)
        
        if current_L > best_set_L:
            best_set_L = current_L
            best_set_residues = residues.copy()
            best_set_cover = cover_count.copy()
            
        current_residues = residues.copy()
        current_cover = cover_count.copy()
        
        # ILS Loop
        # Parameters
        K_perturb = 3 # Number of residues to perturb
        max_iters = 500 # Max iterations per restart? No, time-based.
        
        while time.time() < set_deadline:
            # Local Search (Hill Climbing)
            # We try to improve current_L by changing one residue at a time.
            # If no improvement in a full sweep, we break local search and perturb.
            
            improved = True
            while improved and time.time() < set_deadline:
                improved = False
                # Shuffle primes to avoid bias? Or random order?
                # Random order helps escape local minima slightly, but HC is deterministic if order is fixed.
                # Let's use random order for each sweep.
                P_random = P_sorted.copy()
                random.shuffle(P_random)
                
                for p in P_random:
                    if time.time() > set_deadline:
                        break
                    
                    r_old = current_residues[p]
                    
                    # Try a few random residues? Or all?
                    # All is O(p). Sum p is ~ 2000. 2000 * 100 primes = 200k ops per sweep.
                    # Very fast. Let's try ALL residues to ensure we find the best single-move improvement.
                    best_r = r_old
                    best_l = current_L
                    best_cover_diff = None # We don't want to copy cover array 100 times
                    
                    # To avoid copying cover array for each r, we can:
                    # 1. Update cover with -1 for r_old
                    # 2. Loop r_new:
                    #    Update cover with +1 for r_new
                    #    Calculate L
                    #    If L > best_l, record best_r, best_l
                    #    Update cover with -1 for r_new
                    # 3. After loop, if best_r != r_old, keep the best state.
                    #    Else, revert? No, we already reverted in loop.
                    #    Wait, if we don't change, we need to re-apply r_old?
                    #    Actually, if we don't change, the cover is currently in state "r_old removed, nothing added".
                    #    So we must re-add r_old at the end if we didn't accept a change.
                    
                    update_cover(current_cover, p, r_old, -1)
                    
                    for r_new in range(p):
                        if r_new == r_old:
                            continue
                        update_cover(current_cover, p, r_new, 1)
                        l_new = get_L(current_cover)
                        if l_new > best_l:
                            best_l = l_new
                            best_r = r_new
                        update_cover(current_cover, p, r_new, -1)
                    
                    if best_r != r_old:
                        # Accept the best move
                        update_cover(current_cover, p, best_r, 1)
                        current_residues[p] = best_r
                        current_L = best_l
                        improved = True
                    else:
                        # No improvement, re-apply old residue
                        update_cover(current_cover, p, r_old, 1)
                        
                    if current_L > best_set_L:
                        best_set_L = current_L
                        best_set_residues = current_residues.copy()
                        best_set_cover = current_cover.copy()
                        
            # Perturbation
            # Randomly change K residues
            for _ in range(K_perturb):
                p = random.choice(P_sorted)
                r_old = current_residues[p]
                r_new = random.randint(0, p - 1)
                if r_new != r_old:
                    update_cover(current_cover, p, r_old, -1)
                    update_cover(current_cover, p, r_new, 1)
                    current_residues[p] = r_new
                    current_L = get_L(current_cover)
                    
            # If we hit a very high L, we might want to stop early?
            # But ILS might find better.
            
        # Update global best
        if best_set_L > best_L_global:
            best_L_global = best_set_L
            best_result_global = [[p, best_set_residues[p]] for p in best_set_residues]
            
            # Check if we should break early
            if best_L_global >= 5000: # Very high, unlikely
                break

    # Final check and return
    if not best_result_global:
        # Fallback to seed
        primes_seed = [p for p in primes if p >= 13]
        return [[p, (i + 1) % p] for i, p in enumerate(primes_seed)]
        
    return best_result_global
