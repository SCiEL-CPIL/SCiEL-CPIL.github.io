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
    # Set A: {3, 5} + primes >= 13. Sum ~ 0.97.
    # Set B: {2} + primes >= 67. Sum ~ 0.95.
    # Set C: {3, 5, 7} + primes >= 150. Sum ~ 0.99.
    # Set D: {2, 3} + primes >= 330. Sum ~ 1.0 (approx).
    
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
    # 4 sets * 30s = 120s. 25s margin.
    time_per_set = 30
    
    L_MAX = 10000  # We aim for high L, but cap search space for speed
    # Using NumPy for fast coverage checks
    
    for P in valid_sets:
        if time.time() > deadline - 5:
            break
            
        set_start = time.time()
        set_deadline = set_start + time_per_set
        
        # Sort primes for deterministic greedy init
        P_sorted = sorted(P)
        n_primes = len(P_sorted)
        
        # 1. Greedy Initialization
        # We want to cover 1..L.
        # Greedy: for n=1,2,... if not covered, pick a prime p not yet assigned 
        # (or any prime) and set r = n % p.
        # To be smarter, we should pick the prime that allows the most future coverage?
        # For now, simple greedy: pick smallest available prime? Or random?
        # Let's try picking the smallest prime that hasn't been used recently?
        # Actually, standard greedy: for n=1..L_MAX, if not covered, assign a prime.
        # Which prime? The one with the smallest value? 
        # If we use small primes, they cover dense patterns.
        
        residues = {p: 0 for p in P_sorted}
        # Use NumPy array for coverage counts
        cover_count = np.zeros(L_MAX + 1, dtype=np.int32)
        
        def update_cover(p, r, delta):
            # Update cover_count for prime p with residue r by delta
            # Vectorized update using NumPy slicing
            # Indices: r, r+p, r+2p, ...
            # If r == 0, indices are p, 2p, ...
            # If r > 0, indices are r, r+p, ...
            # Note: NumPy slicing start, stop, step
            # We want indices in [1, L_MAX]
            
            # Calculate start index
            if r == 0:
                start = p
            else:
                start = r
                
            if start > L_MAX:
                return
                
            # Create slice
            # start, L_MAX+1, p
            cover_count[start:L_MAX+1:p] += delta

        # Initialize cover_count to 0 (already done)
        # Greedy assignment
        assigned_primes = set()
        
        # We'll iterate through n and assign primes to uncovered numbers
        # To speed this up, we can check coverage in batches? 
        # No, greedy is sequential.
        # But we can optimize the check "is n covered?"
        # cover_count[n] > 0
        
        for n in range(1, L_MAX + 1):
            if cover_count[n] == 0:
                # Find a prime to assign
                # Heuristic: pick the smallest prime not in assigned_primes?
                # Or pick the prime that minimizes "conflict"?
                # Let's just pick the smallest prime from P_sorted.
                # If all are used, we break.
                
                # Optimization: Keep a list of available primes?
                # Since n_primes is small (~100), linear search is fine.
                chosen_p = None
                for p in P_sorted:
                    if p not in assigned_primes:
                        chosen_p = p
                        break
                
                if chosen_p is None:
                    break # No more primes to assign
                
                r = n % chosen_p
                residues[chosen_p] = r
                assigned_primes.add(chosen_p)
                update_cover(chosen_p, r, 1)
            
            if time.time() > set_deadline:
                break
        
        # Calculate current L
        def get_L():
            # Find largest L such that cover_count[1..L] > 0
            # NumPy: find first index where cover_count == 0
            # Note: cover_count is 1-indexed for our purposes (0 is unused)
            # We care about 1..L_MAX
            # np.nonzero(cover_count[1:] == 0)[0]
            # If empty, L_MAX. Else, first index + 1 (offset) - 1?
            # Let's say indices are 0..L_MAX.
            # We want the first i in 1..L_MAX such that cover_count[i] == 0.
            # zeros = np.nonzero(cover_count[1:L_MAX+1] == 0)[0]
            # if len(zeros) == 0: return L_MAX
            # else: return zeros[0] (since offset is 1, index 0 in slice is 1 in array)
            # Wait.
            # Slice is cover_count[1:L_MAX+1].
            # Index 0 in slice corresponds to array index 1.
            # If zeros[0] is k, then array index is 1+k.
            # So L = (1+k) - 1 = k.
            
            zeros = np.nonzero(cover_count[1:L_MAX+1] == 0)[0]
            if len(zeros) == 0:
                return L_MAX
            else:
                return zeros[0]
        
        current_L = get_L()
        best_set_L = current_L
        best_set_residues = residues.copy()
        
        # 2. Simulated Annealing with Local Repair
        # Parameters
        T = 5.0  # Start temperature
        T_min = 0.1
        cooling = 0.999
        steps = 0
        
        # Precompute ranges for speed? No, dynamic is fine with incremental update.
        
        # Local Repair: When we have a gap, try to fix it by adjusting residues of nearby primes
        def local_repair():
            nonlocal current_L, best_set_L, best_set_residues
            # Find the first gap
            gap_pos = current_L + 1
            if gap_pos > L_MAX:
                return False
            
            # Try to cover gap_pos by changing residue of some prime
            # For each prime, check if changing its residue can cover gap_pos
            # without creating new gaps before current_L
            for p in P_sorted:
                r_old = residues[p]
                r_new = gap_pos % p
                if r_new == r_old:
                    continue
                
                # Check if changing p to r_new covers gap_pos
                # And doesn't uncover any n in [1, current_L]
                # We need to check if any n in [1, current_L] is ONLY covered by p with r_old
                # This is expensive. Let's do a quick check:
                # If cover_count[n] > 1 for all n in [1, current_L] that are covered by p,
                # then we can safely change p.
                
                # Quick check: Is there any n in [1, current_L] such that
                # cover_count[n] == 1 and n is covered by p with r_old?
                # We can check this by looking at the intersection of:
                # - numbers covered by p with r_old
                # - numbers with cover_count == 1
                # - numbers in [1, current_L]
                
                # For speed, let's just try the move and see if L decreases
                # If L decreases, revert.
                
                update_cover(p, r_old, -1)
                update_cover(p, r_new, 1)
                
                new_L = get_L()
                
                if new_L >= current_L:
                    # Accept the move
                    residues[p] = r_new
                    if new_L > current_L:
                        current_L = new_L
                        if current_L > best_set_L:
                            best_set_L = current_L
                            best_set_residues = residues.copy()
                    return True
                else:
                    # Revert
                    update_cover(p, r_new, -1)
                    update_cover(p, r_old, 1)
            
            return False
        
        while time.time() < set_deadline:
            # Pick a random prime to change
            p = random.choice(P_sorted)
            r_old = residues[p]
            
            # Generate new residue
            # Random walk vs Random Jump?
            # Jump is better for SA.
            r_new = random.randint(0, p - 1)
            
            if r_new == r_old:
                continue
            
            # Update cover_count
            update_cover(p, r_old, -1)
            update_cover(p, r_new, 1)
            
            # Calculate new L
            new_L = get_L()
            
            # Acceptance
            if new_L > current_L:
                current_L = new_L
                residues[p] = r_new
                if current_L > best_set_L:
                    best_set_L = current_L
                    best_set_residues = residues.copy()
            else:
                # Accept with probability exp((new_L - current_L) / T)
                # Since new_L <= current_L, delta is negative or zero.
                delta = new_L - current_L
                if delta == 0:
                    # Always accept equal? Or random?
                    # Accepting equal moves helps explore plateaus.
                    if random.random() < 0.5:
                        residues[p] = r_new
                        current_L = new_L # Unchanged
                    else:
                        # Try local repair
                        if local_repair():
                            pass
                        else:
                            # Reject: revert cover_count
                            update_cover(p, r_new, -1)
                            update_cover(p, r_old, 1)
                else:
                    # delta < 0
                    prob = math.exp(delta / T)
                    if random.random() < prob:
                        residues[p] = r_new
                        current_L = new_L
                        # Try local repair to potentially improve
                        if local_repair():
                            pass
                    else:
                        # Reject: revert cover_count
                        update_cover(p, r_new, -1)
                        update_cover(p, r_old, 1)
            
            # Cool down
            T *= cooling
            if T < T_min:
                T = T_min
                # Optional: reheat occasionally?
                # For now, just keep cooling.
            
            steps += 1
            
            # If we hit the global cap or high target, we can stop early for this set
            if best_set_L >= 999 and best_set_L > best_L_global:
                # We found a good solution, but let's see if we can push higher.
                # If we have time, continue.
                pass

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
