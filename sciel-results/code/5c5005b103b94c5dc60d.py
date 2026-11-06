import time
import math
import random
import numpy as np
from fractions import Fraction

def construct():
    t0 = time.time()
    deadline = t0 + 145  # Safety margin for 150s limit

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

    # Define candidate sets of primes with sum(1/p) <= 1
    # Set A: {3, 5} + primes >= 13. Sum ~ 0.97.
    # Set B: {2} + primes >= 67. Sum ~ 0.95.
    # Set C: {3, 5, 7} + primes >= 150. Sum ~ 0.99.
    candidate_sets = [
        [3, 5] + [p for p in primes if p >= 13],
        [2] + [p for p in primes if p >= 67],
        [3, 5, 7] + [p for p in primes if p >= 150]
    ]
    
    # Validate sets with exact rationals
    valid_sets = []
    for P in candidate_sets:
        s = sum(Fraction(1, p) for p in P)
        if s <= 1:
            valid_sets.append(P)
            
    if not valid_sets:
        # Fallback
        primes_seed = [p for p in primes if p >= 13]
        return [[p, (i + 1) % p] for i, p in enumerate(primes_seed)]

    best_L_global = 0
    best_result_global = []

    # We will run SA for each valid set
    # Time allocation:
    # Total ~ 145s.
    # 3 sets.
    # Let's give 40s to the first (best structure), 40s to second, 40s to third, 25s margin.
    
    L_MAX = 10000  # Search space cap for speed

    for P in valid_sets:
        if time.time() > deadline - 5:
            break
            
        set_start = time.time()
        set_deadline = set_start + 40  # 40s per set
        
        # Sort primes for deterministic greedy init
        P_sorted = sorted(P)
        
        # 1. Greedy Initialization
        # We want to cover 1..L.
        # Greedy: for n=1,2,... if not covered, pick a prime p and set r = n % p.
        # To be smarter, we should pick the prime that allows the most future coverage?
        # For now, simple greedy: pick smallest available prime?
        # Actually, standard greedy: for n=1..L_MAX, if not covered, assign a prime.
        # Which prime? The one with the smallest value? 
        # If we use small primes, they cover dense patterns.
        
        residues = {p: 0 for p in P}
        cover_count = [0] * (L_MAX + 1)
        
        def update_cover(p, r, delta):
            # Update cover_count for prime p with residue r by delta
            # Note: r is in [0, p-1].
            # The numbers covered are r, r+p, r+2p, ...
            # If r == 0, the numbers are p, 2p, ... (since we start from 1)
            # Wait, the problem says integers 1..L.
            # If r=0, it covers multiples of p.
            # If r>0, it covers r, r+p, ...
            
            # Optimization: Use numpy if possible? No, L_MAX is small enough for list if optimized.
            # But Python loops are slow. Let's stick to list but be careful.
            
            # First covered number >= 1
            if r == 0:
                first = p
            else:
                first = r
            
            if first > L_MAX:
                return
            
            for n in range(first, L_MAX + 1, p):
                cover_count[n] += delta

        # Greedy assignment
        assigned_primes = set()
        for n in range(1, L_MAX + 1):
            if cover_count[n] == 0:
                # Find a prime to assign
                # Heuristic: pick the smallest prime not in assigned_primes?
                # Or pick the prime that minimizes "conflict"?
                # Let's just pick the smallest prime from P_sorted.
                # If all are used, we break.
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
            for n in range(1, L_MAX + 1):
                if cover_count[n] == 0:
                    return n - 1
            return L_MAX
        
        current_L = get_L()
        best_set_L = current_L
        best_set_residues = residues.copy()
        
        # 2. Simulated Annealing
        # Parameters
        T = 5.0  # Start temperature
        T_min = 0.1
        cooling = 0.999
        steps = 0
        
        # Precompute ranges for speed? No, dynamic is fine with incremental update.
        
        while time.time() < set_deadline:
            # Pick a random prime to change
            p = random.choice(P)
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
                    # delta < 0
                    prob = math.exp(delta / T)
                    if random.random() < prob:
                        residues[p] = r_new
                        current_L = new_L
                    else:
                        # Reject: revert cover_count
                        update_cover(p, r_new, -1)
                        update_cover(p, r_old, 1)
            
            # Cool down
            T *= cooling
            if T < T_min:
                T = T_min
            
            steps += 1

        # Update global best
        if best_set_L > best_L_global:
            best_L_global = best_set_L
            best_result_global = [[p, best_set_residues[p]] for p in best_set_residues]

    # Final check and return
    if not best_result_global:
        # Fallback to seed
        primes_seed = [p for p in primes if p >= 13]
        return [[p, (i + 1) % p] for i, p in enumerate(primes_seed)]
        
    return best_result_global
