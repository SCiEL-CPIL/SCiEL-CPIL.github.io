import time
import random
from fractions import Fraction

# EVOLVE-BLOCK-START

def _get_primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [i for i, is_prime in enumerate(sieve) if is_prime]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    primes = _get_primes(1000)
    prime_set = set(primes)
    
    # Initial solution
    # Try to include as many small primes as possible?
    # Or start with seed and improve.
    
    active_primes = []
    residues = {}
    current_sum = Fraction(0, 1)
    
    # Seed-like initialization
    for p in primes:
        if p < 13:
            continue
        if current_sum + Fraction(1, p) <= 1:
            active_primes.append(p)
            current_sum += Fraction(1, p)
            # Assign residue to cover the index
            # i = len(active_primes) - 1
            # We want to cover i+1?
            # Seed: (i+1) % p
            i = len(active_primes) - 1
            residues[p] = (i + 1) % p
            
    MAX_L = 100000
    covered = [False] * (MAX_L + 1)
    
    def update_coverage(p, r, sign):
        if sign == 1:
            for k in range(r, MAX_L + 1, p):
                if k > 0:
                    covered[k] = True
        else:
            for k in range(r, MAX_L + 1, p):
                if k > 0:
                    covered[k] = False
                    
    for p, r in residues.items():
        update_coverage(p, r, 1)
        
    def calc_L():
        for i in range(1, MAX_L + 1):
            if not covered[i]:
                return i - 1
        return MAX_L
        
    best_L = calc_L()
    
    inactive_primes = [p for p in primes if p not in residues]
    
    # Local Search
    while time.time() < deadline:
        # Choose move
        if not active_primes:
            break
            
        move_type = random.random()
        if move_type < 0.5:
            # Residue change
            p = random.choice(active_primes)
            old_r = residues[p]
            new_r = random.randint(0, p - 1)
            if new_r == old_r:
                continue
                
            update_coverage(p, old_r, -1)
            residues[p] = new_r
            update_coverage(p, new_r, 1)
            
            new_L = calc_L()
            if new_L > best_L:
                best_L = new_L
            else:
                update_coverage(p, new_r, -1)
                residues[p] = old_r
                update_coverage(p, old_r, 1)
                
        elif move_type < 0.9 and inactive_primes:
            # Swap
            p_out = random.choice(active_primes)
            p_in = random.choice(inactive_primes)
            
            old_r_out = residues[p_out]
            
            # Remove p_out
            update_coverage(p_out, old_r_out, -1)
            del residues[p_out]
            active_primes.remove(p_out)
            current_sum -= Fraction(1, p_out)
            
            # Find best residue for p_in
            best_r_in = 0
            best_L_in = 0
            # Try a few residues
            for _ in range(10):
                r_in = random.randint(0, p_in - 1)
                update_coverage(p_in, r_in, 1)
                L_temp = calc_L()
                if L_temp > best_L_in:
                    best_L_in = L_temp
                    best_r_in = r_in
                update_coverage(p_in, r_in, -1)
                
            # Check budget
            if current_sum + Fraction(1, p_in) <= 1:
                # Add p_in
                residues[p_in] = best_r_in
                update_coverage(p_in, best_r_in, 1)
                active_primes.append(p_in)
                current_sum += Fraction(1, p_in)
                inactive_primes.remove(p_in)
                
                new_L = calc_L()
                if new_L > best_L:
                    best_L = new_L
                else:
                    # Revert
                    update_coverage(p_in, best_r_in, -1)
                    del residues[p_in]
                    active_primes.remove(p_in)
                    current_sum -= Fraction(1, p_in)
                    inactive_primes.append(p_in)
                    
                    # Re-add p_out
                    residues[p_out] = old_r_out
                    update_coverage(p_out, old_r_out, 1)
                    active_primes.append(p_out)
                    current_sum += Fraction(1, p_out)
            else:
                # Revert
                residues[p_out] = old_r_out
                update_coverage(p_out, old_r_out, 1)
                active_primes.append(p_out)
                current_sum += Fraction(1, p_out)
                
        else:
            # Add if possible
            if inactive_primes:
                p_in = random.choice(inactive_primes)
                if current_sum + Fraction(1, p_in) <= 1:
                    best_r_in = 0
                    best_L_in = 0
                    for _ in range(10):
                        r_in = random.randint(0, p_in - 1)
                        update_coverage(p_in, r_in, 1)
                        L_temp = calc_L()
                        if L_temp > best_L_in:
                            best_L_in = L_temp
                            best_r_in = r_in
                        update_coverage(p_in, r_in, -1)
                        
                    residues[p_in] = best_r_in
                    update_coverage(p_in, best_r_in, 1)
                    active_primes.append(p_in)
                    current_sum += Fraction(1, p_in)
                    inactive_primes.remove(p_in)
                    
                    new_L = calc_L()
                    if new_L > best_L:
                        best_L = new_L
                    else:
                        update_coverage(p_in, best_r_in, -1)
                        del residues[p_in]
                        active_primes.remove(p_in)
                        current_sum -= Fraction(1, p_in)
                        inactive_primes.append(p_in)
                        
    return [[p, r] for p, r in residues.items()]

# EVOLVE-BLOCK-END
