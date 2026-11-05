import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Primes up to 1000
    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                sieve[i*i:limit+1:i] = [False] * len(sieve[i*i:limit+1:i])
        return [i for i, is_prime in enumerate(sieve) if is_prime]
        
    primes = get_primes(1000)
    n_primes = len(primes)
    
    MAX_L = 100000
    
    # Initialize residues
    # Using a simple heuristic: r_p = p // 2 or random
    residues = [random.randint(0, p - 1) for p in primes]
    
    # Initialize counts
    counts = np.zeros(MAX_L + 1, dtype=np.int32)
    for i, p in enumerate(primes):
        r = residues[i]
        start = r if r > 0 else p
        if start <= MAX_L:
            counts[start::p] += 1
            
    # Function to calculate L
    def calc_L(counts, current_L):
        # Check prefix
        if current_L > 0:
            if np.min(counts[1:current_L+1]) < 2:
                # Find first bad
                bad = np.where(counts[1:current_L+1] < 2)[0]
                if len(bad) > 0:
                    return bad[0]
                else:
                    # Should not happen if min < 2
                    pass
        
        # Extend
        L = current_L
        if L < MAX_L:
            rest = counts[L+1:]
            bad_idx = np.where(rest < 2)[0]
            if len(bad_idx) == 0:
                return MAX_L
            else:
                return L + bad_idx[0]
        return L

    current_L = calc_L(counts, 0)
    best_L = current_L
    best_residues = residues.copy()
    
    # Greedy Extension Phase
    # Find first x > L with count < 2
    # Try to fix it by changing one residue
    
    while time.time() < deadline - 20: # Leave 20s for SA
        # Find first gap
        if current_L < MAX_L:
            rest = counts[current_L+1:]
            bad_idx = np.where(rest < 2)[0]
            if len(bad_idx) == 0:
                current_L = MAX_L
                break
            target_x = current_L + bad_idx[0]
        else:
            break
            
        # Try to cover target_x
        best_move = None
        best_new_L = current_L
        
        for i, p in enumerate(primes):
            r_old = residues[i]
            r_new = target_x % p
            
            if r_new == r_old:
                continue
                
            # Check validity: Does changing r_old to r_new break [1, current_L]?
            # Loss positions: y in [1, current_L] s.t. y % p == r_old
            # We need counts[y] > 2 for all such y
            
            # Generate loss positions
            if r_old == 0:
                start_loss = p
            else:
                start_loss = r_old
                
            if start_loss <= current_L:
                loss_indices = np.arange(start_loss, current_L + 1, p)
                # Check if any count is exactly 2 (would become 1)
                if np.any(counts[loss_indices] == 2):
                    continue # Invalid move
            
            # Move is valid for prefix. Calculate new L
            # Apply change temporarily
            if r_old == 0:
                start_old = p
            else:
                start_old = r_old
            if start_old <= MAX_L:
                counts[start_old::p] -= 1
                
            if r_new == 0:
                start_new = p
            else:
                start_new = r_new
            if start_new <= MAX_L:
                counts[start_new::p] += 1
                
            new_L = calc_L(counts, current_L)
            
            # Revert change
            if start_new <= MAX_L:
                counts[start_new::p] -= 1
            if start_old <= MAX_L:
                counts[start_old::p] += 1
                
            if new_L > best_new_L:
                best_new_L = new_L
                best_move = (i, r_new)
                
        if best_move:
            i, r_new = best_move
            r_old = residues[i]
            p = primes[i]
            
            # Apply best move
            if r_old == 0:
                start_old = p
            else:
                start_old = r_old
            if start_old <= MAX_L:
                counts[start_old::p] -= 1
                
            if r_new == 0:
                start_new = p
            else:
                start_new = r_new
            if start_new <= MAX_L:
                counts[start_new::p] += 1
                
            residues[i] = r_new
            current_L = best_new_L
            
            if current_L > best_L:
                best_L = current_L
                best_residues = residues.copy()
        else:
            # Stuck in greedy. Break to SA
            break

    # SA Phase
    T = 50.0
    cooling = 0.99
    
    while time.time() < deadline:
        i = np.random.randint(n_primes)
        p = primes[i]
        r_old = residues[i]
        r_new = np.random.randint(p)
        
        if r_new == r_old:
            continue
            
        # Check validity for current prefix to avoid obviously bad moves
        # This is an optimization, not a strict requirement for SA
        if r_old == 0:
            start_loss = p
        else:
            start_loss = r_old
            
        if start_loss <= current_L:
            loss_indices = np.arange(start_loss, current_L + 1, p)
            if np.any(counts[loss_indices] == 2):
                # This move would break the prefix. 
                # We can still consider it if we are in SA mode and T is high, 
                # but it's likely to be rejected. 
                # To save time, we might skip these or handle them carefully.
                # Let's just compute new L and let SA decide.
                pass

        # Apply change
        if r_old == 0:
            start_old = p
        else:
            start_old = r_old
        if start_old <= MAX_L:
            counts[start_old::p] -= 1
            
        if r_new == 0:
            start_new = p
        else:
            start_new = r_new
        if start_new <= MAX_L:
            counts[start_new::p] += 1
            
        new_L = calc_L(counts, current_L)
        delta = new_L - current_L
        
        if delta > 0:
            # Accept
            residues[i] = r_new
            current_L = new_L
            if new_L > best_L:
                best_L = new_L
                best_residues = residues.copy()
        else:
            # Accept with probability
            prob = np.exp(delta / T)
            if np.random.random() < prob:
                residues[i] = r_new
                current_L = new_L
            else:
                # Revert
                residues[i] = r_old
                if start_new <= MAX_L:
                    counts[start_new::p] -= 1
                if start_old <= MAX_L:
                    counts[start_old::p] += 1
                # current_L unchanged
                
        T *= cooling
        if T < 1.0:
            T = 1.0

    return best_residues
