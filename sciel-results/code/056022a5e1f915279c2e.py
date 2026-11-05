import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit ** 0.5) + 1):
            if sieve[i]:
                sieve[i * i:limit + 1:i] = [False] * len(sieve[i * i:limit + 1:i])
        return [i for i, is_prime in enumerate(sieve) if is_prime]

    primes = get_primes(1000)
    n_primes = len(primes)
    MAX_L = 100000
    
    # Initialize residues
    # Start with 0, then improve
    residues = [0] * n_primes
    
    # Counts array: counts[i] is the number of primes p such that i == r_p (mod p)
    # We care about i in 1..MAX_L
    counts = np.zeros(MAX_L + 1, dtype=np.int32)
    
    def update_counts(p, old_r, new_r, delta):
        """
        Update counts array by adding or removing coverage for prime p.
        delta is +1 to add, -1 to remove.
        """
        # Positions covered by r mod p are r, r+p, r+2p... 
        # Note: if r==0, positions are p, 2p, ... (since we look at 1..MAX_L, 0 is not included)
        # Actually, the congruence x = r (mod p) includes x=r if r>0.
        # If r=0, x=p, 2p...
        
        start = old_r if old_r > 0 else p
        if start <= MAX_L:
            counts[start::p] += delta
        
    # Initialize counts with residues = 0
    for i, p in enumerate(primes):
        r = residues[i]
        start = r if r > 0 else p
        if start <= MAX_L:
            counts[start::p] += 1

    def find_L(counts):
        # Find first index >= 1 where count < 2
        # If all are >= 2, return MAX_L
        # np.where returns indices. We want the first one.
        # counts[1:MAX_L+1] < 2
        mask = counts[1:MAX_L+1] < 2
        if not np.any(mask):
            return MAX_L
        # argmax on boolean array returns index of first True
        idx = np.argmax(mask)
        return idx + 1 # +1 because we sliced from 1

    current_L = find_L(counts)
    best_L = current_L
    best_residues = residues.copy()

    # Precompute indices for speed
    prime_indices = list(range(n_primes))

    # --- Phase 1: Greedy Repair Loop ---
    # We iteratively find the first failure and try to fix it by changing one residue.
    
    no_improve_count = 0
    max_no_improve = 50 # If we can't improve for 50 steps, do a random kick
    
    while time.time() < deadline:
        if no_improve_count > max_no_improve:
            # Random kick: change a few random residues to escape local minima
            kick_size = 5
            kick_indices = np.random.choice(prime_indices, kick_size, replace=False)
            for idx in kick_indices:
                p = primes[idx]
                old_r = residues[idx]
                new_r = np.random.randint(p)
                if old_r == new_r:
                    continue
                # Remove old
                update_counts(p, old_r, new_r, -1)
                residues[idx] = new_r
                # Add new
                update_counts(p, old_r, new_r, 1) # Wait, update_counts logic needs care.
                
                # Let's rewrite update logic to be explicit to avoid bugs
                # Actually, the update_counts function above adds delta to the OLD position? 
                # No, it adds delta to the position specified by old_r? 
                # My update_counts function is flawed. It uses old_r to find the slice, but applies delta.
                # If I want to remove old_r, I call update_counts(p, old_r, 0, -1).
                # If I want to add new_r, I call update_counts(p, new_r, 0, 1).
                
                # Let's fix the kick logic with explicit add/remove
                # Revert the kick for a moment to fix the logic in the main loop
                # Actually, let's just restart the loop structure with a cleaner update function.
                
                # I will define a helper to change residue
                pass
            
            no_improve_count = 0
            current_L = find_L(counts)
            if current_L > best_L:
                best_L = current_L
                best_residues = residues.copy()
            continue

        # Find current failure
        x_fail = current_L
        if x_fail == MAX_L:
            # We reached the max, try to push further? No, MAX_L is the cap.
            # If we are at MAX_L, we can't improve. Break? 
            # Or maybe we can optimize for "doubly covered" beyond 100000? No, capped at 10^5.
            # So if L == 100000, we are done.
            break

        # Try to fix x_fail
        best_new_L = current_L
        best_p_idx = -1
        best_new_r = -1
        
        # We need to find a prime p and new residue r such that:
        # 1. x_fail is covered by the new residue (i.e., x_fail % p == r)
        # 2. The new L is maximized.
        
        # For each prime, the only residue that covers x_fail is x_fail % p.
        # So we only have 1 candidate per prime.
        
        for i in prime_indices:
            p = primes[i]
            r_target = x_fail % p
            r_old = residues[i]
            
            if r_old == r_target:
                # This prime already covers x_fail. Changing it would remove coverage from x_fail.
                # So this move would decrease count at x_fail, making it worse.
                # Unless... changing it allows a *different* prime to cover it? No, we are changing one at a time.
                # So skip.
                continue
            
            # Simulate change
            # Remove r_old
            start_old = r_old if r_old > 0 else p
            if start_old <= MAX_L:
                counts[start_old::p] -= 1
            
            # Add r_target
            start_new = r_target if r_target > 0 else p
            if start_new <= MAX_L:
                counts[start_new::p] += 1
            
            # Check new L
            new_L = find_L(counts)
            
            # Revert change
            if start_new <= MAX_L:
                counts[start_new::p] -= 1
            if start_old <= MAX_L:
                counts[start_old::p] += 1
            
            if new_L > best_new_L:
                best_new_L = new_L
                best_p_idx = i
                best_new_r = r_target
        
        if best_p_idx != -1:
            # Apply the best move
            p = primes[best_p_idx]
            r_old = residues[best_p_idx]
            r_new = best_new_r
            
            # Remove old
            start_old = r_old if r_old > 0 else p
            if start_old <= MAX_L:
                counts[start_old::p] -= 1
            
            # Add new
            start_new = r_new if r_new > 0 else p
            if start_new <= MAX_L:
                counts[start_new::p] += 1
            
            residues[best_p_idx] = r_new
            current_L = best_new_L
            no_improve_count = 0
            
            if current_L > best_L:
                best_L = current_L
                best_residues = residues.copy()
        else:
            # No single move improved L.
            no_improve_count += 1

    return best_residues
