import time
import random
import math
from fractions import Fraction
import numpy as np

def construct():
    t0 = time.time()
    deadline = t0 + 115.0

    def _primes_upto(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, n+1, i):
                    sieve[j] = False
        return [i for i in range(2, n+1) if sieve[i]]

    all_primes = _primes_upto(1000)
    # We need sum(1/p) <= 1.
    # Greedy selection of smallest primes minimizes the "waste" in terms of coverage density?
    # Actually, smaller primes cover more numbers per unit cost.
    # Let's select primes until cost exceeds 1, then remove the largest (most expensive) to get back under?
    # Or just take as many small primes as possible.
    
    # Strategy: Start with all primes >= 13 (2,3,5,7 are too small? No, 1/2+1/3+1/5+1/7 = 0.5+0.33+0.2+0.14 = 1.17 > 1).
    # So we must exclude some small primes or use a subset.
    # To maximize coverage length L, we want primes that are "well distributed".
    # Common heuristic: Use primes up to a certain limit.
    # Let's try to include as many small primes as possible without exceeding cost 1.
    # Sort primes by 1/p descending (i.e., p ascending).
    # Add them until sum > 1. Then remove the last one added?
    # Actually, to maximize L, we want high coverage density. Small primes provide high density.
    # So we want the set of primes with smallest values such that sum(1/p) <= 1.
    
    selected_primes = []
    cost = Fraction(0, 1)
    for p in all_primes:
        if cost + Fraction(1, p) <= Fraction(1, 1):
            selected_primes.append(p)
            cost += Fraction(1, p)
        else:
            # Try swapping? For simplicity, just stop.
            # This gives us 2, 3, 5, 7? 1/2+1/3+1/5+1/7 = 173/210 ~ 0.82.
            # Next is 11: 173/210 + 1/11 = (1903+210)/2310 = 2113/2310 ~ 0.91.
            # Next is 13: 2113/2310 + 1/13 = (27469+2310)/30030 = 29779/30030 ~ 0.99.
            # Next is 17: 29779/30030 + 1/17 > 1.
            # So {2,3,5,7,11,13} is a good set.
            pass

    # Let's verify the set {2,3,5,7,11,13}
    # Sum = 1/2+1/3+1/5+1/7+1/11+1/13
    # This is a very dense set.
    # Is it better to swap 13 for something else? No, 13 is the next smallest.
    # What if we exclude 2? 1/3+1/5+1/7+1/11+1/13+1/17...
    # 1/2 is huge. It covers half the integers. It's very efficient.
    # So we should definitely include 2, 3, 5, 7, 11, 13.
    
    # Let's stick with the greedy fill: smallest primes first.
    # The code above does this.
    
    if not selected_primes:
        return []

    N = len(selected_primes)
    MAX_L = 100000 # Cap for evaluation, but we can search up to this.
    # However, creating an array of size 100000 is fine.
    # But wait, the task says "capped at 10^5".
    # Using MAX_L = 100000 is safe.
    
    # Optimization: We don't need to go up to 100000 immediately if we know L is small.
    # But let's use a reasonable bound for the search, say 20000, and expand if needed?
    # No, fixed array is faster for NumPy. 100000 int32 is 400KB. Very small.
    
    counts = np.zeros(MAX_L + 2, dtype=np.int32)
    residues = [0] * N

    # Initialize residues.
    # A good initialization is to set residues to 0? Or random?
    # If residue is 0, it covers p, 2p, ...
    # If residue is 1, it covers 1, 1+p, ...
    # To cover 1..L, we need 1 to be covered.
    # Let's initialize with random residues to break symmetry.
    for i in range(N):
        residues[i] = random.randint(0, selected_primes[i] - 1)

    # Apply initial residues to counts
    for i in range(N):
        p = selected_primes[i]
        r = residues[i]
        if r == 0:
            start = p
        else:
            start = r
        if start <= MAX_L:
            idxs = np.arange(start, MAX_L + 1, p)
            counts[idxs] += 1

    def calc_L():
        # Find first index k in 1..MAX_L where counts[k] == 0
        # If none, return MAX_L
        # We can use np.nonzero on the slice
        # But checking for the first zero is tricky with nonzero.
        # Better: find indices where counts > 0.
        # If the first element of that list is not 0 (relative to start 1), then 1 is not covered.
        
        # Efficient way:
        # If counts[1] == 0, return 0.
        if counts[1] == 0:
            return 0
            
        # Find the first index i > 1 such that counts[i] == 0.
        # We can iterate? No, too slow in Python.
        # Use NumPy:
        # Create a boolean array where counts > 0
        # Find the first False in the range 1..MAX_L
        
        # np.argmax on a boolean array returns the index of the first True.
        # We want the first False.
        # So, find the first True in (counts == 0).
        # But we must start from index 1.
        
        zero_mask = counts[1:MAX_L+1] == 0
        if not np.any(zero_mask):
            return MAX_L
        first_zero_idx = np.argmax(zero_mask)
        # first_zero_idx is 0-based in the slice [1..MAX_L]
        # So the actual index is first_zero_idx + 1
        # L is the index before the first zero.
        # If first zero is at index 1 (value 1), L=0.
        # If first zero is at index 2 (value 2), L=1.
        # So L = first_zero_idx
        return first_zero_idx

    def apply_change(i, old_r, new_r):
        p = selected_primes[i]
        
        # Remove old
        if old_r == 0:
            start_old = p
        else:
            start_old = old_r
        if start_old <= MAX_L:
            idxs = np.arange(start_old, MAX_L + 1, p)
            counts[idxs] -= 1
            
        # Add new
        if new_r == 0:
            start_new = p
        else:
            start_new = new_r
        if start_new <= MAX_L:
            idxs = np.arange(start_new, MAX_L + 1, p)
            counts[idxs] += 1

    L = calc_L()
    best_L = L
    best_residues = residues[:]

    def greedy_extend(current_L):
        extended = True
        L_local = current_L
        while extended and time.time() < deadline:
            extended = False
            # Try to cover L_local + 1
            target_val = L_local + 1
            if target_val > MAX_L:
                break
                
            # Find primes that can cover target_val
            # Any prime p can cover target_val by setting residue to target_val % p.
            # We need to check if changing residue is safe.
            # Safe if all previously covered numbers by the old residue are still covered by others.
            
            # To speed up, we can shuffle the primes.
            order = list(range(N))
            random.shuffle(order)
            
            for i in order:
                if time.time() > deadline:
                    return L_local
                    
                p = selected_primes[i]
                r_old = residues[i]
                r_new = target_val % p
                
                if r_old == r_new:
                    continue
                
                # Check safety:
                # All n in (old residue class) intersect [1, L_local] must have count > 1.
                # Note: If r_old == 0, the class is p, 2p, ...
                # If r_old != 0, the class is r_old, r_old+p, ...
                
                if r_old == 0:
                    start_chk = p
                else:
                    start_chk = r_old
                    
                if start_chk > L_local:
                    # No overlap with 1..L_local, so it's safe to change
                    safe = True
                else:
                    idxs = np.arange(start_chk, L_local + 1, p)
                    if len(idxs) > 0:
                        if np.min(counts[idxs]) > 1:
                            safe = True
                        else:
                            safe = False
                    else:
                        safe = True

                if safe:
                    apply_change(i, r_old, r_new)
                    residues[i] = r_new
                    
                    # Extend L
                    while L_local < MAX_L and counts[L_local + 1] > 0:
                        L_local += 1
                    extended = True
                    break # Restart search for next gap with new L
                
        return L_local

    # Main Loop
    while time.time() < deadline:
        new_L = greedy_extend(L)
        
        if new_L > L:
            L = new_L
            if L > best_L:
                best_L = L
                best_residues = residues[:]
        else:
            # Stuck. Kick.
            save_residues = residues[:]
            save_counts = counts.copy()
            save_L = L
            
            # Kick: Change a few random residues
            K = random.randint(2, 5)
            for _ in range(K):
                i_kick = random.randint(0, N - 1)
                p_kick = selected_primes[i_kick]
                old_r_kick = residues[i_kick]
                new_r_kick = random.randint(0, p_kick - 1)
                
                if old_r_kick != new_r_kick:
                    apply_change(i_kick, old_r_kick, new_r_kick)
                    residues[i_kick] = new_r_kick
            
            # Repair: Try to extend from the new state
            L_temp = calc_L()
            L_repaired = greedy_extend(L_temp)
            
            if L_repaired > best_L:
                L = L_repaired
                best_L = L
                best_residues = residues[:]
            elif L_repaired >= L:
                # Keep the new state if it's not worse
                L = L_repaired
            else:
                # Revert
                residues[:] = save_residues
                counts[:] = save_counts
                L = save_L

    return [[selected_primes[i], best_residues[i]] for i in range(N)]
