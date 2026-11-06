import numpy as np
import time
import random

def construct():
    # Generate primes up to 1000
    primes = []
    for p in range(2, 1001):
        is_prime = True
        for q in range(2, int(p**0.5) + 1):
            if p % q == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(p)
    
    n_primes = len(primes)  # 168
    MAX_X = 20000  # Allow searching up to 20000 to find high L
    
    # Convert to numpy for speed
    primes_np = np.array(primes, dtype=np.int64)
    
    def compute_L(cov):
        """Compute the largest L such that cov[1..L] >= 2."""
        # cov is 0-indexed, we care about indices 1..MAX_X
        # Use numpy to find first index where cov < 2
        # cov[1:MAX_X+1]
        if MAX_X == 0:
            return 0
        # Find first position where cov < 2
        below = np.where(cov[1:MAX_X+1] < 2)[0]
        if len(below) == 0:
            return MAX_X
        return below[0]  # This is 0-indexed relative to cov[1:], so actual x = below[0] + 1? 
        # Wait: cov[1:MAX_X+1] is indices 1..MAX_X.
        # If below[0] is 0, it means cov[1] < 2, so L=0.
        # If below[0] is 5, it means cov[1+5]=cov[6] < 2, so L=5.
        # So L = below[0].
    
    def update_cov(cov, p, old_r, new_r, delta):
        """Update coverage array for prime p changing residue from old_r to new_r by delta."""
        # delta is -1 for removing, +1 for adding
        if old_r == 0:
            start = p
        else:
            start = old_r
        # Indices: start, start+p, start+2p, ... <= MAX_X
        if start <= MAX_X:
            idx = np.arange(start, MAX_X + 1, p, dtype=np.int64)
            cov[idx] += delta
    
    def greedy_construct():
        """Greedy construction: assign residues to maximize coverage."""
        cov = np.zeros(MAX_X + 1, dtype=np.int32)
        residues = [0] * n_primes
        
        # We want to cover 1..MAX_X at least twice.
        # Strategy: For each x from 1 to MAX_X, if cov[x] < 2, assign a prime to cover it.
        # Which prime? Pick one that causes least damage or is "cheap".
        # Actually, let's iterate x and fix gaps.
        
        # Better: Iterate primes and assign residues to maximize coverage of low numbers?
        # No, the standard approach for this problem is often to start with r_p = 1 or 0 and then repair.
        # Let's try a greedy repair from r_p = 1 for all p.
        
        residues = [1] * n_primes
        # Initialize cov
        for i, p in enumerate(primes):
            r = residues[i]
            if r == 0:
                start = p
            else:
                start = r
            if start <= MAX_X:
                idx = np.arange(start, MAX_X + 1, p, dtype=np.int64)
                cov[idx] += 1
        
        L = compute_L(cov)
        
        # Repair loop
        max_iter = 5000
        for _ in range(max_iter):
            # Find first gap
            if L >= MAX_X:
                break
            # Check if L+1 is covered
            if cov[L+1] >= 2:
                # Extend L
                L += 1
                continue
            
            # Need to cover L+1
            target_x = L + 1
            need = 2 - cov[target_x]
            
            # Find primes that can cover target_x
            # new_r = target_x % p
            # We want to pick a prime where changing residue to new_r doesn't break existing coverage too much.
            # Heuristic: Pick largest p? Smallest p?
            # Let's try picking the prime with the largest p that currently doesn't cover target_x.
            # Or pick the one with minimal "damage".
            
            best_idx = -1
            best_score = -1
            best_new_r = -1
            
            for i, p in enumerate(primes):
                new_r = target_x % p
                if residues[i] == new_r:
                    continue # Already covers it? No, if it already covers it, cov[target_x] would be higher.
                             # Wait, if residues[i] == new_r, then this prime ALREADY covers target_x.
                             # So cov[target_x] already includes this prime.
                             # So we should only consider primes that DO NOT currently cover target_x.
                
                # Calculate damage: how many y in [1, L] with cov[y] == 2 would lose coverage?
                # This is expensive to compute exactly.
                # Approximation: Prefer large p (less frequent hits) or small p?
                # Large p hits fewer points in [1, L], so less likely to break something.
                # Let's score by p (larger is better).
                
                score = p
                if score > best_score:
                    best_score = score
                    best_idx = i
                    best_new_r = new_r
            
            if best_idx == -1:
                # All primes already cover target_x? Then cov[target_x] >= n_primes.
                # But cov[target_x] < 2. Contradiction.
                # Unless target_x is not covered by any prime?
                # If target_x < 2, no prime covers it?
                # Primes >= 2.
                # If target_x = 1, r_p = 1 covers it.
                # So this shouldn't happen.
                break
            
            # Apply change
            p = primes[best_idx]
            old_r = residues[best_idx]
            new_r = best_new_r
            
            update_cov(cov, p, old_r, new_r, -1)
            update_cov(cov, p, new_r, new_r, 1)
            residues[best_idx] = new_r
            
            # L might have decreased if we broke a spot in [1, L]
            # Recalculate L
            new_L = compute_L(cov)
            if new_L < L:
                # We broke the prefix. This is bad.
                # For now, just accept it and try to fix it in the next iteration.
                # But if we keep breaking, we might loop.
                # Let's just update L.
                L = new_L
            else:
                L = new_L
        
        return residues, L
    
    def local_search(residues, L):
        """Local search to improve L."""
        cov = np.zeros(MAX_X + 1, dtype=np.int32)
        for i, p in enumerate(primes):
            r = residues[i]
            if r == 0:
                start = p
            else:
                start = r
            if start <= MAX_X:
                idx = np.arange(start, MAX_X + 1, p, dtype=np.int64)
                cov[idx] += 1
        
        # Ensure L is correct
        L = compute_L(cov)
        
        max_iter = 10000
        for _ in range(max_iter):
            if L >= MAX_X:
                break
            
            # Find first gap
            target_x = L + 1
            if cov[target_x] >= 2:
                L += 1
                continue
            
            need = 2 - cov[target_x]
            
            # Try to find a prime to change
            best_idx = -1
            best_score = -1
            best_new_r = -1
            
            # Sort primes by size? We want to minimize damage.
            # Damage is proportional to the number of points in [1, L] that are currently at cov==2
            # and are hit by this prime with the old residue.
            
            # Heuristic: Try all primes, pick the one with minimal estimated damage.
            # Damage estimate: count of y in [1, L] where y % p == old_r and cov[y] == 2.
            
            for i, p in enumerate(primes):
                new_r = target_x % p
                if residues[i] == new_r:
                    continue
                
                old_r = residues[i]
                
                # Calculate damage
                # Indices y in [1, L] such that y % p == old_r
                if old_r == 0:
                    start_y = p
                else:
                    start_y = old_r
                
                if start_y > L:
                    damage = 0
                else:
                    # y = start_y, start_y+p, ... <= L
                    # Check how many of these have cov[y] == 2
                    # This is a sparse check.
                    # np.arange(start_y, L+1, p)
                    # Then check cov[indices] == 2
                    # This is fast if L/p is small.
                    idx = np.arange(start_y, L + 1, p, dtype=np.int64)
                    if len(idx) > 0:
                        damage = np.sum(cov[idx] == 2)
                    else:
                        damage = 0
                
                # Score: minimize damage, maximize p (larger p is safer generally)
                score = -damage * 1000 + p
                if score > best_score:
                    best_score = score
                    best_idx = i
                    best_new_r = new_r
            
            if best_idx == -1:
                break
            
            # Apply change
            p = primes[best_idx]
            old_r = residues[best_idx]
            new_r = best_new_r
            
            update_cov(cov, p, old_r, new_r, -1)
            update_cov(cov, p, new_r, new_r, 1)
            residues[best_idx] = new_r
            
            # Update L
            new_L = compute_L(cov)
            if new_L < L:
                # Revert? Or accept?
                # If we break the prefix, L drops.
                # Let's accept and hope to fix it.
                L = new_L
            else:
                L = new_L
        
        return residues, L
    
    best_L = 0
    best_res = None
    start_time = time.time()
    time_budget = 115
    
    # Strategy 1: Greedy from 1s
    res, L = greedy_construct()
    if L > best_L:
        best_L = L
        best_res = res[:]
    
    # Strategy 2: Local search on result of Strategy 1
    if time.time() - start_time < time_budget:
        res2, L2 = local_search(res, L)
        if L2 > best_L:
            best_L = L2
            best_res = res2[:]
    
    # Strategy 3: Random restarts
    while time.time() - start_time < time_budget:
        # Random init
        res_init = [random.randint(0, p - 1) for p in primes]
        # Quick local search
        res3, L3 = local_search(res_init, 0)
        if L3 > best_L:
            best_L = L3
            best_res = res3[:]
        
        # If we hit a high score, we can stop early?
        # No, keep trying to maximize.
    
    if best_res is None:
        return [0] * 168
        
    return best_res
