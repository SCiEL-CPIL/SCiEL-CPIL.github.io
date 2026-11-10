# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _primes_upto(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    return [i for i in range(n + 1) if sieve[i]]

def construct():
    deadline = time.time() + 145
    primes = _primes_upto(1000)
    n = len(primes)
    
    # Initial assignment: r_p = 1 for all (gives count >= 2 for all x >= 2)
    residues = [1] * n
    
    # Precompute for fast incremental scoring
    # For each prime index, store (p, r) pairs
    # We'll maintain a counts array and update incrementally
    
    def score_with_counts(counts, cap):
        """Find L from existing counts array."""
        # Find first position where count < 2
        below = np.where(counts[1:cap+1] < 2)[0]
        if len(below) == 0:
            return cap
        return int(below[0])
    
    def rebuild_counts(residues, cap):
        counts = np.zeros(cap + 1, dtype=np.int32)
        for p, r in zip(primes, residues):
            start = r
            if start < 1:
                start += ((1 - start + p - 1) // p) * p
            if start <= cap:
                counts[start::p] += 1
        return counts
    
    def update_counts(counts, old_p, old_r, new_p, new_r, cap):
        """Update counts when changing one residue."""
        # Remove old
        start = old_r
        if start < 1:
            start += ((1 - start + old_p - 1) // old_p) * old_p
        if start <= cap:
            counts[start::old_p] -= 1
        # Add new
        start = new_r
        if start < 1:
            start += ((1 - start + new_p - 1) // new_p) * new_p
        if start <= cap:
            counts[start::new_p] += 1
    
    # Start with a reasonable cap for search
    search_cap = 5000
    counts = rebuild_counts(residues, search_cap)
    best_score = score_with_counts(counts, search_cap)
    best_residues = residues[:]
    
    # Order primes: small primes first for gap-filling (they're more effective)
    # For gap filling, try small primes first as they have more "leverage"
    prime_order = list(range(n))
    # Sort by prime value ascending - small primes cover more positions per unit change
    prime_order.sort(key=lambda i: primes[i])
    
    # Phase 1: Greedy gap-filling
    for iteration in range(5000):
        if time.time() > deadline - 30:
            break
        
        sc = score_with_counts(counts, search_cap)
        if sc >= search_cap:
            # Try to extend
            new_cap = min(sc + 1000, 20000)
            if new_cap > search_cap:
                # Rebuild with larger cap
                counts = rebuild_counts(residues, new_cap)
                search_cap = new_cap
                sc = score_with_counts(counts, search_cap)
            if sc >= 20000:
                break
            continue
        
        # Find first gap
        check_L = min(sc + 500, search_cap)
        below = np.where(counts[1:check_L+1] < 2)[0]
        if len(below) == 0:
            # No gap in check range, try extending
            new_cap = min(sc + 1000, 20000)
            if new_cap > search_cap:
                counts = rebuild_counts(residues, new_cap)
                search_cap = new_cap
                sc = score_with_counts(counts, search_cap)
            continue
        
        bad_x = int(below[0]) + 1
        fixed = False
        
        # Try to fix by changing one residue - small primes first
        for idx in prime_order:
            p = primes[idx]
            old_r = residues[idx]
            new_r = bad_x % p
            if old_r == new_r:
                continue
            
            # Check if this change would help (new_r covers bad_x)
            # It does by construction
            update_counts(counts, p, old_r, p, new_r, search_cap)
            residues[idx] = new_r
            
            new_sc = score_with_counts(counts, search_cap)
            if new_sc > sc:
                sc = new_sc
                fixed = True
                break
            else:
                # Revert
                update_counts(counts, p, new_r, p, old_r, search_cap)
                residues[idx] = old_r
        
        if not fixed:
            # Try changing a residue to cover a slightly different position
            # or try multiple positions
            improved = False
            for offset in range(1, 20):
                if time.time() > deadline - 30:
                    break
                target = bad_x + offset
                if target > check_L:
                    break
                for idx in prime_order[:50]:  # Only try first 50 primes
                    p = primes[idx]
                    old_r = residues[idx]
                    new_r = target % p
                    if old_r == new_r:
                        continue
                    update_counts(counts, p, old_r, p, new_r, search_cap)
                    residues[idx] = new_r
                    new_sc = score_with_counts(counts, search_cap)
                    if new_sc > sc:
                        sc = new_sc
                        fixed = True
                        improved = True
                        break
                    else:
                        update_counts(counts, p, new_r, p, old_r, search_cap)
                        residues[idx] = old_r
                if improved:
                    break
            if not improved:
                break
        
        if sc > best_score:
            best_score = sc
            best_residues = residues[:]
    
    # Phase 2: Local search with random perturbations
    if time.time() < deadline - 20:
        current = best_residues[:]
        current_counts = rebuild_counts(current, search_cap)
        current_score = score_with_counts(current_counts, search_cap)
        
        T = 50.0
        T_min = 1.0
        cooling = 0.999
        
        while T > T_min and time.time() < deadline - 10:
            idx = random.randint(0, n - 1)
            p = primes[idx]
            old_r = current[idx]
            new_r = random.randint(0, p - 1)
            if new_r == old_r:
                continue
            
            update_counts(current_counts, p, old_r, p, new_r, search_cap)
            current[idx] = new_r
            
            new_score = score_with_counts(current_counts, search_cap)
            delta = new_score - current_score
            
            if delta >= 0 or random.random() < np.exp(delta / max(T, 0.1)):
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_residues = current[:]
            else:
                update_counts(current_counts, p, new_r, p, old_r, search_cap)
                current[idx] = old_r
            
            T *= cooling
    
    # Phase 3: Final verification with full cap
    if time.time() < deadline - 3:
        final_counts = rebuild_counts(best_residues, 100000)
        final_score = score_with_counts(final_counts, 100000)
        if final_score > best_score:
            best_score = final_score
    
    return best_residues
# EVOLVE-BLOCK-END
