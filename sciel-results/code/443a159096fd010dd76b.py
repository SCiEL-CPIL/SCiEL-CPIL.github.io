"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
import numpy as np

def _primes_upto(n):
    sieve = bytearray(b'\x01') * (n + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(2, n+1) if sieve[i]]

def _build_counts_np(residues, primes, max_L):
    """Build count array using numpy for speed."""
    counts = np.zeros(max_L + 1, dtype=np.int32)
    for p, r in zip(primes, residues):
        if r == 0:
            start = p
        else:
            start = r
        if start > max_L:
            continue
        counts[start::p] += 1
    return counts

def _find_gap(counts, max_L):
    """Find the first index x in 1..max_L where counts[x] < 2."""
    # Using numpy to find first index where condition is true
    # counts is 1-indexed effectively (0 is unused)
    # We want min x such that counts[x] < 2
    # Create a boolean mask
    # To avoid creating a full mask if possible, we can just loop if max_L is small (10000)
    # Loop in Python for 10000 items is very fast (~0.5ms)
    for x in range(1, max_L + 1):
        if counts[x] < 2:
            return x
    return max_L + 1

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safe margin for 120s limit
    
    primes = _primes_upto(1000)
    n_primes = len(primes)
    
    # We aim for L >= 1000, but let's work with a slightly larger window to ensure stability
    WORK_L = 20000 
    
    # Phase 1: Greedy Initialization
    # Strategy: For each prime, choose the residue that helps cover the current "frontier".
    # A simpler, effective heuristic: Random init + immediate repair.
    
    # Better Init: 
    # We want high coverage everywhere.
    # Let's try a "balanced" random init first, then repair.
    
    residues = [random.randint(0, p - 1) for p in primes]
    counts = _build_counts_np(residues, primes, WORK_L)
    
    current_gap = _find_gap(counts, WORK_L)
    best_score = current_gap - 1
    best_residues = residues[:]
    
    # Phase 2: Rapid Local Search
    # Loop until time runs out.
    # Each step:
    # 1. Find gap G.
    # 2. If G > WORK_L, we are done (or hit limit).
    # 3. Identify primes that DO NOT cover G.
    # 4. Pick one such prime (randomly or best candidate) and change its residue to cover G.
    # 5. Update counts.
    # 6. Check if we broke the prefix. If yes, maybe revert or accept if we still improve overall "min gap".
    
    # To be fast, we will accept moves that fix the current gap, even if they break earlier spots, 
    # BUT we will prioritize moves that don't break earlier spots if possible.
    # Given the complexity, a simple "fix the first gap" strategy often works well for this density.
    
    steps = 0
    while time.time() < deadline:
        steps += 1
        if current_gap > WORK_L:
            break
            
        gap = current_gap
        
        # Find primes that cover the gap
        coverers = []
        for i, (p, r) in enumerate(zip(primes, residues)):
            if gap % p == r:
                coverers.append(i)
        
        # We need at least one more coverer.
        # Candidates: all primes NOT in coverers.
        # We want to pick a prime i and set residues[i] = gap % primes[i].
        # Risk: This removes coverage of gap % primes[i] elsewhere.
        
        # Heuristic: Pick a prime that covers few "critical" spots.
        # For speed, let's just pick a random candidate from non-coverers.
        # To improve, we can check the cost (how many x < gap have count 2 and are covered by this prime).
        
        candidates = [i for i in range(n_primes) if i not in coverers]
        if not candidates:
            # Should not happen if gap is a gap (count < 2 implies not all primes cover it)
            break
            
        # Try to find a 0-cost move (no loss of double coverage in [1, gap-1])
        best_move = None
        best_cost = float('inf')
        
        # Sample a few candidates if list is long to save time, or check all if fast enough
        # n_primes = 168. Checking cost for 168 primes involves slicing numpy arrays.
        # This might be too slow if done every step.
        # Optimization: Only check cost if we are stuck? Or just pick random?
        # Let's try random first, it's O(1).
        
        idx = random.choice(candidates)
        p = primes[idx]
        old_r = residues[idx]
        new_r = gap % p
        
        # Calculate cost quickly? 
        # Cost = number of x in [1, gap-1] such that x % p == old_r and counts[x] == 2.
        # If cost > 0, we might break the prefix.
        # Let's just apply it and see. If the new gap is smaller than the old gap, we might revert.
        
        # Apply move
        if old_r == 0:
            start_old = p
        else:
            start_old = old_r
        if start_old <= WORK_L:
            counts[start_old::p] -= 1
            
        if new_r == 0:
            start_new = p
        else:
            start_new = new_r
        if start_new <= WORK_L:
            counts[start_new::p] += 1
            
        residues[idx] = new_r
        
        # Find new gap
        new_gap = _find_gap(counts, WORK_L)
        
        if new_gap > current_gap:
            # Improvement
            current_gap = new_gap
            if current_gap - 1 > best_score:
                best_score = current_gap - 1
                best_residues = residues[:]
        else:
            # Regretted move: revert
            residues[idx] = old_r
            if old_r == 0:
                start_old = p
            else:
                start_old = old_r
            if start_old <= WORK_L:
                counts[start_old::p] += 1
                
            if new_r == 0:
                start_new = p
            else:
                start_new = new_r
            if start_new <= WORK_L:
                counts[start_new::p] -= 1
                
            # If we can't improve by fixing the current gap directly without regression,
            # try a random perturbation to escape local minimum?
            # For now, just break out of this specific gap fixing loop and try a different strategy?
            # Actually, if random move fails, try a few more random candidates.
            improved = False
            for _ in range(10): # Try 10 random candidates
                idx2 = random.choice(candidates)
                if idx2 == idx: continue
                p2 = primes[idx2]
                old_r2 = residues[idx2]
                new_r2 = gap % p2
                
                # Apply
                if old_r2 == 0:
                    s_o2 = p2
                else:
                    s_o2 = old_r2
                if s_o2 <= WORK_L:
                    counts[s_o2::p2] -= 1
                if new_r2 == 0:
                    s_n2 = p2
                else:
                    s_n2 = new_r2
                if s_n2 <= WORK_L:
                    counts[s_n2::p2] += 1
                residues[idx2] = new_r2
                
                new_gap2 = _find_gap(counts, WORK_L)
                
                if new_gap2 > current_gap:
                    current_gap = new_gap2
                    if current_gap - 1 > best_score:
                        best_score = current_gap - 1
                        best_residues = residues[:]
                    improved = True
                    break
                else:
                    # Revert
                    residues[idx2] = old_r2
                    if old_r2 == 0:
                        s_o2 = p2
                    else:
                        s_o2 = old_r2
                    if s_o2 <= WORK_L:
                        counts[s_o2::p2] += 1
                    if new_r2 == 0:
                        s_n2 = p2
                    else:
                        s_n2 = new_r2
                    if s_n2 <= WORK_L:
                        counts[s_n2::p2] -= 1
            
            if not improved:
                # Stuck? Try a random global move to escape?
                # Or just accept that we are stuck at this gap and try to find a different path?
                # Let's try a random single-residue change anywhere.
                rand_idx = random.randint(0, n_primes - 1)
                p_rand = primes[rand_idx]
                old_rand = residues[rand_idx]
                new_rand = random.randint(0, p_rand - 1)
                
                if new_rand == old_rand:
                    continue
                    
                if old_rand == 0:
                    s_or = p_rand
                else:
                    s_or = old_rand
                if s_or <= WORK_L:
                    counts[s_or::p_rand] -= 1
                if new_rand == 0:
                    s_nr = p_rand
                else:
                    s_nr = new_rand
                if s_nr <= WORK_L:
                    counts[s_nr::p_rand] += 1
                residues[rand_idx] = new_rand
                
                new_gap3 = _find_gap(counts, WORK_L)
                
                if new_gap3 > current_gap:
                    current_gap = new_gap3
                    if current_gap - 1 > best_score:
                        best_score = current_gap - 1
                        best_residues = residues[:]
                else:
                    # Revert random move
                    residues[rand_idx] = old_rand
                    if old_rand == 0:
                        s_or = p_rand
                    else:
                        s_or = old_rand
                    if s_or <= WORK_L:
                        counts[s_or::p_rand] += 1
                    if new_rand == 0:
                        s_nr = p_rand
                    else:
                        s_nr = new_rand
                    if s_nr <= WORK_L:
                        counts[s_nr::p_rand] -= 1

    # Ensure proper ints
    residues = best_residues
    residues = [int(r) % p for r, p in zip(residues, primes)]
    
    return residues
# EVOLVE-BLOCK-END
