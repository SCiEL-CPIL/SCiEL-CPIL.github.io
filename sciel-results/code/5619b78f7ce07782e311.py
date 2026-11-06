"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def _primes_up_to(n):
    sieve = np.ones(n + 1, dtype=bool)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = False
    return [i for i in range(n + 1) if sieve[i]]

def _compute_L(residues, primes, cap=100000):
    """Compute the largest L such that every integer 1..L has count >= 2."""
    # We only need to check up to the first failure.
    # For efficiency, we can check in chunks or just up to a reasonable limit if L is small.
    # But the problem asks for exact count capped at 10^5.
    # However, our search targets L <= 1000.
    # Let's compute up to 2000 to be safe, or just until failure.
    L_max_check = min(cap, 2000)
    counts = np.zeros(L_max_check + 1, dtype=np.int32)
    for r, p in zip(residues, primes):
        if r == 0:
            start = p
        else:
            start = r
        if start <= L_max_check:
            counts[start::p] += 1
            
    if counts[1] < 2:
        return 0
        
    # Find first index where count < 2
    # counts[1:] < 2
    mask = counts[1:] < 2
    if not mask.any():
        return L_max_check
    first_bad = np.argmax(mask) + 1
    return first_bad

def _solve_for_L(L, primes_all, deadline):
    """
    Attempt to find residues for all primes such that 1..L are doubly covered.
    Returns list of residues or None if failed.
    """
    # Split primes
    P_S = [p for p in primes_all if p <= L]
    P_L = [p for p in primes_all if p > L]
    N_S = len(P_S)
    N_L = len(P_L)
    
    # Target deficit
    # We need sum(max(0, 2 - count[x]) for x in 1..L) <= N_L
    
    # Backtracking state
    counts = np.zeros(L + 1, dtype=np.int32)
    residues_S = [0] * N_S
    
    def backtrack(idx):
        if time.time() > deadline:
            return False
            
        if idx == N_S:
            # Check final deficit
            deficit = 0
            for x in range(1, L + 1):
                if counts[x] < 2:
                    deficit += 2 - counts[x]
            return deficit <= N_L
        
        p = P_S[idx]
        
        # Pruning: Max possible reduction from remaining P_S
        # Current deficit
        curr_deficit = 0
        for x in range(1, L + 1):
            if counts[x] < 2:
                curr_deficit += 2 - counts[x]
                
        # Max reduction from P_S[idx+1:]
        max_red = 0
        for j in range(idx + 1, N_S):
            max_red += L // P_S[j]
            
        if curr_deficit - max_red > N_L:
            return False
            
        # Generate candidates
        # We want to cover positions with low counts.
        # Score each residue r
        # r covers positions r, r+p, ...
        
        # Optimization: Only consider residues that cover at least one position with count < 2?
        # No, we must consider all, but sort by heuristic.
        
        candidates = []
        for r in range(p):
            # Count how many deficit positions are covered
            # Deficit position x: counts[x] < 2
            # Covered if x % p == r % p (with 0 handling)
            
            # Efficient check:
            # Positions covered: r, r+p, ...
            # We can just iterate through them
            n_cover_0 = 0 # counts[x] == 0
            n_cover_1 = 0 # counts[x] == 1
            n_cover_2 = 0 # counts[x] >= 2
            
            start = r if r > 0 else p
            if start <= L:
                for x in range(start, L + 1, p):
                    if counts[x] == 0:
                        n_cover_0 += 1
                    elif counts[x] == 1:
                        n_cover_1 += 1
                    else:
                        n_cover_2 += 1
            
            # Heuristic: Prioritize covering 0s, then 1s.
            # Score = 100 * n_cover_0 + 10 * n_cover_1 - n_cover_2
            score = 100 * n_cover_0 + 10 * n_cover_1 - n_cover_2
            candidates.append((score, r))
            
        # Sort by score descending
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        for score, r in candidates:
            # Apply change
            # Save old counts? No, we can just update and revert.
            # But we need to know which positions changed to revert.
            
            # Actually, for small L, copying counts is fast.
            old_counts = counts.copy()
            
            start = r if r > 0 else p
            if start <= L:
                counts[start::p] += 1
                
            residues_S[idx] = r
            
            if backtrack(idx + 1):
                return True
                
            # Revert
            counts[:] = old_counts
            
        return False

    if backtrack(0):
        # Construct full residues
        # P_S residues are in residues_S
        # P_L residues: assign to cover deficit
        # Calculate final deficit
        deficit_positions = []
        for x in range(1, L + 1):
            if counts[x] < 2:
                for _ in range(2 - counts[x]):
                    deficit_positions.append(x)
                    
        # We have N_L primes and len(deficit_positions) <= N_L slots to fill.
        # Assign P_L primes to these slots.
        residues_L = [0] * N_L
        for i, p in enumerate(P_L):
            if i < len(deficit_positions):
                x = deficit_positions[i]
                residues_L[i] = x % p # which is x since p > L >= x
            else:
                residues_L[i] = 0 # arbitrary, doesn't cover anything in 1..L
                
        # Combine
        final_residues = residues_S + residues_L
        return final_residues
    else:
        return None

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    primes = _primes_up_to(1000)
    
    best_L = 0
    best_res = [0] * len(primes)
    
    # Start with a reasonable guess, e.g., 100
    # If it works, try higher. If not, try lower.
    # Let's do a linear scan with step 10, then refine.
    
    # First, try to find a feasible L
    # Start at 100
    L = 100
    
    # If 100 is too hard, maybe start lower?
    # Let's try 50 first to ensure we have a baseline.
    L = 50
    while L < 1000 and time.time() < deadline:
        res = _solve_for_L(L, primes, deadline)
        if res is not None:
            # Verify
            actual_L = _compute_L(res, primes, cap=100000)
            if actual_L >= L:
                best_L = actual_L
                best_res = res
                # Try to go higher
                L += 10
            else:
                # Solution found for L but actual L is lower?
                # This can happen if our deficit logic was too optimistic or verification failed.
                # Use the actual L found and try to improve from there?
                # Or just accept it and try a slightly higher target?
                best_L = actual_L
                best_res = res
                L += 5
        else:
            # Failed to find solution for L
            if L == 50:
                # Even 50 failed? Return zeros or try 10.
                # Try smaller
                L = 10
            else:
                # Back off
                L -= 10
                if L < 10:
                    break
                    
    # Refine: Try to extend best_L by 1
    # This is expensive, so only do it if we have time
    if time.time() < deadline - 10:
        # We can try a local search on best_res to extend L
        # But _solve_for_L is already a strong solver.
        # Let's just try L = best_L + 1, + 2 ...
        for target_L in range(best_L + 1, best_L + 20):
            if time.time() > deadline - 5:
                break
            res = _solve_for_L(target_L, primes, deadline)
            if res is not None:
                actual_L = _compute_L(res, primes, cap=100000)
                if actual_L > best_L:
                    best_L = actual_L
                    best_res = res
            else:
                break
                
    return best_res
# EVOLVE-BLOCK-END
