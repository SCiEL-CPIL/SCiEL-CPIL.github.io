"""
Strategy: Greedy Conflict Resolution with Neighborhood Optimization.

Instead of random hill-climbing, we:
1. Identify the first position $L_{fail}$ where coverage drops below 2.
2. Identify the set of primes $P_{active}$ that cover positions in $[L_{fail}, L_{fail} + W]$ 
   (where W is a window size, e.g., 100).
3. For these active primes, we perform a coordinated local search: 
   we try to assign residues to these primes such that they maximize the 
   minimum coverage in the window $[L_{fail}, L_{fail} + W]$.
4. We use a fast incremental update mechanism to evaluate these changes.
"""

import time
import numpy as np

def _get_primes(n):
    """Sieve of Eratosthenes to get primes up to n."""
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    return [i for i, is_prime in enumerate(sieve) if is_prime]

def _find_first_failure(counts, max_L):
    """Find the first index i in 1..max_L where counts[i] < 2. Returns max_L if none."""
    # counts is 1-indexed effectively, but we use 0-indexed array where counts[x] is count for x
    # We check from 1 to max_L
    if max_L == 0:
        return 0
    
    # We want the first x in [1, max_L] such that counts[x] < 2
    # Using numpy to find first index where condition is met
    # Note: counts[0] is for x=0, which we ignore. We look at counts[1:max_L+1]
    below_two = np.where(counts[1:max_L+1] < 2)[0]
    if len(below_two) == 0:
        return max_L
    return below_two[0] + 1

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    primes = _get_primes(1000)
    n_primes = len(primes)
    max_L_check = 100000  # We evaluate up to this, but the score is capped by the first failure
    
    # Initialize with a simple heuristic: r_p = p // 2 or similar
    # Let's start with residues that are spread out
    residues = [0] * n_primes
    for i, p in enumerate(primes):
        residues[i] = i % p
        
    # Precompute positions for each prime to speed up updates
    # positions[i] is a boolean mask or list of indices? 
    # For memory efficiency and speed, let's just recompute the mask for the active window when needed,
    # or maintain a full count array.
    
    # Full count array approach for fast global evaluation
    counts = np.zeros(max_L_check + 1, dtype=np.int32)
    
    # Helper to update counts for a specific prime
    def update_counts(residues, counts, add=True):
        """Add or subtract the coverage of the current residues from counts."""
        sign = 1 if add else -1
        for i, p in enumerate(primes):
            r = residues[i]
            if r == 0:
                start = p
            else:
                start = r
            if start > max_L_check:
                continue
            # Use numpy to add sign to the positions
            # np.arange(start, max_L_check + 1, p)
            # This creates a temporary array. For 168 primes, this is okay.
            pos = np.arange(start, max_L_check + 1, p)
            if len(pos) > 0:
                counts[pos] += sign

    # Initialize counts
    update_counts(residues, counts, add=True)
    
    # Initial score
    current_L = _find_first_failure(counts, max_L_check)
    best_L = current_L
    best_residues = residues[:]
    
    # Main loop: Greedy Conflict Resolution
    # We focus on the window around the first failure
    window_size = 200
    
    while (time.time() - start_time) < deadline - 5:
        current_L = _find_first_failure(counts, max_L_check)
        
        if current_L > best_L:
            best_L = current_L
            best_residues = residues[:]
        
        # If we've reached the max check limit or a very high score, we might stop or continue
        if current_L >= 1000: 
            # We have achieved the target. Try to push further if time permits?
            # The task says "Maximize", so we should keep trying to go higher.
            # But if we hit 1000, that's a huge jump.
            pass
        
        # Identify the failure point
        fail_pos = current_L
        if fail_pos > max_L_check:
            break
            
        # Define a window around the failure point to optimize
        # We want to fix the coverage in [fail_pos, fail_pos + window_size]
        win_start = max(1, fail_pos - 10)
        win_end = min(max_L_check, fail_pos + window_size)
        
        # Identify which primes are "active" in this window.
        # A prime p is active if it has at least one multiple in [win_start, win_end].
        # Actually, we should consider primes that *could* be changed to cover fail_pos.
        # To cover fail_pos, we must set r_p = fail_pos % p.
        # But changing r_p affects other positions.
        
        # Strategy: 
        # 1. Find all primes p such that fail_pos % p is a "good" candidate.
        #    Actually, let's look at the primes that currently cover positions in the window.
        #    And primes that *could* cover fail_pos (which is all of them, since we can set r_p = fail_pos % p).
        
        # Let's pick a subset of primes to optimize:
        # The ones with smallest p (most frequent hits) are the most powerful levers.
        # Let's take the first K primes (smallest primes) as the "optimizable set".
        K = 20  # Optimize the first 20 primes (2, 3, ..., 71)
        opt_indices = list(range(K))
        
        # For these K primes, we want to find residues that maximize the minimum coverage in the window.
        # This is a small-scale local search.
        
        # Current residues for these primes
        current_opt_residues = [residues[i] for i in opt_indices]
        
        # We will do a few random restarts / local searches on this small subproblem.
        # Objective: Minimize the number of "gaps" (positions with count < 2) in the window,
        # or maximize the minimum count in the window.
        
        # To evaluate a candidate set of residues for the K primes:
        # 1. Subtract their current contribution from counts.
        # 2. Add their new contribution.
        # 3. Check the window.
        # 4. Revert counts.
        
        best_sub_L = -1
        best_sub_residues = current_opt_residues[:]
        
        # We can't try all p^20 combinations. We need a heuristic.
        # Heuristic: For each prime in the set, try to set r_p to cover the "most urgent" uncovered position.
        
        # Let's identify the uncovered positions in the window for the current state
        # Note: counts currently reflects the FULL solution.
        # We need to know the "base" count (excluding the K primes) to evaluate the K primes' contribution.
        
        base_counts = counts.copy()
        # Subtract the K primes' current contribution
        for i in opt_indices:
            p = primes[i]
            r = residues[i]
            if r == 0:
                start = p
            else:
                start = r
            if start <= max_L_check:
                pos = np.arange(start, max_L_check + 1, p)
                if len(pos) > 0:
                    base_counts[pos] -= 1
        
        # Now, for a candidate set of residues for the K primes, the total count is base_counts + sum(contributions)
        
        # Local search on the K primes
        # We'll do a few iterations of "fix the worst hole"
        
        improved = True
        iter_count = 0
        while improved and (time.time() - start_time) < deadline - 10 and iter_count < 50:
            improved = False
            iter_count += 1
            
            # Calculate current coverage in the window for the candidate solution
            # Start with base_counts
            temp_counts = base_counts.copy()
            for i in opt_indices:
                p = primes[i]
                r = residues[i] # Using current global residues as starting point for this sub-search?
                # Wait, we are searching for new residues. Let's maintain a local variable.
                pass
            
            # Let's maintain a local array of residues for the K primes
            # We start with the current global ones
            local_residues = [residues[i] for i in opt_indices]
            
            # Evaluate current local solution
            temp_counts = base_counts.copy()
            for idx, i in enumerate(opt_indices):
                p = primes[i]
                r = local_residues[idx]
                if r == 0:
                    start = p
                else:
                    start = r
                if start <= max_L_check:
                    pos = np.arange(start, max_L_check + 1, p)
                    if len(pos) > 0:
                        temp_counts[pos] += 1
                        
            # Find the first failure in the window [win_start, win_end]
            # Actually, we care about the global first failure, but optimizing the window helps.
            # Let's find the position in [win_start, win_end] with the minimum count.
            window_slice = temp_counts[win_start:win_end+1]
            min_count = np.min(window_slice)
            min_pos_rel = np.argmin(window_slice)
            min_pos_abs = win_start + min_pos_rel
            
            # Try to improve the count at min_pos_abs by changing one of the K primes
            # The prime p can cover min_pos_abs if we set r_p = min_pos_abs % p.
            # Let's try setting each of the K primes to cover min_pos_abs and see if it improves the min count.
            
            best_delta = 0
            best_new_idx = -1
            best_new_r = -1
            
            for idx, i in enumerate(opt_indices):
                p = primes[i]
                new_r = min_pos_abs % p
                if new_r == local_residues[idx]:
                    continue
                
                # Simulate change
                old_r = local_residues[idx]
                
                # Remove old
                if old_r == 0:
                    start_old = p
                else:
                    start_old = old_r
                if start_old <= max_L_check:
                    pos_old = np.arange(start_old, max_L_check + 1, p)
                    if len(pos_old) > 0:
                        temp_counts[pos_old] -= 1
                        
                # Add new
                if new_r == 0:
                    start_new = p
                else:
                    start_new = new_r
                if start_new <= max_L_check:
                    pos_new = np.arange(start_new, max_L_check + 1, p)
                    if len(pos_new) > 0:
                        temp_counts[pos_new] += 1
                        
                # Check improvement in the window
                new_window_slice = temp_counts[win_start:win_end+1]
                new_min_count = np.min(new_window_slice)
                
                # We want to increase the minimum count in the window
                if new_min_count > min_count:
                    # Accept this change temporarily to see if it's the best
                    # But we need to compare all candidates. 
                    # Let's store the best one.
                    if new_min_count > best_delta:
                        best_delta = new_min_count
                        best_new_idx = idx
                        best_new_r = new_r
                        # Store the temp_counts state? No, just revert and re-apply best later.
                        
                # Revert
                if new_r == 0:
                    start_new = p
                else:
                    start_new = new_r
                if start_new <= max_L_check:
                    pos_new = np.arange(start_new, max_L_check + 1, p)
                    if len(pos_new) > 0:
                        temp_counts[pos_new] -= 1
                        
                if old_r == 0:
                    start_old = p
                else:
                    start_old = old_r
                if start_old <= max_L_check:
                    pos_old = np.arange(start_old, max_L_check + 1, p)
                    if len(pos_old) > 0:
                        temp_counts[pos_old] += 1
            
            if best_new_idx != -1:
                # Apply the best change
                local_residues[best_new_idx] = best_new_r
                # Update global residues for these indices
                # And update the global counts array?
                # It's better to keep global counts in sync.
                
                i_global = opt_indices[best_new_idx]
                old_global_r = residues[i_global]
                residues[i_global] = best_new_r
                
                # Update global counts
                p = primes[i_global]
                
                # Remove old
                if old_global_r == 0:
                    start_old = p
                else:
                    start_old = old_global_r
                if start_old <= max_L_check:
                    pos_old = np.arange(start_old, max_L_check + 1, p)
                    if len(pos_old) > 0:
                        counts[pos_old] -= 1
                        
                # Add new
                if best_new_r == 0:
                    start_new = p
                else:
                    start_new = best_new_r
                if start_new <= max_L_check:
                    pos_new = np.arange(start_new, max_L_check + 1, p)
                    if len(pos_new) > 0:
                        counts[pos_new] += 1
                        
                improved = True
                # Update base_counts? 
                # base_counts was calculated once. If we change residues, base_counts becomes stale.
                # Since we are doing a local search within this window, and we update global counts,
                # we should probably restart the local search or update base_counts.
                # For simplicity, let's break out and re-evaluate the whole thing if we made progress.
                break
                
    # Final evaluation
    final_L = _find_first_failure(counts, max_L_check)
    if final_L > best_L:
        best_L = final_L
        best_residues = residues[:]
        
    return best_residues

# Run a quick self-test if executed directly
if __name__ == "__main__":
    res = construct()
    print(f"Result length: {len(res)}")
    print(f"First 10 residues: {res[:10]}")
