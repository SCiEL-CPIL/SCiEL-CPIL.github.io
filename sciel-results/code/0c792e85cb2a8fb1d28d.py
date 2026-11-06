import numpy as np
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # 1. Generate Primes
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(math.isqrt(n)) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]
    
    primes = sieve(1000)
    n_primes = len(primes)
    primes_np = np.array(primes, dtype=np.int64)
    
    # We aim for L >= 1000. Let's check up to 2000 to be safe.
    # If we can extend the frontier, we do so greedily.
    MAX_CHECK = 2000
    x_values = np.arange(1, MAX_CHECK + 1)
    
    # Precompute x mod p for all x and p.
    # Shape: (MAX_CHECK, n_primes)
    x_mod_p = x_values[:, None] % primes_np[None, :]
    
    # Residues array, initialized to -1 (unassigned)
    residues = np.full(n_primes, -1, dtype=np.int64)
    
    # Helper to compute the current double-covered prefix length
    # Returns the largest L such that 1..L are all covered >= 2 times.
    # If no gap found in 1..MAX_CHECK, returns MAX_CHECK.
    def compute_frontier():
        # Only consider assigned residues
        valid_mask = residues >= 0
        if not np.any(valid_mask):
            return 0
        
        # For each x, count how many assigned primes cover it
        # match[i, j] is True if x_values[i] % primes[j] == residues[j] AND residues[j] != -1
        # We can optimize by only checking valid primes, but vectorized over all is fine for 168 primes
        
        # Create a mask of valid residues
        # Note: residues == -1 will never match x_mod_p (which is 0..p-1), so we can just compare directly
        # But we must be careful: if residues[j] is -1, (x_mod_p[:, j] == -1) is always False.
        # So we can just do:
        match = (x_mod_p == residues[None, :])
        counts = match.sum(axis=1)
        
        # Find first index where counts < 2
        invalid = counts < 2
        if not np.any(invalid):
            return MAX_CHECK
        else:
            first_invalid_idx = np.argmax(invalid)
            return int(first_invalid_idx)
    
    # Initial frontier is 0
    current_L = 0
    
    # Greedy Frontier Extension
    # We try to extend the frontier one step at a time.
    # To extend to L+1, the integer x = L+1 must be covered by at least 2 primes.
    # Currently, it might be covered by 0, 1, or more.
    # If it's covered by >= 2, we just advance L.
    # If it's covered by 1, we need to find another prime to cover it.
    # If it's covered by 0, we need to find two primes to cover it.
    
    # However, changing a residue for a prime might break coverage for previous numbers.
    # To minimize breakage, we should prefer primes that are "less critical" for the previous prefix.
    # Or, we can use a repair strategy:
    # 1. Check if x = current_L + 1 is covered >= 2.
    # 2. If yes, increment current_L.
    # 3. If no, find primes that can cover x.
    #    - If 0 primes cover x, pick 2 primes.
    #    - If 1 prime covers x, pick 1 more prime.
    #    For each candidate prime, check if changing its residue to x % p breaks the double coverage of any x' in [1, current_L].
    #    If it doesn't, apply the change.
    #    If multiple candidates work, pick the one that maximizes some heuristic (e.g., covers more future numbers, or is smaller prime).
    #    If no single change works, we might need to backtrack or use a more complex repair.
    #    Given the density of primes, a single change usually works.
    
    # Let's implement a more robust version:
    # Maintain the current set of covered counts for 1..current_L.
    # When we change a residue for prime p, we update the counts for all x in 1..MAX_CHECK where x % p == old_r or x % p == new_r.
    
    # For simplicity and speed, we will recompute the frontier from scratch after each successful move,
    # but we will use a smarter move selection.
    
    # Actually, recomputing from scratch is O(MAX_CHECK * n_primes) = 2000 * 168 ~ 336k ops.
    # This is very fast in NumPy.
    
    # Strategy:
    # While current_L < MAX_CHECK and time < deadline:
    #   x_next = current_L + 1
    #   count_next = number of primes p such that residues[p] == x_next % p
    #   if count_next >= 2:
    #       current_L += 1
    #   else:
    #       needed = 2 - count_next
    #       candidates = []
    #       for each prime index i:
    #           if residues[i] == x_next % primes[i]: continue # already covers
    #           # Check if changing residues[i] to x_next % primes[i] breaks coverage of 1..current_L
    #           # We need to ensure that for all x in 1..current_L, the count of covers remains >= 2
    #           # Old count for x was C_old[x].
    #           # If x % p == old_r, then C_new[x] = C_old[x] - 1
    #           # If x % p == new_r, then C_new[x] = C_old[x] + 1
    #           # We need C_new[x] >= 2 for all x in 1..current_L
    #           # So if x % p == old_r, we need C_old[x] >= 3
    #           # If x % p == new_r, no constraint (count increases)
    #           # If x % p != old_r and != new_r, no change
            
    #           # Let's precompute C_old for 1..current_L
    #           # This is expensive to do for every candidate.
    #           # Alternative: Just try the change, recompute frontier, if it drops below current_L, revert.
    #           # This is O(1) move + O(MAX_CHECK * n_primes) check.
    #           # With 168 primes, we might try up to 168 moves per step.
    #           # 168 * 336k ops = 56M ops per step.
    #           # If we have 2000 steps, that's 112B ops. Too slow.
            
    #   We need a faster check.
    #   Let's maintain the counts array `counts` for 1..MAX_CHECK.
    #   When we change residue for prime i from old_r to new_r:
    #       For all x in 1..MAX_CHECK:
    #           if x % p == old_r: counts[x] -= 1
    #           if x % p == new_r: counts[x] += 1
    #   This can be vectorized.
    #   x_mod_p[:, i] == old_r gives a boolean mask.
    #   counts[mask] -= 1
    #   counts[x_mod_p[:, i] == new_r] += 1
    
    #   Then we check if min(counts[1:current_L+1]) >= 2.
    #   If yes, accept the move.
    #   If no, revert.
    
    #   This is much faster.
    
    counts = np.zeros(MAX_CHECK + 1, dtype=np.int32) # 1-indexed
    # Initialize counts with residues = -1 (all 0)
    
    # Helper to apply a move and return success (True if frontier doesn't drop)
    def try_move(i, new_r):
        old_r = residues[i]
        if old_r == new_r:
            return True
        
        p_val = primes[i]
        mask_old = (x_mod_p[:, i] == old_r)
        mask_new = (x_mod_p[:, i] == new_r)
        
        # Update counts
        counts[mask_old] -= 1
        counts[mask_new] += 1
        
        # Check if 1..current_L are still double covered
        # Note: current_L is global
        if np.min(counts[1:current_L+1]) < 2:
            # Revert
            counts[mask_old] += 1
            counts[mask_new] -= 1
            return False
        
        # Accept move
        residues[i] = new_r
        return True
    
    # Main Loop
    while current_L < MAX_CHECK and time.time() < deadline:
        x_next = current_L + 1
        
        # Count how many primes currently cover x_next
        # This is counts[x_next]
        cur_count = counts[x_next]
        
        if cur_count >= 2:
            current_L += 1
            continue
        
        # Need to add (2 - cur_count) more covers
        # Let's try to add one at a time.
        # Find a prime i such that residues[i] != x_next % p_i
        # and changing it to x_next % p_i doesn't break the prefix.
        
        # To be efficient, we can iterate through primes.
        # Prefer primes that are "loose" in the prefix.
        # A prime i is "loose" at x if counts[x] > 2 when x % p_i == residues[i].
        
        # Let's just iterate through all primes and try the first one that works.
        # To make it deterministic and maybe better, we can sort primes by some heuristic?
        # Smaller primes have more frequent occurrences, so changing them affects more x's.
        # Larger primes affect fewer x's.
        # So it's safer to change larger primes first?
        # Let's iterate from largest prime to smallest.
        
        found = False
        for i in range(n_primes - 1, -1, -1):
            if residues[i] == x_next % primes[i]:
                continue # already covers
            
            new_r = x_next % primes[i]
            if try_move(i, new_r):
                found = True
                break
        
        if not found:
            # If no single move works, we are stuck.
            # This can happen if changing any prime to cover x_next breaks the double coverage of some x' < x_next.
            # In this case, we might need to change multiple primes simultaneously or use a more advanced repair.
            # For now, break.
            break
    
    # After the greedy phase, we might be stuck at some L.
    # We can try to improve by looking ahead or using local search on the current solution.
    # But the greedy phase is usually strong.
    
    # If we have time, we can try to extend further by allowing temporary drops in coverage?
    # No, the constraint is hard.
    
    # Let's check the final score
    final_score = compute_frontier()
    
    # If we didn't reach MAX_CHECK, we might have stopped early.
    # But the greedy logic should push it as far as possible.
    
    return [int(r) for r in residues]

if __name__ == "__main__":
    res = construct()
    print(len(res))
