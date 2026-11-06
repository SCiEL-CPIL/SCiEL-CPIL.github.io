"""
Strategy: Deterministic Greedy Set-Cover with Conflict Resolution.
Instead of random local search, we construct the solution by iteratively
identifying the first uncovered (or singly covered) position in the prefix
and assigning residues to primes to cover it, prioritizing assignments
that minimize future conflicts (interference with other positions).
"""
import time
import numpy as np

def _get_primes(limit):
    """Sieve of Eratosthenes to get primes up to limit."""
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    return [i for i in range(2, limit + 1) if is_prime[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s kill
    
    primes = _get_primes(1000)
    n_primes = len(primes)
    
    # We aim for L >= 1000. 
    # To be safe and allow for "repair" of the tail, we monitor up to 2000.
    max_check = 2000
    
    # residues[i] is the residue for primes[i]
    residues = np.zeros(n_primes, dtype=np.int64)
    
    # counts[x] = number of primes p such that x == residues[idx] (mod p)
    counts = np.zeros(max_check + 1, dtype=np.int32)
    
    # Precompute indices for each prime's residue positions in [1, max_check]
    # pos_map[i] = array of x in [1, max_check] such that x % primes[i] == residues[i]
    # Note: This changes as residues change, so we can't fully precompute static arrays.
    # Instead, we compute on the fly but efficiently.
    
    def update_counts(idx, old_r, new_r, delta):
        """
        Update counts array.
        delta = +1 means adding the residue, -1 means removing.
        """
        p = primes[idx]
        r = new_r if delta == 1 else old_r
        
        # Find first occurrence of r in [1, max_check]
        # If r == 0, the numbers are p, 2p, ...
        # If r > 0, the numbers are r, r+p, ... (if r >= 1)
        if r == 0:
            start = p
        else:
            start = r
            
        if start <= max_check:
            counts[start::p] += delta

    def get_prefix_score():
        """
        Find the largest L such that counts[1..L] >= 2.
        """
        # We want the first index in 1..max_check where count < 2
        # Using np.where is fast
        below_two = np.where(counts[1:max_check+1] < 2)[0]
        if len(below_two) == 0:
            return max_check
        return below_two[0]

    # Initialize counts
    for i in range(n_primes):
        update_counts(i, 0, 0, 1)
        
    best_score = get_prefix_score()
    best_residues = residues.copy()

    # --- Phase 1: Greedy Construction ---
    # Goal: Extend the prefix as far as possible.
    # We iterate: find the first gap (count < 2), try to fill it.
    
    # We maintain a list of "active" primes that haven't been "locked in" or 
    # we just iterate through all primes to find the best candidate to fix the gap.
    
    # To speed up, we track the current prefix score.
    current_score = best_score
    
    # Optimization: We will try to fix the gap at `current_score + 1`.
    # If we can't fix it, we try to fix `current_score + 2`, etc.
    # Actually, the prefix score is defined by the FIRST failure.
    # So we must fix `current_score + 1`.
    
    # We will perform a limited number of "repair" iterations.
    # In each iteration, we look at the first gap x = current_score + 1.
    # We try all primes p that do NOT currently cover x (or even those that do, 
    # if we can swap to a better position? No, we just want to ADD coverage to x).
    # So we need primes p where residues[i] != x % p.
    # Changing residues[i] to x % p will add coverage to x.
    # BUT it will REMOVE coverage from the old position y = residues[i] (and y+p, ...).
    # This removal might break the prefix if any y in [1, current_score] had count 2.
    
    # So, for a prime i to be a valid candidate to cover x:
    # 1. new_r = x % p
    # 2. old_r = residues[i]
    # 3. If old_r == new_r, skip.
    # 4. Check if removing coverage from old_r breaks any z in [1, current_score] where count[z] == 2.
    #    The positions affected are z such that z % p == old_r and z <= current_score.
    
    max_repair_steps = 5000
    step = 0
    
    while time.time() < deadline - 10.0 and step < max_repair_steps:
        step += 1
        
        current_score = get_prefix_score()
        if current_score > best_score:
            best_score = current_score
            best_residues = residues.copy()
            
        if current_score >= max_check:
            break
            
        x_target = current_score + 1
        
        # Find a prime to cover x_target
        # We prefer primes with larger p? Or smaller?
        # Larger p means fewer positions affected by the "removal" part, so less risk of breaking prefix.
        # Smaller p means more frequent coverage, potentially better for global density.
        # Let's iterate primes in descending order of p (largest first) to minimize conflict risk.
        # This is a structural choice: "Large primes are safer to move".
        
        found = False
        # Sort indices by prime value descending
        # primes is already sorted ascending. So iterate backwards.
        for i in range(n_primes - 1, -1, -1):
            p = primes[i]
            old_r = residues[i]
            new_r = x_target % p
            
            if old_r == new_r:
                continue
                
            # Check validity: Does changing old_r to new_r break any prefix point?
            # Affected positions (losing coverage): x in [1, current_score] where x % p == old_r
            # If any such x has count[x] == 2, it becomes 1, breaking prefix.
            
            # Calculate first position in [1, current_score] with residue old_r mod p
            if old_r == 0:
                start_pos = p
            else:
                start_pos = old_r
                
            if start_pos > current_score:
                # No positions in prefix are affected by removal
                valid = True
            else:
                # Check counts at these positions
                positions = np.arange(start_pos, current_score + 1, p)
                if np.any(counts[positions] == 2):
                    valid = False
                else:
                    valid = True
            
            if valid:
                # Apply change
                update_counts(i, old_r, new_r, -1)
                residues[i] = new_r
                update_counts(i, new_r, new_r, 1)
                found = True
                break
        
        if not found:
            # Stuck! No single prime move can fix the gap without breaking the prefix.
            # This is a "local minimum" for the prefix score.
            # Strategy: "Sacrificial Move" or "Random Kick"
            # We must break the current structure to escape.
            # Option A: Randomly change a residue of a large prime to a random value, 
            #           accepting a temporary drop in score, hoping to find a better path.
            # Option B: Try to fix the gap by changing a prime that DOES break the prefix,
            #           but then immediately try to repair the broken part. (Complex)
            
            # Let's try a targeted "kick":
            # Pick a prime that is NOT critical for the current prefix (i.e., its removal 
            # doesn't break anything, or it's already not covering the gap).
            # Actually, if we are stuck, it means for ALL primes, either:
            # 1. They already cover x_target (new_r == old_r)
            # 2. Changing them to cover x_target breaks the prefix.
            
            # If 1 is true for many primes, we are in trouble.
            # Let's try to "shuffle" a prime that has high redundancy in the prefix.
            # Find a prime i such that changing its residue doesn't break the prefix, 
            # but moves it to a random spot. This might open up a slot for another prime.
            
            kicked = False
            # Try random large primes
            for _ in range(10):
                i = np.random.randint(n_primes // 2, n_primes) # Large primes
                p = primes[i]
                old_r = residues[i]
                new_r = np.random.randint(p)
                
                if new_r == old_r:
                    continue
                    
                # Check if this random move breaks the prefix
                start_pos = old_r if old_r > 0 else p
                valid = True
                if start_pos <= current_score:
                    positions = np.arange(start_pos, current_score + 1, p)
                    if np.any(counts[positions] == 2):
                        valid = False
                
                if valid:
                    update_counts(i, old_r, new_r, -1)
                    residues[i] = new_r
                    update_counts(i, new_r, new_r, 1)
                    kicked = True
                    break
            
            if not kicked:
                # If we can't even do a safe random kick, we are severely stuck.
                # Try a "dangerous" kick: pick a prime, change it randomly, 
                # and see if we can repair. If score drops too much, revert.
                # For now, if we can't kick safely, we might as well break the loop 
                # and rely on the best found so far.
                break

    # --- Phase 2: Fine-tuning / Global Improvement ---
    # If we have time, try to improve the score further by looking beyond the immediate gap.
    # Or, try to increase the "margin" (min count in prefix) to make the structure more robust.
    
    # Let's try a simple hill-climbing on the prefix score with more aggressive moves.
    # If we are stuck at L, maybe we can extend to L+2 by fixing L+1 and L+2 simultaneously?
    # That's hard. 
    
    # Instead, let's try to "push" the prefix further by checking if we can cover 
    # the gap at current_score + 1 using a combination of moves?
    # No, the greedy approach is single-step.
    
    # Let's just run a few iterations of the same greedy loop, 
    # but this time allowing "unsafe" moves if they result in a net gain after a repair step.
    # This is getting complex. Given the time constraint, the greedy construction 
    # with large-prime priority is likely the most effective structural change.
    
    # Final check
    final_score = get_prefix_score()
    if final_score > best_score:
        best_score = final_score
        best_residues = residues.copy()

    return [int(r) for r in best_residues]
