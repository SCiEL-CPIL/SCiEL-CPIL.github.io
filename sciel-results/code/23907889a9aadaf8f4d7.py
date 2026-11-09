import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s/150s kill
    
    # Generate primes <= 1000
    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit ** 0.5) + 1):
            if sieve[i]:
                for j in range(i * i, limit + 1, i):
                    sieve[j] = False
        return [i for i, is_p in enumerate(sieve) if is_p]
    
    primes = get_primes(1000)
    n_primes = len(primes)
    primes_arr = np.array(primes, dtype=np.int64)
    
    # We evaluate up to a large L to allow for growth.
    # The task caps score at 10^5, but we can stop earlier if we hit a wall.
    # Let's set a high limit for the array to allow checking up to ~50k quickly.
    MAX_EVAL = 50000 
    xs = np.arange(1, MAX_EVAL + 1, dtype=np.int64)
    
    def compute_coverage(residues):
        """
        Vectorized coverage calculation.
        Returns an array of length MAX_EVAL where coverage[i] is the count
        of primes p_j such that (xs[i] - residues[j]) % p_j == 0.
        """
        # xs shape: (MAX_EVAL,)
        # residues shape: (n_primes,)
        # diff shape: (n_primes, MAX_EVAL)
        diff = xs[None, :] - residues[:, None]
        mods = np.remainder(diff, primes_arr[:, None])
        hits = (mods == 0)
        coverage = np.sum(hits, axis=0)
        return coverage

    def get_score(coverage):
        """
        Finds the largest L such that 1..L are doubly covered.
        """
        # coverage[0] corresponds to x=1
        # We need coverage[i] >= 2 for all i in [0, L-1]
        # Find first index where coverage < 2
        mask = coverage < 2
        if not np.any(mask):
            return MAX_EVAL
        first_fail_idx = np.argmax(mask)
        # If first_fail_idx is 0, L=0.
        # If first_fail_idx is k, then x=1..k are covered, x=k+1 is not.
        # So L = k.
        return int(first_fail_idx)

    # Initialization:
    # A good start is crucial for greedy. 
    # Let's try a "dense small" initialization.
    # For small x, we want many residues to match x % p.
    # If we set r_p = 1 for all p, x=1 is covered by all 168 primes.
    # x=2 is covered by primes p where 2 % p == r_p. If r_p=1, only p=1? No p=1 not prime.
    # So x=2 is covered by 0 primes if all r_p=1.
    # This is bad.
    
    # Better init: Random, but we will quickly fix it.
    # Or, use a heuristic: for each prime p, pick r_p to cover a specific small set.
    # Let's just start with all zeros and let the greedy repair handle it.
    # Actually, all zeros means x=0 is covered by all. x=1 is covered by none.
    # The first failure is x=1.
    # Repair x=1: pick a prime p, set r_p = 1 % p = 1.
    # Now x=1 is covered by 1 prime. Need 2.
    # Pick another prime q, set r_q = 1.
    # Now x=1 covered by 2.
    # Next failure x=2.
    # This sequential repair is exactly what we want to automate.
    
    residues = np.zeros(n_primes, dtype=np.int64)
    coverage = compute_coverage(residues)
    current_L = get_score(coverage)
    best_L = current_L
    best_residues = residues.copy()
    
    # Greedy Sequential Extension
    # We loop until we hit deadline or we can't extend L anymore (stuck).
    # To avoid getting stuck in a local minimum where no single swap improves L,
    # we might need to allow "neutral" moves or small regressions?
    # The prompt asks for a structural change. Pure greedy might get stuck early.
    # However, the "min-impact" heuristic is strong.
    # Let's add a "kick" mechanism: if stuck for N steps, try a random swap that
    # doesn't decrease L too much, or just accept a decrease if it opens up future possibilities?
    # Actually, if L is stuck, it means for x = L+1, any change to increase its coverage
    # breaks something in 1..L.
    # Let's implement the greedy step first.
    
    def try_extend_L(residues, coverage, current_L):
        """
        Attempts to increase current_L by 1.
        Returns (success, new_residues, new_coverage, new_L)
        """
        if current_L >= MAX_EVAL:
            return False, residues, coverage, current_L
            
        target_x = current_L + 1
        target_idx = target_x - 1
        
        # Current coverage of target_x
        c_target = coverage[target_idx]
        
        # We need c_target >= 2.
        # If c_target >= 2, then L should have been higher. 
        # So c_target must be 0 or 1.
        
        # Candidates: Primes that do NOT currently cover target_x.
        # Changing their residue to target_x % p will increase c_target by 1.
        # We need to pick one such prime i, and set residues[i] = target_x % p_i.
        # We must ensure that this change does NOT reduce coverage of any x in 1..current_L to below 2.
        
        # For each candidate prime i:
        # 1. Calculate new_r = target_x % p_i
        # 2. Check if old_r was covering some x in 1..current_L.
        #    The set of x in 1..current_L covered by prime i with old_r is:
        #    x such that x % p_i == old_r.
        #    Let S_old = {x in 1..current_L | x % p_i == old_r}
        # 3. Check if new_r covers some x in 1..current_L.
        #    S_new = {x in 1..current_L | x % p_i == new_r}
        # 4. The net change in coverage for any x is:
        #    +1 if x in S_new and x not in S_old
        #    -1 if x in S_old and x not in S_new
        #    0 otherwise
        # 5. We require that for all x in S_old \ S_new, coverage[x] - 1 >= 2 => coverage[x] >= 3.
        #    (Because we remove one cover).
        #    And for x in S_new \ S_old, coverage[x] + 1 >= 2 (always true).
        
        # Efficient check:
        # We can precompute the list of x in 1..current_L for each residue class mod p_i?
        # That's expensive to do for every step.
        
        # Alternative: Just simulate the change on the coverage array for the prefix 1..current_L.
        # But doing this for all 168 primes every step is O(168 * current_L).
        # If current_L is 1000, that's 168k ops. Fast.
        # If current_L is 10000, that's 1.68M ops. Still fast in NumPy if vectorized per prime?
        # Or vectorized across primes?
        
        # Let's try a vectorized approach for candidate evaluation.
        # For each prime i, we want to know:
        # Does changing r_i to new_r_i break any x in 1..L?
        # Break condition: exists x in 1..L such that x % p_i == old_r_i AND coverage[x] == 2.
        # (If coverage[x] > 2, removing one cover is safe).
        
        # So, for each prime i, let's find the set of x in 1..L that are "critically covered" by i.
        # Critically covered by i means:
        # 1. x % p_i == residues[i]
        # 2. coverage[x] == 2
        
        # If this set is empty, then changing residues[i] is SAFE for the prefix 1..L.
        # If it's not empty, we need to check if the new residue covers those x's?
        # No, if we change r_i, we lose coverage for those x's.
        # Unless new_r_i also covers them? 
        # x % p_i == old_r_i and x % p_i == new_r_i implies old_r_i == new_r_i.
        # But we only consider candidates where new_r_i != old_r_i (since they don't cover target_x, 
        # and if they did, they would cover target_x, but we established they don't).
        # Wait, if old_r_i covered target_x, then i is not a candidate.
        # So for candidates, new_r_i != old_r_i.
        # Therefore, S_old and S_new are disjoint (for a given p_i).
        # So if there is ANY x in 1..L such that x % p_i == old_r_i and coverage[x] == 2,
        # then changing i will break x.
        
        # So the condition for a safe move is:
        # For prime i, the set {x in 1..L | x % p_i == residues[i] AND coverage[x] == 2} is empty.
        
        # Let's precompute this for all primes in one go?
        # We have `coverage` array of length MAX_EVAL.
        # We only care about 1..current_L.
        # Let `critical_mask` be a boolean array of length current_L where critical_mask[j] is True
        # if coverage[j] == 2. (j is 0-indexed, x = j+1).
        
        # For each prime i, we want to check if any x in 1..L satisfies:
        # (x % p_i == residues[i]) AND critical_mask[x-1]
        
        # We can do this by iterating through primes? 168 iterations.
        # Inside, we check the subset of xs.
        
        # Optimization:
        # Precompute for the current state:
        # `is_crit` = (coverage[:current_L] == 2)
        
        # For each prime i:
        #   x_vals = np.arange(1, current_L + 1)
        #   mods = x_vals % p_i
        #   matches = (mods == residues[i])
        #   bad = np.any(matches & is_crit)
        #   if not bad: candidate!
        
        # This is O(N_primes * L).
        # If L=10000, 168 * 10000 = 1.68M operations.
        # In Python loop over primes, this might be slow (168 loops).
        # We can vectorize the "bad" check?
        
        # Let's try to vectorize the candidate search.
        # Create a matrix `M` of shape (n_primes, current_L).
        # M[i, j] = ( (j+1) % p_i == residues[i] ) AND (coverage[j] == 2)
        # If any M[i, :] is True, then prime i is "blocked".
        # We want primes where M[i, :] is all False.
        
        # Constructing M:
        # x_vals = np.arange(1, current_L + 1)
        # mods_matrix = x_vals[None, :] % primes_arr[:, None]  # Shape (168, L)
        # match_matrix = (mods_matrix == residues[:, None])
        # crit_row = (coverage[:current_L] == 2)
        # blocked_matrix = match_matrix & crit_row[None, :]
        # is_blocked = np.any(blocked_matrix, axis=1)
        
        # Memory: 168 * L booleans. If L=10000, 1.68MB. Fine.
        # Time: Modulo on 1.68M elements. Fast.
        
        x_vals = np.arange(1, current_L + 1, dtype=np.int64)
        # Only compute if L > 0
        if current_L > 0:
            mods_matrix = x_vals[None, :] % primes_arr[:, None]
            match_matrix = (mods_matrix == residues[:, None])
            crit_row = (coverage[:current_L] == 2)
            blocked_matrix = match_matrix & crit_row[None, :]
            is_blocked = np.any(blocked_matrix, axis=1)
        else:
            is_blocked = np.zeros(n_primes, dtype=bool)
            
        # Now, find candidates:
        # 1. Not covering target_x: (target_x - residues) % p_i != 0
        # 2. Not blocked: ~is_blocked
        
        diff_target = target_x - residues
        mods_target = np.remainder(diff_target, primes_arr)
        covers_target = (mods_target == 0)
        
        candidates = np.where((~covers_target) & (~is_blocked))[0]
        
        if len(candidates) == 0:
            return False, residues, coverage, current_L
            
        # Pick the first candidate (or random? First is deterministic, good for reproducibility)
        # Actually, let's pick the one that maximizes future potential?
        # For now, just pick the first one.
        i = candidates[0]
        new_r = target_x % primes_arr[i]
        
        # Apply move
        old_r = residues[i]
        residues[i] = new_r
        
        # Update coverage efficiently?
        # We only changed one prime.
        # We can update the coverage array incrementally.
        # But for simplicity and correctness, let's just recompute the relevant part or full?
        # Full recompute is O(168 * MAX_EVAL).
        # MAX_EVAL is 50000. 168 * 50000 = 8.4M ops.
        # This is ~1-2ms in NumPy.
        # If we do 1000 steps, that's 1-2 seconds.
        # We have 115 seconds. So we can do ~50,000 steps.
        # This is plenty to reach L=1000 or more.
        
        coverage = compute_coverage(residues)
        new_L = get_score(coverage)
        
        if new_L > current_L:
            return True, residues, coverage, new_L
        else:
            # Should not happen if logic is correct, but revert just in case
            residues[i] = old_r
            return False, residues, coverage, current_L

    # Main Loop
    stuck_count = 0
    max_stuck = 10000 # If we are stuck for this many tries, we might need a different strategy
    
    while time.time() < deadline:
        if current_L >= MAX_EVAL:
            break
            
        success, residues, coverage, current_L = try_extend_L(residues, coverage, current_L)
        
        if success:
            stuck_count = 0
            if current_L > best_L:
                best_L = current_L
                best_residues = residues.copy()
        else:
            stuck_count += 1
            # If stuck, we are in a local minimum.
            # The greedy "safe move" didn't find a way to extend L.
            # This means for x=L+1, any prime that could cover it (by changing residue)
            # is currently "blocked" because it critically covers some x in 1..L.
            
            # Strategy to escape:
            # 1. Try to "shift" a blocked prime to a different residue that is safe,
            #    even if it doesn't cover L+1 immediately, hoping to unblock it?
            #    No, that doesn't help L+1 directly.
            # 2. Accept a "bad" move that decreases L by 1 or 2, but allows a larger jump later?
            #    This is hill-climbing with restarts/kicks.
            
            # Simple Kick: Pick a random prime that is NOT blocked (safe to change),
            # and change its residue to a random value.
            # This might break nothing (if safe) but won't increase L.
            # It might change the "blocked" status of other primes?
            # Changing a safe prime doesn't change coverage of any x in 1..L to <2.
            # It might increase coverage of some x to 3, which unblocks other primes!
            # This is a powerful insight.
            
            # If we increase coverage of a critically covered x (cov=2) to 3,
            # then the prime covering it is no longer "blocked".
            
            # So, if we are stuck:
            # Find a prime i that is SAFE (not blocked).
            # Change its residue to cover some x in 1..L that is currently cov=2.
            # This increases cov[x] to 3.
            # Now, the prime j that was covering x (and was blocked because cov[x]=2)
            # is now unblocked!
            # Then we can try to change j to cover L+1.
            
            # Implementation of Kick:
            # 1. Find safe primes: ~is_blocked
            # 2. If any safe primes exist:
            #    a. Pick a random safe prime i.
            #    b. Find an x in 1..L where coverage[x] == 2.
            #    c. Set residues[i] = x % p_i.
            #    d. This increases coverage[x] to 3.
            #    e. Recompute coverage and blocked status.
            #    f. This doesn't change L (since all 1..L still cov>=2).
            #    g. But it might unblock primes.
            # 3. If no safe primes exist (all primes are blocked):
            #    This means every prime critically covers some x in 1..L.
            #    This is a very tight constraint.
            #    We might need to accept a temporary decrease in L.
            #    Let's just do a random move and accept the decrease.
            
            if stuck_count % 10 == 0: # Do kick every 10 failed attempts
                if current_L > 0:
                    # Recompute is_blocked for current state (might have changed if we did kicks)
                    x_vals_kick = np.arange(1, current_L + 1, dtype=np.int64)
                    mods_kick = x_vals_kick[None, :] % primes_arr[:, None]
                    match_kick = (mods_kick == residues[:, None])
                    crit_kick = (coverage[:current_L] == 2)
                    blocked_kick = match_kick & crit_kick[None, :]
                    is_blocked_kick = np.any(blocked_kick, axis=1)
                    
                    safe_primes = np.where(~is_blocked_kick)[0]
                    
                    if len(safe_primes) > 0:
                        # Pick a random safe prime
                        i = safe_primes[np.random.randint(len(safe_primes))]
                        # Find a critical x
                        crit_indices = np.where(crit_kick)[0]
                        if len(crit_indices) > 0:
                            x_idx = crit_indices[np.random.randint(len(crit_indices))]
                            x_val = x_idx + 1
                            new_r = x_val % primes_arr[i]
                            # Apply
                            old_r = residues[i]
                            residues[i] = new_r
                            coverage = compute_coverage(residues)
                            # L should be same
                            # Do not update best_L
                        else:
                            # No critical x? All cov >= 3?
                            # Then why are we stuck?
                            # If all cov >= 3, then no prime is blocked.
                            # So safe_primes should be all primes.
                            # We can pick any prime and change it to cover L+1?
                            # Wait, if all cov >= 3, then is_blocked is all False.
                            # So in try_extend_L, candidates would be all primes not covering L+1.
                            # So we wouldn't be stuck.
                            # Contradiction.
                            pass
                    else:
                        # All primes blocked.
                        # Random kick: change a random prime to random residue.
                        i = np.random.randint(n_primes)
                        old_r = residues[i]
                        residues[i] = np.random.randint(primes_arr[i])
                        coverage = compute_coverage(residues)
                        new_L = get_score(coverage)
                        if new_L > current_L:
                            current_L = new_L
                            if current_L > best_L:
                                best_L = current_L
                                best_residues = residues.copy()
                        else:
                            # Accept the decrease? Or revert?
                            # If we revert, we are stuck in a loop.
                            # Let's accept the decrease to explore.
                            current_L = new_L
                            # Don't update best_L if it decreased
                            
                # If current_L == 0, just random move
                else:
                    i = np.random.randint(n_primes)
                    residues[i] = np.random.randint(primes_arr[i])
                    coverage = compute_coverage(residues)
                    current_L = get_score(coverage)
                    if current_L > best_L:
                        best_L = current_L
                        best_residues = residues.copy()

    return best_residues.tolist()
