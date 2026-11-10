import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Generate primes up to 1000
    def get_primes(n):
        sieve = np.ones(n + 1, dtype=bool)
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                sieve[i*i:n+1:i] = False
        return np.array([i for i, is_prime in enumerate(sieve) if is_prime], dtype=np.int64)
    
    primes = get_primes(1000)
    n_primes = len(primes)
    
    # We will work with a window of size W to compute scores efficiently.
    # The goal is L >= 1000. Let's use W = 2000 to be safe for intermediate steps.
    # If we exceed W, we cap the score at W for optimization, but final verification can be larger.
    # However, the task asks for the exact L. 
    # To be safe and fast, we optimize on W=2000. If we reach 2000, we can try to extend or just return.
    # Given the difficulty, reaching 1000 is the target.
    W = 2000
    x_vals = np.arange(1, W + 1, dtype=np.int64)
    
    # Initialize residues to 0
    residues = np.zeros(n_primes, dtype=np.int64)
    
    # Precompute masks for each prime and each residue? 
    # 168 primes * avg 500 residues * 2000 length = 168M booleans. 168MB. 
    # This is feasible and makes local search extremely fast.
    # masks[i, r, :] is a boolean array of length W indicating where x % p_i == r.
    
    # Let's check memory: 168 * 1000 * 2000 * 1 byte = 336 MB. 
    # This might be too high for some environments, but usually okay.
    # Alternatively, we can compute masks on the fly for the specific prime being changed.
    # On-the-fly: (x_vals - r) % p == 0. This is O(W).
    # If we do this for every candidate r in local search, it's O(p * W) per prime update.
    # Sum p_i ~ 76000. 76000 * 2000 = 152M ops per full pass.
    # In NumPy, this is fast (~0.1s per pass).
    # Let's stick to on-the-fly to avoid memory issues, but optimize the inner loop.
    
    def compute_counts(res_vec):
        counts = np.zeros(W, dtype=np.int32)
        for i in range(n_primes):
            p = primes[i]
            r = res_vec[i]
            counts += ((x_vals - r) % p == 0).astype(np.int32)
        return counts

    def get_score_from_counts(counts):
        # Find first x where count < 2
        fail = counts < 2
        if not np.any(fail):
            return W
        first_fail_idx = np.argmax(fail)
        return x_vals[first_fail_idx] - 1

    # Phase 1: Greedy Construction
    # Process primes in increasing order.
    # For each prime i, choose r_i that maximizes the resulting prefix length,
    # given the previous choices.
    
    # To make this efficient, we maintain `counts` incrementally.
    counts = np.zeros(W, dtype=np.int32)
    current_score = 0
    
    for i in range(n_primes):
        if time.time() > deadline:
            break
            
        p = primes[i]
        
        # Try all residues for this prime
        best_r = 0
        best_local_score = -1
        
        # We can vectorize the score calculation for all r?
        # For each r, new_counts = counts + mask_r.
        # We want max L such that min(new_counts[0:L]) >= 2.
        
        # Let's just loop r, it's fast enough for 168 primes * 1000 residues.
        # 168,000 iterations. Each O(W) = 2000 ops. 336M ops total.
        # In Python loop, this might be slow.
        # Let's vectorize over r for a fixed i.
        
        # masks shape: (p, W)
        # We can compute masks for all r in 0..p-1 at once.
        # r_vals = np.arange(p)
        # diff = x_vals[None, :] - r_vals[:, None]
        # masks = (diff % p == 0).astype(np.int32)
        
        # new_counts = counts[None, :] + masks  # shape (p, W)
        # For each row, find first index where value < 2.
        
        try:
            r_vals = np.arange(p, dtype=np.int64)
            # Compute masks
            # To save memory, we can do it in chunks if p is large, but p<=1000 is fine.
            diff = x_vals[None, :] - r_vals[:, None]
            masks = (diff % p == 0).astype(np.int32)
            
            new_counts = counts[None, :] + masks
            
            # Find first failure for each candidate
            # fail = new_counts < 2
            # We want the first True index for each row.
            # If no True, score is W.
            
            fail = new_counts < 2
            
            # np.argmax returns 0 if no True is found. 
            # We need to distinguish between "first fail at 0" and "no fail".
            # Let's use a trick:
            # If any(fail), then first_fail = argmax(fail). Else W.
            
            has_fail = np.any(fail, axis=1)
            first_fail_indices = np.argmax(fail, axis=1)
            
            # Scores are x_vals[first_fail_indices] - 1
            # If has_fail is False, score is W.
            scores = np.where(has_fail, x_vals[first_fail_indices] - 1, W)
            
            best_idx = np.argmax(scores)
            best_r = r_vals[best_idx]
            best_local_score = scores[best_idx]
            
        except Exception:
            # Fallback to loop if vectorization fails (unlikely)
            for r in range(p):
                mask = ((x_vals - r) % p == 0).astype(np.int32)
                nc = counts + mask
                s = get_score_from_counts(nc)
                if s > best_local_score:
                    best_local_score = s
                    best_r = r
        
        # Apply best choice
        residues[i] = best_r
        mask = ((x_vals - best_r) % p == 0).astype(np.int32)
        counts += mask
        current_score = best_local_score
        
        # Early stopping if we've covered the whole window?
        if current_score >= W:
            # We can break, but maybe we can do better? 
            # The window is capped. If we reach W, we've achieved at least W.
            # Since target is 1000 and W=2000, we are good.
            # But we should still fill the rest of the residues with something valid (e.g., 0) 
            # or continue to see if we can push it further? 
            # The score is capped at W in our internal logic. 
            # The actual score might be higher. 
            # Let's just fill the rest with 0 and break to save time for local search?
            # Or continue greedy to see if we can maintain W?
            # If we break, we need to set remaining residues.
            # Let's just set remaining to 0 and break out of greedy.
            for j in range(i + 1, n_primes):
                residues[j] = 0
            break

    # Phase 2: Local Search / Repair
    # Now that we have a good initial configuration, we try to improve it.
    # The greedy method is myopic. Local search can fix issues.
    
    # We will iterate through primes and try to change their residue to improve the score.
    # To be efficient, we only try to change a prime if it might help.
    # A prime helps if changing it increases the count at the first failure point or earlier.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n_primes):
            if time.time() > deadline:
                break
                
            p = primes[i]
            r_old = residues[i]
            
            # Remove old coverage
            old_mask = ((x_vals - r_old) % p == 0).astype(np.int32)
            counts -= old_mask
            
            best_r = r_old
            best_local_score = get_score_from_counts(counts) # Score without this prime's contribution
            
            # Try all other residues
            # We can vectorize this as in Phase 1.
            try:
                r_vals = np.arange(p, dtype=np.int64)
                diff = x_vals[None, :] - r_vals[:, None]
                masks = (diff % p == 0).astype(np.int32)
                
                new_counts = counts[None, :] + masks
                fail = new_counts < 2
                has_fail = np.any(fail, axis=1)
                first_fail_indices = np.argmax(fail, axis=1)
                scores = np.where(has_fail, x_vals[first_fail_indices] - 1, W)
                
                best_idx = np.argmax(scores)
                best_r = r_vals[best_idx]
                best_local_score = scores[best_idx]
                
            except Exception:
                for r in range(p):
                    if r == r_old: continue
                    mask = ((x_vals - r) % p == 0).astype(np.int32)
                    nc = counts + mask
                    s = get_score_from_counts(nc)
                    if s > best_local_score:
                        best_local_score = s
                        best_r = r

            # Restore best
            new_mask = ((x_vals - best_r) % p == 0).astype(np.int32)
            counts += new_mask
            residues[i] = best_r
            
            # Check if global score improved
            # Note: best_local_score is the score *with* this prime's best residue.
            # We need to compare it to the current global score?
            # Actually, `counts` now reflects the new state.
            # Let's compute the global score from `counts`.
            current_global_score = get_score_from_counts(counts)
            
            if current_global_score > best_local_score:
                # This can happen if other primes' changes (none in this loop iteration) 
                # Wait, we are changing one prime at a time.
                # The score of the system after changing prime i to best_r is best_local_score.
                # So if best_local_score > previous_global_score, we improved.
                pass # We don't track previous_global_score explicitly in this loop structure easily.
                # Let's just track if we found a better residue for this prime than the old one.
                if best_r != r_old:
                    improved = True
                # Or if the score increased.
                # Since we don't store prev_score, let's assume if we changed residue, it's worth noting.
                # Actually, we should store the score before the change.
                
            # Let's refine the improvement check.
            # We removed r_old. Let's calculate score with r_old to compare?
            # No, we already have best_local_score (max over all r).
            # If best_r != r_old, it means some other r gave a better score than r_old 
            # in the context of the other fixed primes.
            # So if best_r != r_old, we improved (or tied, but changed).
            # To be safe, only mark improved if we actually changed the residue.
            if best_r != r_old:
                improved = True

    return [int(r) for r in residues]

# Keep the function name as required
# The task says "Keep the function names and signatures of the seed program."
# Seed program has `def construct():`
# So I just need to output the code block with `construct`.
