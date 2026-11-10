import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit

    def get_primes(n):
        sieve = bytearray([1]) * (n + 1)
        sieve[0] = sieve[1] = 0
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
        return [i for i in range(2, n+1) if sieve[i]]

    primes = get_primes(1000)
    n_primes = len(primes)
    p_list = np.array(primes, dtype=np.int32)
    
    # Working limit for fast evaluation during optimization
    # We want L >= 1000, but to be safe and maximize, we look at a larger window.
    # However, the score is capped at 10^5, but we likely won't reach that.
    # Let's use a limit that is large enough to capture the "true" boundary if it's near 1000-5000.
    # If the boundary is < LIMIT, we find it exactly. If >= LIMIT, we know it's at least LIMIT.
    # For optimization, a smaller LIMIT is faster. Let's start with 5000.
    # We can increase it if we get stuck.
    OPT_LIMIT = 5000
    EVAL_LIMIT = 100000

    # --- Phase 1: Greedy Initialization ---
    # Assign residues to maximize immediate coverage of singly-covered points.
    coverage = np.zeros(OPT_LIMIT + 1, dtype=np.int16)
    residues = np.zeros(n_primes, dtype=np.int32)
    
    for i, p in enumerate(primes):
        if time.time() > deadline - 30:
            break
            
        # Calculate gain for each residue r in 0..p-1
        # gain[r] = number of x in [1, OPT_LIMIT] with x % p == r and coverage[x] == 1
        # We can compute this by looking at where coverage is 1 and binning by x % p
        # But doing this for every prime is O(N * LIMIT).
        # Faster: 
        # For each r, the points are r, r+p, r+2p...
        # We want to count how many of these have coverage == 1.
        
        # Vectorized approach for greedy:
        # Create a boolean mask of where coverage == 1
        # For a specific p, we can't easily vectorize across all r without a large intermediate array.
        # Given p is small (max 1000) and LIMIT is 5000, a simple loop over r is 1000 * (5000/p) ops.
        # Actually, simpler:
        # points_with_cov_1 = np.where(coverage == 1)[0]
        # residues_of_points = points_with_cov_1 % p
        # counts = np.bincount(residues_of_points, minlength=p)
        
        cov_1_indices = np.where(coverage[1:OPT_LIMIT+1] == 1)[0] + 1
        if len(cov_1_indices) > 0:
            rem = cov_1_indices % p
            counts = np.bincount(rem, minlength=p)
            best_r = int(np.argmax(counts))
        else:
            # No singly covered points, pick random or 0
            best_r = 0
            
        residues[i] = best_r
        
        # Update coverage
        r = best_r
        if r == 0:
            start = p
        else:
            start = r
        if start <= OPT_LIMIT:
            coverage[start::p] += 1

    # --- Phase 2: Vectorized Coordinate Descent ---
    # We will iterate through primes, and for each prime, try all possible residues.
    # We accept the residue that maximizes the "doubly covered prefix" L.
    
    def get_score(coverage_arr):
        """
        Returns (L, total_cov_in_prefix) where L is the largest integer such that
        coverage[1..L] >= 2.
        """
        # Find first index where coverage < 2
        # Note: coverage[0] is unused. We care about 1..LIMIT
        bad = np.where(coverage_arr[1:OPT_LIMIT+1] < 2)[0]
        if len(bad) == 0:
            L = OPT_LIMIT
        else:
            L = int(bad[0])
        
        # Secondary metric: sum of coverage in 1..L (higher is better for robustness)
        # This helps break ties and push towards "thicker" coverage
        if L > 0:
            sec_score = np.sum(coverage_arr[1:L+1])
        else:
            sec_score = 0
            
        return L, sec_score

    current_L, current_sec = get_score(coverage)
    
    # Optimization Loop
    # We perform multiple sweeps. In each sweep, we visit each prime once.
    # For each prime, we evaluate all p residues.
    
    # Precompute indices for speed?
    # For prime p, residue r covers indices: r, r+p, ...
    # We need to update coverage quickly.
    
    num_sweeps = 50
    for sweep in range(num_sweeps):
        if time.time() > deadline - 20:
            break
            
        improved = False
        
        for i in range(n_primes):
            p = int(p_list[i])
            old_r = int(residues[i])
            
            # We want to find the new_r in [0, p-1] that maximizes score.
            # Instead of trying all p residues one by one (which is slow if p is large),
            # we can vectorize the evaluation if p is small.
            # Max p is 997. Evaluating 997 candidates * 5000 limit = 5M ops per prime.
            # 168 primes * 5M = 840M ops per sweep. This might be too slow in Python/NumPy overhead.
            
            # Alternative: Only try a subset of residues?
            # Or, optimize the evaluation.
            
            # Let's try a "smart" local search first:
            # 1. Try the current residue (baseline)
            # 2. Try residues that cover the "first bad point" (current_L + 1).
            
            if current_L >= OPT_LIMIT:
                # We've hit the limit, try to push further? 
                # For now, if we hit OPT_LIMIT, we are in a good state. 
                # We might increase OPT_LIMIT later if time permits.
                pass
            
            # The bottleneck is finding the first bad point for EVERY candidate residue.
            # Can we do better?
            
            # Let's stick to a simpler, faster local search strategy within the vectorized framework:
            # For each prime, try:
            # 1. The residue that covers current_L + 1 (if any).
            # 2. The residue that maximizes coverage of the "frontier" [current_L - p, current_L + p].
            
            # Actually, let's just do the full vectorized evaluation but with a smaller OPT_LIMIT 
            # or fewer candidates if it's too slow.
            
            # Let's estimate speed:
            # For p=997, candidates = 997.
            # For each candidate, we need to compute coverage change and find min.
            # This is O(p * LIMIT) if done naively.
            
            # Better approach:
            # We maintain coverage.
            # For a given p, the set of points covered by residue r is S_r = {x in [1, LIMIT] | x % p == r}.
            # Changing residue from old_r to new_r means:
            # coverage[S_old] -= 1
            # coverage[S_new] += 1
            
            # We want to maximize L.
            # L is determined by the first x where coverage[x] < 2.
            
            # Let's try a "Greedy Local Search" that is vectorized across candidates:
            # We can't easily vectorize the "find first bad" step across candidates without storing
            # the full coverage array for each candidate (memory heavy: 1000 * 5000 * 2 bytes = 10MB, ok).
            
            # Let's try it for small p. For large p, p is large, so the step size is large.
            # The number of points covered by each residue is LIMIT/p.
            
            # Strategy:
            # If p < 100: Try all residues.
            # If p >= 100: Try only residues that cover current_L+1 or current_L+2, plus a few randoms.
            
            if p < 100:
                # Vectorized evaluation of all residues
                # This is expensive but p is small.
                # Actually, even for p=97, 97 * 5000 is ~500k ops.
                # 168 primes * 500k = 84M ops. In NumPy, this is fast (<1s per sweep).
                
                # We will simulate the change for each r.
                # But we can't easily do "if coverage[r] < 2" for all r at once without a loop or broadcasting.
                # Broadcasting: coverage shape (LIMIT,), candidates shape (p,).
                
                # Let's just loop over candidates for now, but use NumPy for the inner "find first bad"
                # This is effectively the same as the previous program's local search but more exhaustive.
                
                best_new_r = old_r
                best_L = current_L
                best_sec = current_sec
                
                # To speed up, we only need to check if the new L is > best_L.
                # We can compute the new coverage for each r?
                # No, that's O(p * LIMIT) memory/time if we store arrays.
                
                # Let's just iterate.
                for new_r in range(p):
                    if new_r == old_r:
                        continue
                    
                    # Apply change
                    if old_r == 0:
                        old_start = p
                    else:
                        old_start = old_r
                    if new_r == 0:
                        new_start = p
                    else:
                        new_start = new_r
                    
                    if old_start <= OPT_LIMIT:
                        coverage[old_start::p] -= 1
                    if new_start <= OPT_LIMIT:
                        coverage[new_start::p] += 1
                    
                    new_L, new_sec = get_score(coverage)
                    
                    if new_L > best_L or (new_L == best_L and new_sec > best_sec):
                        best_L = new_L
                        best_sec = new_sec
                        best_new_r = new_r
                        # Don't revert yet, keep this state
                    else:
                        # Revert
                        if new_start <= OPT_LIMIT:
                            coverage[new_start::p] -= 1
                        if old_start <= OPT_LIMIT:
                            coverage[old_start::p] += 1
                
                if best_new_r != old_r:
                    # Apply the best change found
                    # Note: coverage is currently reverted to original state if best was not found in the last iteration,
                    # OR it is in the state of best_new_r if the last iteration was the best.
                    # To be safe, let's revert to original, then apply best_new_r.
                    # Wait, the loop above reverts if not best. If it IS best, it leaves it applied.
                    # But if a LATER iteration is better, it reverts the previous best and applies the new best.
                    # So at the end, coverage reflects best_new_r.
                    # However, if best_new_r == old_r, coverage is original.
                    
                    # We need to update current_L and current_sec
                    current_L = best_L
                    current_sec = best_sec
                    residues[i] = best_new_r
                    if best_new_r != old_r:
                        improved = True
                        # Ensure coverage is correct.
                        # If best_new_r was found, coverage is already updated to it.
                        # If best_new_r == old_r, coverage is original.
                        # So we are good.
                        
            else:
                # Large prime: targeted search
                # Candidates: 
                # 1. Residue covering current_L + 1
                # 2. Residue covering current_L + 2
                # 3. Residue covering current_L + 3
                # 4. old_r
                # 5. A few randoms
                
                candidates = set()
                candidates.add(old_r)
                
                for offset in range(1, 6):
                    target = current_L + offset
                    if target >= 1:
                        r = target % p
                        candidates.add(r)
                
                # Add 5 random candidates
                for _ in range(5):
                    candidates.add(np.random.randint(0, p))
                
                best_new_r = old_r
                best_L = current_L
                best_sec = current_sec
                
                for new_r in candidates:
                    if new_r == old_r:
                        continue
                    
                    if old_r == 0:
                        old_start = p
                    else:
                        old_start = old_r
                    if new_r == 0:
                        new_start = p
                    else:
                        new_start = new_r
                    
                    if old_start <= OPT_LIMIT:
                        coverage[old_start::p] -= 1
                    if new_start <= OPT_LIMIT:
                        coverage[new_start::p] += 1
                    
                    new_L, new_sec = get_score(coverage)
                    
                    if new_L > best_L or (new_L == best_L and new_sec > best_sec):
                        best_L = new_L
                        best_sec = new_sec
                        best_new_r = new_r
                    else:
                        if new_start <= OPT_LIMIT:
                            coverage[new_start::p] -= 1
                        if old_start <= OPT_LIMIT:
                            coverage[old_start::p] += 1
                
                if best_new_r != old_r:
                    current_L = best_L
                    current_sec = best_sec
                    residues[i] = best_new_r
                    improved = True
                    # Coverage is already updated to best_new_r

        if not improved:
            # No improvement in this sweep. 
            # Maybe we are stuck. 
            # Try increasing OPT_LIMIT?
            # Or just break.
            # Let's try a "random restart" of a few residues if stuck for a while?
            # For now, break to save time.
            break

    # --- Phase 3: Final Evaluation ---
    # Compute the actual score on the full 100,000 range
    
    def full_score(residues, limit=EVAL_LIMIT):
        cov = np.zeros(limit + 1, dtype=np.int16)
        for i, p in enumerate(primes):
            r = int(residues[i])
            if r == 0:
                start = p
            else:
                start = r
            if start <= limit:
                cov[start::p] += 1
        
        bad = np.where(cov[1:limit+1] < 2)[0]
        if len(bad) == 0:
            return limit
        return int(bad[0])

    final_score = full_score(residues)
    
    # If we have time, do a quick check if we can improve further with a larger OPT_LIMIT?
    # Probably not worth the complexity risk.
    
    return residues.tolist()

if __name__ == "__main__":
    # Quick test
    import time
    start = time.time()
    res = construct()
    end = time.time()
    print(f"Time: {end - start:.2f}s")
    # Print score
    def get_primes(n):
        sieve = bytearray([1]) * (n + 1)
        sieve[0] = sieve[1] = 0
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
        return [i for i in range(2, n+1) if sieve[i]]
    primes = get_primes(1000)
    cov = np.zeros(100001, dtype=np.int16)
    for i, p in enumerate(primes):
        r = res[i]
        if r == 0:
            start = p
        else:
            start = r
        if start <= 100000:
            cov[start::p] += 1
    bad = np.where(cov[1:100001] < 2)[0]
    score = 100000 if len(bad) == 0 else int(bad[0])
    print(f"Score: {score}")
