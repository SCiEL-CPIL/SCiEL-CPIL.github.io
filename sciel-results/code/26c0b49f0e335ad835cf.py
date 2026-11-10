import numpy as np

def construct():
    # Generate primes up to 1000
    primes = [p for p in range(2, 1001) if all(p % q for q in range(2, int(p ** 0.5) + 1))]
    n_primes = len(primes)
    
    # We need to support L up to at least 1000, likely more.
    # Let's set a safe upper bound.
    MAX_L = 2000
    
    x = np.arange(1, MAX_L + 1)
    
    coverage = np.zeros(MAX_L, dtype=np.int32)
    residues = np.zeros(n_primes, dtype=np.int64)
    
    def compute_L(cov):
        # Find largest L such that all x in [1,L] have coverage >= 2
        # cov is 0-indexed for x=1..MAX_L
        # cov[0] corresponds to x=1
        bad = np.where(cov < 2)[0]
        if len(bad) == 0:
            return MAX_L
        return bad[0]

    # Phase 1: Greedy Construction with Lookahead
    # We iterate through primes. For each, we pick the best residue.
    
    for i in range(n_primes):
        p = primes[i]
        
        best_r = 0
        best_score = -1
        best_L = -1
        best_potential = -1
        
        # We evaluate all residues r in [0, p)
        # To speed this up, we can precompute x % p
        x_mod_p = x % p
        
        for r in range(p):
            # Calculate the change in coverage if we pick r
            # mask = (x_mod_p == r)
            # new_cov = coverage + mask
            
            # We want to find the new L.
            # Instead of creating a full new_cov array, we can reason:
            # L is the first index where cov < 2.
            # Let current_L = compute_L(coverage)
            
            # If we add mask, cov[i] increases by 1 if x[i]%p==r.
            
            # Let's just compute the new L efficiently.
            # We can check the prefix up to current_L + some margin.
            
            # Actually, for small p (which early primes are), p is small, so the loop is fast.
            # For large p, p is large, but we only have ~168 primes.
            # Total operations: sum(p) for p in primes <= 1000.
            # Sum of primes up to 1000 is approx 76127.
            # Inside the loop, we do O(MAX_L) work if we are not careful.
            # 76127 * 2000 = 152 million operations. This might be too slow in Python.
            
            # Optimization:
            # We only need to check the prefix up to `current_L + margin`.
            # If current_L is small, we check a small prefix.
            # As L grows, the prefix grows.
            
            current_L = compute_L(coverage)
            
            # We are interested in the prefix [0, current_L + 50]
            # Because if we can't extend L by more than 50, we look at potential.
            
            # Let's try a different scoring:
            # 1. Does this r fix the current bottleneck?
            # 2. How much does it extend L?
            
            # Let's compute new_L for this r.
            # We can do this by checking the coverage in the prefix.
            
            # Let's just do it simply first, and optimize if needed.
            # Since MAX_L is 2000, and sum(p) is 76k, 76k * 2000 = 152M.
            # Numpy operations are fast. Let's try to vectorize the inner loop?
            # No, the loop is over r.
            
            # Alternative:
            # For each position x in [1, MAX_L], we know if it's covered by r.
            # Let's precompute the "impact" of each r? No, r changes.
            
            # Let's stick to the simple loop but limit the check range.
            check_limit = min(MAX_L, current_L + 100)
            
            # Get the relevant slice of x and coverage
            x_slice = x[:check_limit]
            cov_slice = coverage[:check_limit]
            
            # For this r, new_cov_slice = cov_slice + (x_slice % p == r)
            # We can compute this vectorized.
            
            mask = (x_slice % p == r)
            new_cov_slice = cov_slice + mask
            
            new_bad = np.where(new_cov_slice < 2)[0]
            if len(new_bad) == 0:
                new_L = check_limit
            else:
                new_L = new_bad[0]
                
            # Score:
            # Primary: new_L
            # Secondary: Sum of min(new_cov_slice, 2) in [new_L, new_L+10]
            
            if new_L > best_L:
                best_L = new_L
                # Calculate potential for tie-breaking
                # Look at the next 10 points after new_L
                start_pot = new_L
                end_pot = min(check_limit, new_L + 10)
                if start_pot < end_pot:
                    potential = np.sum(np.minimum(new_cov_slice[start_pot:end_pot], 2))
                else:
                    potential = 0
                best_potential = potential
                best_r = r
            elif new_L == best_L:
                # Tie-breaker: potential
                start_pot = new_L
                end_pot = min(check_limit, new_L + 10)
                if start_pot < end_pot:
                    potential = np.sum(np.minimum(new_cov_slice[start_pot:end_pot], 2))
                else:
                    potential = 0
                if potential > best_potential:
                    best_potential = potential
                    best_r = r
                    best_L = new_L # Update best_L to match? It's the same.

        residues[i] = best_r
        mask = (x % p) == best_r
        coverage[mask] += 1

    # Phase 2: Local Search
    L = compute_L(coverage)
    
    improved = True
    while improved and L < MAX_L:
        improved = False
        x_fail = L + 1
        if x_fail > MAX_L:
            break
            
        # Find a prime to swap
        # We want to change a prime p to cover x_fail.
        # r_new = x_fail % p
        # Check safety: all x in [1, L] with x % p == r_old must have coverage >= 3.
        
        found_swap = False
        for i in range(n_primes):
            p = primes[i]
            r_old = residues[i]
            r_new = x_fail % p
            
            if r_new == r_old:
                continue
            
            # Check safety
            # Find indices in [0, L-1] where x % p == r_old
            # x_val = r_old + k*p
            # 1 <= x_val <= L
            
            if r_old == 0:
                k_start = 1
            else:
                k_start = 0
                
            k_end = (L - r_old) // p
            
            if k_start <= k_end:
                ks = np.arange(k_start, k_end + 1)
                x_vals = r_old + ks * p
                idxs = x_vals - 1
                
                # Check min coverage
                if np.min(coverage[idxs]) < 3:
                    continue
            # Else safe
            
            # Apply change
            residues[i] = r_new
            
            old_mask = (x % p) == r_old
            new_mask = (x % p) == r_new
            
            coverage[old_mask] -= 1
            coverage[new_mask] += 1
            
            new_L = compute_L(coverage)
            
            if new_L > L:
                L = new_L
                improved = True
                found_swap = True
                break
            else:
                # Revert
                coverage[old_mask] += 1
                coverage[new_mask] -= 1
                residues[i] = r_old
                
        if not found_swap:
            # Try a more aggressive search?
            # Maybe try changing a prime that DOESN'T cover x_fail to cover x_fail,
            # even if it breaks some x in [1,L] with coverage 2?
            # No, that would reduce L.
            
            # What if we try to change a prime to cover x_fail, and *also* change another prime to fix the broken one?
            # This is 2-opt.
            
            # For now, let's just stop if no single move works.
            # But we can try to "shake" the state.
            # Or, we can try to improve the potential even if L doesn't increase immediately.
            # This is risky. Let's stick to strict improvement.
            
            # One optimization: try all primes, not just break on first.
            # But if we find one that works, we restart.
            # If none work, we are stuck.
            
            # Let's add a "soft" move: if no hard move works, try to maximize potential at L.
            # This might help in later steps.
            
            best_r = None
            best_p = None
            best_potential = -1
            
            for i in range(n_primes):
                p = primes[i]
                r_old = residues[i]
                r_new = x_fail % p
                
                if r_new == r_old:
                    continue
                
                # Check safety for [1, L]
                if r_old == 0:
                    k_start = 1
                else:
                    k_start = 0
                    
                k_end = (L - r_old) // p
                
                safe = True
                if k_start <= k_end:
                    ks = np.arange(k_start, k_end + 1)
                    x_vals = r_old + ks * p
                    idxs = x_vals - 1
                    if np.min(coverage[idxs]) < 3:
                        safe = False
                
                if not safe:
                    continue
                
                # Calculate potential
                # Apply temporarily
                old_mask = (x % p) == r_old
                new_mask = (x % p) == r_new
                
                # We only care about the prefix [L, L+10]
                # Let's just compute the change in potential for the whole array? No.
                # Let's apply to a copy? Too slow.
                
                # Let's just compute the new coverage at x_fail and neighbors.
                # This is complex. Let's skip soft moves for now.
                pass
            
            break

    return [int(r) for r in residues]
