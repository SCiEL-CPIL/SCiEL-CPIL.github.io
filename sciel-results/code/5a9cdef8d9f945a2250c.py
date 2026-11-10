import numpy as np

def construct():
    # Generate primes up to 1000
    primes = [p for p in range(2, 1001) if all(p % q for q in range(2, int(p ** 0.5) + 1))]
    n_primes = len(primes)
    
    MAX_L = 5000
    x = np.arange(1, MAX_L + 1)
    coverage = np.zeros(MAX_L, dtype=np.int32)
    residues = np.zeros(n_primes, dtype=np.int64)
    
    def compute_L(cov):
        bad = np.where(cov < 2)[0]
        if len(bad) == 0:
            return MAX_L
        return bad[0]
    
    # Phase 1: Fix residues for small primes (first ~30) to maximize coverage density in [1, 2000]
    # Strategy: for each small prime, pick the residue that maximizes the number of points 
    # in [1, 2000] that reach coverage >= 2, weighted by how close they are to the current bottleneck.
    
    SMALL_PRIME_COUNT = 30
    DENSITY_HORIZON = 2000
    
    for i in range(SMALL_PRIME_COUNT):
        p = primes[i]
        best_r = 0
        best_score = -1
        
        # Try each possible residue and evaluate its impact on coverage in [1, DENSITY_HORIZON]
        for r in range(p):
            # Compute what coverage would look like if we add this prime with residue r
            mask = (x <= DENSITY_HORIZON) & (x % p == r)
            new_cov = coverage.copy()
            new_cov[mask] += 1
            
            # Score: number of points in [1, DENSITY_HORIZON] with coverage >= 2
            # Plus bonus for extending the prefix where all points have coverage >= 2
            safe_count = np.sum(new_cov[:DENSITY_HORIZON] >= 2)
            
            # Find the first point where coverage < 2 in the prefix
            bad_in_prefix = np.where(new_cov[:DENSITY_HORIZON] < 2)[0]
            if len(bad_in_prefix) == 0:
                prefix_len = DENSITY_HORIZON
            else:
                prefix_len = bad_in_prefix[0]
            
            # Combine: prioritize extending the prefix, then maximize safe count
            score = prefix_len * 1000 + safe_count
            
            if score > best_score:
                best_score = score
                best_r = r
        
        residues[i] = best_r
        mask = (x % p) == best_r
        coverage[mask] += 1
    
    # Phase 2: For larger primes, use "coverage deficit" heuristic
    # Find the first integer with coverage < 2, and pick the residue that covers it
    # Tie-break: maximize the number of additional points in the deficit region that get covered
    
    for i in range(SMALL_PRIME_COUNT, n_primes):
        p = primes[i]
        
        # Find the first point where coverage < 2
        bad = np.where(coverage < 2)[0]
        
        if len(bad) == 0:
            # All points covered >= 2 in the visible range
            # Pick residue that maximizes coverage in [1, MAX_L]
            best_r = 0
            best_score = -1
            for r in range(min(p, 50)):  # Limit search for speed
                mask = (x % p) == r
                score = np.sum(coverage[mask])
                if score > best_score:
                    best_score = score
                    best_r = r
            residues[i] = best_r
            coverage[(x % p) == best_r] += 1
            continue
        
        first_bad = bad[0]
        target_x = x[first_bad]
        
        # The residue that covers target_x is target_x % p
        r_target = target_x % p
        
        # Evaluate r_target vs a few alternatives
        # We want to cover the first bad point, but also maximize coverage of other deficit points
        candidates = [r_target]
        
        # Add a few other candidates that might cover more deficit points
        deficit_points = np.where(coverage < 2)[0]
        if len(deficit_points) > 0:
            # Look at the first few deficit points and see what residues they suggest
            for j in range(min(5, len(deficit_points))):
                r_alt = x[deficit_points[j]] % p
                if r_alt not in candidates:
                    candidates.append(r_alt)
        
        best_r = r_target
        best_score = -1
        
        for r in candidates:
            mask = (x % p) == r
            new_cov = coverage.copy()
            new_cov[mask] += 1
            
            # Find new L
            new_bad = np.where(new_cov < 2)[0]
            if len(new_bad) == 0:
                new_L = MAX_L
            else:
                new_L = new_bad[0]
            
            # Score: primarily new_L, secondarily total coverage in [1, new_L + 100]
            if new_L < MAX_L:
                cov_region = np.sum(new_cov[:new_L + 100])
            else:
                cov_region = np.sum(new_cov)
            
            score = new_L * 10000 + cov_region
            
            if score > best_score:
                best_score = score
                best_r = r
        
        residues[i] = best_r
        coverage[(x % p) == best_r] += 1
    
    # Phase 3: Targeted local search on primes that cover the current bottleneck
    L = compute_L(coverage)
    
    improved = True
    max_iterations = 500
    iteration = 0
    
    while improved and L < MAX_L and iteration < max_iterations:
        improved = False
        iteration += 1
        
        x_fail = L + 1
        if x_fail > MAX_L:
            break
        
        # Find primes that are NOT currently covering x_fail
        # These are the candidates we can change to cover x_fail
        candidate_indices = []
        for i in range(n_primes):
            p = primes[i]
            r_old = residues[i]
            if x_fail % p != r_old:
                candidate_indices.append(i)
        
        # Sort candidates by "redundancy" - prefer primes whose current coverage is redundant
        # i.e., the points they cover have high coverage
        def redundancy_score(i):
            p = primes[i]
            r_old = residues[i]
            # Find points in [1, L] covered by this prime
            if r_old == 0:
                k_start = 1
            else:
                k_start = 0
            k_end = (L - r_old) // p
            if k_start > k_end:
                return 0  # No points covered in prefix, very safe to change
            ks = np.arange(k_start, k_end + 1)
            x_vals = r_old + ks * p
            idxs = x_vals - 1
            # Return minimum coverage among covered points (higher = more redundant)
            return np.min(coverage[idxs])
        
        # Sort by redundancy (highest first - these are safest to change)
        candidate_indices.sort(key=lambda i: -redundancy_score(i))
        
        for i in candidate_indices:
            p = primes[i]
            r_old = residues[i]
            r_new = x_fail % p
            
            # Check safety: for all x in [1, L] where x % p == r_old, coverage[x] must be >= 3
            if r_old == 0:
                k_start = 1
            else:
                k_start = 0
            k_end = (L - r_old) // p
            
            if k_start <= k_end:
                ks = np.arange(k_start, k_end + 1)
                x_vals = r_old + ks * p
                idxs = x_vals - 1
                if np.min(coverage[idxs]) < 3:
                    continue
            
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
                break
            else:
                # Revert
                coverage[old_mask] += 1
                coverage[new_mask] -= 1
                residues[i] = r_old
    
    return [int(r) for r in residues]
