import sys
import time

def construct():
    # Generate primes up to 1000
    primes = [p for p in range(2, 1001) if all(p % q for q in range(2, int(p ** 0.5) + 1))]
    primes_asc = sorted(primes)
    n_primes = len(primes_asc)
    
    L_limit = 1000
    
    best_score = 0
    best_residues = [0] * n_primes
    
    # Initial greedy to get a good lower bound for pruning
    # Greedy: for each prime, pick r that maximizes the potential prefix length
    cov_init = [0] * (L_limit + 1)
    init_residues = [0] * n_primes
    for i, p in enumerate(primes_asc):
        best_r = 0
        best_potential = -1
        remaining_after = n_primes - i - 1
        for r in range(p):
            # Find the first x where (cov_init[x] + (1 if x%p==r else 0) + remaining_after) < 2
            x_fail = L_limit + 1
            for x in range(1, L_limit + 1):
                nc = cov_init[x] + (1 if x % p == r else 0)
                if nc + remaining_after < 2:
                    x_fail = x
                    break
            if x_fail > best_potential:
                best_potential = x_fail
                best_r = r
        
        init_residues[i] = best_r
        for x in range(1, L_limit + 1):
            if x % p == best_r:
                cov_init[x] += 1
                
    # Calculate score for init
    for x in range(1, L_limit + 1):
        if cov_init[x] < 2:
            break
    else:
        x = L_limit + 1
    init_score = x - 1
    if init_score > best_score:
        best_score = init_score
        best_residues = init_residues[:]

    # Backtracking search with in-place modification
    cov = [0] * (L_limit + 1)
    residues = [0] * n_primes
    
    start_time = time.time()
    time_limit = 110.0
    
    def backtrack_timed(i):
        nonlocal best_score, best_residues
        if time.time() - start_time > time_limit:
            return True # Signal timeout
        
        if i == n_primes:
            # Calculate final score
            score = 0
            for x in range(1, L_limit + 1):
                if cov[x] >= 2:
                    score = x
                else:
                    break
            if score > best_score:
                best_score = score
                best_residues = residues[:]
            return False

        p = primes_asc[i]
        remaining = n_primes - i - 1
        
        candidates = []
        
        # We only need to check feasibility up to best_score to prune.
        # If a candidate fails at x <= best_score, it cannot improve the score.
        # If it survives up to best_score, it is a candidate.
        check_limit = best_score
        
        for r in range(p):
            feasible = True
            # Check only up to check_limit
            for x in range(1, check_limit + 1):
                nc = cov[x] + (1 if x % p == r else 0)
                if nc + remaining < 2:
                    feasible = False
                    break
            
            if feasible:
                # Calculate potential prefix length for sorting
                # Find first x where nc + remaining < 2
                x_fail = check_limit + 1
                # We can start checking from 1, but we know 1..check_limit are feasible.
                # So we can start from check_limit + 1?
                # No, we need to find the *first* failure to determine the exact potential.
                # But for sorting, a higher potential is better.
                # If we don't find a failure up to L_limit, potential is L_limit.
                # We can just scan from 1 to L_limit, but break early if found.
                # Since we already know 1..check_limit are feasible, we can start from check_limit+1?
                # Wait, if check_limit is 0, we start from 1.
                # If check_limit is 500, we know 1..500 are feasible.
                # So the first failure must be > 500.
                # So we can start scanning from check_limit + 1.
                for x in range(check_limit + 1, L_limit + 1):
                    nc = cov[x] + (1 if x % p == r else 0)
                    if nc + remaining < 2:
                        x_fail = x
                        break
                
                candidates.append((x_fail, r))
        
        # Sort candidates by potential prefix length descending
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        for x_fail, r in candidates:
            # Apply change
            residues[i] = r
            start_x = r if r > 0 else p
            for x in range(start_x, L_limit + 1, p):
                cov[x] += 1
            
            timed_out = backtrack_timed(i + 1)
            
            # Undo change
            for x in range(start_x, L_limit + 1, p):
                cov[x] -= 1
                
            if timed_out:
                return True
            
            if best_score == L_limit:
                return False

    backtrack_timed(0)
    
    return best_residues
