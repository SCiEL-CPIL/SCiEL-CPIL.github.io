import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill
    
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
    
    # Evaluation range
    MAX_EVAL = 10000
    
    xs = np.arange(1, MAX_EVAL + 1, dtype=np.int64)
    
    def compute_coverage(residues):
        """Compute coverage array using vectorized operations."""
        # For each prime p_i with residue r_i, mark positions where x % p_i == r_i
        coverage = np.zeros(MAX_EVAL, dtype=np.int32)
        for i in range(n_primes):
            p = primes[i]
            r = residues[i]
            # Positions x in [1, MAX_EVAL] where x % p == r
            # x = k*p + r for k >= 0, and x >= 1
            if r == 0:
                # x must be multiple of p, starting from p
                first = p
            else:
                # x = r, r+p, r+2p, ... but x >= 1
                first = r
                if first < 1:
                    first = r + p
            if first <= MAX_EVAL:
                coverage[first-1::p] += 1
        return coverage

    def get_score(coverage):
        """Find the largest L such that coverage[0:L] >= 2."""
        mask = coverage < 2
        if not np.any(mask):
            return MAX_EVAL
        first_fail_idx = np.argmax(mask)
        return int(xs[first_fail_idx] - 1)

    # Phase 1: Greedy density-based initialization
    # For each prime, choose the residue that maximizes the number of positions in [1, 1000] it covers
    # that are currently under-covered
    residues = np.zeros(n_primes, dtype=np.int64)
    coverage = np.zeros(MAX_EVAL, dtype=np.int32)
    
    # Process primes in order of increasing size (smaller primes cover more positions)
    prime_order = sorted(range(n_primes), key=lambda i: primes[i])
    
    for idx in prime_order:
        p = primes[idx]
        best_r = 0
        best_gain = -1
        for r in range(p):
            # Calculate how many positions in [1, 1000] this residue would cover
            # that currently have coverage < 2
            if r == 0:
                first = p
            else:
                first = r
                if first < 1:
                    first = r + p
            if first > 1000:
                continue
            # Count positions in [1, 1000] with coverage < 2
            count = 0
            x = first
            while x <= 1000:
                if coverage[x-1] < 2:
                    count += 1
                x += p
            if count > best_gain:
                best_gain = count
                best_r = r
        residues[idx] = best_r
        # Update coverage
        if best_r == 0:
            first = p
        else:
            first = best_r
            if first < 1:
                first = best_r + p
        if first <= MAX_EVAL:
            coverage[first-1::p] += 1
    
    current_score = get_score(coverage)
    best_score = current_score
    best_residues = residues.copy()
    
    # Phase 2: Conflict-driven local repair
    # Iteratively find the first gap and try to fix it by reassigning residues
    # of primes that can cover that position
    
    def find_first_gap(coverage):
        mask = coverage < 2
        if not np.any(mask):
            return -1
        idx = np.argmax(mask)
        return int(xs[idx])
    
    def can_cover_prime(x, p):
        """Check if prime p with some residue can cover x."""
        return True  # Any prime can cover any x with appropriate residue
    
    def try_fix_gap(x, residues, coverage, primes):
        """Try to fix a gap at position x by reassigning residues."""
        # Find primes that could cover x
        # A prime p covers x if we set r_p = x % p
        # We need to check if changing r_p would help without breaking other coverage
        
        candidate_primes = []
        for i in range(n_primes):
            p = primes[i]
            old_r = residues[i]
            new_r = x % p
            if new_r == old_r:
                continue
            # Check impact: does this prime currently cover x?
            # If yes, changing it would lose coverage at x
            # If no, changing it would gain coverage at x
            covers_x_old = (x % p == old_r)
            if not covers_x_old:
                # This change would add coverage at x
                candidate_primes.append((i, old_r, new_r))
        
        if not candidate_primes:
            return False
        
        # Try each candidate and see if it improves the score
        best_improvement = 0
        best_change = None
        
        for (i, old_r, new_r) in candidate_primes:
            p = primes[i]
            # Simulate the change
            # Remove old coverage
            if old_r == 0:
                first_old = p
            else:
                first_old = old_r
                if first_old < 1:
                    first_old = old_r + p
            if first_old <= MAX_EVAL:
                coverage[first_old-1::p] -= 1
            
            # Add new coverage
            if new_r == 0:
                first_new = p
            else:
                first_new = new_r
                if first_new < 1:
                    first_new = new_r + p
            if first_new <= MAX_EVAL:
                coverage[first_new-1::p] += 1
            
            new_score = get_score(coverage)
            improvement = new_score - current_score
            
            if improvement > best_improvement:
                best_improvement = improvement
                best_change = (i, old_r, new_r)
            
            # Revert
            if old_r == 0:
                first_old = p
            else:
                first_old = old_r
                if first_old < 1:
                    first_old = old_r + p
            if first_old <= MAX_EVAL:
                coverage[first_old-1::p] += 1
            
            if new_r == 0:
                first_new = p
            else:
                first_new = new_r
                if first_new < 1:
                    first_new = new_r + p
            if first_new <= MAX_EVAL:
                coverage[first_new-1::p] -= 1
        
        if best_change is not None and best_improvement > 0:
            i, old_r, new_r = best_change
            p = primes[i]
            # Apply the change
            if old_r == 0:
                first_old = p
            else:
                first_old = old_r
                if first_old < 1:
                    first_old = old_r + p
            if first_old <= MAX_EVAL:
                coverage[first_old-1::p] -= 1
            
            if new_r == 0:
                first_new = p
            else:
                first_new = new_r
                if first_new < 1:
                    first_new = new_r + p
            if first_new <= MAX_EVAL:
                coverage[first_new-1::p] += 1
            
            residues[i] = new_r
            return True
        
        return False
    
    # Iterative repair loop
    max_iterations = 1000
    iteration = 0
    while time.time() < deadline and iteration < max_iterations:
        gap_x = find_first_gap(coverage)
        if gap_x == -1:
            break
        
        # Try to fix the gap
        fixed = try_fix_gap(gap_x, residues, coverage, primes)
        
        if fixed:
            new_score = get_score(coverage)
            if new_score > best_score:
                best_score = new_score
                best_residues = residues.copy()
            current_score = new_score
        else:
            # If we can't fix with a single change, try a more aggressive approach
            # Randomly perturb a few residues and see if it helps
            improved = False
            for _ in range(10):
                if time.time() >= deadline:
                    break
                i = random.randint(0, n_primes - 1)
                old_r = residues[i]
                p = primes[i]
                new_r = random.randint(0, p - 1)
                if new_r == old_r:
                    continue
                
                # Apply change
                if old_r == 0:
                    first_old = p
                else:
                    first_old = old_r
                    if first_old < 1:
                        first_old = old_r + p
                if first_old <= MAX_EVAL:
                    coverage[first_old-1::p] -= 1
                
                if new_r == 0:
                    first_new = p
                else:
                    first_new = new_r
                    if first_new < 1:
                        first_new = new_r + p
                if first_new <= MAX_EVAL:
                    coverage[first_new-1::p] += 1
                
                residues[i] = new_r
                new_score = get_score(coverage)
                
                if new_score > current_score:
                    improved = True
                    current_score = new_score
                    if new_score > best_score:
                        best_score = new_score
                        best_residues = residues.copy()
                else:
                    # Revert
                    if old_r == 0:
                        first_old = p
                    else:
                        first_old = old_r
                        if first_old < 1:
                            first_old = old_r + p
                    if first_old <= MAX_EVAL:
                        coverage[first_old-1::p] += 1
                    
                    if new_r == 0:
                        first_new = p
                    else:
                        first_new = new_r
                        if first_new < 1:
                            first_new = new_r + p
                    if first_new <= MAX_EVAL:
                        coverage[first_new-1::p] -= 1
                    
                    residues[i] = old_r
            
            if not improved:
                # Try a small simulated annealing step
                T = max(1.0, 50.0 * (1.0 - iteration / max_iterations))
                i = random.randint(0, n_primes - 1)
                old_r = residues[i]
                p = primes[i]
                new_r = random.randint(0, p - 1)
                if new_r == old_r:
                    continue
                
                # Apply change
                if old_r == 0:
                    first_old = p
                else:
                    first_old = old_r
                    if first_old < 1:
                        first_old = old_r + p
                if first_old <= MAX_EVAL:
                    coverage[first_old-1::p] -= 1
                
                if new_r == 0:
                    first_new = p
                else:
                    first_new = new_r
                    if first_new < 1:
                        first_new = new_r + p
                if first_new <= MAX_EVAL:
                    coverage[first_new-1::p] += 1
                
                residues[i] = new_r
                new_score = get_score(coverage)
                delta = new_score - current_score
                
                if delta >= 0 or random.random() < np.exp(delta / T):
                    current_score = new_score
                    if new_score > best_score:
                        best_score = new_score
                        best_residues = residues.copy()
                else:
                    # Revert
                    if old_r == 0:
                        first_old = p
                    else:
                        first_old = old_r
                        if first_old < 1:
                            first_old = old_r + p
                    if first_old <= MAX_EVAL:
                        coverage[first_old-1::p] += 1
                    
                    if new_r == 0:
                        first_new = p
                    else:
                        first_new = new_r
                        if first_new < 1:
                            first_new = new_r + p
                    if first_new <= MAX_EVAL:
                        coverage[first_new-1::p] -= 1
                    
                    residues[i] = old_r
        
        iteration += 1
    
    return best_residues.tolist()
