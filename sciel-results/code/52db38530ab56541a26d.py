# EVOLVE-BLOCK-START
import time
import math
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill

    # Generate primes <= 1000
    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(math.isqrt(limit)) + 1):
            if sieve[i]:
                for j in range(i * i, limit + 1, i):
                    sieve[j] = False
        return [i for i, is_p in enumerate(sieve) if is_p]

    primes = get_primes(1000)
    n = len(primes)  # 168 primes
    primes_arr = np.array(primes)

    # Precompute for fast coverage evaluation
    # For each prime i, precompute the positions it covers for each residue
    # Actually, let's just precompute the "positions covered" for each prime
    # and use efficient updates.

    # Precompute the set of positions covered by each prime for each residue
    # This is too much memory. Instead, let's use a different approach.

    # For fast evaluation, we'll use a coverage array and update it incrementally.
    
    # Let's define a fast score function that works with a coverage array.
    # The idea: maintain a global coverage array. When we change residue for prime i
    # from old_r to new_r, we subtract the old positions and add the new positions.
    
    # Precompute for each prime i the "delta" arrays: for each residue r, 
    # which positions in [1, L_max] are covered.
    # Actually, let's just compute on the fly with numpy, it should be fast enough.

    L_EVAL = 2000  # We'll evaluate up to this for the search
    # For final scoring, we need to go higher, but the search can use a smaller range.

    def compute_coverage(residues, limit):
        """Compute coverage array for residues up to limit."""
        coverage = np.zeros(limit + 1, dtype=np.int32)
        for i in range(n):
            p = primes[i]
            r = residues[i]
            if r == 0:
                start = p
            else:
                start = r
            if start > limit:
                continue
            idx = np.arange(start, limit + 1, p)
            coverage[idx] += 1
        return coverage

    def score_from_coverage(coverage, limit):
        """Find the largest L such that coverage[1..L] >= 2."""
        valid = coverage[1:] >= 2
        if not valid[0]:
            return 0
        false_positions = np.where(~valid)[0]
        if len(false_positions) == 0:
            return limit
        return false_positions[0]

    def full_score(residues):
        """Compute the exact score (largest L), checking up to 10000."""
        limit = 10000
        coverage = compute_coverage(residues, limit)
        L = score_from_coverage(coverage, limit)
        # If L == limit, check further (up to 100000)
        if L == limit:
            # Check up to 100000
            limit2 = 100000
            coverage2 = compute_coverage(residues, limit2)
            L = score_from_coverage(coverage2, limit2)
        return L

    # Initial solution: Greedy construction
    # Process primes in order, choosing the residue that maximizes the current prefix.
    
    def greedy_construct():
        residues = [0] * n
        coverage = np.zeros(L_EVAL + 1, dtype=np.int32)
        
        for i in range(n):
            p = primes[i]
            best_r = 0
            best_prefix = -1
            
            # Find current prefix
            valid = coverage[1:L_EVAL+1] >= 2
            if not valid[0]:
                current_prefix = 0
            else:
                false_pos = np.where(~valid)[0]
                if len(false_pos) == 0:
                    current_prefix = L_EVAL
                else:
                    current_prefix = false_pos[0]
            
            # Try candidate residues
            # For small p, try all. For large p, try a subset.
            if p <= 30:
                candidates = list(range(p))
            else:
                # Try residues that cover the current bottleneck
                bottleneck_x = current_prefix + 1 if current_prefix < L_EVAL else L_EVAL
                candidates = [bottleneck_x % p, 0]
                # Add some random
                for _ in range(20):
                    candidates.append(random.randint(0, p - 1))
                # Remove duplicates
                candidates = list(set(candidates))
            
            for r in candidates:
                # Calculate new prefix
                # new_cov[x] = coverage[x] + (1 if x % p == r else 0)
                # We need the first x where new_cov[x] < 2
                # This happens when coverage[x] == 0 or (coverage[x] == 1 and x % p != r)
                
                new_prefix = current_prefix
                for x in range(current_prefix + 1, L_EVAL + 1):
                    c = coverage[x]
                    if c == 0:
                        break
                    if c == 1:
                        if x % p != r:
                            break
                    new_prefix = x
                
                if new_prefix > best_prefix:
                    best_prefix = new_prefix
                    best_r = r
            
            residues[i] = best_r
            
            # Update coverage
            r = best_r
            if r == 0:
                start = p
            else:
                start = r
            if start <= L_EVAL:
                idx = np.arange(start, L_EVAL + 1, p)
                coverage[idx] += 1
        
        return residues

    best_residues = greedy_construct()
    best_score = full_score(best_residues)
    
    # Simulated Annealing refinement
    # Use incremental coverage updates for speed.
    
    def sa_refine(residues, score, time_limit):
        """Run simulated annealing to improve the solution."""
        current = list(residues)
        current_score = score
        best = list(current)
        best_score_val = current_score
        
        # Compute initial coverage
        coverage = compute_coverage(current, L_EVAL)
        
        T = 100.0  # Initial temperature
        cooling = 0.9995  # Cooling rate
        
        while time.time() < time_limit and T > 0.1:
            # Pick a random prime to change
            i = random.randint(0, n - 1)
            p = primes[i]
            old_r = current[i]
            new_r = random.randint(0, p - 1)
            if new_r == old_r:
                continue
            
            # Update coverage: subtract old, add new
            # Old positions
            old_start = p if old_r == 0 else old_r
            if old_start <= L_EVAL:
                old_idx = np.arange(old_start, L_EVAL + 1, p)
                coverage[old_idx] -= 1
            
            # New positions
            new_start = p if new_r == 0 else new_r
            if new_start <= L_EVAL:
                new_idx = np.arange(new_start, L_EVAL + 1, p)
                coverage[new_idx] += 1
            
            current[i] = new_r
            new_score = score_from_coverage(coverage, L_EVAL)
            
            # Accept or reject
            delta = new_score - current_score
            if delta >= 0 or random.random() < math.exp(delta / T):
                current_score = new_score
                if new_score > best_score_val:
                    best_score_val = new_score
                    best = list(current)
            else:
                # Revert
                if new_start <= L_EVAL:
                    coverage[new_idx] -= 1
                if old_start <= L_EVAL:
                    coverage[old_idx] += 1
                current[i] = old_r
            
            T *= cooling
        
        # Final check with full score
        final_score = full_score(best)
        if final_score > best_score_val:
            best_score_val = final_score
        return best, best_score_val

    # Run SA with time limit
    time_limit = deadline - 15
    best_residues, best_score = sa_refine(best_residues, best_score, time_limit)
    
    # If we still have time, try another round with a different starting point
    if time.time() < deadline - 30:
        # Try a different initial solution: random + greedy repair
        def random_solution():
            return [random.randint(0, primes[i] - 1) for i in range(n)]
        
        def greedy_repair(residues):
            """Repair a random solution by fixing gaps."""
            residues = list(residues)
            coverage = compute_coverage(residues, L_EVAL)
            
            for _ in range(50):  # Max iterations
                L = score_from_coverage(coverage, L_EVAL)
                if L >= L_EVAL:
                    break
                
                # Find the first gap
                gap_x = L + 1
                if gap_x > L_EVAL:
                    break
                
                # Try to fix the gap by changing a random prime's residue
                i = random.randint(0, n - 1)
                p = primes[i]
                old_r = residues[i]
                new_r = gap_x % p
                
                if new_r == old_r:
                    continue
                
                # Update coverage
                old_start = p if old_r == 0 else old_r
                if old_start <= L_EVAL:
                    old_idx = np.arange(old_start, L_EVAL + 1, p)
                    coverage[old_idx] -= 1
                
                new_start = p if new_r == 0 else new_r
                if new_start <= L_EVAL:
                    new_idx = np.arange(new_start, L_EVAL + 1, p)
                    coverage[new_idx] += 1
                
                residues[i] = new_r
                
                # Check if this improved the score
                new_L = score_from_coverage(coverage, L_EVAL)
                if new_L < L:
                    # Revert
                    if new_start <= L_EVAL:
                        coverage[new_idx] -= 1
                    if old_start <= L_EVAL:
                        coverage[old_idx] += 1
                    residues[i] = old_r
            
            return residues
        
        # Try a few random starts
        for _ in range(3):
            if time.time() > deadline - 15:
                break
            random_res = random_solution()
            repaired_res = greedy_repair(random_res)
            res_score = full_score(repaired_res)
            if res_score > best_score:
                best_score = res_score
                best_residues = repaired_res
            # Refine with SA
            if time.time() < deadline - 15:
                refined, refined_score = sa_refine(repaired_res, res_score, deadline - 15)
                if refined_score > best_score:
                    best_score = refined_score
                    best_residues = refined
    
    return best_residues
# EVOLVE-BLOCK-END
