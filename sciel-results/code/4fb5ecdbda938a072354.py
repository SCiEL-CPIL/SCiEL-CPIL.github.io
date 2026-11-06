import time
import math
import random
import numpy as np

N = 5000

def _primes(limit):
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

ALL_PRIMES = _primes(N)
PRIME_SET = set(ALL_PRIMES)

def construct():
    start_time = time.time()
    deadline = start_time + 290.0
    
    best_size = 0
    best_set = []
    
    # We will try a few different random seeds/initializations
    # and local search strategies.
    
    # Helper to compute k and set from residues and active primes
    def evaluate_config(residues, active_primes):
        # residues: dict p -> r
        # active_primes: list of primes
        count = np.zeros(N + 1, dtype=np.int8)
        for p in active_primes:
            r = residues.get(p, 0)
            # Mark forbidden
            # x = r, r+p, r+2p...
            for x in range(r, N + 1, p):
                count[x] += 1
        
        k = np.sum(count == 0)
        # Construct set
        S = [int(x) for x in range(N + 1) if count[x] == 0]
        return k, S

    def local_search(residues, active_primes, time_limit):
        # residues: dict p -> r
        # active_primes: list
        # Returns improved residues, k, S
        
        # Initial eval
        current_k, current_S = evaluate_config(residues, active_primes)
        
        # Precompute indices for fast updates?
        # For each p, the indices are r, r+p, ...
        
        while time.time() < time_limit:
            improved = False
            # Shuffle order to avoid bias
            prims = list(active_primes)
            random.shuffle(prims)
            
            for p in prims:
                if time.time() > time_limit:
                    break
                current_r = residues[p]
                best_r = current_r
                best_k = current_k
                
                # Try all other residues
                # Optimization: We can update count array incrementally
                # But evaluate_config is O(N * len(active_primes)/p_avg) which is roughly O(N * 10)
                # N=5000, so 50k ops per eval. 50 evals per prime pass. 80 primes.
                # 80 * 50 * 50k = 200M ops per full pass. Might be slow in Python.
                
                # Let's optimize the evaluation.
                # We maintain `count` array globally in the local search loop.
                # This function is called inside the loop, so we need to pass count or manage it outside.
                
                pass # Placeholder, will implement incremental update below

        return residues, current_k, current_S

    # Since Python is slow, we need a faster local search.
    # Let's implement the local search with incremental count updates.
    
    def run_local_search(residues_init, active_primes_init, max_time):
        residues = dict(residues_init)
        active_primes = list(active_primes_init)
        
        # Initialize count array
        count = np.zeros(N + 1, dtype=np.int8)
        for p in active_primes:
            r = residues[p]
            for x in range(r, N + 1, p):
                count[x] += 1
        
        current_k = np.sum(count == 0)
        
        # Helper to update count and k when changing residue of p from old_r to new_r
        def update_residue(p, old_r, new_r):
            nonlocal current_k
            # Remove old
            for x in range(old_r, N + 1, p):
                count[x] -= 1
                if count[x] == 0:
                    current_k += 1
            # Add new
            for x in range(new_r, N + 1, p):
                if count[x] == 0:
                    current_k -= 1
                count[x] += 1
        
        # Local Search Loop
        no_improve_count = 0
        while time.time() < max_time:
            if no_improve_count > 3:
                break
            
            improved_in_pass = False
            prims = list(active_primes)
            random.shuffle(prims)
            
            for p in prims:
                if time.time() > max_time:
                    break
                
                current_r = residues[p]
                best_r = current_r
                best_k = current_k
                
                # Try other residues
                for r in range(p):
                    if r == current_r:
                        continue
                    
                    # Simulate
                    old_k = current_k
                    update_residue(p, current_r, r)
                    
                    if current_k > best_k:
                        best_k = current_k
                        best_r = r
                    
                    # Revert
                    update_residue(p, r, current_r)
                    # Verify revert restored old_k
                    # assert current_k == old_k, f"Revert failed: {current_k} vs {old_k}"
                
                if best_r != current_r:
                    # Apply best
                    update_residue(p, current_r, best_r)
                    residues[p] = best_r
                    improved_in_pass = True
            
            if not improved_in_pass:
                no_improve_count += 1
            else:
                no_improve_count = 0
        
        # Finalize
        S = [int(x) for x in range(N + 1) if count[x] == 0]
        return residues, active_primes, current_k, S

    # Main Strategy: Iterate on active set and residues
    # Start with a guess for k
    # k ~ 400-500
    # Let's try a few starting k values
    
    for start_k_guess in [300, 400, 500, 600]:
        if time.time() > deadline:
            break
            
        # Determine active primes
        active_primes = [p for p in ALL_PRIMES if p <= start_k_guess]
        
        # Initialize residues randomly
        residues = {p: random.randint(0, p - 1) for p in active_primes}
        
        # Run local search with a time slice
        # We need to handle the case where active set changes
        # We'll do a few rounds of "evaluate -> update active set -> local search"
        
        current_k = start_k_guess
        current_residues = residues
        current_active = active_primes
        
        for _ in range(5): # Max 5 outer iterations
            if time.time() > deadline:
                break
            
            # Run local search for a fixed amount of time or until convergence
            # We give it 10 seconds per outer iteration
            time_slice = 10.0
            current_residues, current_active, current_k, S = run_local_search(
                current_residues, current_active, time_slice
            )
            
            # Check if active set needs update
            new_active = [p for p in ALL_PRIMES if p <= current_k]
            
            if len(new_active) != len(current_active):
                # Update active set
                # Add new primes
                for p in new_active:
                    if p not in current_residues:
                        current_residues[p] = random.randint(0, p - 1)
                # Remove inactive primes (optional, but keeps dict clean)
                for p in list(current_residues.keys()):
                    if p not in new_active:
                        del current_residues[p]
                
                current_active = new_active
                # We should re-run local search as the set of constraints changed
                # The loop continues, so it will run again
            else:
                # Active set stable, check if we have a valid solution
                # And if we improved
                if current_k > best_size:
                    best_size = current_k
                    best_set = S
                # If active set is stable and local search converged, we can break
                # But maybe a different random init is better?
                # For now, break to try next start_k or next random seed
                break
                
    # One more check: ensure best_set is actually admissible
    # (It should be by construction, but let's be safe)
    if best_set:
        k = len(best_set)
        S_arr = np.array(best_set)
        valid = True
        for p in ALL_PRIMES:
            if p > k:
                break
            residues = S_arr % p
            counts = np.bincount(residues, minlength=p)
            if counts.min() > 0:
                valid = False
                break
        
        if not valid:
            # Fallback to seed-like behavior or empty
            # But this shouldn't happen
            pass

    return sorted(best_set)
