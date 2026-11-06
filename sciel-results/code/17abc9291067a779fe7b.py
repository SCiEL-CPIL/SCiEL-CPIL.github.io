import numpy as np
import time

def construct():
    # Generate primes <= 1000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]
    
    primes = sieve(1000)
    n_primes = len(primes)
    
    # Target length to optimize for
    # We aim for L >= 1000. Let's aim for 3000 to be safe, 
    # but the score is the actual first failure.
    L = 3000
    
    # Indices 1 to L
    indices = np.arange(1, L + 1)
    
    # Initialize residues to 0
    residues = np.zeros(n_primes, dtype=np.int32)
    
    # Initialize counts
    counts = np.zeros(L + 1, dtype=np.int32)
    
    start_time = time.time()
    max_time = 110 # seconds
    
    # Phase 1: Greedy Construction
    # For each prime, pick the residue that covers the most "weak" points (count < 2)
    for i, p in enumerate(primes):
        # Calculate residues of all indices 1..L modulo p
        mods = indices % p
        
        # Identify weak points (currently count < 2)
        # Note: counts is 1-indexed effectively, but array is 0..L
        # We care about indices 1..L
        is_weak = (counts[1:] < 2)
        
        # Count weak points for each residue
        # weights is 1 if weak, 0 otherwise
        weights = is_weak.astype(np.float32)
        
        # bincount gives sum of weights for each residue 0..p-1
        # minlength=p ensures we have an entry for all residues
        gains = np.bincount(mods, weights=weights, minlength=p)
        
        # Pick residue with max gain
        # If tie, pick first one (or randomize slightly for diversity? 
        # For determinism in greedy, first max is fine. 
        # To allow different runs, we could add small random noise, 
        # but let's stick to deterministic greedy first.
        # Actually, random tie-breaking can help escape local minima in subsequent steps?
        # For now, argmax.
        best_r = np.argmax(gains)
        residues[i] = best_r
        
        # Update counts
        # Indices in 1..L where index % p == best_r
        # These are indices: best_r, best_r+p, ... if best_r != 0
        # If best_r == 0, indices are p, 2p, ...
        # Wait, indices are 1..L.
        # x % p == r.
        # If r == 0, x is multiple of p.
        # If r != 0, x = r, r+p, ...
        
        # Vectorized update:
        # Create mask for indices 1..L
        # mods == best_r
        mask = (mods == best_r)
        # Update counts at positions indices[mask]
        # indices is 1..L
        # counts is 0..L, so counts[indices] corresponds to 1..L
        counts[indices[mask]] += 1
        
    # Phase 2: Local Search / Coordinate Descent
    # Iterate until time runs out or no improvement
    best_score = -1
    best_residues = residues.copy()
    
    # Helper to calculate score (first failure index)
    def get_score(counts_arr):
        # Find first index i in 1..L where counts[i] < 2
        # If all >= 2, return L
        # counts is 1-indexed in our usage (counts[1] is for x=1)
        # We check counts[1:]
        failures = np.nonzero(counts_arr[1:] < 2)[0]
        if len(failures) == 0:
            return L
        else:
            return failures[0] + 1 # index in 1..L
        
    current_score = get_score(counts)
    if current_score > best_score:
        best_score = current_score
        best_residues = residues.copy()
        
    stagnation = 0
    max_stagnation = 50
    
    while time.time() - start_time < max_time:
        improved = False
        
        # Shuffle primes to avoid bias? Or iterate in order.
        # Let's iterate in random order to escape local minima faster
        order = np.random.permutation(n_primes)
        
        for i in order:
            if time.time() - start_time > max_time:
                break
                
            p = primes[i]
            r_old = residues[i]
            
            # Remove current residue's contribution
            mods = indices % p
            mask_old = (mods == r_old)
            counts[indices[mask_old]] -= 1
            
            # Find best new residue
            is_weak = (counts[1:] < 2)
            weights = is_weak.astype(np.float32)
            gains = np.bincount(mods, weights=weights, minlength=p)
            
            # To avoid picking the same residue if it was best before, 
            # and to add some randomness, we can perturb gains slightly
            # or just pick argmax. 
            # If we pick argmax, and it's the same as r_old, we revert.
            # Let's try to pick the best residue.
            best_r = np.argmax(gains)
            
            # If best_r is same as r_old, no change.
            # We can also try to force a different residue if stuck?
            # For now, just pick best.
            
            if best_r == r_old:
                # Revert
                counts[indices[mask_old]] += 1
                continue
                
            # Apply new residue
            mask_new = (mods == best_r)
            counts[indices[mask_new]] += 1
            
            new_score = get_score(counts)
            
            if new_score > current_score:
                current_score = new_score
                residues[i] = best_r
                improved = True
                stagnation = 0
                if current_score > best_score:
                    best_score = current_score
                    best_residues = residues.copy()
            else:
                # Revert
                counts[indices[mask_new]] -= 1
                counts[indices[mask_old]] += 1
                stagnation += 1
                
        if not improved:
            stagnation += 1
            if stagnation > max_stagnation:
                # Restart with random residues? Or just break?
                # Let's break and rely on multiple restarts if we had them, 
                # but here we have one long run. 
                # Let's try a random restart of the local search part?
                # Or just break.
                break
                
    return best_residues.tolist()
