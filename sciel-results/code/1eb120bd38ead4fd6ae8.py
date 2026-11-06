"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    # Deadline guard: 150s limit, use 140s for safety
    deadline = time.time() + 140.0
    
    # Generate primes up to 1000
    primes = [p for p in range(2, 1001) if all(p % q for q in range(2, int(p ** 0.5) + 1))]
    n_primes = len(primes)
    primes_arr = np.array(primes, dtype=np.int64)
    
    MAX_VAL = 50000  # We'll track coverage up to 50000
    
    # Precompute for each prime p, the indices x in 1..MAX_VAL where x % p == r
    # for each possible r. Store as lists of indices.
    # Actually, we'll compute coverage incrementally using numpy.
    
    # Initial residues: try r_p = 1 for all
    residues = np.ones(n_primes, dtype=np.int64)
    
    def compute_coverage(res):
        """Compute coverage array using numpy."""
        cov = np.zeros(MAX_VAL + 1, dtype=np.int32)
        for i, p in enumerate(primes):
            r = int(res[i])
            if r == 0:
                start = p
            else:
                start = r
            if start > MAX_VAL:
                continue
            # x = r, r+p, r+2p, ... up to MAX_VAL
            indices = np.arange(start, MAX_VAL + 1, p)
            cov[indices] += 1
        return cov
    
    def get_L(cov):
        """Find largest L such that cov[1..L] >= 2."""
        # Find first index where cov < 2
        under = np.where(cov[1:] < 2)[0]
        if len(under) == 0:
            return MAX_VAL
        return int(under[0])
    
    # Initial coverage
    cov = compute_coverage(residues)
    L = get_L(cov)
    best_L = L
    best_residues = residues.copy()
    
    # Precompute index arrays for each prime and residue
    # For prime p and residue r, the indices are r, r+p, r+2p, ...
    # We'll precompute these as numpy arrays for fast updates
    
    # For each prime i, precompute the base index array for residue 0
    # indices for residue r: if r==0, start at p; else start at r
    # indices = np.arange(start, MAX_VAL+1, p)
    
    # Precompute for each prime the list of index arrays for each residue
    # This uses a lot of memory but speeds things up
    prime_index_cache = {}
    for i, p in enumerate(primes):
        prime_index_cache[i] = {}
        for r in range(p):
            if r == 0:
                start = p
            else:
                start = r
            if start <= MAX_VAL:
                prime_index_cache[i][r] = np.arange(start, MAX_VAL + 1, p)
            else:
                prime_index_cache[i][r] = np.array([], dtype=np.int64)
    
    def update_cov(cov, i, old_r, new_r, delta):
        """Update cov by adding delta to indices of prime i with residue new_r,
        and subtracting delta from indices of prime i with residue old_r."""
        old_indices = prime_index_cache[i][old_r]
        new_indices = prime_index_cache[i][new_r]
        if len(old_indices) > 0:
            cov[old_indices] -= delta
        if len(new_indices) > 0:
            cov[new_indices] += delta
    
    # Main optimization loop
    iter_count = 0
    stuck_count = 0
    
    while time.time() < deadline:
        iter_count += 1
        
        if L >= MAX_VAL:
            break
        
        x_target = L + 1
        
        best_move_gain = 0
        best_move_i = -1
        best_move_r = -1
        best_move_old_r = -1
        
        # Try changing residue for each prime to cover x_target
        for i in range(n_primes):
            p = primes[i]
            old_r = int(residues[i])
            
            # To cover x_target, we need new_r = x_target % p
            new_r = x_target % p
            
            if new_r == old_r:
                continue
            
            # Apply update
            update_cov(cov, i, old_r, new_r, 1)
            
            # Check if x_target is now covered
            if cov[x_target] >= 2:
                # Find how far we can extend
                new_L = L
                x_check = x_target + 1
                while x_check <= MAX_VAL and cov[x_check] >= 2:
                    new_L = x_check
                    x_check += 1
                
                if new_L > best_move_gain:
                    best_move_gain = new_L
                    best_move_i = i
                    best_move_r = new_r
                    best_move_old_r = old_r
            
            # Revert
            update_cov(cov, i, old_r, new_r, -1)
        
        if best_move_i != -1 and best_move_gain > L:
            # Apply best move
            residues[best_move_i] = best_move_r
            update_cov(cov, best_move_i, best_move_old_r, best_move_r, 1)
            L = best_move_gain
            stuck_count = 0
            
            if L > best_L:
                best_L = L
                best_residues = residues.copy()
        else:
            # Stuck - try random perturbations to escape local optimum
            stuck_count += 1
            
            if stuck_count > 50:
                # Try random moves
                made_progress = False
                for _ in range(20):
                    if time.time() >= deadline:
                        break
                    
                    i = random.randint(0, n_primes - 1)
                    p = primes[i]
                    old_r = int(residues[i])
                    new_r = random.randint(0, p - 1)
                    
                    if new_r == old_r:
                        continue
                    
                    update_cov(cov, i, old_r, new_r, 1)
                    new_L = get_L(cov)
                    
                    if new_L > L:
                        # Accept
                        residues[i] = new_r
                        L = new_L
                        made_progress = True
                        stuck_count = 0
                        if L > best_L:
                            best_L = L
                            best_residues = residues.copy()
                        break
                    else:
                        # Revert
                        update_cov(cov, i, old_r, new_r, -1)
                
                if not made_progress:
                    # Give up or try more aggressive random restart
                    if stuck_count > 500:
                        break
    
    return best_residues.tolist()
# EVOLVE-BLOCK-END
