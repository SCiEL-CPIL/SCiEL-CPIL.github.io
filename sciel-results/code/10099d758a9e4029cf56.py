import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    # 1. Primes
    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                sieve[i*i::i] = [False] * len(sieve[i*i::i])
        return [i for i, is_p in enumerate(sieve) if is_p]
        
    primes = get_primes(1000)
    n_primes = len(primes)
    
    L_max = 2000 # Search horizon. We aim for L >= 1000.
    
    # 2. Precompute positions
    # pos[i][r] is a numpy array of indices x in [1, L_max] such that x % p == r
    # Note: if r==0, indices are p, 2p, ...
    pos = []
    for p in primes:
        p_list = []
        for r in range(p):
            if r == 0:
                arr = np.arange(p, L_max + 1, p)
            else:
                arr = np.arange(r, L_max + 1, p)
            p_list.append(arr)
        pos.append(p_list)
        
    # Helper: Fast find_L
    # Returns the largest L such that coverage[1..L] >= 2
    def fast_find_L(coverage):
        # coverage is 1-indexed effectively, we look at 1..L_max
        # We want the first index x >= 1 where coverage[x] < 2
        # If none, return L_max
        # Using np.argmax on (coverage < 2) finds the first True.
        # But we only care about 1..L_max.
        mask = coverage[1:L_max+1] < 2
        if not np.any(mask):
            return L_max
        first_fail = np.argmax(mask)
        return first_fail # This is the 0-based index in the slice, which corresponds to x = first_fail + 1 - 1 = first_fail?
        # Wait.
        # coverage[1] is index 0 in slice.
        # If coverage[1] < 2, mask[0] is True, argmax is 0. L should be 0.
        # If coverage[1]>=2, coverage[2]<2, mask[0]=F, mask[1]=T, argmax=1. L should be 1.
        # So L = argmax(mask).
        
    def evaluate(residues):
        coverage = np.zeros(L_max + 1, dtype=np.int16)
        for i in range(n_primes):
            coverage[pos[i][residues[i]]] += 1
        return fast_find_L(coverage)
        
    # 3. Optimization Strategy: Multi-Start Local Search with Repair
    rng = random.Random(42)
    best_L = 0
    best_residues = None
    
    # Initial greedy seed
    residues = [0] * n_primes
    coverage = np.zeros(L_max + 1, dtype=np.int16)
    
    # Greedy initialization: process primes in order of size (small to large)
    # Small primes have more residues but cover more points? No, small primes cover *denser* points.
    # Actually, for maximizing L, we want to cover the prefix [1..L].
    # Let's just do a simple greedy: for each prime, pick residue that maximizes coverage of [1..current_L+10]
    # Or just random start and let local search do the work.
    
    # We will run multiple restarts.
    num_restarts = 20
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        # Initialize
        if restart == 0:
            # Greedy init
            residues = [0] * n_primes
            coverage = np.zeros(L_max + 1, dtype=np.int16)
            # Sort indices by prime size? 
            # Let's process small primes first to build a base.
            order = list(range(n_primes))
            # Actually, random init is often better for escaping local minima if we have good local search.
            # But a good start helps.
            for i in order:
                p = primes[i]
                best_r = 0
                best_gain = -1
                # Maximize coverage in [1, 1000]
                # Gain = number of x in [1, 1000] that become covered (0->1) or double covered (1->2)
                # Since we start empty, gain is just count of x in pos[i][r] within [1, 1000]
                # Wait, coverage is global.
                # Let's just pick r that maximizes sum(coverage[x] < 2) for x in pos[i][r]
                for r in range(p):
                    gain = np.sum(coverage[pos[i][r]] < 2)
                    if gain > best_gain:
                        best_gain = gain
                        best_r = r
                residues[i] = best_r
                coverage[pos[i][best_r]] += 1
        else:
            # Random init
            residues = [rng.randrange(p) for p in primes]
            coverage = np.zeros(L_max + 1, dtype=np.int16)
            for i in range(n_primes):
                coverage[pos[i][residues[i]]] += 1
        
        current_L = fast_find_L(coverage)
        
        # Local Search Loop
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # 1. Standard Local Search: Try changing each prime's residue to best one
            # To speed up, we only check residues that might improve L.
            # But checking all is safer.
            # Optimization: Only check if the new residue covers current_L+1?
            # No, we might need to fix a hole earlier to allow extending later?
            # No, L is the *first* hole. If we fix a hole at L+1, L increases.
            # If we create a hole at L-1, L decreases.
            # So we want to maximize the *first* hole.
            
            # Let's iterate through primes.
            # For each prime, try all residues.
            # Keep the one that maximizes L.
            
            # To make this faster, we can precompute which residues cover current_L+1.
            # But we need to ensure we don't break [1..current_L].
            
            # Let's do a "Best Improvement" local search.
            # For each i, find r* that maximizes L.
            
            # Optimization: If current_L is high, most residues are bad.
            # We can filter candidates.
            
            for i in range(n_primes):
                p = primes[i]
                old_r = residues[i]
                old_pos = pos[i][old_r]
                
                best_local_L = current_L
                best_local_r = old_r
                
                # Try all residues
                # We can skip if p is large and we have limited time, but 168 primes is small.
                for r in range(p):
                    if r == old_r:
                        continue
                    
                    new_pos = pos[i][r]
                    
                    # Calculate new L efficiently?
                    # We need to update coverage.
                    # coverage[old_pos] -= 1
                    # coverage[new_pos] += 1
                    
                    # We can compute the new L without fully updating coverage array if we track deficits?
                    # It's complex. Given L_max=2000, updating array is fast.
                    
                    coverage[old_pos] -= 1
                    coverage[new_pos] += 1
                    
                    new_L = fast_find_L(coverage)
                    
                    if new_L > best_local_L:
                        best_local_L = new_L
                        best_local_r = r
                    else:
                        # Revert if not better, to keep state clean for next trial?
                        # Or just revert at the end?
                        # Reverting immediately is safer.
                        coverage[new_pos] -= 1
                        coverage[old_pos] += 1
                        
                if best_local_r != old_r:
                    # Apply best move
                    coverage[pos[i][old_r]] -= 1
                    coverage[pos[i][best_local_r]] += 1
                    residues[i] = best_local_r
                    current_L = best_local_L
                    improved = True
                    
                    if current_L > best_L:
                        best_L = current_L
                        best_residues = residues[:]
                        
                    # If we improved L, we can break early in this pass?
                    # No, other primes might allow further improvement.
                    
            # 2. Repair Phase: If stuck, try to specifically fix L+1
            if not improved and current_L < L_max:
                target = current_L + 1
                # Find primes that do NOT cover target
                # i.e., residues[i] != target % primes[i]
                candidates = []
                for i in range(n_primes):
                    if residues[i] != target % primes[i]:
                        candidates.append(i)
                
                if candidates:
                    # Try each candidate, see if it extends L without breaking [1..L]
                    best_repair_L = current_L
                    best_repair_i = -1
                    best_repair_r = -1
                    
                    for i in candidates:
                        p = primes[i]
                        old_r = residues[i]
                        new_r = target % p
                        
                        old_pos = pos[i][old_r]
                        new_pos = pos[i][new_r]
                        
                        coverage[old_pos] -= 1
                        coverage[new_pos] += 1
                        
                        new_L = fast_find_L(coverage)
                        
                        if new_L > best_repair_L:
                            best_repair_L = new_L
                            best_repair_i = i
                            best_repair_r = new_r
                        else:
                            # Revert
                            coverage[new_pos] -= 1
                            coverage[old_pos] += 1
                    
                    if best_repair_i != -1:
                        # Apply repair
                        i = best_repair_i
                        old_r = residues[i]
                        new_r = best_repair_r
                        coverage[pos[i][old_r]] -= 1
                        coverage[pos[i][new_r]] += 1
                        residues[i] = new_r
                        current_L = best_repair_L
                        improved = True # Force another pass of local search
                        
                        if current_L > best_L:
                            best_L = current_L
                            best_residues = residues[:]

    # Final check
    if best_residues is None:
        return [0] * n_primes
    return [int(r) for r in best_residues]
