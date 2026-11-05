import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s limit, 10s margin
    
    # Generate primes <= 1000
    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                sieve[i*i::i] = [False] * len(sieve[i*i::i])
        return [i for i, is_p in enumerate(sieve) if is_p]
    
    primes = get_primes(1000)
    n_primes = len(primes)
    assert n_primes == 168
    
    # Search horizon for coverage
    L_max = 5000
    
    # Precompute positions for each prime and residue
    # positions[i] is a list of arrays: positions[i][r] = np.array of x in [1, L_max] with x ≡ r (mod p)
    positions = []
    for i, p in enumerate(primes):
        pos_list = []
        for r in range(p):
            if r == 0:
                pos_list.append(np.arange(p, L_max + 1, p, dtype=np.int32))
            else:
                pos_list.append(np.arange(r, L_max + 1, p, dtype=np.int32))
        positions.append(pos_list)
    
    def find_L(coverage):
        """Find largest L such that coverage[x] >= 2 for all x in [1, L]."""
        # Vectorized check: find first index where coverage < 2
        # coverage is 1-indexed effectively (index 0 unused)
        # We want the longest prefix [1..L] where coverage >= 2
        valid = coverage >= 2
        # Find first False in valid[1:]
        # If all True, L = L_max
        # Use np.argmax on ~valid to find first False
        # But we need to handle the case where valid[1:] is all True
        # Let's just loop, it's fast enough (5000 iterations)
        L = 0
        for x in range(1, L_max + 1):
            if coverage[x] >= 2:
                L = x
            else:
                break
        return L
    
    # Greedy initialization with lazy evaluation for submodular maximization
    # We want to maximize the "doubly covered prefix length".
    # A good proxy is to maximize the number of positions in [1, K] that have count >= 2.
    # Or better: maximize the minimum count in the prefix? No, that's hard.
    # Let's stick to the "deficit" minimization which is submodular.
    # Deficit(x) = max(0, 2 - coverage[x])
    # Total Deficit = sum(Deficit(x)) for x in 1..L_max
    # Minimizing total deficit is a good proxy for maximizing L.
    
    residues = [0] * n_primes
    coverage = np.zeros(L_max + 1, dtype=np.int32)
    
    # Greedy phase: pick residues to minimize deficit in the early range [1, 200]
    # This focuses on the bottleneck area.
    K_FOCUS = 200
    
    # Precompute "gain" for each prime/residue pair on the focus range
    # gain[i][r] = number of x in [1, K_FOCUS] in positions[i][r] that currently have coverage < 2
    # We will update this lazily as we assign primes.
    
    # Actually, let's just do a standard greedy:
    # For each prime i, pick r that minimizes the total deficit in [1, K_FOCUS] after adding this prime.
    
    # To speed this up, we can precompute the intersection with [1, K_FOCUS]
    focus_positions = []
    for i, p in enumerate(primes):
        focus_pos_list = []
        for r in range(p):
            pos = positions[i][r]
            # Filter to [1, K_FOCUS]
            mask = (pos >= 1) & (pos <= K_FOCUS)
            focus_pos_list.append(pos[mask])
        focus_positions.append(focus_pos_list)
    
    for i, p in enumerate(primes):
        best_r = 0
        best_deficit = float('inf')
        # We can compute the new deficit for each r
        # Current coverage is known.
        # For a candidate r, new_coverage[x] = coverage[x] + (1 if x in positions[i][r] else 0)
        # We only care about x in [1, K_FOCUS] for the decision, but the effect is global.
        # However, for the *choice*, focusing on the bottleneck is usually good.
        # Let's compute the change in deficit for the focus range.
        
        for r in range(p):
            pos = focus_positions[i][r]
            if len(pos) == 0:
                # No change in focus range
                # Deficit change is 0
                # But we need to be careful. If this prime doesn't help the focus, it might be neutral or bad.
                # Let's just calculate the deficit on the focus range for this r.
                # This is O(len(pos)) which is small.
                pass
            
            # Calculate deficit on focus range if we pick r
            # We need to simulate the coverage change on the focus range.
            # Since we are adding one prime, we can just check the positions.
            
            # Optimization: Precompute the "potential" gain?
            # Let's just compute the sum of deficits for the focus range for this r.
            # This is O(K_FOCUS) if we do it naively, but we can do it faster by only looking at affected positions.
            
            # Affected positions in focus range: pos
            # For each x in pos:
            #   old_def = max(0, 2 - coverage[x])
            #   new_def = max(0, 2 - (coverage[x] + 1))
            #   delta_def = new_def - old_def
            #   total_delta += delta_def
            #   # Note: we must not actually update coverage yet
            
            # But what about positions NOT in pos? Their deficit doesn't change.
            # So we only need to sum delta_def for x in pos.
            
            # However, we also need to consider that other primes might have been assigned already.
            # The `coverage` array reflects the current state (primes 0..i-1 assigned).
            
            delta_def = 0
            for x in pos:
                c_old = coverage[x]
                c_new = c_old + 1
                d_old = 0 if c_old >= 2 else (2 - c_old)
                d_new = 0 if c_new >= 2 else (2 - c_new)
                delta_def += (d_new - d_old)
            
            # We want to minimize the total deficit.
            # Since the base deficit is constant for all r, minimizing total_deficit is equivalent to minimizing delta_def.
            # Note: delta_def will be <= 0.
            
            if delta_def < best_deficit:
                best_deficit = delta_def
                best_r = r
        
        residues[i] = best_r
        # Apply the change to coverage
        pos_full = positions[i][best_r]
        coverage[pos_full] += 1
    
    current_L = find_L(coverage)
    best_L = current_L
    best_residues = residues[:]
    
    # Local Search / Simulated Annealing Phase
    # Use the deficit function as the objective.
    # Minimize sum(max(0, 2 - coverage[x])) for x in 1..L_max
    
    T = 10.0
    T_min = 0.01
    alpha = 0.999
    
    rng = random.Random(42)
    
    # Precompute sets for faster delta calculation?
    # Or just use the array slicing.
    
    while time.time() < deadline - 5:
        if T < T_min:
            break
        
        # Pick random prime
        i = rng.randrange(n_primes)
        p = primes[i]
        r_old = residues[i]
        
        # Pick new residue
        r_new = rng.randrange(p)
        if r_old == r_new:
            continue
        
        old_pos = positions[i][r_old]
        new_pos = positions[i][r_new]
        
        # Calculate delta in deficit
        # We need to consider the union of old_pos and new_pos.
        # To do this efficiently, we can create a boolean mask or just iterate over the union.
        # Since L_max is 5000, and p is at most 1000, the sets are small.
        # Union size is at most len(old_pos) + len(new_pos).
        
        # Create a set of unique indices in the union for O(1) lookup?
        # Or just iterate and check.
        
        # Let's use a temporary array to mark affected indices?
        # Or just iterate over the union of indices.
        
        # Efficient way:
        # affected = set(old_pos).union(set(new_pos))
        # This is O(N) where N is the size of the sets.
        
        old_set = set(old_pos)
        new_set = set(new_pos)
        affected = old_set.union(new_set)
        
        delta_E = 0
        for x in affected:
            c0 = coverage[x]
            c1 = c0
            if x in old_set:
                c1 -= 1
            if x in new_set:
                c1 += 1
            
            d0 = 0 if c0 >= 2 else (2 - c0)
            d1 = 0 if c1 >= 2 else (2 - c1)
            delta_E += (d1 - d0)
        
        # Decide accept/reject
        accept = False
        if delta_E < 0:
            accept = True
        elif delta_E == 0:
            accept = True
        else:
            if delta_E < T * 5:
                if rng.random() < np.exp(-delta_E / T):
                    accept = True
        
        if accept:
            # Apply change
            coverage[old_pos] -= 1
            coverage[new_pos] += 1
            residues[i] = r_new
            
            # Check L improvement periodically or on every improvement
            # Checking L is O(L_max) = 5000 ops.
            # We can check if delta_E < 0, as that's the only time we *might* improve L.
            # Actually, even if delta_E > 0, L could theoretically improve? No, if deficit increases, coverage is worse, so L cannot improve.
            # If delta_E == 0, L might improve or stay same.
            # So we only need to check if delta_E <= 0.
            
            if delta_E <= 0:
                cur_L = find_L(coverage)
                if cur_L > best_L:
                    best_L = cur_L
                    best_residues = residues[:]
        
        T *= alpha
    
    # Final Refinement: Local Search with Best Response
    # For each prime, try all residues and pick the one that minimizes deficit (or maximizes L).
    # This is a 1-step local search.
    
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        for i in range(n_primes):
            if time.time() > deadline - 2:
                break
            p = primes[i]
            old_r = residues[i]
            best_r = old_r
            best_deficit = float('inf')
            
            # Calculate current deficit contribution?
            # No, we calculate the total deficit change.
            # We want to minimize total deficit.
            # So we want to minimize delta_def.
            
            # Actually, let's just compute the new total deficit for the affected positions.
            # Or simpler: compute the new L?
            # Computing L is O(5000). Trying all p residues is O(p * 5000).
            # Sum p over all primes is ~ 76k.
            # 76k * 5000 = 380M ops. Too slow.
            
            # So we stick to deficit minimization for the selection.
            # Then we check L once per prime.
            
            best_delta = float('inf')
            
            old_pos = positions[i][old_r]
            old_set = set(old_pos)
            
            for r in range(p):
                if r == old_r:
                    continue
                new_pos = positions[i][r]
                new_set = set(new_pos)
                affected = old_set.union(new_set)
                
                delta_E = 0
                for x in affected:
                    c0 = coverage[x]
                    c1 = c0
                    if x in old_set:
                        c1 -= 1
                    if x in new_set:
                        c1 += 1
                    d0 = 0 if c0 >= 2 else (2 - c0)
                    d1 = 0 if c1 >= 2 else (2 - c1)
                    delta_E += (d1 - d0)
                
                if delta_E < best_delta:
                    best_delta = delta_E
                    best_r = r
            
            if best_r != old_r:
                # Apply best
                old_pos = positions[i][old_r]
                new_pos = positions[i][best_r]
                coverage[old_pos] -= 1
                coverage[new_pos] += 1
                residues[i] = best_r
                improved = True
                
                # Check L
                cur_L = find_L(coverage)
                if cur_L > best_L:
                    best_L = cur_L
                    best_residues = residues[:]
    
    # Final check
    cur_L = find_L(coverage)
    if cur_L > best_L:
        best_L = cur_L
        best_residues = residues[:]
    
    return [int(r) for r in best_residues]
