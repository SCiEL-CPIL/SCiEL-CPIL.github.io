import time
import random
import numpy as np
import math

def construct():
    n = 256
    t0 = time.time()
    deadline = t0 + 55.0

    # Precompute APs
    ap_indices = []
    for d in range(1, n + 1):
        for a in range(n):
            idx = np.arange(a, n, d)
            if len(idx) > 0:
                ap_indices.append(idx)
    
    M = len(ap_indices)
    
    affected_aps = [[] for _ in range(n)]
    for k, idx in enumerate(ap_indices):
        for pos in idx:
            affected_aps[pos].append(k)
    
    affected_aps_np = [np.array(ks, dtype=np.int32) if ks else np.array([], dtype=np.int32) for ks in affected_aps]

    def compute_ap_sums(s):
        sums = np.zeros(M, dtype=np.int32)
        for k in range(M):
            sums[k] = np.sum(s[ap_indices[k]])
        return sums

    # Multi-start with different initializations
    rng = random.Random(12345)
    
    best_s = None
    best_max = n  # worst case
    
    # Try several different initialization strategies
    init_strategies = []
    
    # Strategy 1: Thue-Morse
    def thue_morse(n):
        signs = np.zeros(n, dtype=np.int8)
        for i in range(n):
            signs[i] = 1 if bin(i).count('1') % 2 == 0 else -1
        return signs
    
    init_strategies.append(thue_morse(n))
    
    # Strategy 2: Random with balanced constraint
    for _ in range(3):
        signs = np.random.choice([-1, 1], size=n)
        init_strategies.append(signs)
    
    # Strategy 3: Block-based alternating
    for block in [16, 32, 64]:
        signs = np.zeros(n, dtype=np.int8)
        for i in range(n):
            signs[i] = 1 if (i // block) % 2 == 0 else -1
        init_strategies.append(signs)
    
    # For each initialization, do aggressive local search
    for init_s in init_strategies:
        if time.time() > deadline - 5.0:
            break
        
        s = init_s.copy()
        ap_sums = compute_ap_sums(s)
        current_max = int(np.max(np.abs(ap_sums)))
        
        if current_max < best_max:
            best_max = current_max
            best_s = s.copy()
        
        # Phase 1: Greedy descent - flip positions that reduce max discrepancy
        for iteration in range(2000):
            if time.time() > deadline - 5.0:
                break
            
            # Find positions where flipping would help
            best_pos = -1
            best_new_max = current_max
            
            # Sample a subset of positions to keep it fast
            sample_size = min(32, n)
            candidates = rng.sample(range(n), sample_size)
            
            for i in candidates:
                old_sign = s[i]
                new_sign = -old_sign
                delta = -2 * old_sign
                
                # Compute what the new max would be
                ks = affected_aps_np[i]
                if len(ks) == 0:
                    continue
                    
                # Check if any affected AP would exceed current max
                new_sums_affected = ap_sums[ks] + delta
                max_affected = int(np.max(np.abs(new_sums_affected)))
                
                # Estimate new global max: max of (unaffected max, affected new max)
                # This is an approximation - the true new max could be lower
                # if the current max was in an affected AP
                estimated_new_max = max(max_affected, current_max)
                
                # If this flip definitely helps, take it
                if estimated_new_max < best_new_max:
                    best_new_max = estimated_new_max
                    best_pos = i
            
            if best_pos >= 0 and best_new_max < current_max:
                # Apply the flip
                i = best_pos
                old_sign = s[i]
                s[i] = -old_sign
                delta = -2 * old_sign
                ks = affected_aps_np[i]
                if len(ks) > 0:
                    ap_sums[ks] += delta
                current_max = best_new_max
                
                if current_max < best_max:
                    best_max = current_max
                    best_s = s.copy()
            else:
                # No improvement found in sample, try a random flip with acceptance
                i = rng.randint(0, n - 1)
                old_sign = s[i]
                new_sign = -old_sign
                delta = -2 * old_sign
                
                ks = affected_aps_np[i]
                if len(ks) > 0:
                    ap_sums[ks] += delta
                
                new_max = int(np.max(np.abs(ap_sums)))
                
                # Accept if it doesn't make things too bad
                if new_max <= current_max + 1:
                    current_max = new_max
                else:
                    # Revert
                    s[i] = old_sign
                    ap_sums[ks] -= delta
        
        # Phase 2: Multi-flip moves for escaping local optima
        for iteration in range(500):
            if time.time() > deadline - 5.0:
                break
            
            # Flip 2-4 random positions
            num_flips = rng.randint(2, 4)
            positions = rng.sample(range(n), num_flips)
            
            old_signs = s[positions].copy()
            s[positions] = -old_signs
            
            # Update AP sums
            for i in positions:
                delta = -2 * old_signs[positions.tolist().index(i)]
                ks = affected_aps_np[i]
                if len(ks) > 0:
                    ap_sums[ks] += delta
            
            new_max = int(np.max(np.abs(ap_sums)))
            
            if new_max < current_max:
                current_max = new_max
                if current_max < best_max:
                    best_max = current_max
                    best_s = s.copy()
            else:
                # Revert
                for idx, i in enumerate(positions):
                    s[i] = old_signs[idx]
                    delta = 2 * old_signs[idx]
                    ks = affected_aps_np[i]
                    if len(ks) > 0:
                        ap_sums[ks] += delta
    
    if best_s is None:
        best_s = init_strategies[0]
    
    return [int(x) for x in best_s]
