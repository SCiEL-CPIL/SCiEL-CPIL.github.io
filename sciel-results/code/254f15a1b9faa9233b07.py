import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    # Generate candidates with squared norm 4
    candidates = []

    # Type 1: One 2, rest 0: ±2e_i
    for i in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[i] = 2 * s
            candidates.append(v)

    # Type 2: Four 1s, rest 0: ±e_i ± e_j ± e_k ± e_l
    for idxs in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            candidates.append(v)

    candidates = np.array(candidates, dtype=np.int32)
    N = len(candidates)

    # Compute dot product matrix M[i,j] = candidates[i] . candidates[j]
    # Constraint: for i != j, M[i,j] <= 2
    M = candidates @ candidates.T

    # Build compatibility matrix: compat[i,j] = True if M[i,j] <= 2
    # Note: M[i,i] = 4, so we must exclude self or handle it.
    # We want i != j.
    compat = (M <= 2)
    # Ensure diagonal is False for the "available" mask logic, 
    # but for the final check we just need pairwise.
    # Let's set diagonal to False so we don't select the same vector twice.
    np.fill_diagonal(compat, False)

    best_set = None
    best_count = 0

    rng = np.random.default_rng(42)

    # Strategy: Masked Greedy with degree-based heuristic
    # We maintain a mask of "available" candidates.
    # Initially all True.
    # Pick a candidate, add to set, AND mask with its compatibility row.
    
    # Precompute degrees (number of conflicts) for each candidate
    # Lower degree = fewer conflicts = easier to place
    degrees = np.sum(~compat, axis=1)
    
    num_trials = 50
    for trial in range(num_trials):
        if time.time() > deadline - 5:
            break
            
        if trial == 0:
            # Deterministic order: by degree (fewest conflicts first)
            order = np.argsort(degrees, kind='stable')
        elif trial == 1:
            # Reverse degree order
            order = np.argsort(-degrees, kind='stable')
        elif trial == 2:
            # Index order
            order = np.arange(N)
        else:
            # Random perturbation: mix of degree-based and random
            if trial % 3 == 0:
                # Pure random
                order = rng.permutation(N)
            elif trial % 3 == 1:
                # Degree-based with small random perturbation
                noise = rng.random(N) * 0.1
                order = np.argsort(degrees + noise, kind='stable')
            else:
                # Reverse degree-based with small random perturbation
                noise = rng.random(N) * 0.1
                order = np.argsort(-degrees + noise, kind='stable')

        selected = []
        available = np.ones(N, dtype=bool)
        
        # We iterate through the order, but we only pick if available
        # This is a greedy independent set construction.
        # To make it faster, we can just pick the first available one in the order.
        
        for idx in order:
            if not available[idx]:
                continue
                
            # Select idx
            selected.append(idx)
            # Update available: only those compatible with idx remain
            # available = available & compat[idx, :]
            # But we must also remove idx itself (already handled by diag=False)
            available = available & compat[idx, :]
            
            # Optimization: if no available candidates left, break
            if not np.any(available):
                break

        count = len(selected)
        if count > best_count:
            best_count = count
            best_set = selected[:]

    # If we found a set, return it
    if best_set is not None:
        return candidates[best_set].astype(np.int64)

    # Fallback: just the Type 1 vectors (22 vectors)
    # They are all orthogonal, so dot product 0 <= 2.
    return candidates[:22].astype(np.int64)
