"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # Generate candidate vectors
    candidates = []
    
    # All vectors with exactly 2 non-zero entries (±1) - norm² = 2
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                candidates.append(v)
    
    # All vectors with exactly 3 non-zero entries (±1) - norm² = 3
    for i, j, k in itertools.combinations(range(D), 3):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    v = [0] * D
                    v[i], v[j], v[k] = si, sj, sk
                    candidates.append(v)
    
    candidates = np.array(candidates, dtype=np.int64)
    n = len(candidates)
    
    # Compute norms squared
    norms_sq = np.sum(candidates ** 2, axis=1)
    
    # Compute all pairwise dot products
    dots = candidates @ candidates.T
    
    # Conflict: for vectors i, j, conflict if ||v_i - v_j|| < max(||v_i||, ||v_j||)
    # i.e., norms_sq[i] + norms_sq[j] - 2*dots[i,j] < max(norms_sq[i], norms_sq[j])
    # i.e., 2*dots[i,j] > norms_sq[i] + norms_sq[j] - max(norms_sq[i], norms_sq[j])
    # i.e., 2*dots[i,j] > min(norms_sq[i], norms_sq[j])
    
    conflict = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i+1, n):
            if 2 * dots[i, j] > min(norms_sq[i], norms_sq[j]):
                conflict[i, j] = True
                conflict[j, i] = True
    
    # Greedy selection: pick vectors that don't conflict with already-picked ones
    selected = set()
    # Sort candidates by degree (fewest conflicts first) for better greedy
    degrees = np.sum(conflict, axis=1)
    order = np.argsort(degrees)
    
    for idx in order:
        if time.time() > deadline:
            break
        # Check if idx conflicts with any selected
        if not np.any(conflict[idx, list(selected)]):
            selected.add(idx)
    
    best_set = set(selected)
    best_count = len(best_set)
    
    # Local search: try to add unselected vectors, or swap
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try adding each unselected vector
        for idx in range(n):
            if idx in best_set:
                continue
            if not np.any(conflict[idx, list(best_set)]):
                best_set.add(idx)
                improved = True
        
        # Try swapping: remove one, add another
        if time.time() > deadline - 30:
            break
        
        for idx in range(n):
            if idx in best_set:
                continue
            # Try removing each selected vector
            for rem in list(best_set):
                if time.time() > deadline:
                    break
                new_set = best_set - {rem}
                if not np.any(conflict[idx, list(new_set)]):
                    # Check if this swap improves (it doesn't change count, but might enable future adds)
                    # Actually, swap keeps same count. Let's try: remove rem, add idx, then try to add more
                    candidate_set = best_set - {rem} | {idx}
                    # Try to greedily add more
                    added = 0
                    for other in range(n):
                        if other in candidate_set:
                            continue
                        if not np.any(conflict[other, list(candidate_set)]):
                            candidate_set.add(other)
                            added += 1
                    if len(candidate_set) > best_count:
                        best_set = candidate_set
                        best_count = len(best_set)
                        improved = True
                        break
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
            if improved:
                break
    
    result = candidates[list(best_set)]
    return result.astype(np.int64)

# EVOLVE-BLOCK-END
