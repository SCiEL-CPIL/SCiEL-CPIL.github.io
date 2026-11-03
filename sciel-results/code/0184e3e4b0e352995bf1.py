"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
import random

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # Generate all candidate vectors with 4 ±1 entries (norm² = 4)
    all_vectors = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for p, s in zip(positions, signs):
                v[p] = s
            all_vectors.append(v)
    
    all_vectors = np.array(all_vectors, dtype=np.int64)
    n_candidates = len(all_vectors)
    
    # Compute inner product matrix
    inner_matrix = all_vectors @ all_vectors.T  # shape (n, n)
    
    # Compatibility: inner[i,j] <= 2 means compatible
    # We want a large clique in the compatibility graph
    
    np.random.seed(42)
    
    best_size = 0
    best_mask = None
    
    def greedy_clique():
        remaining = np.ones(n_candidates, dtype=bool)
        selected = []
        while remaining.any():
            candidates_remaining = np.where(remaining)[0]
            idx = candidates_remaining[np.random.randint(len(candidates_remaining))]
            selected.append(idx)
            # Remove incompatible neighbors
            remaining[inner_matrix[idx] > 2] = False
            remaining[idx] = False
        return selected
    
    # Multiple random greedy trials
    n_trials = 200
    for trial in range(n_trials):
        if time.time() > deadline:
            break
        selected = greedy_clique()
        if len(selected) > best_size:
            best_size = len(selected)
            best_mask = np.zeros(n_candidates, dtype=bool)
            best_mask[selected] = True
    
    # Local search: try to improve by swapping
    if best_mask is not None and time.time() < deadline - 5:
        mask = best_mask.copy()
        selected_set = set(np.where(mask)[0].tolist())
        unselected_set = set(np.where(~mask)[0].tolist())
        
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            # Try to add unselected vertices
            for idx in list(unselected_set):
                if time.time() > deadline - 2:
                    break
                # Check if idx is compatible with all selected
                if np.all(inner_matrix[idx, list(selected_set)] <= 2):
                    mask[idx] = True
                    selected_set.add(idx)
                    unselected_set.discard(idx)
                    improved = True
                    best_size += 1
            # Try swaps: remove one selected, add two unselected
            if not improved and time.time() < deadline - 2:
                for sel_idx in list(selected_set):
                    if time.time() > deadline - 2:
                        break
                    # Temporarily remove
                    selected_set.discard(sel_idx)
                    new_adds = []
                    for uns_idx in list(unselected_set):
                        if time.time() > deadline - 2:
                            break
                        # Check compatibility with remaining selected + new_adds
                        compat = True
                        for s in selected_set:
                            if inner_matrix[uns_idx, s] > 2:
                                compat = False
                                break
                        if not compat:
                            continue
                        for a in new_adds:
                            if inner_matrix[uns_idx, a] > 2:
                                compat = False
                                break
                        if compat:
                            new_adds.append(uns_idx)
                        if len(new_adds) >= 2:
                            break
                    if len(new_adds) >= 2:
                        for a in new_adds[:2]:
                            mask[a] = True
                            selected_set.add(a)
                            unselected_set.discard(a)
                        mask[sel_idx] = False
                        unselected_set.add(sel_idx)
                        best_size += 1
                        improved = True
                    else:
                        selected_set.add(sel_idx)  # restore
    
    if best_mask is None or best_size == 0:
        # Fallback: return seed result
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
    
    result = all_vectors[best_mask]
    return result

# EVOLVE-BLOCK-END
