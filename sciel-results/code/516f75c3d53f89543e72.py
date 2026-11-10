import numpy as np
import itertools
import time
import random

def kissing_number11() -> np.ndarray:
    d = 11
    # Generate all vectors with exactly 4 non-zero entries (+/- 1)
    # Norm squared is 4.
    # Condition: dot product <= 2.
    
    vectors = []
    # Indices for the 4 non-zero positions
    for support in itertools.combinations(range(d), 4):
        # Signs for these 4 positions
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(d, dtype=np.int64)
            for i, s in zip(support, signs):
                v[i] = s
            vectors.append(v)
            
    vectors = np.array(vectors)
    N_vecs = len(vectors)
    
    best_set = []
    deadline = time.time() + 590 # Safety margin
    
    # We will try multiple random orderings
    # A single greedy pass takes ~0.1s
    # We can do ~1000 passes.
    
    # Pre-compute dot products? 5280x5280 matrix is 27M entries.
    # 27M * 8 bytes = 216 MB. Might be too much or slow to compute.
    # Let's just compute on the fly or use a smarter check.
    
    # Actually, we can pre-compute the "conflict" matrix if memory allows.
    # But 5280^2 dot products of dim 11 is 5280*5280*11 ~ 300M operations.
    # In numpy, this is fast.
    # Let's compute the full dot product matrix M.
    # M[i, j] = vectors[i] . vectors[j]
    
    # 5280 * 5280 matrix of int64 is ~216MB.
    # This might be risky if memory is limited, but usually 512MB+ is available.
    # If it fails, we fall back to on-the-fly.
    
    try:
        # Compute dot products
        # vectors is (5280, 11)
        # M = vectors @ vectors.T
        # This creates a (5280, 5280) array.
        M = vectors @ vectors.T
        # We need M[i, j] <= 2 for all i, j in the set.
        # Note: M[i, i] = 4.
        
        # Greedy selection
        # We want to find a large subset of indices such that for all i, j in subset, M[i, j] <= 2.
        # This is a Maximum Independent Set problem in a graph where edges connect conflicting pairs (M[i,j] > 2).
        # Greedy: pick a node, remove it and its neighbors.
        
        # To make it fast, we can represent the "available" nodes as a bitset or boolean array.
        # However, standard Python loops for 5280 nodes are slow if done naively.
        
        # Let's use a simple greedy with numpy boolean masks.
        # available = np.ones(N_vecs, dtype=bool)
        # selected = []
        # order = np.random.permutation(N_vecs)
        # for idx in order:
        #     if available[idx]:
        #         selected.append(idx)
        #         # Mark conflicts
        #         # Conflicts are where M[idx, :] > 2
        #         # We need to update available: available &= ~(M[idx, :] > 2)
        #         # This is a vectorized operation on a 5280-length array.
        #         # 5280 iterations * 5280 ops = 27M ops. Very fast.
        
        best_size = 0
        best_indices = []
        
        num_restarts = 0
        while time.time() < deadline:
            num_restarts += 1
            available = np.ones(N_vecs, dtype=bool)
            selected = []
            order = np.random.permutation(N_vecs)
            
            for idx in order:
                if available[idx]:
                    selected.append(idx)
                    # Update available
                    # M[idx, :] > 2 indicates conflicts
                    # We want to remove these from available
                    # available = available & ~(M[idx, :] > 2)
                    # Optimization: only update if we haven't already removed them?
                    # The boolean AND is fast.
                    available &= ~(M[idx, :] > 2)
                    
                    if len(selected) > best_size:
                        best_size = len(selected)
                        best_indices = selected[:]
                        
            # If we found a very large set, we might stop early?
            # Theoretical max is likely around 220-300.
            # If best_size > 250, we can stop.
            if best_size > 250:
                break
                
    except MemoryError:
        # Fallback to on-the-fly dot product check
        best_size = 0
        best_indices = []
        
        num_restarts = 0
        while time.time() < deadline:
            num_restarts += 1
            available = np.ones(N_vecs, dtype=bool)
            selected = []
            order = np.random.permutation(N_vecs)
            
            for idx in order:
                if available[idx]:
                    v = vectors[idx]
                    # Check against all currently selected
                    # This is O(K) per step, K grows.
                    # Total O(N*K).
                    # If K ~ 200, N=5280, 1M ops per restart.
                    # 100 restarts = 100M ops. OK.
                    
                    conflict = False
                    for sel_idx in selected:
                        # dot product
                        if np.dot(v, vectors[sel_idx]) > 2:
                            conflict = True
                            break
                    if not conflict:
                        selected.append(idx)
                        # Update available for future checks?
                        # We can just keep a list of selected and check against them.
                        # The 'available' array is not strictly needed if we check against 'selected'.
                        # But 'available' allows us to skip already rejected nodes?
                        # No, a node rejected by one selected node might be compatible with others?
                        # No, if it conflicts with ANY selected node, it cannot be added.
                        # So we can mark it as unavailable.
                        # But we need to know which nodes conflict with the new selection.
                        # This requires dot products with all unselected nodes.
                        # That's expensive.
                        # So the boolean mask approach in the try block is much better.
                        # If we are in the except block, we just use the list.
                        
                        if len(selected) > best_size:
                            best_size = len(selected)
                            best_indices = selected[:]
                        
            if best_size > 250:
                break

    # Construct the result
    if best_indices:
        points = vectors[best_indices]
    else:
        # Fallback to the seed solution if something went wrong
        points = np.array([np.ones(d), -1 * np.ones(d)]).astype(np.int64)
        
    return points
