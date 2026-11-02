import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    # Generate candidates with squared norm 4
    candidates = []
    
    # Type 1: One 2, rest 0
    for i in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[i] = 2 * s
            candidates.append(v)
            
    # Type 2: Four 1s, rest 0
    for idxs in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int32)
    N = len(candidates)
    
    # Compute conflict matrix
    # M[i, j] = candidates[i] . candidates[j]
    # Conflict if M[i, j] > 2
    M = candidates @ candidates.T
    C = (M > 2).astype(np.int8)
    # Set diagonal to 0 (no self-conflict in logic, but we don't pick same node)
    np.fill_diagonal(C, 0)
    
    # Precompute degrees for sorting (optional, but helpful for greedy)
    degrees = C.sum(axis=1)
    
    best_set = []
    rng = np.random.default_rng(42)
    
    # Helper: Greedy completion from a starting set
    def greedy_complete(start_indices, order=None):
        if order is None:
            # Sort by degree (ascending) to pick "easy" nodes first?
            # Or random?
            # Let's use random for ILS diversity
            order = rng.permutation(N)
            
        selected_mask = np.zeros(N, dtype=bool)
        selected = []
        
        # Initialize counts based on start_indices
        if start_indices:
            selected_mask[start_indices] = True
            selected = list(start_indices)
            # counts[j] = sum of C[i, j] for i in start_indices
            counts = C[start_indices, :].sum(axis=0).astype(np.int32)
        else:
            counts = np.zeros(N, dtype=np.int32)
            
        for idx in order:
            if selected_mask[idx]:
                continue
            if counts[idx] > 0:
                continue
            
            # Add idx
            selected.append(idx)
            selected_mask[idx] = True
            counts += C[idx, :]
            
        return selected

    # Initial greedy passes with different orderings
    for _ in range(10):
        if time.time() > deadline:
            break
        # Try degree-sorted order (lowest degree first)
        order = np.argsort(degrees)
        s = greedy_complete([], order)
        if len(s) > len(best_set):
            best_set = s
            
    # Iterated Local Search
    # Strategy: Remove k random nodes, re-greedy
    k_removal = 2
    while time.time() < deadline:
        if not best_set:
            break
            
        # Perturb: remove k random nodes
        start_indices = list(best_set)
        if len(start_indices) <= k_removal:
            # If set is too small, just restart
            s = greedy_complete([])
        else:
            # Remove k random nodes
            to_remove = rng.choice(len(start_indices), size=k_removal, replace=False)
            to_remove_indices = [start_indices[i] for i in to_remove]
            start_indices = [x for x in start_indices if x not in set(to_remove_indices)]
            
            # Re-greedy
            # Use random order for re-greedy to escape local optima
            s = greedy_complete(start_indices)
            
        if len(s) > len(best_set):
            best_set = s
            # Optional: reset k_removal or adapt?
            # For now, keep fixed.
            
    if not best_set:
        # Fallback
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
        
    return candidates[best_set].astype(np.int64)
