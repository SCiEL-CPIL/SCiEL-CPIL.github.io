import numpy as np
import time
from itertools import combinations, product

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    D = 11
    
    # Generate all weight 3 vectors
    # Support: choose 3 indices from 11
    # Signs: 2^3 = 8 patterns
    candidates = []
    indices = list(range(D))
    
    for support in combinations(indices, 3):
        for signs in product([1, -1], repeat=3):
            v = np.zeros(D, dtype=np.int8)
            for idx, s in zip(support, signs):
                v[idx] = s
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int8)
    N = len(candidates)
    
    # Precompute incompatibility matrix: two vectors are incompatible if their dot product > 1
    # For weight-3 vectors: ||u-v||^2 = 6 - 2*u.v >= 3 iff u.v <= 1
    # So incompatible if u.v > 1 (i.e., u.v >= 2)
    dots = candidates @ candidates.T
    incompat = (dots > 1).astype(np.int8)
    np.fill_diagonal(incompat, 0)
    
    # Use a better greedy strategy: start with the candidate that has the fewest conflicts
    # This is a heuristic for maximum independent set
    
    # Count conflicts for each candidate
    conflict_counts = incompat.sum(axis=1)
    
    # Sort candidates by conflict count (ascending) - prefer those with fewer conflicts
    order = np.argsort(conflict_counts)
    
    # Greedy selection using the sorted order
    selected = []
    selected_set = set()
    
    for i in order:
        if time.time() > deadline - 10.0:
            break
            
        # Check if this candidate is compatible with all selected candidates
        if i in selected_set:
            continue
            
        # Check compatibility with all selected vectors
        valid = True
        for j in selected:
            if incompat[i, j] == 1:
                valid = False
                break
        
        if valid:
            selected.append(i)
            selected_set.add(i)
    
    # Local search: try to improve by swapping
    # For each selected vector, try replacing it with an unselected one
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        for idx in range(len(selected)):
            if time.time() > deadline - 5.0:
                break
            current = selected[idx]
            # Try to replace current with a better candidate
            best_swap = None
            best_score = 0
            
            # Try unselected candidates
            for i in range(N):
                if i in selected_set:
                    continue
                
                # Check if i is compatible with all other selected vectors
                compatible = True
                for j, s in enumerate(selected):
                    if j == idx:
                        continue
                    if incompat[i, s] == 1:
                        compatible = False
                        break
                
                if compatible:
                    # Score: how many conflicts does i have with unselected candidates?
                    # We want to minimize conflicts to allow more future additions
                    # But for now, just accept if it works
                    best_swap = i
                    break
            
            if best_swap is not None and best_swap != current:
                selected[idx] = best_swap
                selected_set.remove(current)
                selected_set.add(best_swap)
                improved = True
    
    if not selected:
        # Fallback to seed solution
        vectors = []
        for i in range(D):
            for j in range(i + 1, D):
                for s1 in [1, -1]:
                    for s2 in [1, -1]:
                        v = np.zeros(D, dtype=np.int8)
                        v[i] = s1
                        v[j] = s2
                        vectors.append(v)
        return np.array(vectors, dtype=np.int8)
        
    return candidates[selected]
