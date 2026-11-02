import numpy as np
import itertools
import time
import random

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_vectors = []
    
    def solve_for_weight(k):
        # Generate candidates
        candidates = []
        indices_combos = list(itertools.combinations(range(D), k))
        for combo in indices_combos:
            for signs in itertools.product([1, -1], repeat=k):
                v = [0] * D
                for idx, s in zip(combo, signs):
                    v[idx] = s
                candidates.append(v)
        
        if not candidates:
            return []
            
        V = np.array(candidates, dtype=np.int8)
        N = len(V)
        
        # Compute dot products
        # V is N x D
        # Dot = V @ V.T
        # This is N x N
        if N > 8000:
             # Too big for dense matrix? 8000^2 = 64M. 512MB. Risky.
             # Use greedy loop
             return greedy_solve(candidates, k)
        
        dots = V @ V.T
        threshold = k / 2.0
        # Conflict if dot > threshold
        # Note: diagonal is k, which is > k/2. We ignore self.
        conflict_mask = dots > threshold
        
        # Greedy MIS using conflict mask
        # We want to pick a set of indices such that no two are connected in conflict_mask
        # Heuristic: pick node with min degree in remaining graph
        
        remaining = np.ones(N, dtype=bool)
        selected_indices = []
        
        while np.any(remaining):
            # Count conflicts for each remaining node
            # conflict_mask[:, remaining] sum
            degrees = np.sum(conflict_mask[:, remaining], axis=1)
            # Only consider remaining nodes
            degrees[~remaining] = np.inf
            # Pick min degree
            idx = np.argmin(degrees)
            if not remaining[idx]:
                break
            selected_indices.append(idx)
            # Remove idx and its neighbors
            remaining[idx] = False
            neighbors = np.where(conflict_mask[idx, :])[0]
            remaining[neighbors] = False
            
        return [candidates[i] for i in selected_indices]

    def greedy_solve(candidates, k):
        random.shuffle(candidates)
        selected = []
        threshold = k / 2.0
        for v in candidates:
            valid = True
            for u in selected:
                dot = 0
                for i in range(D):
                    dot += u[i] * v[i]
                if dot > threshold:
                    valid = False
                    break
            if valid:
                selected.append(v)
        return selected

    for k in range(2, D + 1):
        if time.time() > deadline - 5.0:
            break
        v = solve_for_weight(k)
        if len(v) > len(best_vectors):
            best_vectors = v

    return np.array(best_vectors, dtype=np.int64)
