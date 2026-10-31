import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 120s limit
    
    # Generate all weight-4 vectors in {-1, 0, 1}^11 (norm^2 = 4)
    candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates.append(v)
    
    vectors = np.array(candidates, dtype=np.int64)
    N = len(vectors)
    
    if N == 0:
        return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)
    
    # Compute norms
    norms2 = (vectors ** 2).sum(axis=1)
    max_norm2 = norms2.max()
    
    # Compute dot products
    dots = vectors @ vectors.T
    
    # dist^2(i,j) = norms2[i] + norms2[j] - 2*dots[i,j]
    dist2 = norms2[:, None] + norms2[None, :] - 2 * dots
    np.fill_diagonal(dist2, 0)
    
    min_dist2 = dist2.min()
    
    if min_dist2 >= max_norm2:
        # All vectors are valid together
        return vectors
    
    # Conflict matrix: dist2[i,j] < max_norm2
    conflicts = dist2 < max_norm2
    np.fill_diagonal(conflicts, False)
    
    # Greedy MIS with minimum degree heuristic, multiple restarts
    def greedy_mis(conflicts, N, seed, deadline):
        alive = np.ones(N, dtype=bool)
        selected = []
        degrees = conflicts.sum(axis=1).astype(np.int32)
        rng = np.random.default_rng(seed)
        
        while np.any(alive) and time.time() < deadline:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] == min_deg]
            chosen = cands[rng.integers(len(cands))]
            selected.append(chosen)
            neighbors = np.where(conflicts[chosen])[0]
            to_remove = np.unique(np.concatenate(([chosen], neighbors)))
            alive[to_remove] = False
            rem = np.where(alive)[0]
            if len(rem) > 0:
                degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)
        
        return selected
    
    def local_improve(selected, conflicts, N, deadline):
        """Try to add more vectors by checking if they conflict with any selected."""
        selected_set = set(selected)
        selected_arr = np.array(selected)
        # Find all non-selected vectors that don't conflict with any selected
        # This is expensive, so do it in batches
        improved = list(selected)
        
        # For each non-selected vector, check if it conflicts with any in selected
        # Use the conflict matrix directly
        for v in range(N):
            if v in selected_set:
                continue
            if time.time() > deadline:
                break
            # Check if v conflicts with any selected vector
            if not np.any(conflicts[v, selected_arr]):
                improved.append(v)
                selected_set.add(v)
                selected_arr = np.append(selected_arr, v)
        
        return improved
    
    best_selected = None
    best_len = 0
    
    # Try multiple seeds
    for seed in range(20):
        if time.time() > deadline - 10:
            break
        selected = greedy_mis(conflicts, N, seed, deadline - 10)
        if len(selected) > best_len:
            best_len = len(selected)
            best_selected = selected
    
    if best_selected is None:
        best_selected = list(range(N))  # fallback
    
    # Local improvement pass
    if time.time() < deadline - 5:
        best_selected = local_improve(best_selected, conflicts, N, deadline - 5)
    
    result_vectors = vectors[best_selected]
    return result_vectors
