import numpy as np
import itertools
import time

D = 11

def _greedy_mis(conflicts, N, deadline, rng_seed=42):
    """Greedy MIS with minimum degree heuristic."""
    alive = np.ones(N, dtype=bool)
    selected = []
    degrees = conflicts.sum(axis=1).astype(np.int32)
    rng = np.random.default_rng(rng_seed)
    
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

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Generate all weight-4 vectors in {-1, 0, 1}^11
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
    
    # Need to find a large subset with min dist^2 >= max_norm2
    # Conflict: dist2[i,j] < max_norm2
    conflicts = dist2 < max_norm2
    np.fill_diagonal(conflicts, False)
    
    # Try multiple random restarts of greedy MIS with different seeds
    best_selected = []
    num_restarts = 20
    restart_deadline = deadline - 5  # Leave 5s margin for final processing
    
    for restart in range(num_restarts):
        if time.time() > restart_deadline:
            break
        seed = 42 + restart
        selected = _greedy_mis(conflicts, N, restart_deadline, rng_seed=seed)
        if len(selected) > len(best_selected):
            best_selected = selected
    
    result_vectors = vectors[best_selected]
    return result_vectors
