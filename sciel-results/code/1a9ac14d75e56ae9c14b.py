"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product, combinations

def construct():
    deadline = time.time() + 130  # 20s safety margin under 150s kill
    start_time = time.time()
    
    # Generate candidate integer vectors with norm^2 = 4 (norm = 2)
    # In 11D, integer vectors with norm^2 = 4 can be:
    # Type A: two coordinates are +/-1, rest 0 (weight 2)
    # Type B: one coordinate is +/-2, rest 0 (weight 1)
    # Type C: four coordinates are +/-1, rest 0 (weight 4) -- wait, 1^2*4 = 4, yes
    
    candidates = []
    
    # Type A: weight 2, entries in {-1, 0, 1}, exactly two nonzeros
    # For each pair of positions (i, j) and signs (si, sj)
    for i, j in combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i] = si
                v[j] = sj
                candidates.append(v)
    
    # Type B: weight 1, one coordinate is +/-2
    for i in range(D):
        for s in (2, -2):
            v = np.zeros(D, dtype=np.int64)
            v[i] = s
            candidates.append(v)
    
    # Type C: weight 4, four coordinates are +/-1
    # This is C(11,4) * 2^4 = 330 * 16 = 5280 vectors
    for positions in combinations(range(D), 4):
        for signs in product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates.append(v)
    
    candidates = np.array(candidates, dtype=np.int64)
    n_cand = len(candidates)
    
    # We want to select a subset where min pairwise distance >= max norm
    # All candidates have norm^2 = 4, so max norm = 2
    # We need min pairwise distance >= 2, i.e., min pairwise squared distance >= 4
    # Distance squared between two vectors u, v with norm^2=4 each:
    # ||u-v||^2 = ||u||^2 + ||v||^2 - 2 u·v = 8 - 2 u·v
    # We need 8 - 2 u·v >= 4 => u·v <= 2
    
    # So we need a set of vectors where all pairwise dot products <= 2
    
    # Greedy farthest-point sampling:
    # Start with a random vector, then iteratively add the candidate that maximizes
    # the minimum distance to the current set
    
    # Precompute all pairwise dot products? That's n_cand^2 which might be large.
    # n_cand = C(11,2)*4 + 11*2 + C(11,4)*16 = 55*4 + 22 + 330*16 = 220 + 22 + 5280 = 5522
    # 5522^2 = ~30M, which is manageable.
    
    # Compute Gram matrix (dot products)
    # G = candidates @ candidates.T
    # This is 5522 x 5522, ~30M elements, fine.
    
    G = candidates @ candidates.T
    
    # Initialize: pick first candidate
    selected = [0]
    selected_set = {0}
    
    # For efficiency, track for each candidate the minimum dot product with selected set
    # min_dot[c] = min over s in selected of G[c, s]
    # We want to maximize min distance = 8 - 2*max_dot, where max_dot[c] = max over s in selected of G[c,s]
    # Wait: distance squared = 8 - 2*G[c,s]. We want this >= 4 for all s, so G[c,s] <= 2 for all s.
    # The constraint is G[c,s] <= 2 for all s in selected.
    # To maximize the minimum distance, we want to minimize the maximum G[c,s] over s in selected.
    # Because min distance = min_s (8 - 2*G[c,s]) = 8 - 2*max_s G[c,s]
    
    # So we track max_dot[c] = max over s in selected of G[c,s]
    # And we add the candidate with the smallest max_dot[c] (among those with max_dot[c] <= 2)
    
    max_dot = np.full(n_cand, -np.inf, dtype=np.int64)
    for s in selected:
        max_dot = np.maximum(max_dot, G[s, :])
    
    while time.time() < deadline:
        # Find candidate with smallest max_dot that is <= 2 and not yet selected
        # Mask out selected
        available_mask = np.ones(n_cand, dtype=bool)
        for s in selected:
            available_mask[s] = False
        
        # Among available, find those with max_dot <= 2
        valid = available_mask & (max_dot <= 2)
        
        if not np.any(valid):
            break
        
        # Among valid, pick the one with smallest max_dot
        valid_indices = np.where(valid)[0]
        if len(valid_indices) == 0:
            break
        
        # Pick the one with minimum max_dot
        min_val = np.min(max_dot[valid_indices])
        candidates_with_min = valid_indices[max_dot[valid_indices] == min_val]
        
        # Pick the first one (or random for diversity)
        next_idx = candidates_with_min[0]
        
        selected.append(next_idx)
        selected_set.add(next_idx)
        
        # Update max_dot
        max_dot = np.maximum(max_dot, G[next_idx, :])
        
        # Check if we're running out of time
        if time.time() > deadline - 5:
            break
    
    result = candidates[selected]
    return result
# EVOLVE-BLOCK-END
