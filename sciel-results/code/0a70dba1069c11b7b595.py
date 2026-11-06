import time
import random
import numpy as np
from itertools import combinations

V = 16
ALL_TRIPLES = list(combinations(range(V), 3))
N_TRIPLES = len(ALL_TRIPLES)

# Map triple to index
triple_to_idx = {t: i for i, t in enumerate(ALL_TRIPLES)}

# Precompute conflict structures as numpy array
# conflict_data[i] is a (286, 7) array of indices
conflict_data_list = []
for i, t in enumerate(ALL_TRIPLES):
    a, b, c = t
    others = [v for v in range(V) if v not in (a, b, c)]
    groups = []
    for x, y, z in combinations(others, 3):
        idxs = []
        for comb in [(a, b, z), (a, y, c), (a, y, z), (x, b, c), (x, b, z), (x, y, c), (x, y, z)]:
            idxs.append(triple_to_idx[tuple(sorted(comb))])
        groups.append(idxs)
    conflict_data_list.append(np.array(groups, dtype=np.int32))

# Stack into a single (560, 286, 7) array?
# This might be large: 560 * 286 * 7 * 4 bytes = ~4.5 MB. Fine.
# But accessing a slice `conflict_data[idx]` is fast.
# Let's keep it as a list of arrays to avoid huge memory if not needed, or just use the list.
# Actually, a list of numpy arrays is fine.

def construct():
    start = time.time()
    deadline = start + 115  # Safety margin
    
    best_count = 0
    best_presence = None
    
    # Use a fixed seed for reproducibility in debugging, but random for search?
    # The task says "Maximize", so we should use random restarts.
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        order = list(range(N_TRIPLES))
        random.shuffle(order)
        
        presence = np.zeros(N_TRIPLES, dtype=bool)
        count = 0
        
        for idx in order:
            if time.time() >= deadline:
                break
            
            # Check conflicts using numpy
            # conflict_data_list[idx] is (286, 7)
            # presence[conflict_data_list[idx]] is (286, 7)
            # np.all(..., axis=1) is (286,)
            # np.any(...) is scalar
            
            # Optimization: If count is low, conflicts are unlikely.
            # But we must check.
            
            # Note: presence is updated as we go.
            # conflict_data_list[idx] contains indices of OTHER triples.
            # Since we are adding idx, and idx is not in its own conflict list,
            # we don't need to worry about self-reference.
            
            # Check if any group is fully present
            # This is equivalent to: does there exist a K(2,2,2) missing only 'idx'?
            
            # Vectorized check
            if np.any(np.all(presence[conflict_data_list[idx]], axis=1)):
                continue
            else:
                presence[idx] = True
                count += 1
        
        if count > best_count:
            best_count = count
            best_presence = presence.copy()
            
    if best_presence is None:
        return []
        
    return [list(ALL_TRIPLES[i]) for i in range(N_TRIPLES) if best_presence[i]]
