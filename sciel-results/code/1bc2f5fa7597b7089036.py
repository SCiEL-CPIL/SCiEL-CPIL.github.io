import numpy as np
import time
import itertools

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Generate all vectors with exactly 3 non-zero entries (+/- 1)
    # Number of vectors: C(11, 3) * 2^3 = 165 * 8 = 1320
    vectors = []
    supports = []
    idx_to_support = {}
    
    vec_id = 0
    for support in itertools.combinations(range(D), 3):
        supports.append(support)
        for signs in itertools.product([1, -1], repeat=3):
            v = [0] * D
            for k, s in zip(support, signs):
                v[k] = s
            vectors.append(v)
            idx_to_support[vec_id] = support
            vec_id += 1
            
    N = len(vectors)
    
    # Build incompatibility graph
    # Incompatible if dot product >= 2
    # For norm sqrt(3) vectors, dot product >= 2 implies:
    # 1. Same vector (dot 3) - not in set as we pick distinct
    # 2. Support overlap 2, and signs on overlap are same (dot 2)
    
    # We will store neighbors for each node
    # neighbors[i] = list of j such that i and j are incompatible
    
    # Optimization: Group by support
    # But simpler: just check pairs? 1320^2 / 2 is ~ 870k checks. Fast enough.
    # Or use the structure.
    
    # Let's use the structure to be safe and fast.
    # Map: (coord1, coord2, sign1, sign2) -> list of vector indices
    # If two vectors share (c1, c2) with (s1, s2), they are incompatible.
    
    key_to_indices = {}
    for i in range(N):
        sup = idx_to_support[i]
        v = vectors[i]
        # Generate all pairs of coordinates in this support
        # There are 3 pairs
        for p in range(3):
            for q in range(p+1, 3):
                c1 = sup[p]
                c2 = sup[q]
                s1 = v[c1]
                s2 = v[c2]
                key = (c1, c2, s1, s2)
                if key not in key_to_indices:
                    key_to_indices[key] = []
                key_to_indices[key].append(i)
                
    neighbors = [set() for _ in range(N)]
    
    for key, indices in key_to_indices.items():
        if len(indices) > 1:
            # All pairs in this list are incompatible
            # Note: A vector can appear in multiple keys?
            # Yes, a vector with support {1,2,3} contributes to keys (1,2), (1,3), (2,3).
            # If i and j share key K, they are incompatible.
            for u in range(len(indices)):
                for w in range(u+1, len(indices)):
                    i = indices[u]
                    j = indices[w]
                    neighbors[i].add(j)
                    neighbors[j].add(i)
    
    # Greedy Independent Set
    # Sort by degree initially?
    # Standard greedy: pick min degree, remove neighbors.
    
    remaining = set(range(N))
    independent_set = []
    
    # To speed up, we can just iterate.
    # Since N is small (1320), even O(N^2) is fine.
    
    # Heuristic: Pick node with min degree in remaining graph.
    # To make it faster, we can just pick random or first available if we don't care about optimal.
    # But min degree usually gives better results.
    
    # Let's try a simple greedy with randomization to avoid getting stuck in bad local minima if any.
    # Actually, with degree 6, it's very open.
    
    for _ in range(N):
        if not remaining:
            break
        if time.time() > deadline:
            break
            
        # Find min degree node in remaining
        min_deg = float('inf')
        best_node = -1
        # Sample if large? N is small, just scan.
        for node in remaining:
            deg = len(neighbors[node] & remaining)
            if deg < min_deg:
                min_deg = deg
                best_node = node
                if min_deg == 0:
                    break
                    
        if best_node == -1:
            break
            
        independent_set.append(best_node)
        remaining.remove(best_node)
        # Remove neighbors
        for nb in neighbors[best_node]:
            if nb in remaining:
                remaining.remove(nb)
                
    # Construct result array
    if not independent_set:
        # Fallback to seed if something went wrong
        vectors_seed = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors_seed.append(v)
        return np.array(vectors_seed, dtype=np.int64)
        
    result = np.array([vectors[i] for i in independent_set], dtype=np.int64)
    return result
