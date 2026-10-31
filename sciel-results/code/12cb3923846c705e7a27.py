import numpy as np
import itertools
import time
import heapq

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 120s limit
    
    # Generate all weight-4 vectors in {-1, 0, 1}^11 (Norm^2 = 4)
    # Represented by (positions_tuple, signs_tuple)
    v4_candidates = []
    pos_to_idx = {}
    idx_to_pos = []
    idx_to_signs = []
    
    # We need to map vector to index for conflict graph
    # And store positions and signs for fast conflict checking
    
    # 1. Generate all position sets of size 4
    pos_sets = list(itertools.combinations(range(D), 4))
    pos_set_to_idx = {p: i for i, p in enumerate(pos_sets)}
    
    # 2. For each position set, generate all 16 sign patterns
    # Total vectors = 330 * 16 = 5280
    vectors = []
    signs_list = list(itertools.product([1, -1], repeat=4))
    
    for p in pos_sets:
        for s in signs_list:
            # Create vector
            v = [0] * D
            for pos, sig in zip(p, s):
                v[pos] = sig
            vectors.append(v)
            
    N4 = len(vectors)
    vectors_v4 = np.array(vectors, dtype=np.int64)
    
    # Generate all weight-1 vectors in {-2, 0, 2}^11 (Norm^2 = 4)
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)
            
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)
    
    # Build conflict graph for V4
    # Conflict if dot > 2
    # We identified that conflicts occur if:
    # 1. Identical (ignore)
    # 2. Share 3 positions with same signs.
    
    # To build graph efficiently:
    # Group vectors by position set
    groups = {p: [] for p in pos_sets}
    for i, v in enumerate(vectors):
        # Find position set
        # We stored p in loop, so we can just use that
        pass
        
    # Let's re-iterate to populate groups
    idx = 0
    for p in pos_sets:
        for s in signs_list:
            groups[p].append(idx)
            idx += 1
            
    # Precompute intersection size 3 pairs of position sets
    # And the intersection tuple
    conflict_pairs = []
    for i, p1 in enumerate(pos_sets):
        for j, p2 in enumerate(pos_sets):
            if i >= j: continue
            inter = tuple(set(p1) & set(p2))
            if len(inter) == 3:
                conflict_pairs.append((p1, p2, inter))
                
    # Build adjacency list
    # N=5280. List of lists.
    adj = [[] for _ in range(N4)]
    
    for p1, p2, inter in conflict_pairs:
        # Get indices for p1 and p2
        idxs1 = groups[p1]
        idxs2 = groups[p2]
        
        # For each u in idxs1, v in idxs2
        # Check if signs match on inter
        for i1 in idxs1:
            v1 = vectors[i1]
            # Extract signs of v1 on inter
            # inter is a tuple of 3 positions
            s1 = tuple(v1[pos] for pos in inter)
            
            for i2 in idxs2:
                v2 = vectors[i2]
                s2 = tuple(v2[pos] for pos in inter)
                
                if s1 == s2:
                    adj[i1].append(i2)
                    adj[i2].append(i1)
                    
    # Greedy MIS
    alive = [True] * N4
    degrees = [len(a) for a in adj]
    
    # Heap of (degree, index)
    heap = [(degrees[i], i) for i in range(N4)]
    heapq.heapify(heap)
    
    selected = []
    
    while heap and time.time() < deadline:
        deg, i = heapq.heappop(heap)
        
        # Lazy deletion check
        if not alive[i] or degrees[i] != deg:
            continue
            
        selected.append(i)
        alive[i] = False
        
        # Remove neighbors
        for j in adj[i]:
            if alive[j]:
                alive[j] = False
                degrees[j] -= 1
                # Push updated degree
                heapq.heappush(heap, (degrees[j], j))
                
    result_v4 = vectors_v4[selected]
    
    # Combine V4 and V1
    # V1 vectors are all compatible with V4 (dot <= 2).
    # V1 vectors are compatible with each other (dot <= 0 for distinct axes, -4 for opposite).
    # So we can take all 22.
    
    final_vectors = np.vstack([result_v4, vectors_v1])
    
    return final_vectors
