import numpy as np
import time
import random

D = 11

# EVOLVE-BLOCK-START
import itertools

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # 1. Generate all weight-3 vectors
    # Supports: all combinations of 3 indices from 11
    supports = list(itertools.combinations(range(D), 3))
    vectors = []
    vector_to_id = {}
    
    for supp in supports:
        # 8 sign combinations
        for signs in itertools.product([1, -1], repeat=3):
            v = [0] * D
            for idx, val in zip(supp, signs):
                v[idx] = val
            vec_tuple = tuple(v)
            vector_to_id[vec_tuple] = len(vectors)
            vectors.append(vec_tuple)
            
    n_vectors = len(vectors)
    
    # 2. Build conflict graph
    # Edge exists if two vectors share 2 indices and have matching signs on those indices.
    # Optimization: Group vectors by their "partial signatures" on pairs?
    # Or just iterate over pairs of supports.
    
    # Map support to list of vector IDs
    supp_to_vecs = {supp: [] for supp in supports}
    for i, v in enumerate(vectors):
        # Find support
        # Since we generated them in order, we can track this
        pass
        
    # Re-generate with tracking
    vectors = []
    supp_to_vecs = {supp: [] for supp in supports}
    vec_id_counter = 0
    for supp in supports:
        for signs in itertools.product([1, -1], repeat=3):
            v = [0] * D
            for idx, val in zip(supp, signs):
                v[idx] = val
            vectors.append(tuple(v))
            supp_to_vecs[supp].append(vec_id_counter)
            vec_id_counter += 1
            
    n_vectors = len(vectors)
    
    # Build adjacency list
    # adj[i] = set of j such that i and j conflict
    adj = [set() for _ in range(n_vectors)]
    
    # Iterate over pairs of supports sharing 2 indices
    # To do this efficiently, map each pair of indices to the supports containing it
    # pair_to_supports = { (i, j): [supp1, supp2, ...] }
    pair_to_supports = {}
    for supp in supports:
        for i in range(3):
            for j in range(i+1, 3):
                pair = (supp[i], supp[j])
                if pair not in pair_to_supports:
                    pair_to_supports[pair] = []
                pair_to_supports[pair].append(supp)
                
    for pair, supps in pair_to_supports.items():
        # For each pair of supports in this list
        for i in range(len(supps)):
            for j in range(i+1, len(supps)):
                s1 = supps[i]
                s2 = supps[j]
                # Find the 2 shared indices
                # We know they share 'pair'
                shared = pair
                # Get vector IDs for s1 and s2
                v_ids_1 = supp_to_vecs[s1]
                v_ids_2 = supp_to_vecs[s2]
                
                # For each vector in s1 and s2, check conflict
                # Conflict if signs match on shared indices
                # We can pre-compute the sign pattern on 'shared' for each vector
                # But it's small (8x8=64 checks per pair of supports)
                for id1 in v_ids_1:
                    v1 = vectors[id1]
                    sig1 = (v1[shared[0]], v1[shared[1]])
                    for id2 in v_ids_2:
                        v2 = vectors[id2]
                        sig2 = (v2[shared[0]], v2[shared[1]])
                        if sig1 == sig2:
                            adj[id1].add(id2)
                            adj[id2].add(id1)
                            
    # 3. Find Independent Set
    # Greedy: Pick random node, add to set, remove it and neighbors
    # Repeat until no nodes left
    # Do this multiple times and keep best
    
    best_set = set()
    num_trials = 50
    
    available = set(range(n_vectors))
    
    for trial in range(num_trials):
        if time.time() > deadline:
            break
            
        current_set = set()
        rem = set(available)
        
        while rem:
            # Pick random node
            node = random.choice(list(rem))
            current_set.add(node)
            # Remove node and its neighbors
            neighbors = adj[node]
            to_remove = {node} | neighbors
            rem -= to_remove
            
        if len(current_set) > len(best_set):
            best_set = current_set
            if len(best_set) > 1000: # Good enough?
                break
                
    # 4. Construct output array
    # best_set contains IDs
    result_vectors = [vectors[i] for i in best_set]
    result_array = np.array(result_vectors, dtype=np.int64)
    
    return result_array
# EVOLVE-BLOCK-END
