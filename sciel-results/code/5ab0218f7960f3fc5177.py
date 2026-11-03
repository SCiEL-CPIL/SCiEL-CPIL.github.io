import numpy as np
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    D = 11
    
    # Generate all supports (combinations of 4 indices from 11)
    supports = list(itertools.combinations(range(D), 4))
    num_supports = len(supports)
    
    # Pre-compute triples for each support
    # support[i] is a tuple of 4 indices
    # triples[i] is a list of 4 tuples, each of 3 indices
    triples_for_support = []
    for s in supports:
        trs = list(itertools.combinations(s, 3))
        triples_for_support.append(trs)
        
    # Pre-compute the 16 sign vectors for each support
    # We represent a vector by its support index and a 4-bit integer representing signs
    # 0: -1, 1: 1
    # For each support, we have 16 "sign configs"
    
    # Data structure for blocks
    # Key: (triple_tuple, sign_pattern_int)
    # Value: support_index (the owner)
    # sign_pattern_int: 3 bits, 0 for -1, 1 for 1
    
    # There are 165 triples, 8 patterns.
    # We can use a dictionary or a 2D array.
    # Let's map triple to an ID.
    triple_to_id = {}
    all_triples = list(itertools.combinations(range(D), 3))
    for i, t in enumerate(all_triples):
        triple_to_id[t] = i
        
    # blocks[triple_id][pattern] = owner_support_idx or -1
    blocks = [[-1] * 8 for _ in range(len(all_triples))]
    
    best_vectors = []
    best_score = 0
    
    # We will try a few random orderings of supports to find a better packing
    # Or just one greedy pass?
    # Given the structure, one greedy pass is likely to be good.
    # Let's try 10 random restarts if time permits.
    
    rng = np.random.default_rng(42)
    
    for restart in range(10):
        if time.time() > deadline - 10.0:
            break
            
        # Reset blocks
        for i in range(len(all_triples)):
            for j in range(8):
                blocks[i][j] = -1
                
        current_vectors = []
        
        # Random order of supports
        order = rng.permutation(num_supports)
        
        for sup_idx in order:
            s = supports[sup_idx]
            trs = triples_for_support[sup_idx]
            
            # For this support, determine which of the 16 sign configs are valid
            # A config is valid if for all 4 triples T in s,
            # the pattern p of the config on T is not blocked by another support.
            
            valid_configs = []
            for cfg in range(16):
                is_valid = True
                # Extract signs for the 4 positions
                signs = [(cfg >> i) & 1 for i in range(4)]
                # signs[i] corresponds to s[i]
                
                for t in trs:
                    # Determine the pattern of signs on t
                    # t is a tuple of 3 indices from s
                    # We need to map the indices in t to the positions in s
                    # t contains values from s
                    # Find positions
                    pos_in_s = [s.index(val) for val in t]
                    # Extract signs at these positions
                    pattern_bits = 0
                    for k, pos in enumerate(pos_in_s):
                        if signs[pos]:
                            pattern_bits |= (1 << k)
                    
                    t_id = triple_to_id[t]
                    owner = blocks[t_id][pattern_bits]
                    if owner != -1 and owner != sup_idx:
                        is_valid = False
                        break
                        
                if is_valid:
                    valid_configs.append(cfg)
            
            # Add all valid configs to current_vectors
            for cfg in valid_configs:
                # Create the vector
                v = np.zeros(D, dtype=np.int8)
                for i in range(4):
                    sgn = 1 if ((cfg >> i) & 1) else -1
                    v[s[i]] = sgn
                current_vectors.append(v)
                
                # Update blocks
                trs = triples_for_support[sup_idx]
                signs = [(cfg >> i) & 1 for i in range(4)]
                for t in trs:
                    pos_in_s = [s.index(val) for val in t]
                    pattern_bits = 0
                    for k, pos in enumerate(pos_in_s):
                        if signs[pos]:
                            pattern_bits |= (1 << k)
                    
                    t_id = triple_to_id[t]
                    blocks[t_id][pattern_bits] = sup_idx

        if len(current_vectors) > best_score:
            best_score = len(current_vectors)
            best_vectors = current_vectors
            
    if not best_vectors:
        # Fallback to seed logic if something went wrong
        vectors = []
        for i in range(D):
            for j in range(i + 1, D):
                for s_i in (-1, 1):
                    for s_j in (-1, 1):
                        v = np.zeros(D, dtype=np.int8)
                        v[i] = s_i
                        v[j] = s_j
                        vectors.append(v)
        return np.array(vectors, dtype=np.int8)

    return np.array(best_vectors, dtype=np.int8)
