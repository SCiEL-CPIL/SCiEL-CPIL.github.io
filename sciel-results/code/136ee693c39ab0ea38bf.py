"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time


def construct():
    deadline = time.time() + 115  # Leave 5s buffer
    
    # Start with the 220 weight-2 vectors (norm^2 = 2)
    base_vectors = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                base_vectors.append(v)
    
    selected = [np.array(v, dtype=np.int64) for v in base_vectors]
    selected_norms_sq = [2.0] * len(selected)
    max_norm_sq = 2.0
    
    # Precompute dot products with base vectors for efficiency
    # For adding new vectors, we need to check against all selected
    
    # Try to add weight-4 vectors (norm^2 = 4)
    # For a weight-4 vector w, need w·v <= 1 for all weight-2 vectors v
    # and w·w' <= 2 for all other weight-4 vectors w'
    
    candidates_w4 = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates_w4.append(v)
    
    # Sort candidates to prioritize those that might fit better
    # Greedily add weight-4 vectors
    added_w4 = []
    for v in candidates_w4:
        if time.time() > deadline:
            break
        v_arr = np.array(v, dtype=np.int64)
        v_norm_sq = 4.0
        
        # Check against all selected vectors
        ok = True
        for i, sv in enumerate(selected):
            dot = np.dot(v_arr, sv)
            # Need ‖v - sv‖^2 >= max_norm_sq_new
            # max_norm_sq_new = max(max_norm_sq, v_norm_sq) = 4
            # ‖v-sv‖^2 = v_norm_sq + sv_norm_sq - 2*dot >= 4
            # 4 + sv_norm_sq - 2*dot >= 4 => sv_norm_sq >= 2*dot
            # For sv with norm_sq=2: 2 >= 2*dot => dot <= 1
            # If we had higher norm vectors, different constraint
            if dot > selected_norms_sq[i] / 2.0:
                ok = False
                break
        
        if ok:
            selected.append(v_arr)
            selected_norms_sq.append(v_norm_sq)
            max_norm_sq = max(max_norm_sq, v_norm_sq)
            added_w4.append(v)
    
    if time.time() > deadline:
        return np.array(selected, dtype=np.int64)
    
    # Now try to add weight-6 vectors (norm^2 = 6)
    # For weight-6 vector w, need ‖w - sv‖^2 >= 6 for all sv
    # 6 + sv_norm_sq - 2*(w·sv) >= 6 => sv_norm_sq >= 2*(w·sv)
    # For sv with norm_sq=2: 2 >= 2*(w·sv) => w·sv <= 1
    # For sv with norm_sq=4: 4 >= 2*(w·sv) => w·sv <= 2
    # For sv with norm_sq=6: 6 >= 2*(w·sv) => w·sv <= 3
    
    # This is getting complex; let's try a few weight-6 vectors
    candidates_w6 = []
    for positions in itertools.combinations(range(D), 6):
        for signs in itertools.product([1, -1], repeat=6):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates_w6.append(v)
    
    # This could be too many (C(11,6)*64 = 462*64 = 29568), do a random sample
    import random
    random.seed(42)
    random.shuffle(candidates_w6)
    
    for v in candidates_w6:
        if time.time() > deadline:
            break
        v_arr = np.array(v, dtype=np.int64)
        v_norm_sq = 6.0
        
        ok = True
        for i, sv in enumerate(selected):
            dot = np.dot(v_arr, sv)
            # Need: v_norm_sq + sv_norm_sq - 2*dot >= max(v_norm_sq, max_norm_sq)
            # Since v_norm_sq = 6 >= max_norm_sq likely, need:
            # 6 + sv_norm_sq - 2*dot >= 6 => sv_norm_sq >= 2*dot
            if dot > selected_norms_sq[i] / 2.0:
                ok = False
                break
        
        if ok:
            selected.append(v_arr)
            selected_norms_sq.append(v_norm_sq)
            max_norm_sq = max(max_norm_sq, v_norm_sq)
    
    return np.array(selected, dtype=np.int64)

# EVOLVE-BLOCK-END
