import numpy as np
import itertools
import random
import time

D = 11

def generate_vectors(weight):
    vectors = []
    # Choose indices
    for indices in itertools.combinations(range(D), weight):
        # Choose signs
        for signs in itertools.product([1, -1], repeat=weight):
            v = [0] * D
            for idx, s in zip(indices, signs):
                v[idx] = s
            # Store as list of non-zero tuples for fast overlap check
            nz = [(idx, s) for idx, s in zip(indices, signs)]
            vectors.append((v, nz))
    return vectors

def overlap_count(nz1, nz2):
    # Convert nz2 to dict for O(1) lookup? 
    # Since weight is small, linear scan is fine too.
    # But dict is safer for speed in inner loop.
    d2 = {idx: val for idx, val in nz2}
    count = 0
    for idx, val in nz1:
        if idx in d2 and d2[idx] == val:
            count += 1
    return count

def greedy_select(vectors, max_overlap, seed_val=42):
    random.seed(seed_val)
    v_list = vectors[:]
    random.shuffle(v_list)
    
    selected_nz = []
    selected_vecs = []
    
    for v, nz in v_list:
        ok = True
        for u_nz in selected_nz:
            if overlap_count(nz, u_nz) > max_overlap:
                ok = False
                break
        if ok:
            selected_nz.append(nz)
            selected_vecs.append(v)
            
    return selected_vecs

def construct():
    start_time = time.time()
    deadline = 140.0 # 10s margin
    
    # Weight 2: All valid
    w2_vecs, _ = zip(*generate_vectors(2)) if D >= 2 else ([], [])
    # Actually generate_vectors returns list of (v, nz)
    w2_data = generate_vectors(2)
    count_w2 = len(w2_data)
    
    best_count = count_w2
    best_vectors = [v for v, _ in w2_data]
    
    # Weight 3
    w3_data = generate_vectors(3)
    max_overlap_w3 = 1
    
    # Weight 4
    w4_data = generate_vectors(4)
    max_overlap_w4 = 2
    
    # Try a few random shuffles for W3 and W4
    # Budget check
    iterations = 5
    for i in range(iterations):
        if time.time() > deadline:
            break
            
        # W3
        res3 = greedy_select(w3_data, max_overlap_w3, seed_val=100+i)
        if len(res3) > best_count:
            best_count = len(res3)
            best_vectors = res3
            
        # W4
        res4 = greedy_select(w4_data, max_overlap_w4, seed_val=200+i)
        if len(res4) > best_count:
            best_count = len(res4)
            best_vectors = res4
            
    return np.array(best_vectors, dtype=np.int64)
