import numpy as np
import random
import itertools

def construct():
    D = 11
    best_vectors = []
    
    # Try different target norms
    # Norm 4 is the baseline (220 vectors)
    # We try to beat 348 (current score 1.58 * 220 approx)
    
    # Strategy: 
    # 1. Start with the 220 D11 vectors (norm 4).
    # 2. Try to add random vectors of norm 4.
    # 3. If stuck, try norm 8 vectors from scratch?
    
    # Given time constraints, a robust simple improvement is to use 
    # a better known construction if available, or a fast greedy search.
    
    # Let's use the D11 set as a seed and try to grow it with random vectors of norm 4.
    # If we can't grow, we try norm 8.
    
    def get_d11_vectors():
        vecs = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i] = 2 * si
                    v[j] = 2 * sj
                    vecs.append(v)
        return vecs

    seed_vecs = get_d11_vectors()
    current_set = list(seed_vecs)
    max_norm = 4
    
    # Greedy addition of random norm 4 vectors
    # Norm 4 integer vectors: 
    # 1. Two 2s (D11 type)
    # 2. One 2, two 1s? 4+1+1=6 (No)
    # 3. Four 1s? 1+1+1+1=4. Yes.
    # 4. One 2, one 2? (Covered)
    # So we can add vectors with four 1s.
    
    # Check distance constraint: min dist >= 4.
    # Existing min dist is 4.
    # New vector must be dist >= 4 from all existing.
    
    attempts = 0
    max_attempts = 50000
    while attempts < max_attempts:
        attempts += 1
        # Generate random norm 4 vector
        # Type 1: Four 1s
        if random.random() < 0.5:
            idxs = random.sample(range(D), 4)
            signs = [random.choice([1, -1]) for _ in range(4)]
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = s
        else:
            # Type 2: Two 2s (already in seed, but maybe we missed some? No, seed has all)
            # So we only add Type 1.
            continue
            
        # Check norm
        norm_sq = sum(x*x for x in v)
        if norm_sq != 4:
            continue
            
        # Check distance to all existing vectors
        valid = True
        for u in current_set:
            dist_sq = sum((a-b)**2 for a, b in zip(v, u))
            if dist_sq < 16: # dist < 4
                valid = False
                break
                
        if valid:
            current_set.append(v)
            
    return np.array(current_set, dtype=np.int64)
