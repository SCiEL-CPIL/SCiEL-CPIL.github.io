import numpy as np
import itertools
import random

D = 11

def construct():
    # Generate candidate pool with bitmasks
    # v is represented by (pos_mask, neg_mask, norm_sq)
    candidates = []
    
    for w in [2, 3, 4, 5]:
        for support in itertools.combinations(range(D), w):
            for signs in itertools.product([1, -1], repeat=w):
                pos_mask = 0
                neg_mask = 0
                for idx, sign in zip(support, signs):
                    if sign == 1:
                        pos_mask |= (1 << idx)
                    else:
                        neg_mask |= (1 << idx)
                candidates.append((pos_mask, neg_mask, w))
                
    # Helper to compute dot product from masks
    def dot_product(p1, n1, p2, n2):
        # popcount
        return (p1 & p2).bit_count() - (p1 & n2).bit_count() - (n1 & p2).bit_count() + (n1 & n2).bit_count()

    best_set = []
    best_size = 0
    
    # Run multiple random restarts
    # We can reduce pool size or restarts if time is tight
    # Let's try 20 restarts first, can increase if fast
    for _ in range(20):
        random.shuffle(candidates)
        current_set = [] # list of (pos, neg, norm_sq)
        
        for p, n, w in candidates:
            compatible = True
            for p2, n2, w2 in current_set:
                dot = dot_product(p, n, p2, n2)
                limit = min(w, w2) // 2
                if dot > limit:
                    compatible = False
                    break
            
            if compatible:
                current_set.append((p, n, w))
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
            if best_size > 550:
                break
                
    # Convert back to vectors
    if not best_set:
        return np.array([], dtype=np.int64)
        
    vecs = []
    for p, n, w in best_set:
        v = [0] * D
        for i in range(D):
            if p & (1 << i):
                v[i] = 1
            elif n & (1 << i):
                v[i] = -1
        vecs.append(v)
        
    return np.array(vecs, dtype=np.int64)
