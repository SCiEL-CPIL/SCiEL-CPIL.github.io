"""Program for kissing number in dimension 11 using greedy augmentation of D_11 lattice."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import random

def construct():
    # Generate all integer vectors with norm squared 2
    vecs_norm2 = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i] = si
                v[j] = sj
                vecs_norm2.append(v)
    
    # Generate all integer vectors with norm squared 3
    vecs_norm3 = []
    for i, j, k in itertools.combinations(range(D), 3):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    v = [0] * D
                    v[i] = si
                    v[j] = sj
                    v[k] = sk
                    vecs_norm3.append(v)
    
    vecs_norm2 = np.array(vecs_norm2, dtype=np.int64)
    vecs_norm3 = np.array(vecs_norm3, dtype=np.int64)
    
    # Strategy: Greedily select Norm 3 vectors first (setting R^2=3)
    # Then add compatible Norm 2 vectors.
    
    selected = []
    
    # 1. Select Norm 3 vectors
    # We need dist sq >= 3 between selected Norm 3 vectors.
    # Dist sq between two norm 3 vectors is 6 - 2*u.v.
    # Dist sq < 3 means 6 - 2*u.v < 3 => 2*u.v > 3 => u.v >= 2.
    # So we must avoid u.v >= 2.
    # u.v can be -3, -1, 1, 3.
    # u.v = 3 implies u=v (not possible in set) or u=-v? 
    # If u=(1,1,1,0..), v=(-1,-1,-1,0..), u.v = -3.
    # If u=(1,1,1,0..), v=(1,1,1,0..) is same.
    # If u=(1,1,1,0..), v=(1,1,-1,0..), u.v = 1. Dist sq = 6-2=4. OK.
    # If u=(1,1,1,0..), v=(1,-1,1,0..), u.v = 1. Dist sq = 4. OK.
    # If u=(1,1,1,0..), v=(1,1,0,1..), u.v = 2. Dist sq = 6-4=2. FAIL.
    # So we must avoid u.v >= 2.
    
    random.seed(42)
    indices3 = list(range(len(vecs_norm3)))
    random.shuffle(indices3)
    
    for idx in indices3:
        v = vecs_norm3[idx]
        ok = True
        for w in selected:
            # Check dist sq >= 3
            # w is norm 3 (from previous iterations)
            # dist_sq = 6 - 2 * dot(w, v)
            dot = np.dot(w, v)
            if dot >= 2:
                ok = False
                break
        if ok:
            selected.append(v)
            
    # 2. Add Norm 2 vectors
    # Now selected contains Norm 3 vectors. R^2 = 3.
    # We need to add Norm 2 vectors v such that:
    # a) dist(v, w) >= sqrt(3) for all w in selected (Norm 3)
    # b) dist(v, u) >= sqrt(3) for all u in selected that are Norm 2 (none yet)
    # c) dist(v, u) >= sqrt(3) for all other Norm 2 vectors we add.
    
    # Condition a: dist(v, w)^2 >= 3.
    # v is norm 2, w is norm 3.
    # dist_sq = 2 + 3 - 2*v.w = 5 - 2*v.w.
    # 5 - 2*v.w >= 3 => 2*v.w <= 2 => v.w <= 1.
    # So we must avoid v.w >= 2.
    
    # Condition b/c: dist(v, u)^2 >= 3 for Norm 2 vectors.
    # v, u norm 2.
    # dist_sq = 4 - 2*v.u.
    # 4 - 2*v.u >= 3 => 2*v.u <= 1 => v.u <= 0.
    # So we must avoid v.u > 0. i.e., v.u must be 0 or negative.
    # Since v, u are integer, v.u is integer.
    # v.u = 1 is forbidden.
    # v.u = 0 is OK (dist sq 4).
    # v.u = -1 is OK (dist sq 6).
    
    indices2 = list(range(len(vecs_norm2)))
    random.shuffle(indices2)
    
    for idx in indices2:
        v = vecs_norm2[idx]
        ok = True
        for w in selected:
            # Check if w is norm 2 or 3
            norm_sq_w = np.dot(w, w)
            if norm_sq_w == 3:
                # Condition a: v.w <= 1
                dot = np.dot(v, w)
                if dot > 1:
                    ok = False
                    break
            else: # norm_sq_w == 2
                # Condition c: v.w <= 0
                dot = np.dot(v, w)
                if dot > 0:
                    ok = False
                    break
        if ok:
            selected.append(v)
            
    return np.array(selected, dtype=np.int64)
# EVOLVE-BLOCK-END
