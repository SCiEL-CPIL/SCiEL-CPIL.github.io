import itertools
import numpy as np
import time

def construct(v, k, t):
    start_time = time.time()
    timeout = 115.0

    # Precompute all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    n_subs = len(all_t_subs)
    sub_to_idx = {s: i for i, s in enumerate(all_t_subs)}
    
    # Precompute mapping for gain calculation
    # Key: (tuple of t-1 elements, point p) -> index of t-subset
    # Only valid if p not in tuple
    gain_map = {}
    for i, S in enumerate(all_t_subs):
        for p in range(v):
            if p in S:
                continue
            # S is the t-subset. We want to find C (size t-1) and p such that S = C U {p}
            # This is not quite right. 
            # In the loop, we have B and candidate p.
            # We iterate C in combinations(B, t-1).
            # We form T = sorted(C + (p,)).
            # We need index of T.
            # So we can precompute: for every C of size t-1, and every p not in C, 
            # what is the index of sorted(C + (p,))?
            
    # Let's just precompute for all C of size t-1
    all_t_minus_1_subs = list(itertools.combinations(range(v), t - 1))
    # Map (C, p) -> index
    # C is a tuple of length t-1
    # p is an int
    # T = sorted(C + (p,))
    # This map will have size binom(v, t-1) * (v - (t-1))
    # For v=19, t=4: binom(19,3) * 16 = 969 * 16 = 15504 entries. Very small.
    
    c_p_to_idx = {}
    for C in all_t_minus_1_subs:
        C_set = set(C)
        for p in range(v):
            if p in C_set:
                continue
            T = tuple(sorted(C + (p,)))
            c_p_to_idx[(C, p)] = sub_to_idx[T]

    coverage = np.zeros(n_subs, dtype=np.int32)
    blocks = []
    
    while True:
        if time.time() - start_time > timeout:
            break
            
        if np.min(coverage) >= 1:
            break
            
        idx = np.where(coverage == 0)[0][0]
        S = all_t_subs[idx]
        B = list(S)
        B_set = set(B)
        
        while len(B) < k:
            best_p = -1
            best_gain = -1
            
            for p in range(v):
                if p in B_set:
                    continue
                
                gain = 0
                if len(B) >= t - 1:
                    for C in itertools.combinations(B, t - 1):
                        # C is a tuple
                        i = c_p_to_idx.get((C, p), -1)
                        if i != -1 and coverage[i] == 0:
                            gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_p = p
            
            B.append(best_p)
            B_set.add(best_p)
        
        B.sort()
        blocks.append(B)
        
        for T in itertools.combinations(B, t):
            i = sub_to_idx[T]
            coverage[i] += 1
            
    return blocks
