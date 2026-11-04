import itertools
import time
import numpy as np

D = 11
DEADLINE_S = 118.0

def construct():
    t0 = time.perf_counter()
    
    def is_deadline():
        return time.perf_counter() - t0 > DEADLINE_S

    # Generate candidates for norm^2 = 4 (4 non-zero entries)
    # v has 4 entries of +/-1
    candidates = []
    for support in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(support, signs):
                v[idx] = s
            candidates.append(v)
            
    V = np.array(candidates, dtype=np.int64)
    n_cand = len(V)
    
    # Norms are all 4
    # Condition: u . v <= 2
    # Conflict if u . v >= 3
    
    # Precompute Gram matrix
    G = V @ V.T
    
    # Conflict matrix: True if G[i,j] >= 3
    # Note: G[i,i] = 4, so self is conflict, but we don't check self.
    conflict = G >= 3
    
    # Greedy MIS
    best_set = []
    best_len = 0
    
    rng = np.random.RandomState(42)
    
    # Try multiple random orders
    for trial in range(20):
        if is_deadline():
            break
            
        if trial == 0:
            order = list(range(n_cand))
        else:
            order = rng.permutation(n_cand).tolist()
            
        chosen_mask = np.zeros(n_cand, dtype=bool)
        chosen = []
        
        for idx in order:
            if is_deadline():
                break
            if chosen_mask[idx]:
                continue
            
            # Check if idx conflicts with any chosen
            # conflict[idx, chosen]
            # If any True, skip
            if not np.any(conflict[idx, chosen]):
                chosen.append(idx)
                chosen_mask[idx] = True
                
        if len(chosen) > best_len:
            best_len = len(chosen)
            best_set = chosen[:]
            
    # Also try the norm^2=2 baseline just in case
    base = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i], v[j] = si, sj
                base.append(v)
    base_arr = np.array(base, dtype=np.int64)
    
    if best_len > base_arr.shape[0]:
        return V[best_set]
    else:
        return base_arr
