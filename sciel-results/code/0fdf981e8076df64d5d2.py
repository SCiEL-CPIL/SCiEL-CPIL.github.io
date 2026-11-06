import time
from itertools import combinations
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    # Precompute all 4-subsets of [1,16] as bitmasks
    all_sets = []
    for combo in combinations(range(1, 17), 4):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        all_sets.append((mask, combo))
    
    N = len(all_sets)
    masks = [m for m, _ in all_sets]
    
    # Vectorized Precomputation of Intersection Matrix
    masks_np = np.array(masks, dtype=np.uint32)
    
    # Calculate pairwise ANDs
    and_matrix = np.bitwise_and(masks_np[:, None], masks_np[None, :])
    
    # Popcount using lookup table
    popcount_8 = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)
    
    m0 = and_matrix & 0xFF
    m1 = (and_matrix >> 8) & 0xFF
    m2 = (and_matrix >> 16) & 0xFF
    m3 = (and_matrix >> 24) & 0xFF
    
    inter = popcount_8[m0] + popcount_8[m1] + popcount_8[m2] + popcount_8[m3]
    inter = inter.astype(np.uint8)
    
    del and_matrix, m0, m1, m2, m3
    
    # Precompute Adjacency sets for each intersection size k
    # Adj[k][i] = set of j such that inter[i][j] == k
    Adj = [set() for _ in range(4)]
    
    for i in range(N):
        row = inter[i]
        for k in range(4):
            indices = np.nonzero(row == k)[0]
            if len(indices) > 0:
                Adj[k].update(indices.tolist())
                
    # Remove self-loops (though inter[i][i]=4, so no self loop in k=0..3)
    # But just in case:
    for k in range(4):
        for i in range(N):
            if i in Adj[k][i]:
                Adj[k][i].remove(i)

    def is_valid_addition(C, F):
        """
        Check if adding C to F is valid.
        F is a list of indices.
        """
        if not F:
            return True
            
        # Get intersection sizes of C with all in F
        # F is a list of ints. inter[C, F] works.
        sizes = inter[C, F]
        
        # Group indices in F by intersection size with C
        groups = [[] for _ in range(4)]
        for idx, s in zip(F, sizes):
            if s < 4: 
                groups[s].append(idx)
                
        # Check for sunflower
        for k in range(4):
            g = groups[k]
            if len(g) < 2:
                continue
            
            # Check if any pair in g has intersection size k
            neighbors_cache = [Adj[k][idx] for idx in g]
            for i in range(len(g)):
                n_i = neighbors_cache[i]
                for j in range(i+1, len(g)):
                    if g[j] in n_i:
                        return False
        return True

    best_family = []
    best_size = 0
    random.seed(42)
    
    num_trials = 0
    while time.time() < deadline:
        num_trials += 1
        indices = list(range(N))
        random.shuffle(indices)
        
        F = []
        
        for idx in indices:
            if time.time() > deadline - 5: # Leave time for return
                break
            if is_valid_addition(idx, F):
                F.append(idx)
                
        # Local Search: Try to add more
        unchosen = [x for x in range(N) if x not in F]
        random.shuffle(unchosen)
        for cand in unchosen:
            if time.time() > deadline - 5:
                break
            if is_valid_addition(cand, F):
                F.append(cand)
                
        if len(F) > best_size:
            best_size = len(F)
            best_family = F
            
    return [sorted(all_sets[i][1]) for i in best_family]
