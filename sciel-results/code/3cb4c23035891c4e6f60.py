"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools

def construct():
    """
    Construct nonzero integer vectors in R^11 whose minimum pairwise distance
    is at least the maximum vector norm.
    
    Strategy:
    1. Generate a large set of candidate integer vectors with squared norms 2, 4, and 6.
    2. Use a vectorized greedy algorithm to select a maximal independent set.
    3. Perform multiple random restarts to find a larger set.
    """
    candidates = []
    
    # Helper to add vectors
    def add_vectors(positions, values):
        for pos in positions:
            for vals in values:
                v = [0] * D
                for p, val in zip(pos, vals):
                    v[p] = val
                candidates.append(v)

    # Type 1: Norm^2 = 2 (Two +/-1s)
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i] = si
                v[j] = sj
                candidates.append(v)
                
    # Type 2: Norm^2 = 4
    # 2a: One +/-2
    for i in range(D):
        for s in (1, -1):
            v = [0] * D
            v[i] = s * 2
            candidates.append(v)
            
    # 2b: Four +/-1s
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, sign in zip(positions, signs):
                v[pos] = sign
            candidates.append(v)

    # Type 3: Norm^2 = 6 (One +/-2, Two +/-1s)
    for i in range(D):
        for si in (1, -1):
            # Choose 2 other positions for +/-1
            others = [j for j in range(D) if j != i]
            for j, k in itertools.combinations(others, 2):
                for sj in (1, -1):
                    for sk in (1, -1):
                        v = [0] * D
                        v[i] = si * 2
                        v[j] = sj
                        v[k] = sk
                        candidates.append(v)

    candidates = np.array(candidates, dtype=np.int64)
    norms_sq = np.sum(candidates**2, axis=1)
    max_norm_sq = np.max(norms_sq) # Will be 6

    def greedy_select(candidates, norms_sq, max_norm_sq, seed):
        np.random.seed(seed)
        # Randomize order
        indices = np.arange(len(candidates))
        np.random.shuffle(indices)
        
        selected = []
        selected_norms = []
        
        # To speed up, we can pre-compute dot products or use vectorized distance
        # But with ~10k candidates, a simple loop with vectorized check against selected is fine.
        
        for idx in indices:
            c = candidates[idx]
            c_norm_sq = norms_sq[idx]
            
            if not selected:
                selected.append(idx)
                selected_norms.append(c_norm_sq)
                continue
                
            # Check compatibility with all selected
            # dist_sq = c_norm_sq + sel_norm_sq - 2 * dot(c, sel)
            # We need dist_sq >= max_norm_sq for all sel
            
            # Vectorized check against selected vectors
            sel_vecs = candidates[selected]
            sel_norms = np.array(selected_norms)
            
            dots = np.dot(sel_vecs, c)
            dist_sqs = c_norm_sq + sel_norms - 2 * dots
            
            if np.all(dist_sqs >= max_norm_sq):
                selected.append(idx)
                selected_norms.append(c_norm_sq)
                
        return candidates[selected]

    best_result = None
    best_size = 0
    
    # Run multiple restarts
    for seed in range(10):
        res = greedy_select(candidates, norms_sq, max_norm_sq, seed)
        if len(res) > best_size:
            best_size = len(res)
            best_result = res
            
    return best_result
# EVOLVE-BLOCK-END
