import itertools
import numpy as np
import scipy.sparse
import time

# Precomputation
def precompute(v, k, t):
    # Map t-subsets to indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_subsets_set = {s: i for i, s in enumerate(t_subsets)}
    n_t = len(t_subsets)
    
    # Generate all k-subsets
    k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(k_subsets)
    
    rows = []
    cols = []
    data = []
    
    for i, block in enumerate(k_subsets):
        # Get all t-subsets within this block
        for t_sub in itertools.combinations(block, t):
            j = t_subsets_set[t_sub]
            rows.append(i)
            cols.append(j)
            data.append(1)
            
    M = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(n_k, n_t))
    return M, k_subsets, n_t

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Precompute
    M, k_subsets, n_t = precompute(v, k, t)
    n_k = len(k_subsets)
    
    best_blocks = None
    best_len = float('inf')
    
    # Multi-start greedy
    # Number of restarts depends on time
    # Estimate time per restart
    # Let's run as many as possible
    
    rng = np.random.default_rng(42)
    
    while time.time() < deadline:
        uncovered = np.ones(n_t, dtype=np.int8)
        blocks = []
        
        while uncovered.sum() > 0:
            # Calculate gains
            gains = M.dot(uncovered)
            
            # Randomize selection among top candidates
            # If gains are all 0, break (should not happen)
            if gains.max() == 0:
                break
                
            # Select top k
            # Use argsort on negative gains to get top k
            top_k = 10
            # Ensure we don't pick more than available
            top_k = min(top_k, n_k)
            # Get indices of top k
            # This is O(N log N) or O(N) with partial sort
            # np.argpartition is faster for top k
            indices = np.argpartition(gains, -top_k)[-top_k:]
            
            # Randomly pick one from top k
            # Weighted by gain? Or uniform?
            # Uniform among top k is a good exploration strategy
            best_idx = rng.choice(indices)
            
            blocks.append(k_subsets[best_idx])
            
            # Update uncovered
            row = M[best_idx]
            # row.indices are the columns where M has 1
            uncovered[row.indices] = 0
            
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
            # Optional: if we hit a very good score, maybe break early?
            # Lower bound is 208. If we get close, keep trying.
            
    return best_blocks
