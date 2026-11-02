import itertools
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy import sparse
import time

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115
    
    # 1. Generate all blocks and subsets
    all_blocks = list(itertools.combinations(range(v), k))
    all_subsets = list(itertools.combinations(range(v), t))
    
    n_blocks = len(all_blocks)
    n_subsets = len(all_subsets)
    
    # Map subsets to indices
    subset_index = {s: i for i, s in enumerate(all_subsets)}
    
    # 2. Build sparse matrix A
    rows = []
    cols = []
    data = []
    
    for j, block in enumerate(all_blocks):
        for sub in itertools.combinations(block, t):
            i = subset_index[sub]
            rows.append(i)
            cols.append(j)
            data.append(1)
            
    A = sparse.coo_matrix((data, (rows, cols)), shape=(n_subsets, n_blocks), dtype=np.int8).tocsr()
    
    # 3. Define ILP
    c = np.ones(n_blocks)
    constraints = [LinearConstraint(A, lb=np.ones(n_subsets), ub=np.inf)]
    integrality = np.ones(n_blocks)
    bounds = Bounds(lb=0, ub=1)
    
    # 4. Solve
    try:
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
        
        if res.status == 0:
            x = res.x
            selected_blocks = [all_blocks[i] for i in range(n_blocks) if x[i] > 0.5]
            return [list(b) for b in selected_blocks]
        else:
            # Fallback
            pass
            
    except Exception:
        # Fallback
        pass

    # Fallback: Fast Greedy (First Available)
    uncovered = set(all_subsets)
    used_blocks = set()
    blocks = []
    
    while uncovered:
        s = next(iter(uncovered))
        i = subset_index[s]
        candidates = A[i].indices
        
        best_block_idx = -1
        for b_idx in candidates:
            if b_idx not in used_blocks:
                best_block_idx = b_idx
                break
                
        if best_block_idx == -1:
            break
            
        used_blocks.add(best_block_idx)
        blocks.append(all_blocks[best_block_idx])
        
        block = all_blocks[best_block_idx]
        for sub in itertools.combinations(block, t):
            uncovered.discard(sub)
            
    return [list(b) for b in blocks]
