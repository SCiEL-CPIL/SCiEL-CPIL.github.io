import itertools
import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s margin

    points = list(range(12))
    
    # Generate all 6-subsets (blocks)
    all_blocks = list(itertools.combinations(points, 6))
    n_blocks = len(all_blocks)
    
    # Generate all 5-subsets
    all_5sets = list(itertools.combinations(points, 5))
    n_5sets = len(all_5sets)
    
    # Map 5-set to index for quick lookup
    # Actually, we need to build the matrix A where A[i, j] = 1 if 5set[i] is in block[j]
    # It is more efficient to iterate over blocks and their 5-subsets.
    
    # Build sparse matrix A (n_5sets x n_blocks)
    # lil_matrix is good for row-wise construction, but we are adding column-wise (block by block)
    # Actually, lil_matrix is row-oriented. 
    # Let's use coo_matrix or just a list of (row, col, val) and convert.
    
    rows = []
    cols = []
    vals = []
    
    # Index of each 5-set
    # We need a mapping from 5-tuple to index.
    # Since 792 is small, a dict is fine.
    idx_5set = {s: i for i, s in enumerate(all_5sets)}
    
    for j, block in enumerate(all_blocks):
        # Get all 5-subsets of this block
        for sub in itertools.combinations(block, 5):
            i = idx_5set[sub]
            rows.append(i)
            cols.append(j)
            vals.append(1)
            
    # Convert to CSC or CSR? scipy.optimize.milp accepts sparse matrices.
    # Let's use coo_matrix.
    from scipy.sparse import coo_matrix
    A = coo_matrix((vals, (rows, cols)), shape=(n_5sets, n_blocks))
    
    # Objective: maximize sum of x_j
    c = np.ones(n_blocks)
    
    # Constraints: A @ x <= 1
    # A is (792, 924)
    # x is (924,)
    # A @ x has length 792
    cons = LinearConstraint(A, -np.inf, 1)
    
    # Bounds: 0 <= x_j <= 1
    bounds = Bounds(0, 1)
    
    # Integrality: all binary
    integrality = np.ones(n_blocks)
    
    # Solve
    try:
        res = milp(c=c, constraints=cons, integrality=integrality, bounds=bounds)
        
        if res.success:
            x = res.x
            # Extract blocks where x_j is close to 1
            selected_blocks = []
            for j in range(n_blocks):
                if x[j] > 0.5:
                    selected_blocks.append(list(all_blocks[j]))
            
            # Verify validity just in case
            if len(selected_blocks) > 0:
                # Check no 5-subset is repeated
                seen_5sets = set()
                valid = True
                for b in selected_blocks:
                    for s in itertools.combinations(b, 5):
                        if s in seen_5sets:
                            valid = False
                            break
                        seen_5sets.add(s)
                    if not valid:
                        break
                
                if valid:
                    return selected_blocks
            
            # If ILP returned a solution but it was invalid or empty, fall back
            # Or if it didn't reach optimum?
            # The objective is maximize count. If it returns 132, it's optimal.
            # If it returns less, it might be a local optimum or solver issue.
            # But for this small problem, it should find 132.
            
    except Exception as e:
        pass
    
    # Fallback: Greedy with restarts (improved version)
    # If ILP fails, we need a good heuristic.
    # The previous greedy was weak.
    # Let's try a "Maximal Independent Set" approach on the conflict graph?
    # Nodes: 924 blocks.
    # Edge between two blocks if they share a 5-subset.
    # We want Max Independent Set.
    # This is NP-hard, but heuristics work well.
    
    # However, ILP is so likely to work that the fallback is just for safety.
    # Let's implement a simple greedy as a last resort.
    
    def greedy_fallback():
        best = []
        for _ in range(50):
            if time.time() > deadline:
                break
            random.shuffle(all_blocks)
            covered = set()
            blocks = []
            for b in all_blocks:
                subs = list(itertools.combinations(b, 5))
                if not any(s in covered for s in subs):
                    blocks.append(b)
                    covered.update(subs)
            if len(blocks) > len(best):
                best = blocks
                if len(best) == 132:
                    return [list(b) for b in best]
        return [list(b) for b in best]
        
    import random
    return greedy_fallback()
