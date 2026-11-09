import time
import random
import numpy as np

def construct():
    n = 10
    start_time = time.time()
    deadline = start_time + 50
    
    # Base Latin Square
    base = np.array([[ (i+j)%n for j in range(n)] for i in range(n)])
    
    best_result = [base.tolist()]
    
    # Try to find a second square
    # We will search in the space of isotopes of the base square
    # Row perm, Col perm, Symbol perm
    
    for _ in range(50000):
        if time.time() > deadline:
            break
        
        row_perm = np.random.permutation(n)
        col_perm = np.random.permutation(n)
        sym_perm = np.random.permutation(n)
        
        # Apply permutations
        # base[row_perm[i], col_perm[j]] gives the symbol
        # Then apply sym_perm to the symbol
        cand_np = sym_perm[base[row_perm][:, col_perm]]
        
        # Check orthogonality
        # pairs = base * 10 + cand_np
        # if len(np.unique(pairs)) == 100:
        #    return [base.tolist(), cand_np.tolist()]
        
        # Optimization: check unique pairs
        pairs = base * 10 + cand_np
        if len(np.unique(pairs)) == 100:
            best_result = [base.tolist(), cand_np.tolist()]
            break
            
    return best_result
