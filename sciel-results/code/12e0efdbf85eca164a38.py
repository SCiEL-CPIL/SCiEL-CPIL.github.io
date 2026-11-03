import time
import numpy as np
import itertools

_t_start = time.time()
TIME_LIMIT = 110.0

def _time_left():
    return TIME_LIMIT - (time.time() - _t_start)

def construct():
    n, m, p = 3, 3, 8
    
    # Schoolbook rank
    schoolbook_rank = n * m * p  # 72
    
    # Define all possible Strassen blocks
    # I: pair of rows of A (0..2)
    # J: pair of cols of A (0..2) which are rows of B
    # K: pair of cols of B (0..7)
    
    row_pairs = [(0,1), (0,2), (1,2)]
    col_pairs_A = [(0,1), (0,2), (1,2)]
    col_pairs_B = list(itertools.combinations(range(8), 2))
    
    blocks = []
    for I in row_pairs:
        for J in col_pairs_A:
            for K in col_pairs_B:
                # The set of schoolbook indices covered by this block
                # (i, j, k) where i in I, j in J, k in K
                covered = set()
                for i in I:
                    for j in J:
                        for k in K:
                            covered.add((i, j, k))
                blocks.append((I, J, K, covered))
                
    # Find the maximum number of non-overlapping blocks
    # Use a simple greedy or backtracking search
    # Given the small size, we can try to find a large independent set
    
    # Let's try to find 8 blocks
    # We will use a backtracking search
    
    max_blocks = 0
    best_set = []
    
    # Sort blocks to improve pruning? Maybe by size of covered set (all are 8)
    # Or just iterate
    
    def backtrack(start, current_set):
        nonlocal max_blocks, best_set
        if _time_left() < 10.0:
            return
            
        if len(current_set) > max_blocks:
            max_blocks = len(current_set)
            best_set = current_set.copy()
            
        # Pruning: if we can't beat max_blocks, return
        # Max possible additional blocks is (len(blocks) - start)
        if len(current_set) + (len(blocks) - start) <= max_blocks:
            return
            
        for i in range(start, len(blocks)):
            # Check if block i is compatible with all in current_set
            compatible = True
            for j in range(len(current_set)):
                if blocks[i][3].intersection(blocks[current_set[j]][3]):
                    compatible = False
                    break
            if compatible:
                current_set.append(i)
                backtrack(i + 1, current_set)
                current_set.pop()
                
    backtrack(0, [])
    
    # If we didn't find enough, we might have to settle for less
    # But 8 should be findable
    
    num_blocks = max_blocks
    selected_blocks = [blocks[i] for i in best_set]
    
    # Construct the decomposition
    # Rank = 72 - num_blocks
    rank = 72 - num_blocks
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # First, add the Strassen terms for the selected blocks
    for I, J, K, covered in selected_blocks:
        i0, i1 = I
        j0, j1 = J
        k0, k1 = K
        
        # Strassen terms for 2x2x2 block
        # We need to map the local indices to global indices
        # Local U: (0,0), (0,1), (1,0), (1,1) -> global (i0,j0), (i0,j1), (i1,j0), (i1,j1)
        # Local V: (0,0), (0,1), (1,0), (1,1) -> global (j0,k0), (j0,k1), (j1,k0), (j1,k1)
        # Local W: (0,0), (0,1), (1,0), (1,1) -> global (k0,i0), (k0,i1), (k1,i0), (k1,i1)
        
        # Strassen coefficients
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)(B00)
        # M3 = (A00)(B10-B11)
        # M4 = (A11)(B01-B00)
        # M5 = (A00+A01)(B11)
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        
        # Term 1
        U[i0*m+j0, col] = 1.0
        U[i1*m+j1, col] = 1.0
        V[j0*p+k0, col] = 1.0
        V[j1*p+k1, col] = 1.0
        W[k0*n+i0, col] = 1.0
        W[k1*n+i1, col] = 1.0
        col += 1
        
        # Term 2
        U[i1*m+j0, col] = 1.0
        U[i1*m+j1, col] = 1.0
        V[j0*p+k0, col] = 1.0
        W[k0*n+i1, col] = 1.0
        col += 1
        
        # Term 3
        U[i0*m+j0, col] = 1.0
        V[j1*p+k0, col] = 1.0
        V[j1*p+k1, col] = -1.0
        W[k0*n+i0, col] = 1.0
        W[k1*n+i0, col] = -1.0
        col += 1
        
        # Term 4
        U[i1*m+j1, col] = 1.0
        V[j0*p+k1, col] = 1.0
        V[j0*p+k0, col] = -1.0
        W[k1*n+i1, col] = 1.0
        W[k0*n+i1, col] = -1.0
        col += 1
        
        # Term 5
        U[i0*m+j0, col] = 1.0
        U[i0*m+j1, col] = 1.0
        V[j1*p+k1, col] = 1.0
        W[k1*n+i0, col] = 1.0
        col += 1
        
        # Term 6
        U[i1*m+j0, col] = 1.0
        U[i0*m+j0, col] = -1.0
        V[j0*p+k0, col] = 1.0
        V[j1*p+k0, col] = 1.0
        W[k0*n+i1, col] = 1.0
        W[k0*n+i0, col] = -1.0
        col += 1
        
        # Term 7
        U[i0*m+j1, col] = 1.0
        U[i1*m+j1, col] = -1.0
        V[j0*p+k1, col] = 1.0
        V[j1*p+k1, col] = 1.0
        W[k1*n+i0, col] = 1.0
        W[k1*n+i1, col] = -1.0
        col += 1
        
    # Then, add the schoolbook terms for the remaining indices
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check if this term is covered by any selected block
                covered_by_block = False
                for I, J, K, cov in selected_blocks:
                    if i in I and j in J and k in K:
                        covered_by_block = True
                        break
                if not covered_by_block:
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
                    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
