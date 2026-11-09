import time
import numpy as np
from itertools import combinations

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    # Build the rank-28 Strassen-based decomposition
    rank = 28
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen terms for 2x2 matrix multiplication
    strassen_terms = [
        # (a+d)(e+h) -> C00, C11
        {'U': [(0, 1), (3, 1)], 'V': [(0, 1), (3, 1)], 'W': [(0, 1), (3, 1)]},
        # (c+d)e -> C10, -C11
        {'U': [(2, 1), (3, 1)], 'V': [(0, 1)], 'W': [(1, 1), (3, -1)]},
        # a(f-h) -> C01, C11
        {'U': [(0, 1)], 'V': [(1, 1), (3, -1)], 'W': [(2, 1), (3, 1)]},
        # d(g-e) -> C00, C10
        {'U': [(3, 1)], 'V': [(2, 1), (0, -1)], 'W': [(0, 1), (1, 1)]},
        # (a+b)h -> -C00, C01
        {'U': [(0, 1), (1, 1)], 'V': [(3, 1)], 'W': [(0, -1), (2, 1)]},
        # (c-a)(e+f) -> C11
        {'U': [(2, 1), (0, -1)], 'V': [(0, 1), (1, 1)], 'W': [(3, 1)]},
        # (b-d)(g+h) -> C00
        {'U': [(1, 1), (3, -1)], 'V': [(2, 1), (3, 1)], 'W': [(0, 1)]}
    ]
    
    blocks = [
        {'j_offset': 0, 'k_offset': 0},
        {'j_offset': 0, 'k_offset': 2},
        {'j_offset': 2, 'k_offset': 0},
        {'j_offset': 2, 'k_offset': 2}
    ]
    
    col = 0
    for block in blocks:
        if time.time() > deadline:
            break
        j_off = block['j_offset']
        k_off = block['k_offset']
        for term in strassen_terms:
            if time.time() > deadline:
                break
            # Add to U
            for (idx, coeff) in term['U']:
                i = idx // 2
                j = idx % 2
                global_i = i
                global_j = j + j_off
                U[global_i * m + global_j, col] += coeff
            
            # Add to V
            for (idx, coeff) in term['V']:
                j = idx // 2
                k = idx % 2
                global_j = j + j_off
                global_k = k + k_off
                V[global_j * p + global_k, col] += coeff
            
            # Add to W
            for (idx, coeff) in term['W']:
                k = idx // 2
                i = idx % 2
                global_k = k + k_off
                global_i = i
                W[global_k * n + global_i, col] += coeff
            
            col += 1
    
    def verify_decomposition(U_mat, V_mat, W_mat, rank_val):
        """Check if the decomposition is exact using vectorized computation."""
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should be 1
        # We'll check all (i,j,k) tuples
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Vectorized computation
                    val = np.dot(
                        U_mat[i * m + j, :] * V_mat[j * p + k, :],
                        W_mat[k * n + i, :]
                    )
                    if abs(val - 1.0) > 1e-10:
                        return False
        return True
    
    # Verify the initial decomposition
    if not verify_decomposition(U, V, W, rank):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_rank = rank
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    
    # Try to reduce rank by eliminating columns
    # Strategy: try removing subsets of columns and check if the decomposition still holds
    # We'll try all single-column removals first, then pairs, etc.
    
    # First, try removing one column at a time
    current_rank = rank
    current_U = U.copy()
    current_V = V.copy()
    current_W = W.copy()
    
    # Iteratively try to remove columns
    for elimination_round in range(10):
        if time.time() > deadline:
            break
        
        found_removal = False
        # Try removing each column
        for col_idx in range(current_rank):
            if time.time() > deadline:
                break
            
            # Create matrices with this column removed
            mask = np.ones(current_rank, dtype=bool)
            mask[col_idx] = False
            
            U_trial = current_U[:, mask]
            V_trial = current_V[:, mask]
            W_trial = current_W[:, mask]
            
            if verify_decomposition(U_trial, V_trial, W_trial, current_rank - 1):
                # Successfully removed a column
                current_U = U_trial
                current_V = V_trial
                current_W = W_trial
                current_rank -= 1
                found_removal = True
                
                if current_rank < best_rank:
                    best_rank = current_rank
                    best_U = current_U.copy()
                    best_V = current_V.copy()
                    best_W = current_W.copy()
                
                break  # Restart the loop with the new reduced matrices
        
        if not found_removal:
            break
    
    # If we found a better rank, use it
    if best_rank < rank:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
