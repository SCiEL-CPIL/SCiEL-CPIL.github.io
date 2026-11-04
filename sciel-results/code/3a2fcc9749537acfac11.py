import numpy as np
import random
import time

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    target_rank = n * m * p
    
    # Target tensor T is all ones in the (i,j,k) sense
    # We want sum_r U_r[i,j] V_r[j,k] W_r[i,k] == 1 for all i,j,k
    
    # Try to find a decomposition of rank < target_rank
    # We will try a few random 0-1 decompositions
    # If we find one, return it. Otherwise, return schoolbook.
    
    # Time limit for search
    start_time = time.time()
    max_time = 5.0
    
    # Try ranks from target_rank - 1 down to target_rank - 10
    for rank in range(target_rank - 1, target_rank - 11, -1):
        if time.time() - start_time > max_time:
            break
            
        # Try a few random instances for this rank
        for _ in range(100):
            if time.time() - start_time > max_time:
                break
                
            # Random 0-1 matrices
            # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
            # Note: W indices are k*n + i.
            
            U = np.random.randint(0, 2, (n * m, rank))
            V = np.random.randint(0, 2, (m * p, rank))
            W = np.random.randint(0, 2, (p * n, rank))
            
            # Check condition:
            # For each i, j, k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
            
            # We can vectorize the check
            # Create arrays of indices
            i_vals, j_vals, k_vals = np.meshgrid(np.arange(n), np.arange(m), np.arange(p), indexing='ij')
            i_flat = i_vals.flatten()
            j_flat = j_vals.flatten()
            k_flat = k_vals.flatten()
            
            u_idx = i_flat * m + j_flat
            v_idx = j_flat * p + k_flat
            w_idx = k_flat * n + i_flat
            
            # Extract the relevant rows
            # U_rows: (n*m*p, rank)
            # V_rows: (n*m*p, rank)
            # W_rows: (n*m*p, rank)
            U_rows = U[u_idx, :]
            V_rows = V[v_idx, :]
            W_rows = W[w_idx, :]
            
            # Compute sum over rank
            # prod = U_rows * V_rows * W_rows
            # sum_prod = prod.sum(axis=1)
            # Check if all are 1
            
            prod = U_rows * V_rows * W_rows
            sum_prod = prod.sum(axis=1)
            
            if np.all(sum_prod == 1):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
    # Fallback to schoolbook
    rank = target_rank
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    i_vals, j_vals, k_vals = np.meshgrid(np.arange(n), np.arange(m), np.arange(p), indexing='ij')
    i_flat = i_vals.flatten()
    j_flat = j_vals.flatten()
    k_flat = k_vals.flatten()
    
    u_indices = i_flat * m + j_flat
    v_indices = j_flat * p + k_flat
    w_indices = k_flat * n + i_flat
    col_indices = np.arange(rank)
    
    U[u_indices, col_indices] = 1.0
    V[v_indices, col_indices] = 1.0
    W[w_indices, col_indices] = 1.0
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
