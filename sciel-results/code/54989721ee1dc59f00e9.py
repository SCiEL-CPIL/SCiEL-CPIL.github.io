"""Matrix multiplication 2,5,6 rank-minimal tensor decomposition."""
import numpy as np
import time
import itertools
import random

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS  # 2, 5, 6
    
    # Build rank-54 Strassen decomposition
    rank_base = 54
    U = [[0] * rank_base for _ in range(n * m)]
    V = [[0] * rank_base for _ in range(m * p)]
    W = [[0] * rank_base for _ in range(p * n)]
    col = 0
    
    def add_strassen_block(i_base, j_base, k_base):
        nonlocal col
        # M0: (A00+A11)(B00+B11)
        U[i_base*m+j_base][col] += 1
        U[(i_base+1)*m+(j_base+1)][col] += 1
        V[j_base*p+k_base][col] += 1
        V[(j_base+1)*p+(k_base+1)][col] += 1
        W[k_base*n+i_base][col] += 1
        W[(k_base+1)*n+(i_base+1)][col] += 1
        col += 1
        # M1: (A10+A11)B00
        U[(i_base+1)*m+j_base][col] += 1
        U[(i_base+1)*m+(j_base+1)][col] += 1
        V[j_base*p+k_base][col] += 1
        W[k_base*n+(i_base+1)][col] += 1
        W[(k_base+1)*n+(i_base+1)][col] -= 1
        col += 1
        # M2: A00(B01-B11)
        U[i_base*m+j_base][col] += 1
        V[j_base*p+(k_base+1)][col] += 1
        V[(j_base+1)*p+(k_base+1)][col] -= 1
        W[(k_base+1)*n+i_base][col] += 1
        W[(k_base+1)*n+(i_base+1)][col] += 1
        col += 1
        # M3: A11(B10-B00)
        U[(i_base+1)*m+(j_base+1)][col] += 1
        V[(j_base+1)*p+k_base][col] += 1
        V[j_base*p+k_base][col] -= 1
        W[k_base*n+i_base][col] += 1
        W[k_base*n+(i_base+1)][col] += 1
        col += 1
        # M4: (A00+A01)B11
        U[i_base*m+j_base][col] += 1
        U[i_base*m+(j_base+1)][col] += 1
        V[(j_base+1)*p+(k_base+1)][col] += 1
        W[k_base*n+i_base][col] -= 1
        W[(k_base+1)*n+i_base][col] += 1
        col += 1
        # M5: (A10-A00)(B00+B01)
        U[(i_base+1)*m+j_base][col] += 1
        U[i_base*m+j_base][col] -= 1
        V[j_base*p+k_base][col] += 1
        V[j_base*p+(k_base+1)][col] += 1
        W[(k_base+1)*n+(i_base+1)][col] += 1
        col += 1
        # M6: (A01-A11)(B10+B11)
        U[i_base*m+(j_base+1)][col] += 1
        U[(i_base+1)*m+(j_base+1)][col] -= 1
        V[(j_base+1)*p+k_base][col] += 1
        V[(j_base+1)*p+(k_base+1)][col] += 1
        W[k_base*n+i_base][col] += 1
        col += 1

    def add_naive_block(i_base, j_base, k_base):
        nonlocal col
        U[i_base*m+j_base][col] += 1
        V[j_base*p+k_base][col] += 1
        W[k_base*n+i_base][col] += 1
        col += 1
        U[i_base*m+j_base][col] += 1
        V[j_base*p+(k_base+1)][col] += 1
        W[(k_base+1)*n+i_base][col] += 1
        col += 1
        U[(i_base+1)*m+j_base][col] += 1
        V[j_base*p+k_base][col] += 1
        W[k_base*n+(i_base+1)][col] += 1
        col += 1
        U[(i_base+1)*m+j_base][col] += 1
        V[j_base*p+(k_base+1)][col] += 1
        W[(k_base+1)*n+(i_base+1)][col] += 1
        col += 1

    add_strassen_block(0, 0, 0)
    add_strassen_block(0, 0, 2)
    add_strassen_block(0, 0, 4)
    add_strassen_block(0, 2, 0)
    add_strassen_block(0, 2, 2)
    add_strassen_block(0, 2, 4)
    add_naive_block(0, 4, 0)
    add_naive_block(0, 4, 2)
    add_naive_block(0, 4, 4)
    
    if col != rank_base:
        rank = n*m*p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i*m*p + j*p + k
                    U[i*m+j][idx] = 1
                    V[j*p+k][idx] = 1
                    W[k*n+i][idx] = 1
        return {"U": U, "V": V, "W": W}
    
    # Convert to numpy for verification
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Build target tensor
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
    
    # Verify base decomposition
    T_dec = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    
    if not np.allclose(T_dec, T, atol=1e-10):
        # Fallback to naive
        rank = n*m*p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i*m*p + j*p + k
                    U[i*m+j][idx] = 1
                    V[j*p+k][idx] = 1
                    W[k*n+i][idx] = 1
        return {"U": U, "V": V, "W": W}
    
    best_rank = rank_base
    best_U = [row[:] for row in U]
    best_V = [row[:] for row in V]
    best_W = [row[:] for row in W]
    
    # Try rank reduction via column merging
    # For each pair of columns, check if one can be expressed as a linear combination
    # of the others, and if so, try to eliminate it
    
    if time.time() < deadline - 30.0:
        # Try to reduce rank by finding linearly dependent columns
        # Use SVD to find the rank of the decomposition
        # and try to eliminate redundant columns
        
        # First, check if any column is zero (shouldn't be, but verify)
        for r in range(rank_base):
            if np.allclose(U_arr[:, r], 0) or np.allclose(V_arr[:, r], 0) or np.allclose(W_arr[:, r], 0):
                # This column is zero, remove it
                U_arr = np.delete(U_arr, r, axis=1)
                V_arr = np.delete(V_arr, r, axis=1)
                W_arr = np.delete(W_arr, r, axis=1)
                rank_base -= 1
                break
        
        # Try to find if any two columns can be merged
        # This is a complex optimization problem, so we'll try a simple heuristic
        # Try each column and see if removing it still gives a valid decomposition
        # after adjusting other columns
        
        # For now, stick with the current rank but verify it's correct
        T_dec = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
        if np.allclose(T_dec, T, atol=1e-10):
            best_rank = rank_base
            best_U = U_arr.tolist()
            best_V = V_arr.tolist()
            best_W = W_arr.tolist()
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
