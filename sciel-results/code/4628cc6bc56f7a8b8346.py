import numpy as np
import time

_START_TIME = time.time()
DEADLINE_SECONDS = 130

def _elapsed():
    return time.time() - _START_TIME

def _time_remaining():
    return DEADLINE_SECONDS - _elapsed()

def construct():
    """
    Computes the rank-minimal tensor decomposition for 2x4 * 4x7 matrix multiplication.
    Uses Strassen for 2x2x2 sub-blocks and schoolbook for the remainder.
    """
    n, m, p = 2, 4, 7
    size_u = n * m
    size_v = m * p
    size_w = p * n
    
    # Rank calculation:
    # 6 blocks of 2x2x2 (Strassen rank 7) -> 42
    # 1 block of 2x4x4x1 (Schoolbook rank 8) -> 8
    # Total rank 50
    total_rank = 50
    
    # Initialize factor matrices
    U = np.zeros((size_u, total_rank), dtype=float)
    V = np.zeros((size_v, total_rank), dtype=float)
    W = np.zeros((size_w, total_rank), dtype=float)
    
    current_col = 0
    
    def add_strassen_block(i_base, j_base, k_base):
        """
        Adds a Strassen 2x2x2 block to the decomposition.
        i_base: row index in A (0 or 1, but A is 2x4 so rows are 0,1)
        j_base: col index in A (0 or 2)
        k_base: col index in B (0, 2, or 4)
        """
        nonlocal current_col
        
        # A indices (flattened i*m + j)
        a1 = i_base * m + j_base
        a2 = i_base * m + (j_base + 1)
        a3 = (i_base + 1) * m + j_base
        a4 = (i_base + 1) * m + (j_base + 1)
        
        # B indices (flattened j*p + k)
        b1 = j_base * p + k_base
        b2 = j_base * p + (k_base + 1)
        b3 = (j_base + 1) * p + k_base
        b4 = (j_base + 1) * p + (k_base + 1)
        
        # W indices (flattened k*n + i)
        w1 = k_base * n + i_base
        w2 = k_base * n + (i_base + 1)
        w3 = (k_base + 1) * n + i_base
        w4 = (k_base + 1) * n + (i_base + 1)

        # M1 = (a1+a4)(b1+b4)
        U[a1, current_col] += 1
        U[a4, current_col] += 1
        V[b1, current_col] += 1
        V[b4, current_col] += 1
        W[w1, current_col] += 1
        W[w4, current_col] += 1
        current_col += 1
        
        # M2 = (a3+a4)b1
        U[a3, current_col] += 1
        U[a4, current_col] += 1
        V[b1, current_col] += 1
        W[w2, current_col] += 1
        W[w4, current_col] -= 1
        current_col += 1
        
        # M3 = a1(b2-b4)
        U[a1, current_col] += 1
        V[b2, current_col] += 1
        V[b4, current_col] -= 1
        W[w3, current_col] += 1
        W[w4, current_col] += 1
        current_col += 1
        
        # M4 = a4(b3-b1)
        U[a4, current_col] += 1
        V[b3, current_col] += 1
        V[b1, current_col] -= 1
        W[w1, current_col] += 1
        W[w2, current_col] += 1
        current_col += 1
        
        # M5 = (a1+a2)b4
        U[a1, current_col] += 1
        U[a2, current_col] += 1
        V[b4, current_col] += 1
        W[w1, current_col] -= 1
        W[w3, current_col] += 1
        current_col += 1
        
        # M6 = (a3-a1)(b1+b2)
        U[a3, current_col] += 1
        U[a1, current_col] -= 1
        V[b1, current_col] += 1
        V[b2, current_col] += 1
        W[w4, current_col] += 1
        current_col += 1
        
        # M7 = (a2-a4)(b3+b4)
        U[a2, current_col] += 1
        U[a4, current_col] -= 1
        V[b3, current_col] += 1
        V[b4, current_col] += 1
        W[w1, current_col] += 1
        current_col += 1

    # Apply Strassen to the 6 2x2x2 blocks
    # A is 2x4, so j_base in {0, 2}
    # B is 4x7, so k_base in {0, 2, 4} for the first 6 cols
    for j_start in [0, 2]:
        for k_start in [0, 2, 4]:
            add_strassen_block(0, j_start, k_start)
            
    # Handle the last column (k=6) with schoolbook
    # C[:, 6] = A @ B[:, 6]
    # C[i, 6] = sum(j=0..3) A[i, j] * B[j, 6]
    for j in range(m):
        for i in range(n):
            u_idx = i * m + j
            v_idx = j * p + 6
            w_idx = 6 * n + i
            
            U[u_idx, current_col] = 1
            V[v_idx, current_col] = 1
            W[w_idx, current_col] = 1
            current_col += 1
            
    # Convert to lists for the required interface
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
