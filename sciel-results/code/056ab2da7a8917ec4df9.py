"""Matrix multiplication 2,5,6 - rank-minimal tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START

def _build_strassen_rank54(n, m, p):
    """Build the known rank-54 Strassen-based decomposition."""
    rank = 54
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    current_col = 0
    
    def next_col():
        nonlocal current_col
        current_col += 1

    # Strassen blocks for j-pairs (0,1) and (2,3)
    for j_base in [0, 2]:
        for k_base in [0, 2, 4]:
            i_base = 0

            # M0: (A00+A11)(B00+B11) -> C00, C11
            U[i_base*m+j_base, current_col] = 1
            U[(i_base+1)*m+(j_base+1), current_col] = 1
            V[j_base*p+k_base, current_col] = 1
            V[(j_base+1)*p+(k_base+1), current_col] = 1
            W[k_base*n+i_base, current_col] = 1
            W[(k_base+1)*n+(i_base+1), current_col] = 1
            next_col()

            # M1: (A10+A11)B00
            U[(i_base+1)*m+j_base, current_col] = 1
            U[(i_base+1)*m+(j_base+1), current_col] = 1
            V[j_base*p+k_base, current_col] = 1
            W[k_base*n+(i_base+1), current_col] = 1
            W[(k_base+1)*n+(i_base+1), current_col] = -1
            next_col()

            # M2: A00(B01-B11)
            U[i_base*m+j_base, current_col] = 1
            V[j_base*p+(k_base+1), current_col] = 1
            V[(j_base+1)*p+(k_base+1), current_col] = -1
            W[(k_base+1)*n+i_base, current_col] = 1
            W[(k_base+1)*n+(i_base+1), current_col] = 1
            next_col()

            # M3: A11(B10-B00)
            U[(i_base+1)*m+(j_base+1), current_col] = 1
            V[(j_base+1)*p+k_base, current_col] = 1
            V[j_base*p+k_base, current_col] = -1
            W[k_base*n+i_base, current_col] = 1
            W[k_base*n+(i_base+1), current_col] = 1
            next_col()

            # M4: (A00+A01)B11
            U[i_base*m+j_base, current_col] = 1
            U[i_base*m+(j_base+1), current_col] = 1
            V[(j_base+1)*p+(k_base+1), current_col] = 1
            W[k_base*n+i_base, current_col] = -1
            W[(k_base+1)*n+i_base, current_col] = 1
            next_col()

            # M5: (A10-A00)(B00+B01)
            U[(i_base+1)*m+j_base, current_col] = 1
            U[i_base*m+j_base, current_col] = -1
            V[j_base*p+k_base, current_col] = 1
            V[j_base*p+(k_base+1), current_col] = 1
            W[(k_base+1)*n+(i_base+1), current_col] = 1
            next_col()

            # M6: (A01-A11)(B10+B11)
            U[i_base*m+(j_base+1), current_col] = 1
            U[(i_base+1)*m+(j_base+1), current_col] = -1
            V[(j_base+1)*p+k_base, current_col] = 1
            V[(j_base+1)*p+(k_base+1), current_col] = 1
            W[k_base*n+i_base, current_col] = 1
            next_col()
    
    # Naive blocks for j=4
    for k_base in [0, 2, 4]:
        i_base = 0
        j_idx = 4
        
        U[i_base*m+j_idx, current_col] = 1
        V[j_idx*p+k_base, current_col] = 1
        W[k_base*n+i_base, current_col] = 1
        next_col()
        
        U[i_base*m+j_idx, current_col] = 1
        V[j_idx*p+(k_base+1), current_col] = 1
        W[(k_base+1)*n+i_base, current_col] = 1
        next_col()
        
        U[(i_base+1)*m+j_idx, current_col] = 1
        V[j_idx*p+k_base, current_col] = 1
        W[k_base*n+(i_base+1), current_col] = 1
        next_col()
        
        U[(i_base+1)*m+j_idx, current_col] = 1
        V[j_idx*p+(k_base+1), current_col] = 1
        W[(k_base+1)*n+(i_base+1), current_col] = 1
        next_col()
    
    if current_col != rank:
        raise ValueError(f"Rank mismatch: expected {rank}, got {current_col}")
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    U, V, W = _build_strassen_rank54(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
