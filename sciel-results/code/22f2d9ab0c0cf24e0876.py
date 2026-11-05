"""Matrix multiplication 2,5,6 - rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS
    rank = 54
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    
    # Process Strassen blocks for j-pairs (0,1) and (2,3)
    # j_base in {0, 2}, k_base in {0, 2, 4}
    for j_base in [0, 2]:
        for k_base in [0, 2, 4]:
            i_base = 0
            
            # M0: (A00+A11)(B00+B11)
            U[i_base*m+j_base, col] = 1
            U[(i_base+1)*m+(j_base+1), col] = 1
            V[j_base*p+k_base, col] = 1
            V[(j_base+1)*p+(k_base+1), col] = 1
            W[k_base*n+i_base, col] = 1
            W[(k_base+1)*n+(i_base+1), col] = 1
            col += 1
            
            # M1: (A10+A11)B00
            U[(i_base+1)*m+j_base, col] = 1
            U[(i_base+1)*m+(j_base+1), col] = 1
            V[j_base*p+k_base, col] = 1
            W[k_base*n+(i_base+1), col] = 1
            W[(k_base+1)*n+(i_base+1), col] = -1
            col += 1
            
            # M2: A00(B01-B11)
            U[i_base*m+j_base, col] = 1
            V[j_base*p+(k_base+1), col] = 1
            V[(j_base+1)*p+(k_base+1), col] = -1
            W[(k_base+1)*n+i_base, col] = 1
            W[(k_base+1)*n+(i_base+1), col] = 1
            col += 1
            
            # M3: A11(B10-B00)
            U[(i_base+1)*m+(j_base+1), col] = 1
            V[(j_base+1)*p+k_base, col] = 1
            V[j_base*p+k_base, col] = -1
            W[k_base*n+i_base, col] = 1
            W[k_base*n+(i_base+1), col] = 1
            col += 1
            
            # M4: (A00+A01)B11
            U[i_base*m+j_base, col] = 1
            U[i_base*m+(j_base+1), col] = 1
            V[(j_base+1)*p+(k_base+1), col] = 1
            W[k_base*n+i_base, col] = -1
            W[(k_base+1)*n+i_base, col] = 1
            col += 1
            
            # M5: (A10-A00)(B00+B01)
            U[(i_base+1)*m+j_base, col] = 1
            U[i_base*m+j_base, col] = -1
            V[j_base*p+k_base, col] = 1
            V[j_base*p+(k_base+1), col] = 1
            W[(k_base+1)*n+(i_base+1), col] = 1
            col += 1
            
            # M6: (A01-A11)(B10+B11)
            U[i_base*m+(j_base+1), col] = 1
            U[(i_base+1)*m+(j_base+1), col] = -1
            V[(j_base+1)*p+k_base, col] = 1
            V[(j_base+1)*p+(k_base+1), col] = 1
            W[k_base*n+i_base, col] = 1
            col += 1
            
            # Time check
            if time.time() > 100:
                break
        if time.time() > 100:
            break
    
    # Process Naive blocks for j=4
    # k_bases are 0, 2, 4
    for k_base in [0, 2, 4]:
        i_base = 0
        j_idx = 4
        
        # Term 1: A[i0, j] * B[j, k0] -> C[i0, k0]
        U[i_base*m+j_idx, col] = 1
        V[j_idx*p+k_base, col] = 1
        W[k_base*n+i_base, col] = 1
        col += 1
        
        # Term 2: A[i0, j] * B[j, k1] -> C[i0, k1]
        U[i_base*m+j_idx, col] = 1
        V[j_idx*p+(k_base+1), col] = 1
        W[(k_base+1)*n+i_base, col] = 1
        col += 1
        
        # Term 3: A[i1, j] * B[j, k0] -> C[i1, k0]
        U[(i_base+1)*m+j_idx, col] = 1
        V[j_idx*p+k_base, col] = 1
        W[k_base*n+(i_base+1), col] = 1
        col += 1
        
        # Term 4: A[i1, j] * B[j, k1] -> C[i1, k1]
        U[(i_base+1)*m+j_idx, col] = 1
        V[j_idx*p+(k_base+1), col] = 1
        W[(k_base+1)*n+(i_base+1), col] = 1
        col += 1
        
        if time.time() > 100:
            break
    
    if col != rank:
        raise ValueError(f"Rank mismatch: expected {rank}, got {col}")
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
