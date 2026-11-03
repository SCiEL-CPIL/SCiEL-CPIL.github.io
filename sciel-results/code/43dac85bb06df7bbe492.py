"""Rewritten program using a distinct construction strategy for 2x3x8 matrix multiplication."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = 2, 3, 8
    rank = 36  # 4 blocks of (7 Strassen + 2 schoolbook) = 36
    
    # Build decomposition U, V, W satisfying:
    #   T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # where T[i,j,k] = 1 for all valid (i,j,k)
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    def add_strassen_2x2x2(i_base, j_base, k_base):
        """Add Strassen decomposition for 2x2x2 sub-block"""
        nonlocal col
        # m1: (a+d)(e+h)
        U[(i_base+0)*m + (j_base+0), col] += 1
        U[(i_base+1)*m + (j_base+1), col] += 1
        V[(j_base+0)*p + (k_base+0), col] += 1
        V[(j_base+1)*p + (k_base+1), col] += 1
        W[(k_base+0)*n + (i_base+0), col] += 1
        W[(k_base+1)*n + (i_base+1), col] += 1
        col += 1
        
        # m2: (c+d)e
        U[(i_base+1)*m + (j_base+0), col] += 1
        U[(i_base+1)*m + (j_base+1), col] += 1
        V[(j_base+0)*p + (k_base+0), col] += 1
        W[(k_base+0)*n + (i_base+1), col] += 1
        W[(k_base+1)*n + (i_base+1), col] += -1
        col += 1
        
        # m3: a(f-h)
        U[(i_base+0)*m + (j_base+0), col] += 1
        V[(j_base+0)*p + (k_base+1), col] += 1
        V[(j_base+1)*p + (k_base+1), col] += -1
        W[(k_base+1)*n + (i_base+0), col] += 1
        W[(k_base+1)*n + (i_base+1), col] += 1
        col += 1
        
        # m4: d(g-e)
        U[(i_base+1)*m + (j_base+1), col] += 1
        V[(j_base+1)*p + (k_base+0), col] += 1
        V[(j_base+0)*p + (k_base+0), col] += -1
        W[(k_base+0)*n + (i_base+0), col] += 1
        W[(k_base+0)*n + (i_base+1), col] += 1
        col += 1
        
        # m5: (a+b)h
        U[(i_base+0)*m + (j_base+0), col] += 1
        U[(i_base+0)*m + (j_base+1), col] += 1
        V[(j_base+1)*p + (k_base+1), col] += 1
        W[(k_base+0)*n + (i_base+0), col] += -1
        W[(k_base+1)*n + (i_base+0), col] += 1
        col += 1
        
        # m6: (c-a)(e+f)
        U[(i_base+1)*m + (j_base+0), col] += 1
        U[(i_base+0)*m + (j_base+0), col] += -1
        V[(j_base+0)*p + (k_base+0), col] += 1
        V[(j_base+0)*p + (k_base+1), col] += 1
        W[(k_base+1)*n + (i_base+1), col] += 1
        col += 1
        
        # m7: (b-d)(g+h)
        U[(i_base+0)*m + (j_base+1), col] += 1
        U[(i_base+1)*m + (j_base+1), col] += -1
        V[(j_base+1)*p + (k_base+0), col] += 1
        V[(j_base+1)*p + (k_base+1), col] += 1
        W[(k_base+0)*n + (i_base+0), col] += 1
        col += 1
    
    def add_schoolbook_2x1x2(i_base, j_val, k_base):
        """Add schoolbook for 2x1x2 sub-block"""
        nonlocal col
        for di in range(2):
            for dk in range(2):
                U[(i_base+di)*m + j_val, col] = 1
                V[j_val*p + (k_base+dk), col] = 1
                W[(k_base+dk)*n + (i_base+di), col] = 1
                col += 1
    
    # Process 4 blocks of p=2
    for block in range(4):
        if time.time() > deadline:
            break
        k_base = block * 2
        # Strassen for j=0,1 (2x2x2 part)
        add_strassen_2x2x2(0, 0, k_base)
        # Schoolbook for j=2 (2x1x2 part)
        add_schoolbook_2x1x2(0, 2, k_base)
    
    # Verify correctness if we completed all blocks
    if col == rank:
        # Check: T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        # T[i,j,k] should be 1 for all valid (i,j,k)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - 1.0) > 1e-10:
                        raise ValueError(f"Decomposition failed at ({i},{j},{k}): got {s}")
    
    # Convert to lists of lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
