"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]

import time
import numpy as np

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS
    
    # Rank 21 decomposition for M_{2,2,6}
    # The tensor decomposes into 3 independent 2x2 matrix multiplications.
    # Each 2x2 multiplication has rank 7 (Strassen).
    # Total rank = 3 * 7 = 21.
    
    rank = 21
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 multiplication:
    # P1 = (a+d)(e+h)
    # P2 = (c+d)e
    # P3 = a(f-h)
    # P4 = d(g-e)
    # P5 = (a+b)h
    # P6 = (c-a)(e+f)
    # P7 = (b-d)(g+h)
    #
    # Indices for A (2x2): 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Indices for B (2x2): 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Indices for C (2x2): 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    #
    # Mapping to flat indices:
    # A[i, k] -> i*m + k
    # B[k, j] -> k*p + j
    # C[j, i] -> j*n + i (Note: W convention is W[k*n+i,r] where k is col of C, i is row of C)
    
    for block in range(3):
        # V indices for this block
        # B[k, j]. In block `b`, the local columns are j_local=0,1. Global j = b*2 + j_local.
        # e = B[0, j0] -> 0*p + (b*2+0)
        # f = B[0, j1] -> 0*p + (b*2+1)
        # g = B[1, j0] -> 1*p + (b*2+0)
        # h = B[1, j1] -> 1*p + (b*2+1)
        
        v_e = 0 * p + (block * 2 + 0)
        v_f = 0 * p + (block * 2 + 1)
        v_g = 1 * p + (block * 2 + 0)
        v_h = 1 * p + (block * 2 + 1)
        
        # W indices: W[k*n + i, r] where k is col of C, i is row of C.
        # In block `b`, the relevant C columns are k_global = b*2 and b*2+1.
        
        # c00 (C[0, k0]): row=0, col=k0. Index = k0*n + 0
        # c01 (C[0, k1]): row=0, col=k1. Index = k1*n + 0
        # c10 (C[1, k0]): row=1, col=k0. Index = k0*n + 1
        # c11 (C[1, k1]): row=1, col=k1. Index = k1*n + 1
        
        k0 = block * 2
        k1 = block * 2 + 1
        
        w_c00 = k0 * n + 0
        w_c01 = k1 * n + 0
        w_c10 = k0 * n + 1
        w_c11 = k1 * n + 1
        
        # Step 0: P1 = (a+d)(e+h)
        r = block * 7 + 0
        U[0][r] = 1; U[3][r] = 1
        V[v_e][r] = 1; V[v_h][r] = 1
        W[w_c00][r] = 1; W[w_c11][r] = 1
        
        # Step 1: P2 = (c+d)e
        r = block * 7 + 1
        U[2][r] = 1; U[3][r] = 1
        V[v_e][r] = 1
        W[w_c10][r] = 1; W[w_c11][r] = -1
        
        # Step 2: P3 = a(f-h)
        r = block * 7 + 2
        U[0][r] = 1
        V[v_f][r] = 1; V[v_h][r] = -1
        W[w_c01][r] = 1; W[w_c11][r] = 1
        
        # Step 3: P4 = d(g-e)
        r = block * 7 + 3
        U[3][r] = 1
        V[v_g][r] = 1; V[v_e][r] = -1
        W[w_c00][r] = 1; W[w_c01][r] = 1
        
        # Step 4: P5 = (a+b)h
        r = block * 7 + 4
        U[0][r] = 1; U[1][r] = 1
        V[v_h][r] = 1
        W[w_c00][r] = -1; W[w_c01][r] = 1
        
        # Step 5: P6 = (c-a)(e+f)
        r = block * 7 + 5
        U[2][r] = 1; U[0][r] = -1
        V[v_e][r] = 1; V[v_f][r] = 1
        W[w_c11][r] = 1
        
        # Step 6: P7 = (b-d)(g+h)
        r = block * 7 + 6
        U[1][r] = 1; U[3][r] = -1
        V[v_g][r] = 1; V[v_h][r] = 1
        W[w_c00][r] = 1

    # Verify correctness
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    U3 = U_arr.reshape(n, m, rank)
    V3 = V_arr.reshape(m, p, rank)
    W3 = W_arr.reshape(p, n, rank)
    
    T_decomp = np.einsum('ikr,kjr,jir->ikj', U3, V3, W3)
    T_target = np.ones((n, m, p))
    
    if not np.allclose(T_decomp, T_target, atol=1e-9):
        # Fallback to schoolbook rank 24 if verification fails
        rank24 = 24
        U24 = [[0] * rank24 for _ in range(n * m)]
        V24 = [[0] * rank24 for _ in range(m * p)]
        W24 = [[0] * rank24 for _ in range(p * n)]
        idx = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    U24[i*m+k][idx] = 1
                    V24[k*p+j][idx] = 1
                    W24[j*n+i][idx] = 1
                    idx += 1
        return {"U": U24, "V": V24, "W": W24}
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
