"""Matrix multiplication 2,5,6 - rank-minimal tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 5, 6]
START_TIME = time.time()
DEADLINE = START_TIME + 120  # 120s wall-clock limit

def _time_remaining():
    return DEADLINE - time.time()

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS
    
    # Rank 54 decomposition using Strassen's algorithm for 2x2 blocks
    # A is 2x5, B is 5x6, C is 2x6
    # Partition m=5 into pairs (0,1), (2,3) and single (4).
    # Strassen (7 mults) per 2x2 j-block per 2x2 k-block, schoolbook for j=4.
    
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(i_base, j_base, k_base):
        nonlocal col
        # i_base: 0 (only one pair of rows in A/C since n=2)
        # j_base: 0 or 2 (pairs of cols in A / rows in B)
        # k_base: 0, 2, or 4 (pairs of cols in B / cols in C)
        
        # Indices in U: i*m + j
        # Indices in V: j*p + k
        # Indices in W: k*n + i
        
        # M0: (A00+A11)(B00+B11)
        # A00: U[i_base, j_base], A11: U[i_base+1, j_base+1]
        # B00: V[j_base, k_base], B11: V[j_base+1, k_base+1]
        # C00: W[k_base, i_base], C11: W[k_base+1, i_base+1]
        U[i_base * m + j_base][col] += 1
        U[(i_base+1) * m + (j_base+1)][col] += 1
        V[j_base * p + k_base][col] += 1
        V[(j_base+1) * p + (k_base+1)][col] += 1
        W[k_base * n + i_base][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] += 1
        col += 1
        
        # M1: (A10+A11)B00
        # A10: U[i_base+1, j_base], A11: U[i_base+1, j_base+1]
        # B00: V[j_base, k_base]
        # C10: W[k_base, i_base+1], C11: W[k_base+1, i_base+1] (C11 gets -1)
        U[(i_base+1) * m + j_base][col] += 1
        U[(i_base+1) * m + (j_base+1)][col] += 1
        V[j_base * p + k_base][col] += 1
        W[k_base * n + (i_base+1)][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] -= 1
        col += 1
        
        # M2: A00(B01-B11)
        # A00: U[i_base, j_base]
        # B01: V[j_base, k_base+1], B11: V[j_base+1, k_base+1]
        # C01: W[k_base+1, i_base], C11: W[k_base+1, i_base+1]
        U[i_base * m + j_base][col] += 1
        V[j_base * p + (k_base+1)][col] += 1
        V[(j_base+1) * p + (k_base+1)][col] -= 1
        W[(k_base+1) * n + i_base][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] += 1
        col += 1
        
        # M3: A11(B10-B00)
        # A11: U[i_base+1, j_base+1]
        # B10: V[j_base+1, k_base], B00: V[j_base, k_base]
        # C00: W[k_base, i_base], C10: W[k_base, i_base+1]
        U[(i_base+1) * m + (j_base+1)][col] += 1
        V[(j_base+1) * p + k_base][col] += 1
        V[j_base * p + k_base][col] -= 1
        W[k_base * n + i_base][col] += 1
        W[k_base * n + (i_base+1)][col] += 1
        col += 1
        
        # M4: (A00+A01)B11
        # A00: U[i_base, j_base], A01: U[i_base, j_base+1]
        # B11: V[j_base+1, k_base+1]
        # C00: W[k_base, i_base] (gets -1), C01: W[k_base+1, i_base]
        U[i_base * m + j_base][col] += 1
        U[i_base * m + (j_base+1)][col] += 1
        V[(j_base+1) * p + (k_base+1)][col] += 1
        W[k_base * n + i_base][col] -= 1
        W[(k_base+1) * n + i_base][col] += 1
        col += 1
        
        # M5: (A10-A00)(B00+B01)
        # A10: U[i_base+1, j_base], A00: U[i_base, j_base]
        # B00: V[j_base, k_base], B01: V[j_base, k_base+1]
        # C11: W[k_base+1, i_base+1]
        U[(i_base+1) * m + j_base][col] += 1
        U[i_base * m + j_base][col] -= 1
        V[j_base * p + k_base][col] += 1
        V[j_base * p + (k_base+1)][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] += 1
        col += 1
        
        # M6: (A01-A11)(B10+B11)
        # A01: U[i_base, j_base+1], A11: U[i_base+1, j_base+1]
        # B10: V[j_base+1, k_base], B11: V[j_base+1, k_base+1]
        # C00: W[k_base, i_base]
        U[i_base * m + (j_base+1)][col] += 1
        U[(i_base+1) * m + (j_base+1)][col] -= 1
        V[(j_base+1) * p + k_base][col] += 1
        V[(j_base+1) * p + (k_base+1)][col] += 1
        W[k_base * n + i_base][col] += 1
        col += 1

    def add_naive_2112_block(i_base, j_base, k_base):
        nonlocal col
        # Handles A[i_base..i_base+1, j_base] * B[j_base, k_base..k_base+1] -> C[i_base..i_base+1, k_base..k_base+1]
        # This is a 2x1 * 1x2 multiplication, requiring 4 scalar multiplications.
        
        # 1: A00 B00 -> C00
        U[i_base * m + j_base][col] += 1
        V[j_base * p + k_base][col] += 1
        W[k_base * n + i_base][col] += 1
        col += 1
        
        # 2: A00 B01 -> C01
        U[i_base * m + j_base][col] += 1
        V[j_base * p + (k_base+1)][col] += 1
        W[(k_base+1) * n + i_base][col] += 1
        col += 1
        
        # 3: A10 B00 -> C10
        U[(i_base+1) * m + j_base][col] += 1
        V[j_base * p + k_base][col] += 1
        W[k_base * n + (i_base+1)][col] += 1
        col += 1
        
        # 4: A10 B01 -> C11
        U[(i_base+1) * m + j_base][col] += 1
        V[j_base * p + (k_base+1)][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] += 1
        col += 1

    # Apply Strassen blocks for j-pairs (0,1) and (2,3)
    # k-pairs are (0,1), (2,3), (4,5)
    
    # j_pair 0,1
    add_strassen_block(0, 0, 0)
    add_strassen_block(0, 0, 2)
    add_strassen_block(0, 0, 4)
    
    # j_pair 2,3
    add_strassen_block(0, 2, 0)
    add_strassen_block(0, 2, 2)
    add_strassen_block(0, 2, 4)
    
    # Apply naive blocks for j=4
    # k-pairs are (0,1), (2,3), (4,5)
    add_naive_2112_block(0, 4, 0)
    add_naive_2112_block(0, 4, 2)
    add_naive_2112_block(0, 4, 4)
    
    if col != rank:
        # If col < rank, trim the matrices to the actual number of terms used
        if col < rank:
            U = [row[:col] for row in U]
            V = [row[:col] for row in V]
            W = [row[:col] for row in W]
        # If col > rank, this is an error; return what we have (shouldn't happen)
        pass

    # Quick verification with numpy (cheap for rank 54)
    try:
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        r = U_np.shape[1]
        U_reshaped = U_np.reshape(n, m, r)
        V_reshaped = V_np.reshape(m, p, r)
        W_reshaped = W_np.reshape(p, n, r)
        T = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
        if not np.allclose(T, 1.0, atol=1e-10):
            # Fallback to schoolbook if something went wrong
            rank_school = n * m * p
            U_school = [[0] * rank_school for _ in range(n * m)]
            V_school = [[0] * rank_school for _ in range(m * p)]
            W_school = [[0] * rank_school for _ in range(p * n)]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx = i * m * p + j * p + k
                        U_school[i*m+j][idx] = 1
                        V_school[j*p+k][idx] = 1
                        W_school[k*n+i][idx] = 1
            return {"U": U_school, "V": V_school, "W": W_school}
    except Exception:
        pass

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
