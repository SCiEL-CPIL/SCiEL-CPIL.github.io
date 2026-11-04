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
    # We partition the 'm' dimension (5) into pairs (0,1), (2,3) and single (4).
    # For each pair (j, j+1) in A and corresponding rows in B, we use Strassen (7 mults).
    # For the single column j=4 in A and row j=4 in B, we use schoolbook (12 mults).
    
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
    
    # Trim to actual rank used
    if col < rank:
        U = [row[:col] for row in U]
        V = [row[:col] for row in V]
        W = [row[:col] for row in W]
    
    # Verify the decomposition with a quick check (only if time permits)
    if _time_remaining() > 5:
        try:
            U_np = np.array(U, dtype=float)
            V_np = np.array(V, dtype=float)
            W_np = np.array(W, dtype=float)
            r = U_np.shape[1]
            
            # Build the tensor: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # Check against the expected multiplication tensor
            # Expected: for matrix multiplication C[i,k] = sum_j A[i,j]*B[j,k]
            # The tensor should satisfy: sum_j T[i,j,k] A[i,j] B[j,k] = C[i,k]
            # In the decomposition form, T is implicitly defined by U,V,W.
            # We can verify by checking that the decomposition produces correct results
            # for random matrices.
            
            np.random.seed(42)
            A_test = np.random.randn(n, m)
            B_test = np.random.randn(m, p)
            C_expected = A_test @ B_test
            
            C_computed = np.zeros((n, p))
            for r in range(r):
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            C_computed[i, k] += U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r] * A_test[i,j] * B_test[j,k]
            
            if not np.allclose(C_computed, C_expected, atol=1e-10):
                # Decomposition is incorrect; fall back to schoolbook
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
