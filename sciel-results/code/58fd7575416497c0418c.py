import time
import sys
import numpy as np

# EVOLVE-BLOCK-START

def _trivial_decomposition(n, m, p):
    """
    Return the trivial (rank n*m*p) exact tensor decomposition for n x m x p matrix multiplication.
    
    U: shape (n*m, rank)  -- entries for A[i,j] indexed as i*m+j
    V: shape (m*p, rank)  -- entries for B[j,k] indexed as j*p+k
    W: shape (p*n, rank)  -- entries for C[i,k] indexed as k*n+i (trace convention)
    
    The decomposition satisfies:
    C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]  for all i,j,k
    """
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    t = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx][t] = 1
                V[v_idx][t] = 1
                W[w_idx][t] = 1
                t += 1
    
    return U, V, W

def _strassen_2x2_2x2_UVW():
    """
    Return the Strassen decomposition for 2x2 * 2x2 -> 2x2 multiplication.
    Rank = 7.
    
    U: shape (4, 7) for A[i,j] indexed as i*2+j
    V: shape (4, 7) for B[j,k] indexed as j*2+k  
    W: shape (4, 7) for C[i,k] indexed as k*2+i (trace convention)
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    # M3 = A00(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1
    
    # M4 = A11(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[1][4] = 1; W[3][4] = -1
    
    # M6 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[3][6] = -1
    
    return U, V, W

def _verify_decomposition(n, m, p, U, V, W, num_trials=3, atol=1e-9):
    """
    Verify that the decomposition (U, V, W) correctly computes matrix multiplication.
    Returns True if valid, False otherwise.
    """
    try:
        np.random.seed(42)
        
        # Test with several random matrices
        for trial in range(num_trials):
            A = np.random.randint(-10, 10, (n, m), dtype=float)
            B = np.random.randint(-10, 10, (m, p), dtype=float)
            C_expected = A @ B
            
            rank = len(U[0])
            C_computed = np.zeros((n, p), dtype=float)
            
            for i in range(n):
                for k in range(p):
                    s = 0.0
                    for j in range(m):
                        u_row = U[i * m + j]
                        v_row = V[j * p + k]
                        w_val = W[k * n + i]
                        for r in range(rank):
                            s += u_row[r] * v_row[r] * w_val[r]
                    C_computed[i, k] = s
            
            if not np.allclose(C_computed, C_expected, atol=atol):
                return False
        
        return True
    except Exception:
        return False

def _strassen_4x4_4x4_UVW():
    """
    Return the Strassen-based decomposition for 4x4 * 4x4 -> 4x4.
    Uses block Strassen: treat 4x4 as 2x2 blocks of 2x2 matrices.
    Total rank = 7 * 7 = 49.
    
    U: shape (16, 49) for A[i,j] indexed as i*4+j
    V: shape (16, 49) for B[j,k] indexed as j*4+k
    W: shape (16, 49) for C[i,k] indexed as k*4+i (trace convention)
    """
    U2, V2, W2 = _strassen_2x2_2x2_UVW()
    
    U = [[0] * 49 for _ in range(16)]
    V = [[0] * 49 for _ in range(16)]
    W = [[0] * 49 for _ in range(16)]
    
    # Define the 7 Strassen products for block level
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 - M4 + M6
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M7
    
    # For each block position in A, B, C, determine which Strassen products contribute
    # and with what coefficient
    
    # A blocks:
    # A00 contributes to M1, M3, M5, M6 (with -1)
    # A10 contributes to M2, M6
    # A11 contributes to M1, M2, M4, M7 (with -1)
    # A01 contributes to M5, M7
    
    # B blocks:
    # B00 contributes to M1, M2, M4 (with -1), M6
    # B01 contributes to M3 (with 1), M6 (with 1)
    # B10 contributes to M4, M7
    # B11 contributes to M1, M3 (with -1), M5, M7
    
    # C blocks:
    # C00 = M1 - M4 + M6
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M7
    
    # For each (bi, bj) block of A, (bj2, bk) block of B, (bi2, bk2) block of C
    # we need to compute the combined coefficient
    
    # Let's define the contribution of each block to each Strassen product
    
    # A block (bi, bj) -> which M_r does it contribute to and with what coeff?
    A_contrib = {
        (0,0): {0: 1, 2: 1, 4: 1, 5: -1},   # M1, M3, M5, M6(-1)
        (1,0): {1: 1, 5: 1},                   # M2, M6
        (1,1): {0: 1, 1: 1, 3: 1, 6: -1},      # M1, M2, M4, M7(-1)
        (0,1): {4: 1, 6: 1},                   # M5, M7
    }
    
    B_contrib = {
        (0,0): {0: 1, 1: 1, 3: -1, 5: 1},      # M1, M2, M4(-1), M6
        (0,1): {2: 1, 5: 1},                   # M3, M6
        (1,0): {3: 1, 6: 1},                   # M4, M7
        (1,1): {0: 1, 2: -1, 4: 1, 6: 1},      # M1, M3(-1), M5, M7
    }
    
    C_contrib = {
        (0,0): {0: 1, 3: -1, 5: 1},            # M1, M4(-1), M6
        (0,1): {2: 1, 4: 1},                   # M3, M5
        (1,0): {1: 1, 3: 1},                   # M2, M4
        (1,1): {0: 1, 1: -1, 6: 1},            # M1, M2(-1), M7
    }
    
    # Now fill in U, V, W
    # For each element in A[i,j], it belongs to block (i//2, j//2) and inner position (i%2, j%2)
    # For each element in B[j,k], it belongs to block (j//2, k//2) and inner position (j%2, k%2)
    # For each element in C[i,k], it belongs to block (i//2, k//2) and inner position (i%2, k%2)
    
    for i in range(4):
        for j in range(4):
            bi, bj = i // 2, j // 2
            ii, jj = i % 2, j % 2
            inner_idx = ii * 2 + jj
            for r_block in A_contrib[(bi, bj)]:
                coeff_a = A_contrib[(bi, bj)][r_block]
                for r_inner in range(7):
                    combined_r = r_block * 7 + r_inner
                    U[i*4+j][combined_r] += coeff_a * U2[inner_idx][r_inner]
    
    for j in range(4):
        for k in range(4):
            bj, bk = j // 2, k // 2
            jj, kk = j % 2, k % 2
            inner_idx = jj * 2 + kk
            for r_block in B_contrib[(bj, bk)]:
                coeff_b = B_contrib[(bj, bk)][r_block]
                for r_inner in range(7):
                    combined_r = r_block * 7 + r_inner
                    V[j*4+k][combined_r] += coeff_b * V2[inner_idx][r_inner]
    
    for i in range(4):
        for k in range(4):
            bi, bk = i // 2, k // 2
            ii, kk = i % 2, k % 2
            inner_idx = ii * 2 + kk
            for r_block in C_contrib[(bi, bk)]:
                coeff_c = C_contrib[(bi, bk)][r_block]
                for r_inner in range(7):
                    combined_r = r_block * 7 + r_inner
                    W[k*4+i][combined_r] += coeff_c * W2[inner_idx][r_inner]
    
    return U, V, W

def _extend_4x4_4x4_to_4x4_7():
    """
    Extend the 4x4*4x4 -> 4x4 Strassen decomposition (rank 49) to 4x4*4x7 -> 4x7.
    Uses trivial decomposition (rank 48) for the last 3 columns.
    Total rank = 49 + 48 = 97.
    """
    U4, V4, W4 = _strassen_4x4_4x4_UVW()
    U3, V3, W3 = _trivial_decomposition(4, 4, 3)
    
    rank_total = 49 + 48
    
    U = [row4 + row3 for row4, row3 in zip(U4, U3)]
    V = [row4 + row3 for row4, row3 in zip(V4, V3)]
    
    W = [[0] * rank_total for _ in range(28)]
    
    # Copy W4 into W (rows 0-15, corresponding to k=0,1,2,3)
    for i in range(4):
        for k in range(4):
            row_in_W = k * 4 + i
            for r in range(49):
                W[row_in_W][r] = W4[k*4+i][r]
    
    # Copy W3 into W (rows 16-27, corresponding to k=4,5,6)
    for i3 in range(12):
        k3, i_val = i3 // 4, i3 % 4
        row_in_W = (4 + k3) * 4 + i_val
        for r in range(48):
            W[row_in_W][49 + r] = W3[i3][r]
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = 4, 4, 7
    
    # Start with the trivial decomposition (rank = 4*4*7 = 112)
    U, V, W = _trivial_decomposition(n, m, p)
    
    # Try the extended Strassen decomposition (rank 97)
    try:
        U_strassen, V_strassen, W_strassen = _extend_4x4_4x4_to_4x4_7()
        
        # Verify
        if _verify_decomposition(n, m, p, U_strassen, V_strassen, W_strassen):
            U, V, W = U_strassen, V_strassen, W_strassen
    except Exception:
        pass
    
    # Verify final result
    if not _verify_decomposition(n, m, p, U, V, W):
        # Fallback to trivial
        U, V, W = _trivial_decomposition(n, m, p)
    
    # Return the best decomposition found
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
