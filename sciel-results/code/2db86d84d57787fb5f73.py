"""Matrix multiplication 2,4,5 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_TIME_LIMIT = 140.0

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    rank = 36
    
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    
    # Indices for U (A matrix, 2x4)
    # A[i, j] -> i*m + j
    # A_L: i in {0,1}, j in {0,1} -> indices 0, 1, 4, 5
    # A_R: i in {0,1}, j in {2,3} -> indices 2, 3, 6, 7
    
    # Indices for V (B matrix, 4x5)
    # B[j, k] -> j*p + k
    # B_T: j in {0,1}
    #   B_T1: k in {0,1} -> indices 0, 1, 5, 6
    #   B_T2: k in {2,3} -> indices 2, 3, 7, 8
    #   B_T3: k in {4}   -> indices 4, 9
    # B_B: j in {2,3}
    #   B_B1: k in {0,1} -> indices 10, 11, 15, 16
    #   B_B2: k in {2,3} -> indices 12, 13, 17, 18
    #   B_B3: k in {4}   -> indices 14, 19
    
    # Indices for W (C matrix, 2x5)
    # C[k, i] -> k*n + i
    # C1: k in {0,1} -> indices 0, 1, 2, 3
    # C2: k in {2,3} -> indices 4, 5, 6, 7
    # C3: k in {4}   -> indices 8, 9

    # Helper to add a 2x2x2 Strassen block
    # u_idx: [A00, A01, A10, A11]
    # v_idx: [B00, B01, B10, B11]
    # w_idx: [C00, C01, C10, C11]
    def add_strassen_2x2x2(u_idx, v_idx, w_idx, start_col):
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Coefficients for A (u)
        u_coeffs = [
            [1, 0, 0, 1], # M1
            [0, 0, 1, 1], # M2
            [1, 0, 0, 0], # M3
            [0, 0, 0, 1], # M4
            [1, 1, 0, 0], # M5
            [-1, 0, 1, 0], # M6
            [0, 1, 0, -1]  # M7
        ]
        
        # Coefficients for B (v)
        v_coeffs = [
            [1, 0, 0, 1], # M1
            [1, 0, 0, 0], # M2
            [0, 1, 0, -1], # M3
            [0, 0, 1, -1], # M4
            [0, 0, 0, 1], # M5
            [1, 1, 0, 0], # M6
            [0, 0, 1, 1]  # M7
        ]
        
        # Coefficients for C (w)
        w_coeffs = [
            [1, 0, 0, 1], # M1: C00, C11
            [0, 0, 1, -1], # M2: C10, C11
            [0, 1, 0, 1], # M3: C01, C11
            [1, 0, 1, 0], # M4: C00, C10
            [-1, 1, 0, 0], # M5: C00, C01
            [0, 0, 0, 1], # M6: C11
            [1, 0, 0, 0]  # M7: C00
        ]
        
        for s in range(7):
            col = start_col + s
            for idx in range(4):
                if u_coeffs[s][idx] != 0:
                    U[u_idx[idx]][col] += u_coeffs[s][idx]
                if v_coeffs[s][idx] != 0:
                    V[v_idx[idx]][col] += v_coeffs[s][idx]
                if w_coeffs[s][idx] != 0:
                    W[w_idx[idx]][col] += w_coeffs[s][idx]

    # Helper to add a 2x2x1 Schoolbook block
    # u_idx: [A00, A01, A10, A11]
    # v_idx: [B0, B1]
    # w_idx: [C0, C1]
    def add_schoolbook_2x2x1(u_idx, v_idx, w_idx, start_col):
        # C0 = A00*B0 + A01*B1
        # C1 = A10*B0 + A11*B1
        
        # Term 1: A00*B0 -> C0
        U[u_idx[0]][start_col + 0] += 1
        V[v_idx[0]][start_col + 0] += 1
        W[w_idx[0]][start_col + 0] += 1
        
        # Term 2: A01*B1 -> C0
        U[u_idx[1]][start_col + 1] += 1
        V[v_idx[1]][start_col + 1] += 1
        W[w_idx[0]][start_col + 1] += 1
        
        # Term 3: A10*B0 -> C1
        U[u_idx[2]][start_col + 2] += 1
        V[v_idx[0]][start_col + 2] += 1
        W[w_idx[1]][start_col + 2] += 1
        
        # Term 4: A11*B1 -> C1
        U[u_idx[3]][start_col + 3] += 1
        V[v_idx[1]][start_col + 3] += 1
        W[w_idx[1]][start_col + 3] += 1

    # Define index mappings
    u_L = [0, 1, 4, 5] # A_L
    u_R = [2, 3, 6, 7] # A_R
    
    v_T1 = [0, 1, 5, 6]
    v_T2 = [2, 3, 7, 8]
    v_T3 = [4, 9]
    
    v_B1 = [10, 11, 15, 16]
    v_B2 = [12, 13, 17, 18]
    v_B3 = [14, 19]
    
    w_1 = [0, 1, 2, 3] # C cols 0,1
    w_2 = [4, 5, 6, 7] # C cols 2,3
    w_3 = [8, 9]       # C col 4

    col = 0
    
    # Block 1: A_L * B_T
    add_strassen_2x2x2(u_L, v_T1, w_1, col)
    col += 7
    add_strassen_2x2x2(u_L, v_T2, w_2, col)
    col += 7
    add_schoolbook_2x2x1(u_L, v_T3, w_3, col)
    col += 4
    
    # Block 2: A_R * B_B
    add_strassen_2x2x2(u_R, v_B1, w_1, col)
    col += 7
    add_strassen_2x2x2(u_R, v_B2, w_2, col)
    col += 7
    add_schoolbook_2x2x1(u_R, v_B3, w_3, col)
    col += 4
    
    return {"U": U, "V": V, "W": W}

def _verify_exact(U, V, W):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    rank = len(U[0])
    
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Check compatible triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = np.dot(U_arr[a], V_arr[b] * W_arr[c])
                if abs(s - 1) > 1e-9:
                    return False
    
    # Check incompatible triples
    for a in range(nm):
        i_a = a // m
        j_a = a % m
        for b in range(mp):
            j_b = b // p
            k_b = b % p
            for c in range(pn):
                k_c = c // n
                i_c = c % n
                if j_a == j_b and k_b == k_c and i_c == i_a:
                    continue
                s = np.dot(U_arr[a], V_arr[b] * W_arr[c])
                if abs(s) > 1e-9:
                    return False
    return True

def construct_wrapper():
    if time.time() - _START_TIME > _TIME_LIMIT - 10:
        # Fallback to schoolbook if time is running out (unlikely for this small size)
        n, m, p = DIMENSIONS
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    result = construct()
    
    # Verification is cheap for rank 36
    if not _verify_exact(result["U"], result["V"], result["W"]):
        # Fallback
        n, m, p = DIMENSIONS
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
        
    return result

# The interface requires construct()
# I will rename construct_wrapper to construct and the inner one to _construct_rank36
# But the prompt says "Keep the function names and signatures of the seed program".
# The seed program has `construct`.
# I will replace the body of `construct` with the logic above.
# I will remove the old `construct` and `construct_wrapper` and just define `construct`.

# Re-defining construct to match the requirement
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    rank = 36
    
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    
    u_L = [0, 1, 4, 5]
    u_R = [2, 3, 6, 7]
    
    v_T1 = [0, 1, 5, 6]
    v_T2 = [2, 3, 7, 8]
    v_T3 = [4, 9]
    
    v_B1 = [10, 11, 15, 16]
    v_B2 = [12, 13, 17, 18]
    v_B3 = [14, 19]
    
    w_1 = [0, 1, 2, 3]
    w_2 = [4, 5, 6, 7]
    w_3 = [8, 9]

    def add_strassen_2x2x2(u_idx, v_idx, w_idx, start_col):
        u_coeffs = [
            [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1],
            [1, 1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1]
        ]
        v_coeffs = [
            [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, -1],
            [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]
        ]
        w_coeffs = [
            [1, 0, 0, 1], [0, 0, 1, -1], [0, 1, 0, 1], [1, 0, 1, 0],
            [-1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]
        ]
        
        for s in range(7):
            col = start_col + s
            for idx in range(4):
                if u_coeffs[s][idx] != 0:
                    U[u_idx[idx]][col] += u_coeffs[s][idx]
                if v_coeffs[s][idx] != 0:
                    V[v_idx[idx]][col] += v_coeffs[s][idx]
                if w_coeffs[s][idx] != 0:
                    W[w_idx[idx]][col] += w_coeffs[s][idx]

    def add_schoolbook_2x2x1(u_idx, v_idx, w_idx, start_col):
        U[u_idx[0]][start_col + 0] += 1
        V[v_idx[0]][start_col + 0] += 1
        W[w_idx[0]][start_col + 0] += 1
        
        U[u_idx[1]][start_col + 1] += 1
        V[v_idx[1]][start_col + 1] += 1
        W[w_idx[0]][start_col + 1] += 1
        
        U[u_idx[2]][start_col + 2] += 1
        V[v_idx[0]][start_col + 2] += 1
        W[w_idx[1]][start_col + 2] += 1
        
        U[u_idx[3]][start_col + 3] += 1
        V[v_idx[1]][start_col + 3] += 1
        W[w_idx[1]][start_col + 3] += 1

    col = 0
    add_strassen_2x2x2(u_L, v_T1, w_1, col)
    col += 7
    add_strassen_2x2x2(u_L, v_T2, w_2, col)
    col += 7
    add_schoolbook_2x2x1(u_L, v_T3, w_3, col)
    col += 4
    
    add_strassen_2x2x2(u_R, v_B1, w_1, col)
    col += 7
    add_strassen_2x2x2(u_R, v_B2, w_2, col)
    col += 7
    add_schoolbook_2x2x1(u_R, v_B3, w_3, col)
    col += 4
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
