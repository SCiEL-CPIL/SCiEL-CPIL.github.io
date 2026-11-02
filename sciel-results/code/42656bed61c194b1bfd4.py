"""Matrix multiplication 4x4 * 4x8 - rank-minimal tensor decomposition via two-level Strassen."""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def _strassen_2x2_2x2():
    """
    Strassen decomposition for 2x2 * 2x2 -> 2x2.
    U: (4, 7) - U[i*2+j, c]
    V: (4, 7) - V[j*2+k, c]
    W: (4, 7) - W[k*2+i, c]  (trace convention: k*n+i with n=2)
    """
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # M1 = (a11+a22)(b11+b22) -> contributes to c11 and c22
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1  # W[k*2+i]: c11->k=0,i=0->W[0]; c22->k=1,i=1->W[3]
    
    # M2 = (a21+a22)b11 -> contributes to c21
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1  # W[k*2+i]: c21->k=0,i=1->W[1]
    
    # M3 = a11(b12-b22) -> contributes to c12
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1  # W[k*2+i]: c12->k=1,i=0->W[2]
    
    # M4 = a22(b21-b11) -> contributes to c11 and c21
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # M5 = (a11+a12)b22 -> contributes to c11 (neg) and c12
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6 = (a21-a11)(b11+b12) -> contributes to c22
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (a12-a22)(b21+b22) -> contributes to c11
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _build_2x2_2x4(U_s, V_s, W_s):
    """
    Build decomposition for 2x2 * 2x4 -> 2x4.
    U: (4, 14), V: (8, 14), W: (8, 14)
    U[i*2+j, c], V[j*4+k, c], W[k*2+i, c]
    
    Split 2x4 into two 2x2 halves.
    Left half: cols 0-1 of B, cols 0-1 of C
    Right half: cols 2-3 of B, cols 2-3 of C
    """
    U_2x4 = np.zeros((4, 14), dtype=float)
    V_2x4 = np.zeros((8, 14), dtype=float)
    W_2x4 = np.zeros((8, 14), dtype=float)
    
    # Left half (cols 0-1): embed Strassen 2x2*2x2
    for c in range(7):
        for r in range(4):
            if U_s[r, c] != 0:
                U_2x4[r, c] += U_s[r, c]
            if V_s[r, c] != 0:
                V_2x4[r, c] += V_s[r, c]
            if W_s[r, c] != 0:
                W_2x4[r, c] += W_s[r, c]
    
    # Right half (cols 2-3): embed Strassen 2x2*2x2
    for c in range(7):
        for r in range(4):
            if U_s[r, c] != 0:
                U_2x4[r, 7 + c] += U_s[r, c]
            if V_s[r, c] != 0:
                j = r // 2
                k_old = r % 2
                v_idx = j * 4 + (k_old + 2)
                V_2x4[v_idx, 7 + c] += V_s[r, c]
            if W_s[r, c] != 0:
                # W for 2x2 sub: index k*2+i where k is col within sub, i is row
                # Global W for 2x2*2x4: index k*2+i where k is col 0-3, i is row 0-1
                # Sub col k_old -> global col k_old+2
                # W_global[(k_old+2)*2 + i] = W_sub[k_old*2 + i]
                k_sub = r // 2
                i_sub = r % 2
                W_2x4[(k_sub + 2) * 2 + i_sub, 7 + c] += W_s[r, c]
    
    return U_2x4, V_2x4, W_2x4

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the tensor decomposition is correct."""
    rank = U.shape[1]
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                err = abs(s - 1.0)
                if err > max_err:
                    max_err = err
    return max_err < 1e-9, max_err

def _build_full_decomposition():
    """
    Build the full 4x4 * 4x8 decomposition using two-level Strassen.
    Total rank = 7 * 14 = 98.
    
    Top-level: A is 2x2 blocks of 2x2, B is 2x2 blocks of 2x4, C is 2x2 blocks of 2x4.
    Strassen products:
      P1 = (A11+A22)(B11+B22) -> C11 += P1, C22 += P1
      P2 = (A21+A22)B11 -> C21 += P2
      P3 = A11(B12-B22) -> C12 += P3
      P4 = A22(B21-B11) -> C11 += P4, C21 += P4
      P5 = (A11+A12)B22 -> C11 -= P5, C12 += P5
      P6 = (A21-A11)(B11+B12) -> C22 += P6
      P7 = (A12-A22)(B21+B22) -> C11 += P7
    """
    n, m, p = 4, 4, 8
    
    U_s, V_s, W_s = _strassen_2x2_2x2()
    U_2x4, V_2x4, W_2x4 = _build_2x2_2x4(U_s, V_s, W_s)
    
    total_rank = 7 * 14  # 98
    U = np.zeros((n*m, total_rank), dtype=float)
    V = np.zeros((m*p, total_rank), dtype=float)
    W = np.zeros((p*n, total_rank), dtype=float)
    
    # For each top-level Strassen product s (0..6):
    #   U contribution: for each sub-row r (0..3) with U_2x4[r,c] != 0:
    #     U[global_A_row, col] += U_2x4[r,c]
    #   V contribution: for each sub-row r (0..7) with V_2x4[r,c] != 0:
    #     V[global_B_row, col] += V_2x4[r,c]
    #   W contribution: for each sub-row r (0..7) with W_2x4[r,c] != 0:
    #     W[global_C_row, col] += W_2x4[r,c]
    
    # A block mapping: block (a,b) -> rows a*4+b .. a*4+b+3
    # B block mapping: block (b,c) -> rows b*8+c .. b*8+c+7
    # C block mapping: block (c,a) -> rows c*4+a .. c*4+a+3
    
    # Strassen product definitions: (a_factor, b_factor, c_contributions)
    # a_factor: dict from sub_A_row to coefficient
    # b_factor: dict from sub_B_row to coefficient
    # c_contributions: list of (c_block_row, coefficient)
    
    strassen_products = [
        # P1 = (A11+A22)(B11+B22) -> C11 += P1, C22 += P1
        {
            'a': {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1, 12: 1, 13: 1, 14: 1, 15: 1},
            'b': {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1, 12: 1, 13: 1, 14: 1, 15: 1},
            'c': [(0, 1), (24, 1)]  # C11 at rows 0-3, C22 at rows 24-27
        },
        # P2 = (A21+A22)B11 -> C21 += P2
        {
            'a': {8: 1, 9: 1, 10: 1, 11: 1, 12: 1, 13: 1, 14: 1, 15: 1},
            'b': {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1},
            'c': [(16, 1)]  # C21 at rows 16-19
        },
        # P3 = A11(B12-B22) -> C12 += P3
        {
            'a': {0: 1, 1: 1, 2: 1, 3: 1},
            'b': {8: 1, 9: 1, 10: 1, 11: 1, 12: 1, 13: 1, 14: 1, 15: 1, 16: -1, 17: -1, 18: -1, 19: -1, 20: -1, 21: -1, 22: -1, 23: -1},
            'c': [(8, 1)]  # C12 at rows 8-11
        },
        # P4 = A22(B21-B11) -> C11 += P4, C21 += P4
        {
            'a': {12: 1, 13: 1, 14: 1, 15: 1},
            'b': {0: -1, 1: -1, 2: -1, 3: -1, 4: -1, 5: -1, 6: -1, 7: -1, 16: 1, 17: 1, 18: 1, 19: 1, 20: 1, 21: 1, 22: 1, 23: 1},
            'c': [(0, 1), (16, 1)]  # C11 at rows 0-3, C21 at rows 16-19
        },
        # P5 = (A11+A12)B22 -> C11 -= P5, C12 += P5
        {
            'a': {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1},
            'b': {24: 1, 25: 1, 26: 1, 27: 1, 28: 1, 29: 1, 30: 1, 31: 1},
            'c': [(0, -1), (8, 1)]  # C11 at rows 0-3, C12 at rows 8-11
        },
        # P6 = (A21-A11)(B11+B12) -> C22 += P6
        {
            'a': {0: -1, 1: -1, 2: -1, 3: -1, 8: 1, 9: 1, 10: 1, 11: 1},
            'b': {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1, 12: 1, 13: 1, 14: 1, 15: 1},
            'c': [(24, 1)]  # C22 at rows 24-27
        },
        # P7 = (A12-A22)(B21+B22) -> C11 += P7
        {
            'a': {4: 1, 5: 1, 6: 1, 7: 1, 12: -1, 13: -1, 14: -1, 15: -1},
            'b': {16: 1, 17: 1, 18: 1, 19: 1, 20: 1, 21: 1, 22: 1, 23: 1, 24: 1, 25: 1, 26: 1, 27: 1, 28: 1, 29: 1, 30: 1, 31: 1},
            'c': [(0, 1)]  # C11 at rows 0-3
        },
    ]
    
    for s_idx, prod in enumerate(strassen_products):
        col_base = s_idx * 14
        for c in range(14):
            # U contributions
            for r in range(4):
                if U_2x4[r, c] != 0:
                    for a_row, coeff in prod['a'].items():
                        U[a_row, col_base + c] += U_2x4[r, c] * coeff
            # V contributions
            for r in range(8):
                if V_2x4[r, c] != 0:
                    for b_row, coeff in prod['b'].items():
                        V[b_row, col_base + c] += V_2x4[r, c] * coeff
            # W contributions
            for r in range(8):
                if W_2x4[r, c] != 0:
                    for c_row, coeff in prod['c']:
                        W[c_row + r, col_base + c] += W_2x4[r, c] * coeff
    
    return U, V, W

def _build_schoolbook():
    """Schoolbook decomposition of rank n*m*p."""
    n, m, p = 4, 4, 8
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    return U, V, W

def _to_int_list(mat):
    """Convert numpy matrix to list of lists with integer entries."""
    return [[int(round(mat[i][j])) for j in range(mat.shape[1])] for i in range(mat.shape[0])]

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = 4, 4, 8
    
    try:
        U, V, W = _build_full_decomposition()
        
        valid, max_err = _verify_decomposition(U, V, W, n, m, p)
        
        if valid:
            return {"U": _to_int_list(U), "V": _to_int_list(V), "W": _to_int_list(W)}
        else:
            # Try schoolbook as fallback
            U, V, W = _build_schoolbook()
            return {"U": _to_int_list(U), "V": _to_int_list(V), "W": _to_int_list(W)}
    
    except Exception as e:
        U, V, W = _build_schoolbook()
        return {"U": _to_int_list(U), "V": _to_int_list(V), "W": _to_int_list(W)}
# EVOLVE-BLOCK-END
