"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # For 2x2x15, we can group the 15 columns of B (size 2x15) into:
    # - 7 pairs of columns (14 columns)
    # - 1 single column (1 column)
    # For each pair of columns (col_a, col_b), we can use a 7-term Strassen-like decomposition
    # for the 2x2x2 subproblem where A is 2x2 and B is 2x2 (two column pairs? No, B is 2x2 meaning 2 rows, 2 cols)
    # Wait, B is 2x15. A is 2x2.
    # C is 2x15.
    # C[:, 2k] and C[:, 2k+1] depend on A and B[:, 2k:2k+2].
    # Let B_pair = B[:, 2k:2k+2] which is 2x2.
    # We can compute C_pair = A @ B_pair using Strassen (7 multiplications) instead of 4 multiplications (schoolbook for 2x2x2 is 4? No, 2*2*2=8 scalar mults).
    # Schoolbook for 2x2x2 uses 8 multiplications. Strassen uses 7.
    # So for 7 pairs, we use 7*7 = 49 multiplications.
    # For the 1 remaining column, we use schoolbook for 2x2x1 which is 2*2*1 = 4 multiplications.
    # Total rank = 49 + 4 = 53.
    # This is better than 60.
    
    rank = 7 * 7 + 4  # 49 + 4 = 53
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Helper to add a Strassen decomposition for a 2x2x2 block at column offset col_off
    # A is indexed by (i, j) -> U[i*m+j, c]
    # B is indexed by (j, k) -> V[j*p+k, c]
    # C is indexed by (k, i) -> W[k*n+i, c]
    # For a pair of columns (c1, c2) in B, we treat B_sub as 2x2 with columns c1, c2.
    # Let B_sub[j, 0] = B[j, c1] and B_sub[j, 1] = B[j, c2].
    # Strassen terms:
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B01-B11)
    # P4 = A11(B10-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    # Where A_ij is A[i, j] and B_ij is B_sub[i, j].
    # U coefficients for A:
    # P1: A00:1, A11:1
    # P2: A10:1, A11:1
    # P3: A00:1
    # P4: A11:1
    # P5: A00:1, A01:1
    # P6: A10:1, A00:-1
    # P7: A01:1, A11:-1
    # V coefficients for B:
    # P1: B00:1, B11:1
    # P2: B00:1
    # P3: B01:1, B11:-1
    # P4: B10:1, B00:-1
    # P5: B11:1
    # P6: B00:1, B01:1
    # P7: B10:1, B11:1
    # W coefficients for C:
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    # Note: C indices are (k, i) where k is column index in B (0 or 1 for sub-block) and i is row index in A (0 or 1).
    # So C[0,0] -> W[0*n+0, c], C[0,1] -> W[0*n+1, c], etc.
    # But wait, the global column index is c1 + k.
    # So W[(c1+k)*n + i, c] gets the coefficient.
    
    def add_strassen(col_off, base_c):
        # U coeffs
        u_coeffs = [
            {0: 1, 3: 1},  # P1: A00, A11 (indices 0, 3 in U)
            {2: 1, 3: 1},  # P2: A10, A11 (indices 2, 3)
            {0: 1},        # P3: A00
            {3: 1},        # P4: A11
            {0: 1, 1: 1},  # P5: A00, A01
            {2: 1, 0: -1}, # P6: A10, -A00
            {1: 1, 3: -1}  # P7: A01, -A11
        ]
        # V coeffs
        # B indices: B[j*p+k]. For sub-block, j in {0,1}, k in {0,1} mapped to global col col_off, col_off+1.
        # B00 -> V[0*p + col_off], B01 -> V[0*p + col_off+1]
        # B10 -> V[1*p + col_off], B11 -> V[1*p + col_off+1]
        v_coeffs = [
            {0*p + col_off: 1, 1*p + col_off+1: 1},  # P1: B00, B11
            {0*p + col_off: 1},                       # P2: B00
            {0*p + col_off+1: 1, 1*p + col_off+1: -1},# P3: B01, -B11
            {1*p + col_off: 1, 0*p + col_off: -1},    # P4: B10, -B00
            {1*p + col_off+1: 1},                     # P5: B11
            {0*p + col_off: 1, 0*p + col_off+1: 1},   # P6: B00, B01
            {1*p + col_off: 1, 1*p + col_off+1: 1}    # P7: B10, B11
        ]
        # W coeffs
        # C[k*n+i]. k in {0,1} mapped to global col col_off+k.
        w_coeffs = [
            {(col_off)*n+0: 1, (col_off+1)*n+0: 1, (col_off+1)*n+1: -1, (col_off+1)*n+1: 1}, # P1: C00, C10, -C11? No.
        ]
        # Let's re-derive W carefully.
        # C00 (i=0, k=0) -> W[col_off*n+0, c]
        # C01 (i=0, k=1) -> W[(col_off+1)*n+0, c]
        # C10 (i=1, k=0) -> W[col_off*n+1, c]
        # C11 (i=1, k=1) -> W[(col_off+1)*n+1, c]
        # P1: C00 + C11 - C01? No.
        # C00 = P1 + P4 - P5 + P7
        # C01 = P3 + P5
        # C10 = P2 + P4
        # C11 = P1 - P2 + P3 + P6
        # So:
        # W[col_off*n+0, c] gets: P1:1, P4:1, P5:-1, P7:1
        # W[(col_off+1)*n+0, c] gets: P3:1, P5:1
        # W[col_off*n+1, c] gets: P2:1, P4:1
        # W[(col_off+1)*n+1, c] gets: P1:1, P2:-1, P3:1, P6:1
        
        w_indices = [col_off*n+0, (col_off+1)*n+0, col_off*n+1, (col_off+1)*n+1]
        w_coeffs = [
            {w_indices[0]: 1, w_indices[3]: 1},  # P1
            {w_indices[2]: 1, w_indices[3]: -1}, # P2
            {w_indices[1]: 1, w_indices[3]: 1},  # P3
            {w_indices[0]: 1, w_indices[2]: 1},  # P4
            {w_indices[0]: -1, w_indices[1]: 1}, # P5
            {w_indices[3]: 1},                   # P6
            {w_indices[0]: 1}                    # P7
        ]
        
        for t in range(7):
            c = base_c + t
            for idx, val in u_coeffs[t].items():
                U[idx, c] += val
            for idx, val in v_coeffs[t].items():
                V[idx, c] += val
            for idx, val in w_coeffs[t].items():
                W[idx, c] += val

    # 7 pairs
    for pair in range(7):
        col_off = 2 * pair
        add_strassen(col_off, pair * 7)
    
    # 1 single column at col_off = 14
    # Schoolbook for 2x2x1
    # A is 2x2, B is 2x1 (col 14). C is 2x1.
    # C0 = A00*B0 + A01*B1 (where B0=B[0,14], B1=B[1,14])
    # C1 = A10*B0 + A11*B1
    # 4 terms:
    # T1: A00*B0 -> C0
    # T2: A01*B1 -> C0
    # T3: A10*B0 -> C1
    # T4: A11*B1 -> C1
    base_c = 7 * 7
    col_off = 14
    # T1
    U[0, base_c] = 1
    V[0*p + col_off, base_c] = 1
    W[col_off*n+0, base_c] = 1
    # T2
    U[1, base_c+1] = 1
    V[1*p + col_off, base_c+1] = 1
    W[col_off*n+0, base_c+1] = 1
    # T3
    U[2, base_c+2] = 1
    V[0*p + col_off, base_c+2] = 1
    W[col_off*n+1, base_c+2] = 1
    # T4
    U[3, base_c+3] = 1
    V[1*p + col_off, base_c+3] = 1
    W[col_off*n+1, base_c+3] = 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
