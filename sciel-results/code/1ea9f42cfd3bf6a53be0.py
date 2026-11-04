"""Direct construction of the exact tensor decomposition for (2,3,5) matrix multiplication.

Strategy: rank-28 decomposition using Strassen blocks and schoolbook parts.
Split B (3x5) into B1 (3x3, cols 0-2) and B2 (3x2, cols 3-4).
- A*B1 (2x3 * 3x3 -> 2x3): rank-17 via Strassen(2x2x2) + schoolbook
- A*B2 (2x3 * 3x2 -> 2x2): rank-11 via Strassen(2x2x2) + schoolbook
Total: 28 terms.
"""

import time
import numpy as np


def _verify_decomposition(U, V, W, n=2, m=3, p=5, trials=50):
    """Verify the decomposition produces correct matrix products."""
    rank = U.shape[1]
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    for _ in range(trials):
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_expected = A_test @ B_test
        
        C_computed = np.zeros((n, p))
        for r in range(rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        C_computed[i, k] += (A_test[i, j] * B_test[j, k] 
                                             * U_np[i*m+j, r] 
                                             * V_np[j*p+k, r] 
                                             * W_np[k*n+i, r])
        
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            return False
    return True


def _strassen_2x2x2(A_rows, B_rows, B_cols, out_rows, out_cols, n=2, m=3, p=5):
    """
    Apply Strassen's algorithm for 2x2 * 2x2 -> 2x2 block.
    
    A_rows: list of 2 row indices in A (0,1)
    B_rows: list of 2 row indices in B (e.g., [0,1] or [1,2])
    B_cols: list of 2 col indices in B
    out_rows: list of 2 row indices in C (0,1)  
    out_cols: list of 2 col indices in C
    """
    terms = []
    a00, a01 = A_rows[0], A_rows[1]
    a10, a11 = A_rows[2], A_rows[3] if len(A_rows) > 3 else A_rows[1]
    
    # Actually let's use explicit indices:
    # A' = [[A[a00], A[a01]], [A[a10], A[a11]]]
    # where a00, a01 are column indices for row 0, a10, a11 for row 1
    # In our U indexing: U[i*m + j, r]
    
    # Let me redefine: we're multiplying submatrices
    # A_sub[i, j] corresponds to A[i, A_col_idx[j]] where A_col_idx = [c0, c1]
    # B_sub[j, k] corresponds to B[B_row_idx[j], B_col_idx[k]]
    
    return terms


def construct():
    deadline = time.time() + 110.0
    
    n, m, p = 2, 3, 5
    rank = 28
    U = np.zeros((n * m, rank), dtype=int)   # 6 x 28
    V = np.zeros((m * p, rank), dtype=int)   # 15 x 28
    W = np.zeros((p * n, rank), dtype=int)   # 10 x 28
    
    col = 0
    
    # === PART 1: Columns 0,1,2 of B (3x3 block) -> rank 17 ===
    # Split: j=0,1 -> use Strassen on 2x2x2 part + schoolbook for remaining
    # j=2 -> schoolbook 2x1x3 = 6 terms
    # Total for j=0,1: R(2,2,3) = R(2,2,2)+R(2,2,1) = 7+4 = 11
    # Total: 11 + 6 = 17
    
    # Sub-part 1a: j in {0,1}, k in {0,1}: Strassen 2x2 * 2x2 -> 2x2, rank 7
    # A_sub = [[A[0,0], A[0,1]], [A[1,0], A[1,1]]]
    # B_sub = [[B[0,0], B[0,1]], [B[1,0], B[1,1]]]
    # C_sub = [[C[0,0], C[0,1]], [C[1,0], C[1,1]]]
    
    # Strassen terms for standard 2x2*2x2->2x2:
    # m0 = (a00+a11)*(b00+b11)
    # m1 = (a10+a11)*b00
    # m2 = a00*(b11-b01)
    # m3 = a11*(b10-b00)
    # m4 = (a00+a01)*b11
    # m5 = (a10-a11)*(b00+b10)
    # m6 = (a01-a00)*(b01+b11)
    # C00 = m0+m3-m4+m6, C01 = m2+m4, C10 = m1+m5, C11 = m0-m1+m2+m5
    
    # U indices: A[0,0]->U[0], A[0,1]->U[1], A[1,0]->U[3], A[1,1]->U[4]
    # V indices: B[0,0]->V[0], B[0,1]->V[1], B[1,0]->V[3], B[1,1]->V[4]
    # W indices: C[0,0]->W[0], C[0,1]->W[2], C[1,0]->W[1], C[1,1]->W[3]
    
    strassen1 = [
        # (U_coeffs, V_coeffs, W_coeffs)
        # m0 = (A00+A11)*(B00+B11): C00=+1, C10=-1, C11=+1
        ({0: 1, 4: 1}, {0: 1, 4: 1}, {0: 1, 1: -1, 3: 1}),
        # m1 = (A10+A11)*B00: C10=+1, C11=-1
        ({3: 1, 4: 1}, {0: 1}, {1: 1, 3: -1}),
        # m2 = A00*(B11-B01): C01=+1, C11=+1
        ({0: 1}, {1: -1, 4: 1}, {2: 1, 3: 1}),
        # m3 = A11*(B10-B00): C00=+1
        ({4: 1}, {0: -1, 3: 1}, {0: 1}),
        # m4 = (A00+A01)*B11: C00=-1, C01=+1
        ({0: 1, 1: 1}, {4: 1}, {0: -1, 2: 1}),
        # m5 = (A10-A11)*(B00+B10): C10=+1, C11=+1
        ({3: 1, 4: -1}, {0: 1, 3: 1}, {1: 1, 3: 1}),
        # m6 = (A01-A00)*(B01+B11): C00=+1
        ({0: -1, 1: 1}, {1: 1, 4: 1}, {0: 1}),
    ]
    
    for u_c, v_c, w_c in strassen1:
        for idx, val in u_c.items():
            U[idx, col] += val
        for idx, val in v_c.items():
            V[idx, col] += val
        for idx, val in w_c.items():
            W[idx, col] += val
        col += 1
    
    # Sub-part 1b: j in {0,1}, k=2: schoolbook 2x2*2x1 -> 2x1, rank 4
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    
    schoolbook_1b = [
        ({0: 1}, {0*5+2: 1}, {2*2+0: 1}),   # A00*B02 -> C02
        ({1: 1}, {1*5+2: 1}, {2*2+0: 1}),   # A01*B12 -> C02
        ({3: 1}, {0*5+2: 1}, {2*2+1: 1}),   # A10*B02 -> C12
        ({4: 1}, {1*5+2: 1}, {2*2+1: 1}),   # A11*B12 -> C12
    ]
    
    for u_c, v_c, w_c in schoolbook_1b:
        for idx, val in u_c.items():
            U[idx, col] += val
        for idx, val in v_c.items():
            V[idx, col] += val
        for idx, val in w_c.items():
            W[idx, col] += val
        col += 1
    
    # Sub-part 1c: j=2, k in {0,1,2}: schoolbook 2x1*1x3 -> 2x3, rank 6
    # C[0,k] = A[0,2]*B[2,k] for k=0,1,2
    # C[1,k] = A[1,2]*B[2,k] for k=0,1,2
    
    schoolbook_1c = [
        ({0*3+2: 1}, {2*5+0: 1}, {0*2+0: 1}),   # A02*B20 -> C00
        ({0*3+2: 1}, {2*5+1: 1}, {1*2+0: 1}),   # A02*B21 -> C01
        ({0*3+2: 1}, {2*5+2: 1}, {2*2+0: 1}),   # A02*B22 -> C02
        ({1*3+2: 1}, {2*5+0: 1}, {0*2+1: 1}),   # A12*B20 -> C10
        ({1*3+2: 1}, {2*5+1: 1}, {1*2+1: 1}),   # A12*B21 -> C11
        ({1*3+2: 1}, {2*5+2: 1}, {2*2+1: 1}),   # A12*B22 -> C12
    ]
    
    for u_c, v_c, w_c in schoolbook_1c:
        for idx, val in u_c.items():
            U[idx, col] += val
        for idx, val in v_c.items():
            V[idx, col] += val
        for idx, val in w_c.items():
            W[idx, col] += val
        col += 1
    
    # === PART 2: Columns 3,4 of B (3x2 block) -> rank 11 ===
    # Split: j in {0,1} -> Strassen 2x2*2x2->2x2, rank 7
    # j=2 -> schoolbook 2x1*1x2 -> 2x2, rank 4
    # Total: 7 + 4 = 11
    
    # Sub-part 2a: j in {0,1}, k in {3,4}: Strassen 2x2 * 2x2 -> 2x2, rank 7
    # A_sub = [[A[0,0], A[0,1]], [A[1,0], A[1,1]]]
    # B_sub = [[B[0,3], B[0,4]], [B[1,3], B[1,4]]]
    # C_sub = [[C[0,3], C[0,4]], [C[1,3], C[1,4]]]
    
    # U indices: A[0,0]->U[0], A[0,1]->U[1], A[1,0]->U[3], A[1,1]->U[4]
    # V indices: B[0,3]->V[3], B[0,4]->V[4], B[1,3]->V[8], B[1,4]->V[9]
    # W indices: C[0,3]->W[6], C[0,4]->W[8], C[1,3]->W[7], C[1,4]->W[9]
    
    strassen2 = [
        # m0 = (A00+A11)*(B03+B14): C03=+1, C13=-1, C14=+1
        ({0: 1, 4: 1}, {3: 1, 9: 1}, {6: 1, 7: -1, 9: 1}),
        # m1 = (A10+A11)*B03: C13=+1, C14=-1
        ({3: 1, 4: 1}, {3: 1}, {7: 1, 9: -1}),
        # m2 = A00*(B14-B04): C04=+1, C14=+1
        ({0: 1}, {4: -1, 9: 1}, {8: 1, 9: 1}),
        # m3 = A11*(B13-B03): C03=+1
        ({4: 1}, {3: -1, 8: 1}, {6: 1}),
        # m4 = (A00+A01)*B14: C03=-1, C04=+1
        ({0: 1, 1: 1}, {9: 1}, {6: -1, 8: 1}),
        # m5 = (A10-A11)*(B03+B13): C13=+1, C14=+1
        ({3: 1, 4: -1}, {3: 1, 8: 1}, {7: 1, 9: 1}),
        # m6 = (A01-A00)*(B04+B14): C03=+1
        ({0: -1, 1: 1}, {4: 1, 9: 1}, {6: 1}),
    ]
    
    for u_c, v_c, w_c in strassen2:
        for idx, val in u_c.items():
            U[idx, col] += val
        for idx, val in v_c.items():
            V[idx, col] += val
        for idx, val in w_c.items():
            W[idx, col] += val
        col += 1
    
    # Sub-part 2b: j=2, k in {3,4}: schoolbook 2x1*1x2 -> 2x2, rank 4
    # C[0,3] = A[0,2]*B[2,3]
    # C[0,4] = A[0,2]*B[2,4]
    # C[1,3] = A[1,2]*B[2,3]
    # C[1,4] = A[1,2]*B[2,4]
    
    schoolbook_2b = [
        ({0*3+2: 1}, {2*5+3: 1}, {3*2+0: 1}),   # A02*B23 -> C03
        ({0*3+2: 1}, {2*5+4: 1}, {4*2+0: 1}),   # A02*B24 -> C04
        ({1*3+2: 1}, {2*5+3: 1}, {3*2+1: 1}),   # A12*B23 -> C13
        ({1*3+2: 1}, {2*5+4: 1}, {4*2+1: 1}),   # A12*B24 -> C14
    ]
    
    for u_c, v_c, w_c in schoolbook_2b:
        for idx, val in u_c.items():
            U[idx, col] += val
        for idx, val in v_c.items():
            V[idx, col] += val
        for idx, val in w_c.items():
            W[idx, col] += val
        col += 1
    
    assert col == 28, f"Expected 28 terms, got {col}"
    
    # Verify the decomposition
    if time.time() < deadline - 5.0:
        if not _verify_decomposition(U, V, W, n, m, p, trials=50):
            # Fallback to rank-30 schoolbook
            rank_fb = 30
            U_fb = np.zeros((n * m, rank_fb), dtype=int)
            V_fb = np.zeros((m * p, rank_fb), dtype=int)
            W_fb = np.zeros((p * n, rank_fb), dtype=int)
            c = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_fb[i*m+j, c] = 1
                        V_fb[j*p+k, c] = 1
                        W_fb[k*n+i, c] = 1
                        c += 1
            return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
