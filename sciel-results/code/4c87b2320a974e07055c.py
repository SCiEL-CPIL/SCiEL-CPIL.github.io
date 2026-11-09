"""Matrix multiplication 2x4x6 rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T of shape (n*m, m*p, p*n).
    T[i*m + j, j*p + k, k*n + i] = 1 for all i, j, k, else 0.
    """
    T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1
    return T

def _verify_decomposition(U, V, W, T):
    """Verify that U, V, W exactly decompose T."""
    recon = np.einsum('ic,jc,kc->ijk', U, V, W)
    return np.allclose(recon, T, atol=1e-10)

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_schoolbook = n * m * p
    
    # Strassen-based decomposition for 2x4x6
    # Split A (2x4) into A1 (2x2) and A2 (2x2)
    # Split B (4x6) into B1 (2x6) and B2 (2x6)
    # C = A1 B1 + A2 B2
    # B1 is split into 3 blocks of 2x2: B11, B12, B13
    # B2 is split into 3 blocks of 2x2: B21, B22, B23
    # A1 B1 = A1 B11 + A1 B12 + A1 B13
    # A2 B2 = A2 B21 + A2 B22 + A2 B23
    # Each 2x2x2 multiplication uses rank 7.
    # Total rank = 6 * 7 = 42.
    
    rank = 42
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen factors for 2x2x2
    # U_s (4x7), V_s (4x7), W_s (4x7)
    # Rows of U_s: A00, A01, A10, A11
    # Rows of V_s: B00, B01, B10, B11
    # Rows of W_s: C00, C01, C10, C11
    
    U_s = np.zeros((4, 7))
    V_s = np.zeros((4, 7))
    W_s = np.zeros((4, 7))
    
    # M1 = (A00+A10)*(B00+B10)
    U_s[0, 0] = 1; U_s[2, 0] = 1
    V_s[0, 0] = 1; V_s[2, 0] = 1
    W_s[0, 0] = 1; W_s[3, 0] = 1
    
    # M2 = (A10+A11)*B00
    U_s[2, 1] = 1; U_s[3, 1] = 1
    V_s[0, 1] = 1
    W_s[2, 1] = 1; W_s[3, 1] = -1
    
    # M3 = A00*(B01-B11)
    U_s[0, 2] = 1
    V_s[1, 2] = 1; V_s[3, 2] = -1
    W_s[1, 2] = 1; W_s[3, 2] = 1
    
    # M4 = A11*(B10-B00)
    U_s[3, 3] = 1
    V_s[2, 3] = 1; V_s[0, 3] = -1
    W_s[0, 3] = 1; W_s[2, 3] = 1
    
    # M5 = (A00+A01)*B11
    U_s[0, 4] = 1; U_s[1, 4] = 1
    V_s[3, 4] = 1
    W_s[0, 4] = -1; W_s[1, 4] = 1
    
    # M6 = (A10-A00)*(B00+B01)
    U_s[2, 5] = 1; U_s[0, 5] = -1
    V_s[0, 5] = 1; V_s[1, 5] = 1
    W_s[3, 5] = 1
    
    # M7 = (A01-A11)*(B10+B11)
    U_s[1, 6] = 1; U_s[3, 6] = -1
    V_s[2, 6] = 1; V_s[3, 6] = 1
    W_s[0, 6] = 1; W_s[3, 6] = -1
    
    # Helper to add a Strassen block
    # a_rows: indices in U corresponding to the 2x2 A block
    # b_rows: indices in V corresponding to the 2x2 B block
    # c_rows: indices in W corresponding to the 2x2 C block
    # col_offset: starting column index in U, V, W
    
    def add_strassen_block(a_rows, b_rows, c_rows, col_offset):
        for k in range(7):
            # U
            for r in range(4):
                U[a_rows[r], col_offset + k] += U_s[r, k]
            # V
            for r in range(4):
                V[b_rows[r], col_offset + k] += V_s[r, k]
            # W
            for r in range(4):
                W[c_rows[r], col_offset + k] += W_s[r, k]
                
    # A1: A[:, 0:2]
    # A[0,0] -> idx 0, A[0,1] -> idx 1
    # A[1,0] -> idx 4, A[1,1] -> idx 5
    a1_rows = [0, 1, 4, 5]
    
    # A2: A[:, 2:4]
    # A[0,2] -> idx 2, A[0,3] -> idx 3
    # A[1,2] -> idx 6, A[1,3] -> idx 7
    a2_rows = [2, 3, 6, 7]
    
    # B1: B[0:2, :]
    # B[0,0] -> idx 0, B[0,1] -> idx 1, B[0,2] -> idx 2, B[0,3] -> idx 3, B[0,4] -> idx 4, B[0,5] -> idx 5
    # B[1,0] -> idx 6, B[1,1] -> idx 7, B[1,2] -> idx 8, B[1,3] -> idx 9, B[1,4] -> idx 10, B[1,5] -> idx 11
    
    # B1 blocks:
    # B11: B[0:2, 0:2] -> rows 0,1,6,7
    b11_rows = [0, 1, 6, 7]
    # B12: B[0:2, 2:4] -> rows 2,3,8,9
    b12_rows = [2, 3, 8, 9]
    # B13: B[0:2, 4:6] -> rows 4,5,10,11
    b13_rows = [4, 5, 10, 11]
    
    # B2: B[2:4, :]
    # B[2,0] -> idx 12, B[2,1] -> idx 13, B[2,2] -> idx 14, B[2,3] -> idx 15, B[2,4] -> idx 16, B[2,5] -> idx 17
    # B[3,0] -> idx 18, B[3,1] -> idx 19, B[3,2] -> idx 20, B[3,3] -> idx 21, B[3,4] -> idx 22, B[3,5] -> idx 23
    
    # B2 blocks:
    # B21: B[2:4, 0:2] -> rows 12,13,18,19
    b21_rows = [12, 13, 18, 19]
    # B22: B[2:4, 2:4] -> rows 14,15,20,21
    b22_rows = [14, 15, 20, 21]
    # B23: B[2:4, 4:6] -> rows 16,17,22,23
    b23_rows = [16, 17, 22, 23]
    
    # C blocks
    # C is 2x6.
    # C[0,0] -> idx 0, C[0,1] -> idx 1, C[0,2] -> idx 2, C[0,3] -> idx 3, C[0,4] -> idx 4, C[0,5] -> idx 5
    # C[1,0] -> idx 6, C[1,1] -> idx 7, C[1,2] -> idx 8, C[1,3] -> idx 9, C[1,4] -> idx 10, C[1,5] -> idx 11
    
    # C11: C[:, 0:2] -> rows 0,1,6,7
    c11_rows = [0, 1, 6, 7]
    # C12: C[:, 2:4] -> rows 2,3,8,9
    c12_rows = [2, 3, 8, 9]
    # C13: C[:, 4:6] -> rows 4,5,10,11
    c13_rows = [4, 5, 10, 11]
    
    # A1 B11
    add_strassen_block(a1_rows, b11_rows, c11_rows, 0)
    # A1 B12
    add_strassen_block(a1_rows, b12_rows, c12_rows, 7)
    # A1 B13
    add_strassen_block(a1_rows, b13_rows, c13_rows, 14)
    
    # A2 B21
    add_strassen_block(a2_rows, b21_rows, c11_rows, 21)
    # A2 B22
    add_strassen_block(a2_rows, b22_rows, c12_rows, 28)
    # A2 B23
    add_strassen_block(a2_rows, b23_rows, c13_rows, 35)
    
    # Verify
    T = _build_tensor(n, m, p)
    if _verify_decomposition(U, V, W, T):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook
        U_school = np.zeros((n * m, rank_schoolbook), dtype=int)
        V_school = np.zeros((m * p, rank_schoolbook), dtype=int)
        W_school = np.zeros((p * n, rank_schoolbook), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i * m + j, col] = 1
                    V_school[j * p + k, col] = 1
                    W_school[k * n + i, col] = 1
                    col += 1
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
