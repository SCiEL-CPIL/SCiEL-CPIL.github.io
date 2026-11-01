import time
import numpy as np
from numpy import linalg as LA

DIMENSIONS = [2, 2, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # A is n x m (2x2), B is m x p (2x7), C is n x p (2x7)
    # C[i,j] = sum_k A[i,k] * B[k,j]
    # Decomposition: sum_c U[i*m+k, c] * V[k*p+j, c] * W[j*n+i, c] = 1 for all i,k,j
    
    deadline = time.time() + 145.0
    
    def schoolbook():
        r = n * m * p
        U = [[0]*r for _ in range(n*m)]
        V = [[0]*r for _ in range(m*p)]
        W = [[0]*r for _ in range(p*n)]
        col = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    U[i*m + k][col] = 1
                    V[k*p + j][col] = 1
                    W[j*n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best = schoolbook()
    
    def verify(U_mat, V_mat, W_mat):
        """Check if sum_c U[i*m+k,c]*V[k*p+j,c]*W[j*n+i,c] == 1 for all i,k,j"""
        r = U_mat.shape[1]
        u_r = U_mat.T.reshape(n, m, r)
        v_r = V_mat.T.reshape(m, p, r)
        w_r = W_mat.T.reshape(p, n, r)
        T = np.einsum('ikc,kjc,jic->ijk', u_r, v_r, w_r)
        return np.allclose(T, 1.0, atol=1e-8)
    
    def is_half_integer(mat):
        return np.allclose(mat * 2, np.round(mat * 2), atol=1e-8)
    
    def mat_to_list(mat):
        """Convert matrix to list of lists of Python floats/ints"""
        return mat.tolist()
    
    def strassen_2x2_2x2():
        """
        Returns U, V, W matrices for 2x2 * 2x2 multiplication using Strassen's algorithm.
        Shapes: U (4, 7), V (4, 7), W (4, 7)
        Indices:
        U[i*m+k, c] -> U[i*2+k, c]
        V[k*p+j, c] -> V[k*2+j, c]
        W[j*n+i, c] -> W[j*2+i, c]
        
        Strassen variables:
        M1 = (A11 + A22)(B12 - B22)
        M2 = (A21 + A22)B11
        M3 = A11(B11 + B21)
        M4 = A22(B22 + B12)
        M5 = (A11 + A12)B22
        M6 = (A21 - A11)(B11 + B12)
        M7 = (A12 - A22)(B21 + B22)
        
        C11 = M1 + M4 - M5 + M7
        C12 = M3 + M5
        C21 = M2 + M4
        C22 = M1 - M2 + M3 + M6
        """
        r = 7
        U = np.zeros((4, r))
        V = np.zeros((4, r))
        W = np.zeros((4, r))
        
        # M1 = (A11 + A22)(B12 - B22)
        # U: A11 (0,0) -> 1, A22 (1,1) -> 1
        U[0, 0] = 1.0
        U[3, 0] = 1.0
        # V: B12 (0,1) -> 1, B22 (1,1) -> -1
        V[1, 0] = 1.0
        V[3, 0] = -1.0
        # W: C11 (0,0) -> 1, C22 (1,1) -> 1
        W[0, 0] = 1.0
        W[3, 0] = 1.0
        
        # M2 = (A21 + A22)B11
        # U: A21 (1,0) -> 1, A22 (1,1) -> 1
        U[2, 1] = 1.0
        U[3, 1] = 1.0
        # V: B11 (0,0) -> 1
        V[0, 1] = 1.0
        # W: C21 (1,0) -> 1, C22 (1,1) -> -1
        W[2, 1] = 1.0
        W[3, 1] = -1.0
        
        # M3 = A11(B11 + B21)
        # U: A11 (0,0) -> 1
        U[0, 2] = 1.0
        # V: B11 (0,0) -> 1, B21 (1,0) -> 1
        V[0, 2] = 1.0
        V[2, 2] = 1.0
        # W: C12 (0,1) -> 1, C21 (1,0) -> 1
        W[1, 2] = 1.0
        W[2, 2] = 1.0
        
        # M4 = A22(B22 + B12)
        # U: A22 (1,1) -> 1
        U[3, 3] = 1.0
        # V: B22 (1,1) -> 1, B12 (0,1) -> 1
        V[3, 3] = 1.0
        V[1, 3] = 1.0
        # W: C11 (0,0) -> 1, C21 (1,0) -> 1
        W[0, 3] = 1.0
        W[2, 3] = 1.0
        
        # M5 = (A11 + A12)B22
        # U: A11 (0,0) -> 1, A12 (0,1) -> 1
        U[0, 4] = 1.0
        U[1, 4] = 1.0
        # V: B22 (1,1) -> 1
        V[3, 4] = 1.0
        # W: C11 (0,0) -> -1, C12 (0,1) -> 1
        W[0, 4] = -1.0
        W[1, 4] = 1.0
        
        # M6 = (A21 - A11)(B11 + B12)
        # U: A21 (1,0) -> 1, A11 (0,0) -> -1
        U[2, 5] = 1.0
        U[0, 5] = -1.0
        # V: B11 (0,0) -> 1, B12 (0,1) -> 1
        V[0, 5] = 1.0
        V[1, 5] = 1.0
        # W: C22 (1,1) -> 1
        W[3, 5] = 1.0
        
        # M7 = (A12 - A22)(B21 + B22)
        # U: A12 (0,1) -> 1, A22 (1,1) -> -1
        U[1, 6] = 1.0
        U[3, 6] = -1.0
        # V: B21 (1,0) -> 1, B22 (1,1) -> 1
        V[2, 6] = 1.0
        V[3, 6] = 1.0
        # W: C11 (0,0) -> 1
        W[0, 6] = 1.0
        
        return U, V, W
    
    def schoolbook_2x2_2x1():
        """
        Returns U, V, W matrices for 2x2 * 2x1 multiplication using schoolbook.
        Shapes: U (4, 4), V (2, 4), W (2, 4)
        Note: V has shape (m*p, r) where p=1, so (2, 4). W has shape (p*n, r) where p=1, so (2, 4).
        Indices:
        U[i*m+k, c] -> U[i*2+k, c]
        V[k*p+j, c] -> V[k*1+0, c] = V[k, c]
        W[j*n+i, c] -> W[0*2+i, c] = W[i, c]
        
        C10 = A10*B00 + A11*B10
        C20 = A20*B00 + A21*B10
        
        c=0: A10*B00 -> C10
        c=1: A11*B10 -> C10
        c=2: A20*B00 -> C20
        c=3: A21*B10 -> C20
        """
        r = 4
        U = np.zeros((4, r))
        V = np.zeros((2, r))
        W = np.zeros((2, r))
        
        # c=0: A10 (0,0) * B00 (0,0) -> C10 (0,0)
        U[0, 0] = 1.0
        V[0, 0] = 1.0
        W[0, 0] = 1.0
        
        # c=1: A11 (0,1) * B10 (1,0) -> C10 (0,0)
        U[1, 1] = 1.0
        V[1, 1] = 1.0
        W[0, 1] = 1.0
        
        # c=2: A20 (1,0) * B00 (0,0) -> C20 (1,0)
        U[2, 2] = 1.0
        V[0, 2] = 1.0
        W[1, 2] = 1.0
        
        # c=3: A21 (1,1) * B10 (1,0) -> C20 (1,0)
        U[3, 3] = 1.0
        V[1, 3] = 1.0
        W[1, 3] = 1.0
        
        return U, V, W
    
    # Construct the full 2x2 by 2x7 decomposition
    # Split B into B1 (2x2), B2 (2x2), B3 (2x2), B4 (2x1)
    # Columns 0-1: Block 1 (Strassen)
    # Columns 2-3: Block 2 (Strassen)
    # Columns 4-5: Block 3 (Strassen)
    # Column 6: Block 4 (Schoolbook 2x1)
    
    U_full = np.zeros((4, 25))
    V_full = np.zeros((14, 25))
    W_full = np.zeros((14, 25))
    
    U_s, V_s, W_s = strassen_2x2_2x2()
    U_sb, V_sb, W_sb = schoolbook_2x2_2x1()
    
    # Block 1: cols 0-1
    U_full[:, 0:7] = U_s
    V_full[0:4, 0:7] = V_s
    W_full[0:4, 0:7] = W_s
    
    # Block 2: cols 2-3
    U_full[:, 7:14] = U_s
    V_full[4:8, 7:14] = V_s
    W_full[4:8, 7:14] = W_s
    
    # Block 3: cols 4-5
    U_full[:, 14:21] = U_s
    V_full[8:12, 14:21] = V_s
    W_full[8:12, 14:21] = W_s
    
    # Block 4: col 6
    U_full[:, 21:25] = U_sb
    V_full[12:14, 21:25] = V_sb
    W_full[12:14, 21:25] = W_sb
    
    # Verify
    if verify(U_full, V_full, W_full):
        if is_half_integer(U_full) and is_half_integer(V_full) and is_half_integer(W_full):
            return {
                "U": mat_to_list(U_full),
                "V": mat_to_list(V_full),
                "W": mat_to_list(W_full)
            }
    
    # Fallback to schoolbook if verification fails
    return best
# EVOLVE-BLOCK-END
