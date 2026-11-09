"""Matrix multiplication 4x5x6 tensor decomposition search."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 150.0
    SAFETY = 15.0

    n, m, p = DIMENSIONS  # 4, 5, 6

    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W, rank

    def verify_fast(U, V, W, n, m, p, tol=1e-6):
        """Fast verification using tensor contraction."""
        rank = U.shape[1]
        U_reshaped = U.reshape(n, m, rank)
        V_reshaped = V.reshape(m, p, rank)
        W_reshaped = W.reshape(p, n, rank)
        T = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped, optimize=True)
        return np.allclose(T, 1.0, atol=tol)

    def strassen_2x2x2():
        """
        Strassen decomposition for 2x2 * 2x2 matrix multiplication.
        Returns U (4x7), V (4x7), W (4x7) such that
        T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all i,j,k.
        """
        # Strassen's 7 multiplications for 2x2:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        #
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        #
        # In tensor form, U[i,j,c] corresponds to the coefficient of A[i,j] in the c-th product,
        # V[j,k,c] to the coefficient of B[j,k], and W[k,i,c] to the coefficient of C[i,k].
        #
        # Indices: i,j in {0,1} for A rows/cols; j,k in {0,1} for B rows/cols; k,i in {0,1} for C cols/rows
        # U[i,j,c]: coefficient of A[i,j] in product c
        # V[j,k,c]: coefficient of B[j,k] in product c
        # W[k,i,c]: coefficient of C[i,k] in product c (note: W is indexed by (k*n+i) = (k*2+i))
        
        U = np.zeros((4, 7), dtype=float)
        V = np.zeros((4, 7), dtype=float)
        W = np.zeros((4, 7), dtype=float)
        
        # M1 = (A11+A22)(B11+B22)
        # A11 = A[0,0], A22 = A[1,1]
        # B11 = B[0,0], B22 = B[1,1]
        # U[0,0,0]=1, U[1,1,0]=1  (A11 + A22)
        # V[0,0,0]=1, V[1,1,0]=1  (B11 + B22)
        # C11 += M1, C22 += M1
        # W[0,0,0]=1 (C[0,0]), W[1,1,0]=1 (C[1,1])
        U[0*2+0, 0] = 1.0; U[1*2+1, 0] = 1.0
        V[0*2+0, 0] = 1.0; V[1*2+1, 0] = 1.0
        W[0*2+0, 0] = 1.0; W[1*2+1, 0] = 1.0
        
        # M2 = (A21+A22)B11
        # A21 = A[1,0], A22 = A[1,1]
        # B11 = B[0,0]
        # U[1,0,1]=1, U[1,1,1]=1
        # V[0,0,1]=1
        # C21 += M2, C22 -= M2
        # W[0,1,1]=1 (C[1,0]), W[1,1,1]=-1 (C[1,1])
        U[1*2+0, 1] = 1.0; U[1*2+1, 1] = 1.0
        V[0*2+0, 1] = 1.0
        W[0*2+1, 1] = 1.0; W[1*2+1, 1] = -1.0
        
        # M3 = A11(B12-B22)
        # A11 = A[0,0]
        # B12 = B[0,1], B22 = B[1,1]
        # U[0,0,2]=1
        # V[0,1,2]=1, V[1,1,2]=-1
        # C12 += M3, C22 += M3
        # W[1,0,2]=1 (C[0,1]), W[1,1,2]=1 (C[1,1])
        U[0*2+0, 2] = 1.0
        V[0*2+1, 2] = 1.0; V[1*2+1, 2] = -1.0
        W[1*2+0, 2] = 1.0; W[1*2+1, 2] = 1.0
        
        # M4 = A22(B21-B11)
        # A22 = A[1,1]
        # B21 = B[1,0], B11 = B[0,0]
        # U[1,1,3]=1
        # V[1,0,3]=1, V[0,0,3]=-1
        # C11 += M4, C21 += M4
        # W[0,0,3]=1 (C[0,0]), W[0,1,3]=1 (C[1,0])
        U[1*2+1, 3] = 1.0
        V[1*2+0, 3] = 1.0; V[0*2+0, 3] = -1.0
        W[0*2+0, 3] = 1.0; W[0*2+1, 3] = 1.0
        
        # M5 = (A11+A12)B22
        # A11 = A[0,0], A12 = A[0,1]
        # B22 = B[1,1]
        # U[0,0,4]=1, U[0,1,4]=1
        # V[1,1,4]=1
        # C11 -= M5, C12 += M5
        # W[0,0,4]=-1 (C[0,0]), W[1,0,4]=1 (C[0,1])
        U[0*2+0, 4] = 1.0; U[0*2+1, 4] = 1.0
        V[1*2+1, 4] = 1.0
        W[0*2+0, 4] = -1.0; W[1*2+0, 4] = 1.0
        
        # M6 = (A21-A11)(B11+B12)
        # A21 = A[1,0], A11 = A[0,0]
        # B11 = B[0,0], B12 = B[0,1]
        # U[1,0,5]=1, U[0,0,5]=-1
        # V[0,0,5]=1, V[0,1,5]=1
        # C22 += M6
        # W[1,1,5]=1 (C[1,1])
        U[1*2+0, 5] = 1.0; U[0*2+0, 5] = -1.0
        V[0*2+0, 5] = 1.0; V[0*2+1, 5] = 1.0
        W[1*2+1, 5] = 1.0
        
        # M7 = (A12-A22)(B21+B22)
        # A12 = A[0,1], A22 = A[1,1]
        # B21 = B[1,0], B22 = B[1,1]
        # U[0,1,6]=1, U[1,1,6]=-1
        # V[1,0,6]=1, V[1,1,6]=1
        # C11 += M7
        # W[0,0,6]=1 (C[0,0])
        U[0*2+1, 6] = 1.0; U[1*2+1, 6] = -1.0
        V[1*2+0, 6] = 1.0; V[1*2+1, 6] = 1.0
        W[0*2+0, 6] = 1.0
        
        return U, V, W

    def build_full_decomposition():
        """
        Build a decomposition for 4x5 * 5x6 using Strassen on two disjoint 2x2x2 blocks.
        
        Block 1: i in {0,1}, j in {0,1}, k in {0,1} -> Strassen (rank 7)
        Block 2: i in {2,3}, j in {3,4}, k in {4,5} -> Strassen (rank 7)
        All other (i,j,k) combinations: schoolbook (rank 1 each)
        
        Total rank = 7 + 7 + (4*5*6 - 2*2*2*2) = 7 + 7 + (120 - 16) = 118
        """
        U = np.zeros((n * m, n * m * p), dtype=float)
        V = np.zeros((m * p, n * m * p), dtype=float)
        W = np.zeros((p * n, n * m * p), dtype=float)
        
        # Define which (i,j,k) are handled by Strassen
        strassen1 = set()
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    strassen1.add((i, j, k))
        
        strassen2 = set()
        for i in range(2, 4):
            for j in range(3, 5):
                for k in range(4, 6):
                    strassen2.add((i, j, k))
        
        col = 0
        
        # Add Strassen block 1: i in {0,1}, j in {0,1}, k in {0,1}
        U1, V1, W1 = strassen_2x2x2()
        for c in range(7):
            for i_loc in range(2):
                for j_loc in range(2):
                    U[i_loc * m + j_loc, col + c] += U1[i_loc * 2 + j_loc, c]
            for j_loc in range(2):
                for k_loc in range(2):
                    V[j_loc * p + k_loc, col + c] += V1[j_loc * 2 + k_loc, c]
            for k_loc in range(2):
                for i_loc in range(2):
                    W[k_loc * n + i_loc, col + c] += W1[k_loc * 2 + i_loc, c]
        col += 7
        
        # Add Strassen block 2: i in {2,3}, j in {3,4}, k in {4,5}
        U2, V2, W2 = strassen_2x2x2()
        for c in range(7):
            for i_loc in range(2):
                for j_loc in range(2):
                    U[(i_loc + 2) * m + (j_loc + 3), col + c] += U2[i_loc * 2 + j_loc, c]
            for j_loc in range(2):
                for k_loc in range(2):
                    V[(j_loc + 3) * p + (k_loc + 4), col + c] += V2[j_loc * 2 + k_loc, c]
            for k_loc in range(2):
                for i_loc in range(2):
                    W[(k_loc + 4) * n + (i_loc + 2), col + c] += W2[k_loc * 2 + i_loc, c]
        col += 7
        
        # Add schoolbook for all other (i,j,k)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) in strassen1 or (i, j, k) in strassen2:
                        continue
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        
        total_rank = col
        assert total_rank == 118, f"Expected rank 118, got {total_rank}"
        
        return U[:, :total_rank], V[:, :total_rank], W[:, :total_rank], total_rank

    # Build the Strassen-enhanced decomposition
    U_str, V_str, W_str, rank_str = build_full_decomposition()
    
    # Verify
    if verify_fast(U_str, V_str, W_str, n, m, p):
        return {"U": U_str.tolist(), "V": V_str.tolist(), "W": W_str.tolist()}
    
    # Fallback to schoolbook
    U_fb, V_fb, W_fb, rank_fb = schoolbook()
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
# EVOLVE-BLOCK-END
