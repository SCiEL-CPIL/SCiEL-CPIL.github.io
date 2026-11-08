"""Program for matrix multiplication 3x3x4 tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # 120s limit, use 118s for safety
    
    n, m, p = DIMENSIONS  # n=3, m=3, p=4
    
    # Strategy: Use Bini's rank-23 algorithm for the 3x3x3 part
    # and schoolbook (rank 9) for the 4th column of B, giving total rank 32.
    # This is better than the current rank 34 approach.
    
    def E(i, j):
        M = np.zeros((3, 3))
        M[i, j] = 1.0
        return M
    
    # Bini's rank-23 decomposition for 3x3 matrix multiplication
    bini_terms = []
    bini_terms.append((E(0,0), E(0,0), E(0,0)))
    bini_terms.append((E(0,1), E(1,0), E(0,0)))
    bini_terms.append((E(0,2), E(2,0), E(0,0)))
    bini_terms.append((E(1,0), E(0,1), E(0,1)))
    bini_terms.append((E(1,1), E(1,1), E(1,1)))
    bini_terms.append((E(1,2), E(2,1), E(1,1)))
    bini_terms.append((E(2,0), E(0,2), E(0,2)))
    bini_terms.append((E(2,1), E(1,2), E(2,1)))
    bini_terms.append((E(2,2), E(2,2), E(2,2)))
    bini_terms.append((E(0,0)-E(1,0), E(0,0)-E(1,0), E(0,0)-E(1,0)))
    bini_terms.append((E(0,1)-E(1,1), E(1,0)-E(1,1), E(0,1)-E(1,1)))
    bini_terms.append((E(0,2)-E(1,2), E(2,0)-E(2,1), E(0,2)-E(1,2)))
    bini_terms.append((E(1,0)-E(2,0), E(0,0)-E(0,1), E(1,0)-E(2,0)))
    bini_terms.append((E(1,1)-E(2,1), E(1,0)-E(1,1), E(1,1)-E(2,1)))
    bini_terms.append((E(1,2)-E(2,2), E(2,0)-E(2,1), E(1,2)-E(2,2)))
    bini_terms.append((E(0,0)-E(0,1), E(0,0)-E(0,1), E(0,0)-E(0,1)))
    bini_terms.append((E(1,0)-E(1,1), E(1,0)-E(1,1), E(1,0)-E(1,1)))
    bini_terms.append((E(2,0)-E(2,1), E(2,0)-E(2,1), E(2,0)-E(2,1)))
    bini_terms.append((E(0,0)-E(0,2), E(0,0)-E(0,2), E(0,0)-E(0,2)))
    bini_terms.append((E(1,0)-E(1,2), E(1,0)-E(1,2), E(1,0)-E(1,2)))
    bini_terms.append((E(2,0)-E(2,2), E(2,0)-E(2,2), E(2,0)-E(2,2)))
    bini_terms.append((E(0,1)-E(0,2), E(1,0)-E(2,0), E(0,1)-E(0,2)))
    bini_terms.append((E(1,1)-E(1,2), E(1,1)-E(2,1), E(1,1)-E(1,2)))
    
    # Verify Bini decomposition with a few random tests (vectorized)
    def verify_bini(terms, num_tests=20):
        np.random.seed(42)
        # Pre-compute U and V matrices for all terms
        U_mats = np.array([U_mat.flatten() for U_mat, _, _ in terms])  # (23, 9)
        V_mats = np.array([V_mat.flatten() for _, V_mat, _ in terms])  # (23, 9)
        W_mats = np.array([W_mat.flatten() for _, _, W_mat in terms])  # (23, 9)
        
        for _ in range(num_tests):
            A = np.random.randn(3, 3)
            B = np.random.randn(3, 3)
            C_true = A @ B
            A_flat = A.flatten()
            B_flat = B.flatten()
            u_vals = U_mats @ A_flat  # (23,)
            v_vals = V_mats @ B_flat  # (23,)
            coeffs = u_vals * v_vals  # (23,)
            C_recon = (coeffs[:, None] * W_mats).sum(axis=0).reshape(3, 3)
            if not np.allclose(C_recon, C_true, atol=1e-10):
                return False
        return True
    
    bini_valid = verify_bini(bini_terms)
    
    if bini_valid:
        rank = 23 + 9  # 23 for 3x3x3 part + 9 for 4th column
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # Fill Bini terms (columns 0-22)
        for c, (U_mat, V_mat, W_mat) in enumerate(bini_terms):
            for i in range(3):
                for j in range(3):
                    U[i * m + j, c] = U_mat[i, j]
            for j in range(3):
                for k in range(3):
                    V[j * p + k, c] = V_mat[j, k]
            for k in range(3):
                for i in range(3):
                    W[k * n + i, c] = W_mat[k, i]
        
        # Fill schoolbook terms for 4th column (columns 23-31)
        col = 23
        for i in range(n):
            for j in range(m):
                U[i * m + j, col] = 1
                V[j * p + 3, col] = 1
                W[3 * n + i, col] = 1
                col += 1
        
        # Verify the full decomposition (vectorized)
        def verify_full(U_arr, V_arr, W_arr, num_tests=20):
            np.random.seed(123)
            U_flat = U_arr.T  # (rank, n*m)
            V_flat = V_arr.T  # (rank, m*p)
            W_flat = W_arr.T  # (rank, p*n)
            
            for _ in range(num_tests):
                A = np.random.randn(n, m)
                B = np.random.randn(m, p)
                C_true = A @ B
                A_flat = A.flatten()
                B_flat = B.flatten()
                u_vals = U_flat @ A_flat  # (rank,)
                v_vals = V_flat @ B_flat  # (rank,)
                coeffs = u_vals * v_vals  # (rank,)
                C_recon = (coeffs[:, None] * W_flat).sum(axis=0).reshape(p, n)
                C_recon = C_recon.T  # Convert from W[k*n+i] convention to C[i,k]
                if not np.allclose(C_recon, C_true, atol=1e-10):
                    return False
            return True
        
        full_valid = verify_full(U, V, W)
        
        if full_valid:
            # Convert to lists with exact integer/rational values
            U_list = [[0] * rank for _ in range(n * m)]
            V_list = [[0] * rank for _ in range(m * p)]
            W_list = [[0] * rank for _ in range(p * n)]
            
            for i in range(n * m):
                for c in range(rank):
                    val = U[i, c]
                    if abs(val) < 1e-15:
                        U_list[i][c] = 0
                    elif abs(val - round(val)) < 1e-10:
                        U_list[i][c] = int(round(val))
                    else:
                        U_list[i][c] = val
            
            for i in range(m * p):
                for c in range(rank):
                    val = V[i, c]
                    if abs(val) < 1e-15:
                        V_list[i][c] = 0
                    elif abs(val - round(val)) < 1e-10:
                        V_list[i][c] = int(round(val))
                    else:
                        V_list[i][c] = val
            
            for i in range(p * n):
                for c in range(rank):
                    val = W[i, c]
                    if abs(val) < 1e-15:
                        W_list[i][c] = 0
                    elif abs(val - round(val)) < 1e-10:
                        W_list[i][c] = int(round(val))
                    else:
                        W_list[i][c] = val
            
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: Use the current rank-34 Strassen-based approach
    rank = 34
    U = [[0] * rank for _ in range(n * m)]  # 9 x 34
    V = [[0] * rank for _ in range(m * p)]  # 12 x 34
    W = [[0] * rank for _ in range(p * n)]  # 12 x 34
    
    col = 0
    
    # Strassen Left (A11 * B1L)
    # A11 indices: 0,1,3,4
    # B1L indices: 0,1,4,5
    # C1L indices: 0,3,1,4
    
    # P1
    U[0][col] = 1; U[4][col] = 1
    V[0][col] = 1; V[5][col] = 1
    W[0][col] = 1; W[4][col] = 1
    col += 1
    
    # P2
    U[3][col] = 1; U[4][col] = 1
    V[0][col] = 1
    W[1][col] = 1; W[4][col] = -1
    col += 1
    
    # P3
    U[0][col] = 1
    V[1][col] = 1; V[5][col] = -1
    W[3][col] = 1; W[4][col] = 1
    col += 1
    
    # P4
    U[4][col] = 1
    V[4][col] = 1; V[0][col] = -1
    W[0][col] = 1; W[1][col] = 1
    col += 1
    
    # P5
    U[0][col] = 1; U[1][col] = 1
    V[5][col] = 1
    W[0][col] = -1; W[3][col] = 1
    col += 1
    
    # P6
    U[3][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[4][col] = 1
    col += 1
    
    # P7
    U[1][col] = 1; U[4][col] = -1
    V[4][col] = 1; V[5][col] = 1
    W[0][col] = 1
    col += 1
    
    # A12 * B2L
    # A12 indices: 2, 5
    # B2L indices: 8, 9
    # C1L contribution:
    # C[0,0]->0, C[0,1]->3, C[1,0]->1, C[1,1]->4
    
    # T1: A[0,2] B[2,0] C[0,0]
    U[2][col] = 1
    V[8][col] = 1
    W[0][col] = 1
    col += 1
    
    # T2: A[0,2] B[2,1] C[0,1]
    U[2][col] = 1
    V[9][col] = 1
    W[3][col] = 1
    col += 1
    
    # T3: A[1,2] B[2,0] C[1,0]
    U[5][col] = 1
    V[8][col] = 1
    W[1][col] = 1
    col += 1
    
    # T4: A[1,2] B[2,1] C[1,1]
    U[5][col] = 1
    V[9][col] = 1
    W[4][col] = 1
    col += 1
    
    # A21 * B1L
    # A21 indices: 6, 7
    # B1L indices: 0,1,4,5
    # C2L contribution:
    # C[2,0]->2, C[2,1]->5
    
    # T1: A[2,0] B[0,0] C[2,0]
    U[6][col] = 1
    V[0][col] = 1
    W[2][col] = 1
    col += 1
    
    # T2: A[2,0] B[0,1] C[2,1]
    U[6][col] = 1
    V[1][col] = 1
    W[5][col] = 1
    col += 1
    
    # T3: A[2,1] B[1,0] C[2,0]
    U[7][col] = 1
    V[4][col] = 1
    W[2][col] = 1
    col += 1
    
    # T4: A[2,1] B[1,1] C[2,1]
    U[7][col] = 1
    V[5][col] = 1
    W[5][col] = 1
    col += 1
    
    # A22 * B2L
    # A22 index: 8
    # B2L indices: 8, 9
    # C2L contribution:
    # C[2,0]->2, C[2,1]->5
    
    # T1: A[2,2] B[2,0] C[2,0]
    U[8][col] = 1
    V[8][col] = 1
    W[2][col] = 1
    col += 1
    
    # T2: A[2,2] B[2,1] C[2,1]
    U[8][col] = 1
    V[9][col] = 1
    W[5][col] = 1
    col += 1
    
    # Strassen Right (A11 * B1R)
    # A11 indices: 0,1,3,4
    # B1R indices: 2,3,6,7
    # C1R indices: 6,9,7,10
    
    # P1
    U[0][col] = 1; U[4][col] = 1
    V[2][col] = 1; V[7][col] = 1
    W[6][col] = 1; W[10][col] = 1
    col += 1
    
    # P2
    U[3][col] = 1; U[4][col] = 1
    V[2][col] = 1
    W[7][col] = 1; W[10][col] = -1
    col += 1
    
    # P3
    U[0][col] = 1
    V[3][col] = 1; V[7][col] = -1
    W[9][col] = 1; W[10][col] = 1
    col += 1
    
    # P4
    U[4][col] = 1
    V[6][col] = 1; V[2][col] = -1
    W[6][col] = 1; W[7][col] = 1
    col += 1
    
    # P5
    U[0][col] = 1; U[1][col] = 1
    V[7][col] = 1
    W[6][col] = -1; W[9][col] = 1
    col += 1
    
    # P6
    U[3][col] = 1; U[0][col] = -1
    V[2][col] = 1; V[3][col] = 1
    W[10][col] = 1
    col += 1
    
    # P7
    U[1][col] = 1; U[4][col] = -1
    V[6][col] = 1; V[7][col] = 1
    W[6][col] = 1
    col += 1
    
    # A12 * B2R
    # A12 indices: 2, 5
    # B2R indices: 10, 11
    # C1R contribution:
    # C[0,2]->6, C[0,3]->9, C[1,2]->7, C[1,3]->10
    
    # T1: A[0,2] B[2,2] C[0,2]
    U[2][col] = 1
    V[10][col] = 1
    W[6][col] = 1
    col += 1
    
    # T2: A[0,2] B[2,3] C[0,3]
    U[2][col] = 1
    V[11][col] = 1
    W[9][col] = 1
    col += 1
    
    # T3: A[1,2] B[2,2] C[1,2]
    U[5][col] = 1
    V[10][col] = 1
    W[7][col] = 1
    col += 1
    
    # T4: A[1,2] B[2,3] C[1,3]
    U[5][col] = 1
    V[11][col] = 1
    W[10][col] = 1
    col += 1
    
    # A21 * B1R
    # A21 indices: 6, 7
    # B1R indices: 2,3,6,7
    # C2R contribution:
    # C[2,2]->8, C[2,3]->11
    
    # T1: A[2,0] B[0,2] C[2,2]
    U[6][col] = 1
    V[2][col] = 1
    W[8][col] = 1
    col += 1
    
    # T2: A[2,0] B[0,3] C[2,3]
    U[6][col] = 1
    V[3][col] = 1
    W[11][col] = 1
    col += 1
    
    # T3: A[2,1] B[1,2] C[2,2]
    U[7][col] = 1
    V[6][col] = 1
    W[8][col] = 1
    col += 1
    
    # T4: A[2,1] B[1,3] C[2,3]
    U[7][col] = 1
    V[7][col] = 1
    W[11][col] = 1
    col += 1
    
    # A22 * B2R
    # A22 index: 8
    # B2R indices: 10, 11
    # C2R contribution:
    # C[2,2]->8, C[2,3]->11
    
    # T1: A[2,2] B[2,2] C[2,2]
    U[8][col] = 1
    V[10][col] = 1
    W[8][col] = 1
    col += 1
    
    # T2: A[2,2] B[2,3] C[2,3]
    U[8][col] = 1
    V[11][col] = 1
    W[11][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
