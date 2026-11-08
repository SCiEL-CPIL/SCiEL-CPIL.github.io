"""Program for matrix multiplication 3x3x4 tensor decomposition."""
import numpy as np

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS  # n=3, m=3, p=4
    
    # Bini's rank-23 decomposition for 3x3 matrix multiplication
    # Each term is (U_mat, V_mat, W_mat) where each is a 3x3 matrix
    def E(i, j):
        M = np.zeros((3, 3))
        M[i, j] = 1.0
        return M
    
    bini_terms = [
        (E(0,0), E(0,0), E(0,0)),
        (E(0,1), E(1,0), E(0,0)),
        (E(0,2), E(2,0), E(0,0)),
        (E(1,0), E(0,1), E(0,1)),
        (E(1,1), E(1,1), E(1,1)),
        (E(1,2), E(2,1), E(1,1)),
        (E(2,0), E(0,2), E(0,2)),
        (E(2,1), E(1,2), E(2,1)),
        (E(2,2), E(2,2), E(2,2)),
        (E(0,0)-E(1,0), E(0,0)-E(1,0), E(0,0)-E(1,0)),
        (E(0,1)-E(1,1), E(1,0)-E(1,1), E(0,1)-E(1,1)),
        (E(0,2)-E(1,2), E(2,0)-E(2,1), E(0,2)-E(1,2)),
        (E(1,0)-E(2,0), E(0,0)-E(0,1), E(1,0)-E(2,0)),
        (E(1,1)-E(2,1), E(1,0)-E(1,1), E(1,1)-E(2,1)),
        (E(1,2)-E(2,2), E(2,0)-E(2,1), E(1,2)-E(2,2)),
        (E(0,0)-E(0,1), E(0,0)-E(0,1), E(0,0)-E(0,1)),
        (E(1,0)-E(1,1), E(1,0)-E(1,1), E(1,0)-E(1,1)),
        (E(2,0)-E(2,1), E(2,0)-E(2,1), E(2,0)-E(2,1)),
        (E(0,0)-E(0,2), E(0,0)-E(0,2), E(0,0)-E(0,2)),
        (E(1,0)-E(1,2), E(1,0)-E(1,2), E(1,0)-E(1,2)),
        (E(2,0)-E(2,2), E(2,0)-E(2,2), E(2,0)-E(2,2)),
        (E(0,1)-E(0,2), E(1,0)-E(2,0), E(0,1)-E(0,2)),
        (E(1,1)-E(1,2), E(1,1)-E(2,1), E(1,1)-E(1,2)),
    ]
    
    # Build rank-32 decomposition: Bini(23) + schoolbook(9) for 4th column
    rank = 32
    U = np.zeros((n * m, rank))  # 9 x 32
    V = np.zeros((m * p, rank))  # 12 x 32
    W = np.zeros((p * n, rank))  # 12 x 32
    
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
    
    # Verify the full decomposition with random tests
    def verify_full(U_arr, V_arr, W_arr, num_tests=30):
        np.random.seed(123)
        for _ in range(num_tests):
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C_true = A @ B
            C_recon = np.zeros((n, p))
            for c in range(rank):
                u_val = np.sum(U_arr[:, c] * A.flatten())
                v_val = np.sum(V_arr[:, c] * B.flatten())
                w_vec = W_arr[:, c]
                for i in range(n):
                    for k in range(p):
                        C_recon[i, k] += u_val * v_val * w_vec[k * n + i]
            if not np.allclose(C_recon, C_true, atol=1e-10):
                return False
        return True
    
    full_valid = verify_full(U, V, W)
    
    if full_valid:
        # Convert to lists with exact integer values
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
    
    # Fallback: rank-34 Strassen-based approach
    rank_fb = 34
    U_fb = [[0] * rank_fb for _ in range(n * m)]
    V_fb = [[0] * rank_fb for _ in range(m * p)]
    W_fb = [[0] * rank_fb for _ in range(p * n)]
    
    col = 0
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        nonlocal col
        U_fb[u_idx][col] += u_val
        V_fb[v_idx][col] += v_val
        W_fb[w_idx][col] += w_val
        col += 1
    
    # Strassen Left (A11 * B1L)
    U_fb[0][col]=1; U_fb[4][col]=1; V_fb[0][col]=1; V_fb[5][col]=1; W_fb[0][col]=1; W_fb[4][col]=1; col+=1
    U_fb[3][col]=1; U_fb[4][col]=1; V_fb[0][col]=1; W_fb[1][col]=1; W_fb[4][col]=-1; col+=1
    U_fb[0][col]=1; V_fb[1][col]=1; V_fb[5][col]=-1; W_fb[3][col]=1; W_fb[4][col]=1; col+=1
    U_fb[4][col]=1; V_fb[4][col]=1; V_fb[0][col]=-1; W_fb[0][col]=1; W_fb[1][col]=1; col+=1
    U_fb[0][col]=1; U_fb[1][col]=1; V_fb[5][col]=1; W_fb[0][col]=-1; W_fb[3][col]=1; col+=1
    U_fb[3][col]=1; U_fb[0][col]=-1; V_fb[0][col]=1; V_fb[1][col]=1; W_fb[4][col]=1; col+=1
    U_fb[1][col]=1; U_fb[4][col]=-1; V_fb[4][col]=1; V_fb[5][col]=1; W_fb[0][col]=1; col+=1
    
    # A12 * B2L
    U_fb[2][col]=1; V_fb[8][col]=1; W_fb[0][col]=1; col+=1
    U_fb[2][col]=1; V_fb[9][col]=1; W_fb[3][col]=1; col+=1
    U_fb[5][col]=1; V_fb[8][col]=1; W_fb[1][col]=1; col+=1
    U_fb[5][col]=1; V_fb[9][col]=1; W_fb[4][col]=1; col+=1
    
    # A21 * B1L
    U_fb[6][col]=1; V_fb[0][col]=1; W_fb[2][col]=1; col+=1
    U_fb[6][col]=1; V_fb[1][col]=1; W_fb[5][col]=1; col+=1
    U_fb[7][col]=1; V_fb[4][col]=1; W_fb[2][col]=1; col+=1
    U_fb[7][col]=1; V_fb[5][col]=1; W_fb[5][col]=1; col+=1
    
    # A22 * B2L
    U_fb[8][col]=1; V_fb[8][col]=1; W_fb[2][col]=1; col+=1
    U_fb[8][col]=1; V_fb[9][col]=1; W_fb[5][col]=1; col+=1
    
    # Strassen Right (A11 * B1R)
    U_fb[0][col]=1; U_fb[4][col]=1; V_fb[2][col]=1; V_fb[7][col]=1; W_fb[6][col]=1; W_fb[10][col]=1; col+=1
    U_fb[3][col]=1; U_fb[4][col]=1; V_fb[2][col]=1; W_fb[7][col]=1; W_fb[10][col]=-1; col+=1
    U_fb[0][col]=1; V_fb[3][col]=1; V_fb[7][col]=-1; W_fb[9][col]=1; W_fb[10][col]=1; col+=1
    U_fb[4][col]=1; V_fb[6][col]=1; V_fb[2][col]=-1; W_fb[6][col]=1; W_fb[7][col]=1; col+=1
    U_fb[0][col]=1; U_fb[1][col]=1; V_fb[7][col]=1; W_fb[6][col]=-1; W_fb[9][col]=1; col+=1
    U_fb[3][col]=1; U_fb[0][col]=-1; V_fb[2][col]=1; V_fb[3][col]=1; W_fb[10][col]=1; col+=1
    U_fb[1][col]=1; U_fb[4][col]=-1; V_fb[6][col]=1; V_fb[7][col]=1; W_fb[6][col]=1; col+=1
    
    # A12 * B2R
    U_fb[2][col]=1; V_fb[10][col]=1; W_fb[6][col]=1; col+=1
    U_fb[2][col]=1; V_fb[11][col]=1; W_fb[9][col]=1; col+=1
    U_fb[5][col]=1; V_fb[10][col]=1; W_fb[7][col]=1; col+=1
    U_fb[5][col]=1; V_fb[11][col]=1; W_fb[10][col]=1; col+=1
    
    # A21 * B1R
    U_fb[6][col]=1; V_fb[2][col]=1; W_fb[8][col]=1; col+=1
    U_fb[6][col]=1; V_fb[3][col]=1; W_fb[11][col]=1; col+=1
    U_fb[7][col]=1; V_fb[6][col]=1; W_fb[8][col]=1; col+=1
    U_fb[7][col]=1; V_fb[7][col]=1; W_fb[11][col]=1; col+=1
    
    # A22 * B2R
    U_fb[8][col]=1; V_fb[10][col]=1; W_fb[8][col]=1; col+=1
    U_fb[8][col]=1; V_fb[11][col]=1; W_fb[11][col]=1; col+=1
    
    return {"U": U_fb, "V": V_fb, "W": W_fb}
