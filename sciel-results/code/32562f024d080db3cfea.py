"""Matrix multiplication 2x3x10 tensor decomposition using Strassen for 2x2 blocks."""
import time

DIMENSIONS = [2, 3, 10]

def _strassen_2x2x2():
    """
    Return the Strassen rank-7 decomposition for 2x2x2 matrix multiplication.
    
    Strassen's 7 products:
    M1 = (A11 + A22)(B11 + B22)
    M2 = (A21 + A22)B11
    M3 = A11(B21 - B22)
    M4 = A22(B12 - B11)
    M5 = (A11 + A12)B22
    M6 = (A21 - A11)(B11 + B21)
    M7 = (A12 - A22)(B12 + B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    """
    # U: shape (4, 7) for A indices [A11, A12, A21, A22]
    # V: shape (4, 7) for B indices [B11, B12, B21, B22]
    # W: shape (4, 7) for C indices [C11, C12, C21, C22]
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A11 + A22)(B11 + B22)
    U[0][0] = 1; U[2][0] = 1  # A11, A22
    V[0][0] = 1; V[2][0] = 1  # B11, B22
    W[0][0] = 1; W[3][0] = 1  # C11, C22
    
    # M2 = (A21 + A22)B11
    U[1][1] = 1; U[2][1] = 1  # A21, A22
    V[0][1] = 1              # B11
    W[2][1] = 1; W[3][1] = -1  # C21, -C22
    
    # M3 = A11(B21 - B22)
    U[0][2] = 1              # A11
    V[2][2] = 1; V[3][2] = -1  # B21, -B22
    W[1][2] = 1; W[3][2] = 1  # C12, C22
    
    # M4 = A22(B12 - B11)
    U[2][3] = 1              # A22
    V[1][3] = 1; V[0][3] = -1  # B12, -B11
    W[0][3] = 1; W[2][3] = 1  # C11, C21
    
    # M5 = (A11 + A12)B22
    U[0][4] = 1; U[1][4] = 1  # A11, A12
    V[3][4] = 1              # B22
    W[0][4] = -1; W[1][4] = 1  # -C11, C12
    
    # M6 = (A21 - A11)(B11 + B21)
    U[1][5] = -1; U[2][5] = 1  # -A11, A21
    V[0][5] = 1; V[2][5] = 1  # B11, B21
    W[3][5] = 1              # C22
    
    # M7 = (A12 - A22)(B12 + B22)
    U[1][6] = 1; U[2][6] = -1  # A12, -A22
    V[1][6] = 1; V[3][6] = 1  # B12, B22
    W[0][6] = 1              # C11
    
    return U, V, W

def construct():
    """
    Constructs a rank-55 exact tensor decomposition for 2x3x10 matrix multiplication.
    
    Strategy: Partition the 3-dimensional middle index into 2+1.
    - 2x2x10 part uses Strassen (rank 7 per 2x2x2 block, 5 blocks = 35)
    - 2x1x10 part uses schoolbook (rank 20)
    Total rank = 55
    
    Indexing:
    - A is 2x3: A[i,j] with i in {0,1}, j in {0,1,2}
    - B is 3x10: B[j,k] with j in {0,1,2}, k in {0,...,9}
    - C is 2x10: C[i,k] with i in {0,1}, k in {0,...,9}
    - W uses trace convention: W[k*n+i, r] where n=2
    
    U: shape (n*m, rank) = (6, 55)
    V: shape (m*p, rank) = (30, 55)  
    W: shape (p*n, rank) = (20, 55)
    """
    n, m, p = DIMENSIONS  # n=2, m=3, p=10
    
    # Split middle dimension: first 2 rows (j=0,1) and last 1 row (j=2)
    # Part 1: A[:,0:2] (2x2) * B[0:2,:] (2x10) -> 2x2x10, rank 35
    # Part 2: A[:,2:3] (2x1) * B[2:3,:] (1x10) -> 2x1x10, rank 20
    
    total_rank = 35 + 20
    U = [[0] * total_rank for _ in range(n * m)]  # (6, 55)
    V = [[0] * total_rank for _ in range(m * p)]  # (30, 55)
    W = [[0] * total_rank for _ in range(p * n)]  # (20, 55)
    
    # Get Strassen decomposition
    sU, sV, sW = _strassen_2x2x2()
    
    # Part 1: 2x2x10 using Strassen
    # Partition 10 columns into 5 groups of 2
    col_groups = [(0,1), (2,3), (4,5), (6,7), (8,9)]
    
    for g, (k1, k2) in enumerate(col_groups):
        for r in range(7):
            col = g * 7 + r
            
            # U indices for A[i,j] where i in {0,1}, j in {0,1}
            # A index: i*m + j = i*3 + j
            u_idx_map = {
                0: 0 * 3 + 0,  # A[0,0]
                1: 0 * 3 + 1,  # A[0,1]
                2: 1 * 3 + 0,  # A[1,0]
                3: 1 * 3 + 1,  # A[1,1]
            }
            for local_i, global_i in u_idx_map.items():
                U[global_i][col] += sU[local_i][r]
            
            # V indices for B[j,k] where j in {0,1}, k in {k1,k2}
            # B index: j*p + k = j*10 + k
            v_idx_map = {
                0: 0 * 10 + k1,  # B[0,k1]
                1: 0 * 10 + k2,  # B[0,k2]
                2: 1 * 10 + k1,  # B[1,k1]
                3: 1 * 10 + k2,  # B[1,k2]
            }
            for local_j, global_j in v_idx_map.items():
                V[global_j][col] += sV[local_j][r]
            
            # W indices for C[i,k] where i in {0,1}, k in {k1,k2}
            # W index: k*n + i = k*2 + i (trace convention)
            w_idx_map = {
                0: k1 * 2 + 0,  # C[0,k1]
                1: k1 * 2 + 1,  # C[1,k1]
                2: k2 * 2 + 0,  # C[0,k2]
                3: k2 * 2 + 1,  # C[1,k2]
            }
            for local_c, global_c in w_idx_map.items():
                W[global_c][col] += sW[local_c][r]
    
    # Part 2: 2x1x10 using schoolbook
    # A[:,2] (2x1) * B[2,:] (1x10)
    # Rank = 2*1*10 = 20
    for i in range(2):
        for k in range(10):
            col = 35 + i * 10 + k
            
            # A[i, 2]
            u_idx = i * 3 + 2
            U[u_idx][col] = 1
            
            # B[2, k]
            v_idx = 2 * 10 + k
            V[v_idx][col] = 1
            
            # C[i, k] -> W[k*2 + i]
            w_idx = k * 2 + i
            W[w_idx][col] = 1
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    import numpy as np
    result = construct()
    U = np.array(result["U"])
    V = np.array(result["V"])
    W = np.array(result["W"])
    
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    
    # Verify: for random A, B, check that C = A@B equals sum_r U[:,r] * V[:,r] * W[:,r]
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    C_computed = np.zeros((n, p))
    for r in range(rank):
        u_col = U[:, r].reshape(n, m)
        v_col = V[:, r].reshape(m, p)
        w_col = W[:, r].reshape(p, n)
        # The tensor decomposition: C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # This is: C = sum_r (U_col outer V_col) contracted... 
        # Actually: C[i,k] = sum_{j,r} A[i,j] * U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        # Wait, the standard form is: C[i,k] = sum_r W[k*n+i,r] * (sum_j A[i,j] * U[i*m+j,r]) * (sum_j B[j,k] * V[j*p+k,r])
        # No, the correct interpretation:
        # C[i,k] = sum_r (sum_j A[i,j] * U[i*m+j,r]) * (sum_j B[j,k] * V[j*p+k,r]) * W[k*n+i,r]
        # Actually for matrix multiplication tensor: T[i,j,k] = delta[j,j'] for A[i,j]*B[j,k]
        # The decomposition: T = sum_r U_r ⊗ V_r ⊗ W_r where
        # C[i,k] = sum_r W_r[k*n+i] * (sum_j A[i,j] U_r[i*m+j]) * (sum_j B[j,k] V_r[j*p+k])
        
        # Simpler: just compute directly
        pass
    
    # Direct verification
    C_direct = np.zeros((n, p))
    for r in range(rank):
        for i in range(n):
            for k in range(p):
                val = 0.0
                for j in range(m):
                    val += A[i,j] * U[i*m+j, r] * V[j*p+k, r]
                C_direct[i,k] += W[k*n+i, r] * val
    
    error = np.max(np.abs(C_direct - C_expected))
    print(f"Rank: {rank}")
    print(f"Max error: {error}")
    print(f"Valid: {error < 1e-10}")
