import time
import numpy as np

DIMENSIONS = [4, 4, 6]
_START_TIME = time.time()
_TIME_LIMIT = 140.0  # Safety margin before 150s kill

def _time_left():
    return _TIME_LIMIT - (time.time() - _START_TIME)

# 2) Strassen 2x2x2: rank 7
def _strassen_2x2():
    """Returns U (4x7), V (4x7), W (4x7) for 2x2x2 multiplication using Strassen."""
    rank = 7
    U = np.zeros((4, rank), dtype=float)
    V = np.zeros((4, rank), dtype=float)
    W = np.zeros((4, rank), dtype=float)
    
    # Strassen formulas for 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 - M4 + M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Indexing: U row = i*2+j (i=0,1; j=0,1)
    #            V row = j*2+k (j=0,1; k=0,1)
    #            W row = k*2+i (k=0,1; i=0,1)
    
    # A indices: A11=(0,0), A12=(0,1), A21=(1,0), A22=(1,1)
    # B indices: B11=(0,0), B12=(0,1), B21=(1,0), B22=(1,1)
    # C indices: C11=(0,0), C12=(0,1), C21=(1,0), C22=(1,1)
    
    # u vectors (for A): index = i*2+j
    # v vectors (for B): index = j*2+k
    # w vectors (for C): index = k*2+i
    
    # M1 = (A11+A22)(B11+B22)
    # u: A11=1, A22=1 => [1,0,0,1]
    # v: B11=1, B22=1 => [1,0,0,1]
    # w: C11=1, C22=1 => C11=(i=0,k=0)->k*2+i=0, C22=(i=1,k=1)->k*2+i=3 => [1,0,0,1]
    U[0, 0] += 1; U[3, 0] += 1
    V[0, 0] += 1; V[3, 0] += 1
    W[0, 0] += 1; W[3, 0] += 1
    
    # M2 = (A21+A22)B11
    # u: A21=1, A22=1 => [0,0,1,1]
    # v: B11=1 => [1,0,0,0]
    # w: C21=1 => C21=(i=1,k=0)->k*2+i=1 => [0,1,0,0]
    U[2, 1] += 1; U[3, 1] += 1
    V[0, 1] += 1
    W[1, 1] += 1
    
    # M3 = A11(B12-B22)
    # u: A11=1 => [1,0,0,0]
    # v: B12=1, B22=-1 => [0,1,0,-1]
    # w: C12=1, C22=1 => C12=(i=0,k=1)->k*2+i=2, C22=(i=1,k=1)->3 => [0,0,1,1]
    U[0, 2] += 1
    V[1, 2] += 1; V[3, 2] += -1
    W[2, 2] += 1; W[3, 2] += 1
    
    # M4 = A22(B21-B11)
    # u: A22=1 => [0,0,0,1]
    # v: B21=1, B11=-1 => [-1,0,1,0]
    # w: C11=-1, C21=1 => C11=(0,0)->0, C21=(1,0)->1 => [-1,1,0,0]
    U[3, 3] += 1
    V[0, 3] += -1; V[2, 3] += 1
    W[0, 3] += -1; W[1, 3] += 1
    
    # M5 = (A11+A12)B22
    # u: A11=1, A12=1 => [1,1,0,0]
    # v: B22=1 => [0,0,0,1]
    # w: C11=1, C12=1 => C11=(0,0)->0, C12=(0,1)->2 => [1,0,1,0]
    U[0, 4] += 1; U[1, 4] += 1
    V[3, 4] += 1
    W[0, 4] += 1; W[2, 4] += 1
    
    # M6 = (A21-A11)(B11+B12)
    # u: A21=1, A11=-1 => [-1,0,1,0]
    # v: B11=1, B12=1 => [1,1,0,0]
    # w: C22=1 => C22=(1,1)->3 => [0,0,0,1]
    U[0, 5] += -1; U[2, 5] += 1
    V[0, 5] += 1; V[1, 5] += 1
    W[3, 5] += 1
    
    # M7 = (A12-A22)(B21+B22)
    # u: A12=1, A22=-1 => [0,1,0,-1]
    # v: B21=1, B22=1 => [0,0,1,1]
    # w: C11=1, C22=1 => C11=(0,0)->0, C22=(1,1)->3 => [1,0,0,1]
    U[1, 6] += 1; U[3, 6] += -1
    V[2, 6] += 1; V[3, 6] += 1
    W[0, 6] += 1; W[3, 6] += 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # n=4, m=4, p=6
    
    # Strategy: Use Strassen for the 2x2x2 blocks.
    # The 4x4x6 multiplication can be viewed as:
    # - Split A (4x4) into 2x2 blocks: A_{ij} where i,j in {0,1}, each 2x2
    # - Split B (4x6) into 2x3 blocks: B_{jk} where j in {0,1}, k in {0,1,2}, each 2x2
    # - C (4x6) has 2x2 blocks: C_{ik} where i in {0,1}, k in {0,1,2}, each 2x2
    
    # For each block multiplication C_{ik} = sum_j A_{ij} B_{jk}, we use Strassen (rank 7)
    # Total blocks: 2 (i) * 3 (k) * 2 (j) = 12 block multiplications, but each uses rank 7
    # However, the Strassen decomposition is shared across the j-sum? No, each (i,j,k) 
    # block multiplication is independent.
    
    # Actually, for the full 4x4x6:
    # U has shape (n*m, rank) = (16, rank)
    # V has shape (m*p, rank) = (24, rank)  
    # W has shape (p*n, rank) = (24, rank)
    
    # Using block structure with 2x2 blocks:
    # Number of (i_block, j_block, k_block) triples: 2 * 2 * 3 = 12
    # Each uses Strassen rank 7, so total rank = 12 * 7 = 84
    
    rank = 12 * 7
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Get Strassen 2x2x2 factors
    U2, V2, W2 = _strassen_2x2()
    
    col = 0
    for i_b in range(2):  # block row of A, C
        for j_b in range(2):  # block col of A, block row of B
            for k_b in range(3):  # block col of B, C
                for r in range(7):
                    # U: rows indexed by i*4+j (global), where i = i_b*2 + i_local, j = j_b*2 + j_local
                    # U2 row = i_local*2 + j_local
                    for i_loc in range(2):
                        for j_loc in range(2):
                            global_i = i_b * 2 + i_loc
                            global_j = j_b * 2 + j_loc
                            global_U = global_i * m + global_j  # = global_i * 4 + global_j
                            U2_row = i_loc * 2 + j_loc
                            U[global_U, col] += U2[U2_row, r]
                    
                    # V: rows indexed by j*6+k (global), where j = j_b*2 + j_local, k = k_b*2 + k_local
                    # V2 row = j_local*2 + k_local
                    for j_loc in range(2):
                        for k_loc in range(2):
                            global_j = j_b * 2 + j_loc
                            global_k = k_b * 2 + k_loc
                            global_V = global_j * p + global_k  # = global_j * 6 + global_k
                            V2_row = j_loc * 2 + k_loc
                            V[global_V, col] += V2[V2_row, r]
                    
                    # W: rows indexed by k*4+i (global), where k = k_b*2 + k_local, i = i_b*2 + i_local
                    # W2 row = k_local*2 + i_local
                    for k_loc in range(2):
                        for i_loc in range(2):
                            global_k = k_b * 2 + k_loc
                            global_i = i_b * 2 + i_loc
                            global_W = global_k * n + global_i  # = global_k * 4 + global_i
                            W2_row = k_loc * 2 + i_loc
                            W[global_W, col] += W2[W2_row, r]
                    
                    col += 1
    
    # Verify correctness
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    
    # Test with random matrices
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    # Reconstruct using tensor decomposition
    # C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over j
    # More precisely: C[i,k] = sum_{j,r} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    C_recon = np.zeros((n, p), dtype=float)
    for i in range(n):
        for k in range(p):
            s = 0.0
            for j in range(m):
                for r in range(rank):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
            C_recon[i, k] = s
    
    err = np.max(np.abs(C_recon - C_expected))
    if err > 1e-9:
        # Fallback to schoolbook if Strassen implementation has a bug
        rank_sb = n * m * p
        U_sb = np.zeros((n * m, rank_sb), dtype=float)
        V_sb = np.zeros((m * p, rank_sb), dtype=float)
        W_sb = np.zeros((p * n, rank_sb), dtype=float)
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, c] = 1
                    V_sb[j * p + k, c] = 1
                    W_sb[k * n + i, c] = 1
                    c += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    # Convert to lists with exact integer/half-integer representation
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    # Round to nearest half-integer to ensure exact representation
    U_list = [[round(x * 2) / 2 for x in row] for row in U_list]
    V_list = [[round(x * 2) / 2 for x in row] for row in V_list]
    W_list = [[round(x * 2) / 2 for x in row] for row in W_list]
    
    return {"U": U_list, "V": V_list, "W": W_list}
