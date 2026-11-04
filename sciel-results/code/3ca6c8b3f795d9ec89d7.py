"""
Program for matrix multiplication 3x4 x 4x8 with optimized block-structured decomposition.

Strategy:
1. Split the inner dimension m=4 into two blocks of size 2 (m1, m2).
   This decouples the multiplication into two independent 3x2 * 2x8 products.
2. For each 3x2 * 2x8 product, split the output dimension p=8 into four blocks of size 2.
   This results in 4 independent 3x2 * 2x2 sub-problems per m-block.
3. Each 3x2 * 2x2 sub-problem is decomposed using a rank-11 scheme:
   - The 3x2 * 2x2 tensor is equivalent to a 2x2 * 2x2 tensor (Strassen, rank 7) 
     plus a 1x2 * 2x2 tensor (schoolbook, rank 4).
   - Total rank per 3x2x2 block = 7 + 4 = 11.
4. Total Rank = 2 (m-blocks) * 4 (p-blocks) * 11 (rank per 3x2x2) = 88.

This is lower than the schoolbook rank of 96.
"""

import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def _build_strassen_2x2x2():
    """
    Returns U, V, W for 2x2 * 2x2 matrix multiplication using Strassen's algorithm.
    U: shape (4, 7) where row index is i*2+j for A[i,j]
    V: shape (4, 7) where row index is i*2+j for B[i,j]
    W: shape (4, 7) where row index is i*2+j for C[i,j]
    
    Indices:
    A: [[A00, A01], [A10, A11]] -> indices 0,1,2,3
    B: [[B00, B01], [B10, B11]] -> indices 0,1,2,3
    C: [[C00, C01], [C10, C11]] -> indices 0,1,2,3
    """
    rank = 7
    U = [[0.0] * rank for _ in range(4)]
    V = [[0.0] * rank for _ in range(4)]
    W = [[0.0] * rank for _ in range(4)]
    
    # Strassen terms:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 - M5 + M4 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # t=0: M1
    U[0][0] += 1; U[3][0] += 1  # A00, A11
    V[0][0] += 1; V[3][0] += 1  # B00, B11
    W[0][0] += 1; W[3][0] += 1  # C00, C11
    
    # t=1: M2
    U[2][1] += 1; U[3][1] += 1  # A10, A11
    V[0][1] += 1                # B00
    W[2][1] += 1                # C10
    
    # t=2: M3
    U[0][2] += 1                # A00
    V[2][2] += 1; V[3][2] += -1 # B10, -B11
    W[1][2] += 1; W[3][2] += 1  # C01, C11
    
    # t=3: M4
    U[3][3] += 1                # A11
    V[1][3] += 1; V[2][3] += -1 # B01, -B10
    W[0][3] += 1; W[2][3] += 1  # C00, C10
    
    # t=4: M5
    U[0][4] += 1; U[1][4] += 1  # A00, A01
    V[3][4] += 1                # B11
    W[0][4] += -1; W[1][4] += 1 # -C00, C01
    
    # t=5: M6
    U[2][5] += 1; U[0][5] += -1 # A10, -A00
    V[0][5] += 1; V[1][5] += 1  # B00, B01
    W[3][5] += 1                # C11
    
    # t=6: M7
    U[1][6] += 1; U[3][6] += -1 # A01, -A11
    V[2][6] += 1; V[3][6] += 1  # B10, B11
    W[0][6] += 1                # C00
    
    return U, V, W

def _build_row1_2x2x2():
    """
    Returns U, V, W for 1x2 * 2x2 matrix multiplication (schoolbook).
    A is 1x2, B is 2x2, C is 1x2.
    U: shape (2, 4) where row index is j for A[0,j]
    V: shape (4, 4) where row index is j*2+k for B[j,k]
    W: shape (2, 4) where row index is k for C[0,k]
    
    C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    Terms:
    t=0: A0 * B00 -> C0
    t=1: A0 * B01 -> C1
    t=2: A1 * B10 -> C0
    t=3: A1 * B11 -> C1
    """
    rank = 4
    U = [[0.0] * rank for _ in range(2)] # A indices: 0, 1
    V = [[0.0] * rank for _ in range(4)] # B indices: 0(0,0), 1(0,1), 2(1,0), 3(1,1)
    W = [[0.0] * rank for _ in range(2)] # C indices: 0, 1
    
    # t=0: A[0] * B[0,0] -> C[0]
    U[0][0] = 1.0
    V[0][0] = 1.0
    W[0][0] = 1.0
    
    # t=1: A[0] * B[0,1] -> C[1]
    U[0][1] = 1.0
    V[1][1] = 1.0
    W[1][1] = 1.0
    
    # t=2: A[1] * B[1,0] -> C[0]
    U[1][2] = 1.0
    V[2][2] = 1.0
    W[0][2] = 1.0
    
    # t=3: A[1] * B[1,1] -> C[1]
    U[1][3] = 1.0
    V[3][3] = 1.0
    W[1][3] = 1.0
    
    return U, V, W

def _build_3x2x2_block():
    """
    Builds the decomposition for 3x2 * 2x2.
    This is composed of:
    - A 2x2 * 2x2 block (rows 0,1 of A) using Strassen (rank 7)
    - A 1x2 * 2x2 block (row 2 of A) using schoolbook (rank 4)
    Total rank 11.
    
    U: shape (6, 11) for A (3x2)
    V: shape (4, 11) for B (2x2)
    W: shape (4, 11) for C (3x2 -> flattened? No, C is 3x2, so 6 elements? 
       Wait, the interface for the whole 3x4x8 problem has W shape (p*n, rank).
       Here we are building a local block. Let's return local U, V, W.
       Local A is 3x2 (6 elements).
       Local B is 2x2 (4 elements).
       Local C is 3x2 (6 elements).
       
       Wait, in the full problem:
       U has shape (n*m, rank) = (12, R)
       V has shape (m*p, rank) = (32, R)
       W has shape (p*n, rank) = (24, R)
       
       We are decomposing the tensor T[i, j, k] where i in 0..2, j in 0..3, k in 0..7.
       We split j into j1 (0..1) and j2 (2..3).
       We split k into k1 (0..1), k2 (2..3), k3 (4..5), k4 (6..7).
       
       For a fixed j_block (0 or 1) and k_block (0..3), we have a 3x2 * 2x2 subproblem.
       Let's define the local indices for this subproblem:
       A_local: 3x2 (indices 0..5)
       B_local: 2x2 (indices 0..3)
       C_local: 3x2 (indices 0..5)
       
       We will map these local indices to global indices.
    """
    # Get Strassen part (rank 7) for first 2 rows of A
    Us, Vs, Ws = _build_strassen_2x2x2()
    # Us: 4x7, Vs: 4x7, Ws: 4x7
    
    # Get Schoolbook part (rank 4) for 3rd row of A
    U1, V1, W1 = _build_row1_2x2x2()
    # U1: 2x4, V1: 4x4, W1: 2x4
    
    # Combine into 3x2 * 2x2 block
    # U_block: 6x11
    # Rows 0-3 correspond to A[0,0], A[0,1], A[1,0], A[1,1]
    # Rows 4-5 correspond to A[2,0], A[2,1]
    
    U_block = [[0.0] * 11 for _ in range(6)]
    V_block = [[0.0] * 11 for _ in range(4)]
    W_block = [[0.0] * 11 for _ in range(6)]
    
    # Fill Strassen part (columns 0-6)
    for r in range(4):
        for t in range(7):
            U_block[r][t] = Us[r][t]
            V_block[r][t] = Vs[r][t]
            W_block[r][t] = Ws[r][t]
            
    # Fill Schoolbook part (columns 7-10)
    # U1 rows 0,1 map to A[2,0], A[2,1] which are global rows 4,5 of U_block
    # V1 rows 0-3 map to B[0,0]..B[1,1] which are global rows 0-3 of V_block
    # W1 rows 0,1 map to C[2,0], C[2,1] which are global rows 4,5 of W_block
    
    for r in range(2):
        for t in range(4):
            U_block[4 + r][7 + t] = U1[r][t]
            V_block[r][7 + t] = V1[r][t] # Wait, V1 has 4 rows. U1 has 2. 
            # V1 is 4x4. Rows 0..3.
            # V_block is 4x11. Rows 0..3.
            # So V_block[0..3][7..10] = V1[0..3][0..3]
            pass
            
    # Correct V mapping
    for r in range(4):
        for t in range(4):
            V_block[r][7 + t] = V1[r][t]
            
    # Correct W mapping
    for r in range(2):
        for t in range(4):
            W_block[4 + r][7 + t] = W1[r][t]
            
    return U_block, V_block, W_block

def construct():
    """
    Construct rank-88 decomposition for 3x4 * 4x8.
    """
    n, m, p = 3, 4, 8
    
    # Total rank is 88
    total_rank = 88
    
    # Initialize matrices
    U = [[0.0] * total_rank for _ in range(n * m)] # 12 x 88
    V = [[0.0] * total_rank for _ in range(m * p)] # 32 x 88
    W = [[0.0] * total_rank for _ in range(p * n)] # 24 x 88
    
    # Build the 3x2x2 block once
    U_blk, V_blk, W_blk = _build_3x2x2_block()
    
    # Loop over m-blocks (j_block) and p-blocks (k_block)
    # j_block in {0, 1} corresponds to j in {0,1} and {2,3}
    # k_block in {0, 1, 2, 3} corresponds to k in {0,1}, {2,3}, {4,5}, {6,7}
    
    for j_block in range(2):
        for k_block in range(4):
            # Determine the column offset in the global matrices
            # There are 8 blocks total, each of rank 11.
            col_offset = (j_block * 4 + k_block) * 11
            
            # Map local indices to global indices
            # Local A (3x2) -> Global A (3x4)
            # Local row l (0..5) -> Global row i*m + j
            # l = i*2 + j_local
            # j_global = j_block*2 + j_local
            
            # Local B (2x2) -> Global B (4x8)
            # Local row b (0..3) -> Global row j_global*p + k_global
            # b = j_local*2 + k_local
            # j_global = j_block*2 + j_local
            # k_global = k_block*2 + k_local
            
            # Local C (3x2) -> Global C (8x3) -- Wait, W is (p*n, rank)
            # Global W index is k_global*n + i
            # Local W row w (0..5) -> i*2 + k_local
            # i = w // 2
            # k_local = w % 2
            # k_global = k_block*2 + k_local
            # Global W index = k_global*n + i
            
            for l in range(6):
                i = l // 2
                j_local = l % 2
                j_global = j_block * 2 + j_local
                global_u_idx = i * m + j_global
                for t in range(11):
                    U[global_u_idx][col_offset + t] = U_blk[l][t]
                    
            for b in range(4):
                j_local = b // 2
                k_local = b % 2
                j_global = j_block * 2 + j_local
                k_global = k_block * 2 + k_local
                global_v_idx = j_global * p + k_global
                for t in range(11):
                    V[global_v_idx][col_offset + t] = V_blk[b][t]
                    
            for w in range(6):
                i = w // 2
                k_local = w % 2
                k_global = k_block * 2 + k_local
                global_w_idx = k_global * n + i
                for t in range(11):
                    W[global_w_idx][col_offset + t] = W_blk[w][t]
                    
    # Verification (optional but good for safety)
    # For this specific structured solution, it should be exact.
    # We can do a quick check with numpy if time permits, but construction is O(1).
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    # Quick self-test
    res = construct()
    U, V, W = res['U'], res['V'], res['W']
    n, m, p = 3, 4, 8
    rank = len(U[0])
    
    # Verify with random matrices
    np.random.seed(0)
    A = np.random.randint(-10, 10, size=(n, m))
    B = np.random.randint(-10, 10, size=(m, p))
    C_true = A @ B
    
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    # u[i, t] = sum_j U[i*m+j, t] * A[i, j]
    U_reshaped = U_np.reshape(n, m, rank)
    u = np.einsum('imt,ij->it', U_reshaped, A)
    
    # v[k, t] = sum_j V[j*p+k, t] * B[j, k]
    V_reshaped = V_np.reshape(m, p, rank)
    v = np.einsum('jkt,jk->kt', V_reshaped, B)
    
    # C[i, k] = sum_t W[k*n+i, t] * u[i, t] * v[k, t]
    W_reshaped = W_np.reshape(p, n, rank)
    C_calc = np.einsum('kit,it,kt->ik', W_reshaped, u, v)
    
    if np.allclose(C_calc, C_true):
        print(f"Verification passed. Rank: {rank}")
    else:
        print("Verification failed!")
        print(C_calc)
        print(C_true)

