"""Improved program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _build_strassen_2x2():
    """
    Build Strassen decomposition for 2x2 x 2x2 matrix multiplication.
    Returns U (4,7), V (4,7), W (4,7) where:
    U indexed by (i,k), V by (k,j), W by (j,i)
    T[i,k,j] = sum_r U[i*2+k, r] * V[k*2+j, r] * W[j*2+i, r]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A11+A22)(B11+B22)
    # A part: A11=(i=0,k=0), A22=(i=1,k=1)
    # B part: B11=(k=0,j=0), B22=(k=1,j=1)
    # W part: C11 gets +M1, C22 gets +M1
    # C11: i=0,j=0 -> W[0*2+0, 0] = W[0,0]
    # C22: i=1,j=1 -> W[1*2+1, 0] = W[3,0]
    U[0, 0] = 1  # A11: i=0,k=0
    U[3, 0] = 1  # A22: i=1,k=1
    V[0, 0] = 1  # B11: k=0,j=0
    V[3, 0] = 1  # B22: k=1,j=1
    W[0, 0] = 1  # C11: j=0,i=0
    W[3, 0] = 1  # C22: j=1,i=1
    
    # M2 = (A21+A22)B11
    # A part: A21=(i=1,k=0), A22=(i=1,k=1)
    # B part: B11=(k=0,j=0)
    # C part: C21 gets +M2, C22 gets -M2
    # C21: i=1,j=0 -> W[0*2+1, 1] = W[1,1]
    # C22: i=1,j=1 -> W[1*2+1, 1] = W[3,1]
    U[2, 1] = 1  # A21: i=1,k=0
    U[3, 1] = 1  # A22: i=1,k=1
    V[0, 1] = 1  # B11: k=0,j=0
    W[1, 1] = 1  # C21: j=0,i=1
    W[3, 1] = -1 # C22: j=1,i=1
    
    # M3 = A11(B12-B22)
    # A part: A11=(i=0,k=0)
    # B part: B12=(k=0,j=1), B22=(k=1,j=1)
    # C part: C12 gets +M3, C22 gets +M3
    # C12: i=0,j=1 -> W[1*2+0, 2] = W[2,2]
    # C22: i=1,j=1 -> W[1*2+1, 2] = W[3,2]
    U[0, 2] = 1  # A11: i=0,k=0
    V[1, 2] = 1  # B12: k=0,j=1
    V[3, 2] = -1 # B22: k=1,j=1
    W[2, 2] = 1  # C12: j=1,i=0
    W[3, 2] = 1  # C22: j=1,i=1
    
    # M4 = A22(B21-B11)
    # A part: A22=(i=1,k=1)
    # B part: B21=(k=1,j=0), B11=(k=0,j=0)
    # C part: C11 gets +M4, C21 gets +M4
    # C11: i=0,j=0 -> W[0*2+0, 3] = W[0,3]
    # C21: i=1,j=0 -> W[0*2+1, 3] = W[1,3]
    U[3, 3] = 1  # A22: i=1,k=1
    V[2, 3] = 1  # B21: k=1,j=0
    V[0, 3] = -1 # B11: k=0,j=0
    W[0, 3] = 1  # C11: j=0,i=0
    W[1, 3] = 1  # C21: j=0,i=1
    
    # M5 = (A11+A12)B22
    # A part: A11=(i=0,k=0), A12=(i=0,k=1)
    # B part: B22=(k=1,j=1)
    # C part: C11 gets -M5, C12 gets +M5
    # C11: i=0,j=0 -> W[0*2+0, 4] = W[0,4]
    # C12: i=0,j=1 -> W[1*2+0, 4] = W[2,4]
    U[0, 4] = 1  # A11: i=0,k=0
    U[1, 4] = 1  # A12: i=0,k=1
    V[3, 4] = 1  # B22: k=1,j=1
    W[0, 4] = -1 # C11: j=0,i=0
    W[2, 4] = 1  # C12: j=1,i=0
    
    # M6 = (A21-A11)(B11+B12)
    # A part: A21=(i=1,k=0), A11=(i=0,k=0)
    # B part: B11=(k=0,j=0), B12=(k=0,j=1)
    # C part: C22 gets +M6
    # C22: i=1,j=1 -> W[1*2+1, 5] = W[3,5]
    U[2, 5] = 1  # A21: i=1,k=0
    U[0, 5] = -1 # A11: i=0,k=0
    V[0, 5] = 1  # B11: k=0,j=0
    V[1, 5] = 1  # B12: k=0,j=1
    W[3, 5] = 1  # C22: j=1,i=1
    
    # M7 = (A12-A22)(B21+B22)
    # A part: A12=(i=0,k=1), A22=(i=1,k=1)
    # B part: B21=(k=1,j=0), B22=(k=1,j=1)
    # C part: C11 gets +M7
    # C11: i=0,j=0 -> W[0*2+0, 6] = W[0,6]
    U[1, 6] = 1  # A12: i=0,k=1
    U[3, 6] = -1 # A22: i=1,k=1
    V[2, 6] = 1  # B21: k=1,j=0
    V[3, 6] = 1  # B22: k=1,j=1
    W[0, 6] = 1  # C11: j=0,i=0
    
    return U, V, W

def _build_4x4x4_strassen():
    """
    Build Strassen-based rank-49 decomposition for 4x4 x 4x4.
    Uses Strassen at the 4x4 level (splitting into 2x2 blocks),
    with each 2x2 x 2x2 multiplication done via Strassen (rank 7).
    """
    U2, V2, W2 = _build_strassen_2x2()
    
    # 4x4 A split into 2x2 blocks:
    # A11: rows 0-1, cols 0-1
    # A12: rows 0-1, cols 2-3
    # A21: rows 2-3, cols 0-1
    # A22: rows 2-3, cols 2-3
    
    # The 7 Strassen combinations at the 4x4 level:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Each of these is a 2x2 x 2x2 multiplication with rank 7
    # So total rank = 7 * 7 = 49
    
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    # Define the 7 big Strassen multiplications
    # Each entry: (A_blocks, B_blocks, C_blocks)
    # A_blocks: list of (block_row, block_col, sign)
    # B_blocks: list of (block_row, block_col, sign)
    # C_blocks: list of (block_row, block_col, sign)
    
    big_mults = [
        # M1 = (A11+A22)(B11+B22)
        {
            'A': [((0,0), 1), ((1,1), 1)],
            'B': [((0,0), 1), ((1,1), 1)],
            'C': [((0,0), 1), ((1,1), 1)]
        },
        # M2 = (A21+A22)B11
        {
            'A': [((1,0), 1), ((1,1), 1)],
            'B': [((0,0), 1)],
            'C': [((1,0), 1), ((1,1), -1)]
        },
        # M3 = A11(B12-B22)
        {
            'A': [((0,0), 1)],
            'B': [((0,1), 1), ((1,1), -1)],
            'C': [((0,1), 1), ((1,1), 1)]
        },
        # M4 = A22(B21-B11)
        {
            'A': [((1,1), 1)],
            'B': [((1,0), 1), ((0,0), -1)],
            'C': [((0,0), 1), ((1,0), 1)]
        },
        # M5 = (A11+A12)B22
        {
            'A': [((0,0), 1), ((0,1), 1)],
            'B': [((1,1), 1)],
            'C': [((0,0), -1), ((0,1), 1)]
        },
        # M6 = (A21-A11)(B11+B12)
        {
            'A': [((1,0), 1), ((0,0), -1)],
            'B': [((0,0), 1), ((0,1), 1)],
            'C': [((1,1), 1)]
        },
        # M7 = (A12-A22)(B21+B22)
        {
            'A': [((0,1), 1), ((1,1), -1)],
            'B': [((1,0), 1), ((1,1), 1)],
            'C': [((0,0), 1)]
        }
    ]
    
    for mult_idx, mult in enumerate(big_mults):
        for r2 in range(7):
            r = mult_idx * 7 + r2
            
            # For each A block involved
            for (br, bc, sign_a) in mult['A']:
                for i2 in range(2):
                    for k2 in range(2):
                        i = 2 * br + i2
                        k = 2 * bc + k2
                        u_idx = i * 4 + k
                        u2_idx = i2 * 2 + k2
                        U[u_idx, r] += sign_a * U2[u2_idx, r2]
                        
            # For each B block involved
            for (br, bc, sign_b) in mult['B']:
                for k2 in range(2):
                    for j2 in range(2):
                        k = 2 * br + k2
                        j = 2 * bc + j2
                        v_idx = k * 4 + j
                        v2_idx = k2 * 2 + j2
                        V[v_idx, r] += sign_b * V2[v2_idx, r2]
                        
            # For each C block involved
            for (br, bc, sign_c) in mult['C']:
                for j2 in range(2):
                    for i2 in range(2):
                        j = 2 * br + j2
                        i = 2 * bc + i2
                        w_idx = j * 4 + i
                        w2_idx = j2 * 2 + i2
                        W[w_idx, r] += sign_c * W2[w2_idx, r2]
                        
    return U, V, W

def construct():
    """
    Find a low-rank exact tensor decomposition for 4x4 * 4x9 matrix multiplication.
    Uses Strassen for the 4x4x4 part (rank 49) 
    and schoolbook for the 4x4x5 part (rank 80).
    Total rank: 129.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    # Part 1: 4x4 * 4x4 (columns 0-3 of B)
    # Part 2: 4x4 * 4x5 (columns 4-8 of B)
    
    U1, V1, W1 = _build_4x4x4_strassen()
    
    rank1 = 49
    rank2 = 4 * 4 * 5 # 80
    total_rank = rank1 + rank2
    
    U = [[0.0] * total_rank for _ in range(16)]
    V = [[0.0] * total_rank for _ in range(36)]
    W = [[0.0] * total_rank for _ in range(36)]
    
    # Fill Part 1: Strassen 4x4x4
    # U1 is (16, 49), V1 is (16, 49), W1 is (16, 49)
    # Note: For the sub-problem 4x4 * 4x4, the dimensions are:
    # A: 4x4 (indices 0-15)
    # B_sub: 4x4 (indices 0-15 in the full B, which is 4x9, so rows 0-3, cols 0-3)
    # C_sub: 4x4 (indices 0-15 in the full C, which is 9x4, so rows 0-3, cols 0-3)
    
    for i in range(16):
        for r in range(49):
            U[i][r] = U1[i, r]
            
    # V for Part 1:
    # The B matrix is 4x9. The sub-part is columns 0-3.
    # V1 is indexed by (k, j) where k in 0..3, j in 0..3.
    # In the full V (size 36), the index is j * 9 + k? No, V is (m*p, rank).
    # m=4, p=9. Index is j * p + k? No, standard is usually row*cols + col.
    # Let's check the convention: V[j*p + k].
    # Wait, the task says: "output factor follows the official trace convention W[k*n+i,r]".
    # And "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)".
    # So U is (n*m, rank) -> index i*m + j.
    # V is (m*p, rank) -> index j*p + k.
    # W is (p*n, rank) -> index k*n + i.
    
    # So for Part 1 (cols 0-3 of B):
    # V1 has indices (k, j) for the 4x4 block.
    # In full V, index is j * 9 + k.
    for j in range(4):
        for k in range(4):
            v_idx_full = j * 9 + k
            v_idx_sub = k * 4 + j # Wait, V1 is (16, 49) from _build_4x4x4_strassen.
            # In _build_4x4x4_strassen, V is indexed by v_idx = k * 4 + j?
            # Let's check _build_strassen_2x2: V[k*2+j, r].
            # In _build_4x4x4_strassen: V[v_idx, r] where v_idx = k * 4 + j.
            # So V1 index is k * 4 + j.
            # Full V index is j * 9 + k.
            for r in range(49):
                V[v_idx_full][r] = V1[k * 4 + j, r]
                
    # W for Part 1:
    # W1 has indices (j, i) for the 4x4 block?
    # In _build_strassen_2x2: W[j*2+i, r].
    # In _build_4x4x4_strassen: W[w_idx, r] where w_idx = j * 4 + i.
    # Full W is (p*n, rank) -> index k * n + i.
    # For Part 1, k is 0..3, i is 0..3.
    # So W1 index is j * 4 + i. But wait, in the 4x4x4 subproblem, the output is C_sub (4x4).
    # The indices of C_sub are (k, i) where k is row (0..3), i is col (0..3).
    # In _build_4x4x4_strassen, the "W" matrix corresponds to the output tensor.
    # The output tensor for 4x4x4 is T[k, j, i] = sum U[i,k] V[k,j] W[j,i]?
    # No, standard matrix mult: C[i, j] = sum_k A[i, k] B[k, j].
    # The task convention: W[k*n+i, r]. This implies the third index of the tensor is (k, i).
    # So the tensor is T[i, j, k] = sum U[i*m+j] V[j*p+k] W[k*n+i].
    # This corresponds to C[k, i] = sum_j A[i, j] B[j, k].
    # Wait, A is n x m. B is m x p. C is p x n.
    # C[k, i] = sum_{j=0}^{m-1} A[i, j] B[j, k].
    # This is (A * B^T)[i, k]? No.
    # A is (n, m). B is (m, p).
    # Usually C = A * B is (n, p). C[i, k] = sum_j A[i, j] B[j, k].
    # Here C is (p, n). So C[k, i] = sum_j A[i, j] B[j, k].
    # This matches the tensor definition T[i, j, k] = A[i, j] B[j, k].
    # And the decomposition is T[i, j, k] = sum_r U[i, r] V[j, r] W[k, r].
    # With U indexed by i*m+j, V by j*p+k, W by k*n+i.
    
    # So for Part 1:
    # A is 4x4. i in 0..3, j in 0..3.
    # B is 4x4 (sub-part). j in 0..3, k in 0..3.
    # C is 4x4 (sub-part). k in 0..3, i in 0..3.
    
    # _build_4x4x4_strassen returns U, V, W for this subproblem.
    # U is (16, 49). Index i*4+j.
    # V is (16, 49). Index j*4+k? Or k*4+j?
    # In _build_strassen_2x2, V is (4, 7). Index k*2+j.
    # In _build_4x4x4_strassen, V is (16, 49).
    # Let's trace:
    # v_idx = k * 4 + j.
    # So V1[k*4+j, r].
    # Full V index is j * 9 + k.
    # So we need to map V1[k*4+j, r] to V[j*9+k, r].
    
    # W is (16, 49).
    # In _build_strassen_2x2, W is (4, 7). Index j*2+i.
    # In _build_4x4x4_strassen, W is (16, 49).
    # w_idx = j * 4 + i.
    # So W1[j*4+i, r].
    # Full W index is k * 4 + i.
    # Wait, in the subproblem, the output index is (k, i).
    # But in _build_strassen_2x2, the W index is (j, i) where j is the column of B and i is the row of A.
    # In the subproblem 4x4x4, B is 4x4. Columns are k (0..3).
    # So the "j" in the 2x2 Strassen context maps to "k" in the 4x4 subproblem context?
    # Let's re-verify the indices in _build_strassen_2x2.
    # T[i, k, j] = sum U[i*2+k] V[k*2+j] W[j*2+i].
    # Here A is (i, k), B is (k, j), C is (j, i).
    # So C[j, i] = sum_k A[i, k] B[k, j].
    # This matches C[k, i] = sum_j A[i, j] B[j, k] if we swap j and k in the names.
    # In our problem: C[k, i] = sum_j A[i, j] B[j, k].
    # So in the subproblem:
    # A is (i, j).
    # B is (j, k).
    # C is (k, i).
    # So the "k" in the Strassen code corresponds to "j" in the problem.
    # And the "j" in the Strassen code corresponds to "k" in the problem.
    
    # So:
    # U1 index: i*4 + j (problem indices).
    # V1 index: j*4 + k?
    # In _build_strassen_2x2, V index is k*2 + j.
    # Here "k" is the summation index (j in problem). "j" is the column of B (k in problem).
    # So V1 index is j*4 + k.
    # Full V index is j*9 + k.
    # This matches my previous mapping!
    
    # W1 index: j*4 + i?
    # In _build_strassen_2x2, W index is j*2 + i.
    # Here "j" is the column of B (k in problem). "i" is the row of A (i in problem).
    # So W1 index is k*4 + i.
    # Full W index is k*4 + i.
    # This matches directly!
    
    for k in range(4):
        for i in range(4):
            w_idx_full = k * 4 + i
            w_idx_sub = k * 4 + i
            for r in range(49):
                W[w_idx_full][r] = W1[w_idx_sub, r]
                
    # Part 2: Schoolbook for 4x4 * 4x5
    # A is 4x4. i in 0..3, j in 0..3.
    # B is 4x5. j in 0..3, k in 4..8.
    # C is 5x4. k in 4..8, i in 0..3.
    
    col = rank1
    for i in range(4):
        for j in range(4):
            for k in range(4, 9):
                U[i * 4 + j][col] = 1.0
                V[j * 9 + k][col] = 1.0
                W[k * 4 + i][col] = 1.0
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
