"""Find a rank-minimal exact tensor decomposition for 4x4 * 4x8 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START

def _get_strassen_2x2():
    """Return U, V, W for rank-7 2x2x2 matmul."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # s=1
    U[:, 0] = [1, 0, 0, 1]
    V[:, 0] = [1, 0, 0, 1]
    W[:, 0] = [1, 0, 0, 1] # W[0]=1 (C11), W[3]=1 (C22) -> indices 0, 3
    
    # s=2
    U[:, 1] = [0, 0, 1, 1]
    V[:, 1] = [1, 0, 0, 0]
    W[:, 1] = [0, 1, 0, -1] # W[1]=1 (C21), W[3]=-1 (C22) -> indices 1, 3
    
    # s=3
    U[:, 2] = [1, -1, 0, 0]
    V[:, 2] = [0, 0, 1, 1]
    W[:, 2] = [0, 0, 1, 1] # W[2]=1 (C12), W[3]=1 (C22) -> indices 2, 3
    
    # s=4
    U[:, 3] = [0, 0, 0, 1]
    V[:, 3] = [0, 1, -1, 0]
    W[:, 3] = [1, 1, 0, 0] # W[0]=1 (C11), W[1]=1 (C21) -> indices 0, 1
    
    # s=5
    U[:, 4] = [1, 0, 0, 1]
    V[:, 4] = [0, -1, 0, 1]
    W[:, 4] = [-1, 0, 1, 0] # W[0]=-1 (C11), W[2]=1 (C12) -> indices 0, 2
    
    # s=6
    U[:, 5] = [0, 1, 1, 0]
    V[:, 5] = [1, 0, -1, 0]
    W[:, 5] = [0, 0, 0, 1] # W[3]=1 (C22) -> index 3
    
    # s=7
    U[:, 6] = [1, 0, 1, 0]
    V[:, 6] = [0, 1, 0, -1]
    W[:, 6] = [1, 0, 0, 0] # W[0]=1 (C11) -> index 0
    
    return U, V, W

def _get_strassen_4x4():
    """Return U, V, W for rank-49 4x4x4 matmul using hierarchical Strassen."""
    n = 4
    U_2, V_2, W_2 = _get_strassen_2x2()
    
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    # Strassen 4x4 uses 7 multiplications of 2x2 matrices
    # The 2x2 matrices are formed by linear combinations of 2x2 blocks of the 4x4 matrices.
    # Let A be 4x4, split into A11, A12, A21, A22 (each 2x2).
    # Let B be 4x4, split into B11, B12, B21, B22 (each 2x2).
    
    # The 7 products M1..M7 are:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = (A11-A21)(B22+B21)
    # M4 = A22(B12-B22)
    # M5 = (A11+A22)(B22-B12)
    # M6 = (A12+A21)(B11-B21)
    # M7 = (A11+A21)(B12-B22)
    
    # Each M is a 2x2 matrix product. We decompose each using rank-7.
    
    # Define the linear combinations for U and V for each of the 7 Strassen steps.
    # U vectors are for A (4x4 -> 16 entries).
    # V vectors are for B (4x4 -> 16 entries).
    
    # Indices for 4x4 matrix split into 2x2 blocks:
    # A11: rows 0-1, cols 0-1 -> indices 0,1,4,5
    # A12: rows 0-1, cols 2-3 -> indices 2,3,6,7
    # A21: rows 2-3, cols 0-1 -> indices 8,9,12,13
    # A22: rows 2-3, cols 2-3 -> indices 10,11,14,15
    
    # Helper to add a scaled 2x2 block to a 16-dim vector
    def add_block(vec, block_idx, scale):
        if block_idx == 0: # 11
            vec[0:2, :] += scale * vec[0:2, :] # This is wrong logic for building vectors
            pass

    # Let's build the U and V matrices directly.
    # For each of the 7 Strassen steps s_big in 0..6:
    #   We have linear combinations of blocks for A and B.
    #   Let U_big[s_big] be the 16-dim vector for A.
    #   Let V_big[s_big] be the 16-dim vector for B.
    #   Then for each of the 7 inner Strassen steps s_small in 0..6:
    #     The total vector is U[:, s_big*7 + s_small] = U_big[s_big] * U_2[:, s_small] (element-wise? No, Kronecker product structure)
    #     Actually, the inner product is between the 2x2 blocks.
    #     So U[:, idx] is formed by taking U_2[:, s_small] and placing it in the corresponding 2x2 block positions defined by U_big[s_big].
    
    # U_big coefficients for A blocks (11, 12, 21, 22)
    U_big_coeffs = [
        [1, 0, 0, 1], # M1: A11+A22
        [0, 0, 1, 1], # M2: A21+A22
        [1, 0, -1, 0],# M3: A11-A21
        [0, 0, 0, 1], # M4: A22
        [1, 0, 0, 1], # M5: A11+A22
        [0, 1, 1, 0], # M6: A12+A21
        [1, 0, 1, 0], # M7: A11+A21
    ]
    
    V_big_coeffs = [
        [1, 0, 0, 1], # M1: B11+B22
        [1, 0, 0, 0], # M2: B11
        [0, 0, 1, 1], # M3: B21+B22 (Note: M3=(A11-A21)(B22+B21))
        [0, 1, 0, -1],# M4: B12-B22
        [0, -1, 0, 1],# M5: B22-B12
        [1, 0, -1, 0],# M6: B11-B21
        [0, 1, 0, -1],# M7: B12-B22
    ]
    
    # Block indices in 16-dim vector:
    # Block 0 (11): 0,1,4,5
    # Block 1 (12): 2,3,6,7
    # Block 2 (21): 8,9,12,13
    # Block 3 (22): 10,11,14,15
    
    block_indices = [
        [0, 1, 4, 5],
        [2, 3, 6, 7],
        [8, 9, 12, 13],
        [10, 11, 14, 15]
    ]
    
    for s_big in range(7):
        for s_small in range(7):
            idx = s_big * 7 + s_small
            
            # Construct U[:, idx]
            # U_2[:, s_small] has 4 entries, corresponding to the 2x2 block.
            u_small = U_2[:, s_small]
            for b_idx in range(4):
                scale = U_big_coeffs[s_big][b_idx]
                if scale != 0:
                    U[block_indices[b_idx], idx] = scale * u_small
            
            # Construct V[:, idx]
            v_small = V_2[:, s_small]
            for b_idx in range(4):
                scale = V_big_coeffs[s_big][b_idx]
                if scale != 0:
                    V[block_indices[b_idx], idx] = scale * v_small
            
            # Construct W[:, idx]
            # W for 4x4 is 16-dim.
            # The result C is 4x4, split into C11, C12, C21, C22.
            # The inner Strassen gives coefficients for the 2x2 result block.
            # The outer Strassen combines the 7 M's to form C blocks.
            
            # Outer Strassen combination for C blocks:
            # C11 = M1 + M4 - M5 + M7
            # C12 = M3 + M5
            # C21 = M2 + M4
            # C22 = M1 - M2 + M3 + M6
            
            # So for a given s_big (which corresponds to M_{s_big+1}), we add the inner W to the corresponding C block.
            w_small = W_2[:, s_small]
            
            # Determine which C blocks this M contributes to and with what coefficient
            if s_big == 0: # M1
                C11_coeff = 1; C12_coeff = 0; C21_coeff = 0; C22_coeff = 1
            elif s_big == 1: # M2
                C11_coeff = 0; C12_coeff = 0; C21_coeff = 1; C22_coeff = -1
            elif s_big == 2: # M3
                C11_coeff = 0; C12_coeff = 1; C21_coeff = 0; C22_coeff = 1
            elif s_big == 3: # M4
                C11_coeff = 1; C12_coeff = 0; C21_coeff = 1; C22_coeff = 0
            elif s_big == 4: # M5
                C11_coeff = -1; C12_coeff = 1; C21_coeff = 0; C22_coeff = 0
            elif s_big == 5: # M6
                C11_coeff = 0; C12_coeff = 0; C21_coeff = 0; C22_coeff = 1
            elif s_big == 6: # M7
                C11_coeff = 1; C12_coeff = 0; C21_coeff = 0; C22_coeff = 0
            
            # Add to W
            if C11_coeff != 0:
                W[block_indices[0], idx] += C11_coeff * w_small
            if C12_coeff != 0:
                W[block_indices[1], idx] += C12_coeff * w_small
            if C21_coeff != 0:
                W[block_indices[2], idx] += C21_coeff * w_small
            if C22_coeff != 0:
                W[block_indices[3], idx] += C22_coeff * w_small

    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-8):
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    # Get rank-49 decomposition for 4x4x4
    U_444, V_444, W_444 = _get_strassen_4x4()
    
    # Construct rank-98 decomposition for 4x4x8
    rank = 98
    U = np.zeros((16, rank))
    V = np.zeros((32, rank))
    W = np.zeros((32, rank))
    
    # Block 1: k in [0,4)
    # U[:, 0:49] = U_444
    U[:, 0:49] = U_444
    # V: j in [0,4), k in [0,4). Index j*8+k.
    # V_444 index is j*4+k.
    for j in range(4):
        for k in range(4):
            V[j * 8 + k, 0:49] = V_444[j * 4 + k, :]
    # W: k in [0,4), i in [0,4). Index k*4+i.
    # W_444 index is k*4+i.
    W[0:16, 0:49] = W_444
    
    # Block 2: k in [4,8)
    # U[:, 49:98] = U_444
    U[:, 49:98] = U_444
    # V: j in [0,4), k in [4,8). Index j*8+k.
    # Let k = 4 + k2. k2 in [0,4).
    # V_444 index is j*4+k2.
    for j in range(4):
        for k2 in range(4):
            V[j * 8 + 4 + k2, 49:98] = V_444[j * 4 + k2, :]
    # W: k in [4,8), i in [0,4). Index k*4+i.
    # Let k = 4 + k2. Index (4+k2)*4+i = 16 + k2*4+i.
    # W_444 index is k2*4+i.
    W[16:32, 49:98] = W_444
    
    # Verify
    if _verify(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook
        rank_sb = n * m * p
        U_sb = np.zeros((16, rank_sb))
        V_sb = np.zeros((32, rank_sb))
        W_sb = np.zeros((32, rank_sb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, col] = 1
                    V_sb[j * p + k, col] = 1
                    W_sb[k * n + i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

# EVOLVE-BLOCK-END
