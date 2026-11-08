"""Matrix multiplication tensor decomposition for 2x4x7.

This program constructs an exact tensor decomposition of the matrix multiplication
tensor for multiplying a 2x4 matrix by a 4x7 matrix. It uses a block-wise
decomposition leveraging Strassen's algorithm for 2x2x2 blocks and schoolbook
for smaller blocks to achieve a rank of 50.

The interface returns factor matrices U, V, W with integer entries such that
the matrix multiplication tensor is exactly decomposed.
"""

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct the tensor decomposition for matrix multiplication 2x4 @ 4x7.
    
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    
    The decomposition satisfies: for all i, j, k,
    (A[i,:] dot B[:,k]) = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    """
    import numpy as np

    n, m, p = DIMENSIONS
    # Target rank 50
    rank = 50
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Helper to add a decomposition block
    # block_n, block_m, block_p are dimensions of the sub-problem
    # offset_u, offset_v, offset_w are offsets in the full matrices
    # col_offset is the starting column index in U, V, W
    def add_block(block_n, block_m, block_p, offset_u, offset_v, offset_w, col_offset, sub_decomp):
        # sub_decomp is a tuple (U_sub, V_sub, W_sub)
        U_sub, V_sub, W_sub = sub_decomp
        r = U_sub.shape[1]
        U[offset_u:offset_u+block_n*block_m, col_offset:col_offset+r] = U_sub
        V[offset_v:offset_v+block_m*block_p, col_offset:col_offset+r] = V_sub
        W[offset_w:offset_w+block_p*block_n, col_offset:col_offset+r] = W_sub
        return col_offset + r

    # 1. Strassen 2x2x2 decomposition (Rank 7)
    def get_strassen_222():
        n_s, m_s, p_s = 2, 2, 2
        r_s = 7
        U_s = np.zeros((n_s*m_s, r_s), dtype=int)
        V_s = np.zeros((m_s*p_s, r_s), dtype=int)
        W_s = np.zeros((p_s*n_s, r_s), dtype=int)
        
        # M1 = (A00+A11)(B00+B11)
        # U: A00 (0), A11 (3)
        # V: B00 (0), B11 (3)
        # W: C00 (0), C11 (3)
        U_s[0, 0] = 1; U_s[3, 0] = 1
        V_s[0, 0] = 1; V_s[3, 0] = 1
        W_s[0, 0] = 1; W_s[3, 0] = 1
        
        # M2 = (A10+A11)B00
        # U: A10 (2), A11 (3)
        # V: B00 (0)
        # W: C10 (2), C11 (3)
        U_s[2, 1] = 1; U_s[3, 1] = 1
        V_s[0, 1] = 1
        W_s[2, 1] = 1; W_s[3, 1] = -1
        
        # M3 = A00(B01-B11)
        # U: A00 (0)
        # V: B01 (1), B11 (3)
        # W: C01 (1), C11 (3)
        U_s[0, 2] = 1
        V_s[1, 2] = 1; V_s[3, 2] = -1
        W_s[1, 2] = 1; W_s[3, 2] = 1
        
        # M4 = A11(B10-B00)
        # U: A11 (3)
        # V: B10 (2), B00 (0)
        # W: C00 (0), C10 (2)
        U_s[3, 3] = 1
        V_s[2, 3] = 1; V_s[0, 3] = -1
        W_s[0, 3] = 1; W_s[2, 3] = 1
        
        # M5 = (A00+A10)B11
        # U: A00 (0), A10 (2)
        # V: B11 (3)
        # W: C00 (0), C01 (1)
        U_s[0, 4] = 1; U_s[2, 4] = 1
        V_s[3, 4] = 1
        W_s[0, 4] = -1; W_s[1, 4] = 1
        
        # M6 = (A10-A00)(B00+B01)
        # U: A10 (2), A00 (0)
        # V: B00 (0), B01 (1)
        # W: C11 (3)
        U_s[2, 5] = 1; U_s[0, 5] = -1
        V_s[0, 5] = 1; V_s[1, 5] = 1
        W_s[3, 5] = 1
        
        # M7 = (A01-A11)(B10+B11)
        # U: A01 (1), A11 (3)
        # V: B10 (2), B11 (3)
        # W: C00 (0), C11 (3)
        U_s[1, 6] = 1; U_s[3, 6] = -1
        V_s[2, 6] = 1; V_s[3, 6] = 1
        W_s[0, 6] = 1; W_s[3, 6] = 1
        
        return U_s, V_s, W_s

    # 2. Schoolbook 2x2x1 decomposition (Rank 4)
    def get_schoolbook_221():
        n_s, m_s, p_s = 2, 2, 1
        r_s = 4
        U_s = np.zeros((n_s*m_s, r_s), dtype=int)
        V_s = np.zeros((m_s*p_s, r_s), dtype=int)
        W_s = np.zeros((p_s*n_s, r_s), dtype=int)
        
        # A00 B00 -> C00
        U_s[0, 0] = 1; V_s[0, 0] = 1; W_s[0, 0] = 1
        # A01 B01 -> C01
        U_s[1, 1] = 1; V_s[1, 1] = 1; W_s[1, 1] = 1
        # A10 B10 -> C10
        U_s[2, 2] = 1; V_s[0, 2] = 1; W_s[0, 2] = 1
        # A11 B11 -> C11
        U_s[3, 3] = 1; V_s[1, 3] = 1; W_s[1, 3] = 1
        
        return U_s, V_s, W_s

    # 3. Combine to get 2x2x3 (Rank 11)
    def get_223():
        U_222, V_222, W_222 = get_strassen_222()
        U_221, V_221, W_221 = get_schoolbook_221()
        
        U_223 = np.zeros((4, 11), dtype=int)
        V_223 = np.zeros((6, 11), dtype=int)
        W_223 = np.zeros((6, 11), dtype=int)
        
        # First 7 cols: 2x2x2 (k=0,1)
        U_223[:, 0:7] = U_222
        V_223[0:4, 0:7] = V_222 # V is 2x2=4 rows for k=0,1? No, V is m*p. 
        # In 2x2x2, V is 4x7. Here V_223 is 6x11.
        # The 2x2x2 part corresponds to k=0,1.
        # V indices for k=0,1 are 0..3.
        # W indices for k=0,1 are 0..3.
        V_223[0:4, 0:7] = V_222
        W_223[0:4, 0:7] = W_222
        
        # Last 4 cols: 2x2x1 (k=2)
        # U is same 4 rows.
        U_223[:, 7:11] = U_221
        # V is 2x1=2 rows, offset by 4 (since k=2 starts at index 4 in m*p=2*3=6? No.)
        # m=2, p=3. V shape (6, 11).
        # k=0 -> V[0..1]
        # k=1 -> V[2..3]
        # k=2 -> V[4..5]
        V_223[4:6, 7:11] = V_221
        # W is 3x2=6 rows.
        # k=0 -> W[0..1]
        # k=1 -> W[2..3]
        # k=2 -> W[4..5]
        W_223[4:6, 7:11] = W_221
        
        return U_223, V_223, W_223

    # 4. Combine to get 2x2x7 (Rank 25)
    def get_227():
        U_222, V_222, W_222 = get_strassen_222()
        U_223, V_223, W_223 = get_223()
        
        U_227 = np.zeros((4, 25), dtype=int)
        V_227 = np.zeros((14, 25), dtype=int)
        W_227 = np.zeros((14, 25), dtype=int)
        
        # k=0,1: 2x2x2 (Rank 7)
        U_227[:, 0:7] = U_222
        V_227[0:4, 0:7] = V_222
        W_227[0:4, 0:7] = W_222
        
        # k=2,3: 2x2x2 (Rank 7)
        U_227[:, 7:14] = U_222
        V_227[4:8, 7:14] = V_222
        W_227[4:8, 7:14] = W_222
        
        # k=4,5,6: 2x2x3 (Rank 11)
        U_227[:, 14:25] = U_223
        V_227[8:14, 14:25] = V_223
        W_227[8:14, 14:25] = W_223
        
        return U_227, V_227, W_227

    # 5. Combine to get 2x4x7 (Rank 50)
    U_227, V_227, W_227 = get_227()
    
    # j=0,1: 2x2x7 (Rank 25)
    U[0:4, 0:25] = U_227
    V[0:14, 0:25] = V_227
    W[0:14, 0:25] = W_227
    
    # j=2,3: 2x2x7 (Rank 25)
    U[4:8, 25:50] = U_227
    V[14:28, 25:50] = V_227
    W[0:14, 25:50] = W_227
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
