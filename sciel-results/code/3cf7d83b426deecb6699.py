import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS  # 6, 6, 6
    bs = 3  # block size
    
    # 2x2 block decomposition: 8 schoolbook 3x3 products, rank 9 each = 72 total
    # Block layout:
    #   A = [[A00, A01],    B = [[B00, B01],    C = [[C00, C01],
    #        [A10, A11]]         [B10, B11]]         [C10, C11]]
    #
    # Products needed:
    #   C00 = A00*B00 + A01*B10
    #   C01 = A00*B01 + A01*B11
    #   C10 = A10*B00 + A11*B10
    #   C11 = A10*B01 + A11*B11
    #
    # 8 products total, each 3x3 schoolbook rank 9 => total rank 72
    
    rank3 = 9  # schoolbook rank for 3x3
    total_rank = 8 * rank3  # 72
    
    U6 = np.zeros((36, total_rank))
    V6 = np.zeros((36, total_rank))
    W6 = np.zeros((36, total_rank))
    
    # Block offsets: block index -> (row_offset, col_offset)
    # Block 0: TL (rows 0-2, cols 0-2)
    # Block 1: TR (rows 0-2, cols 3-5)
    # Block 2: BL (rows 3-5, cols 0-2)
    # Block 3: BR (rows 3-5, cols 3-5)
    
    def block_offset(bi):
        return (bi // 2) * bs, (bi % 2) * bs
    
    # The 8 products: (A_block, B_block, C_block)
    products = [
        (0, 0, 0),  # A00 * B00 -> C00
        (1, 2, 0),  # A01 * B10 -> C00
        (0, 1, 1),  # A00 * B01 -> C01
        (1, 3, 1),  # A01 * B11 -> C01
        (2, 0, 2),  # A10 * B00 -> C10
        (3, 2, 2),  # A11 * B10 -> C10
        (2, 1, 3),  # A10 * B01 -> C11
        (3, 3, 3),  # A11 * B11 -> C11
    ]
    
    for prod_idx, (a_blk, b_blk, c_blk) in enumerate(products):
        a_row0, a_col0 = block_offset(a_blk)
        b_row0, b_col0 = block_offset(b_blk)
        c_row0, c_col0 = block_offset(c_blk)
        
        col_start = prod_idx * rank3
        
        # Schoolbook 3x3: for each (i, k, j) in 0..2
        # U index: (a_row0+i)*6 + (a_col0+k)
        # V index: (b_row0+k)*6 + (b_col0+j)
        # W index: (c_col0+j)*6 + (c_row0+i)  [W[k*n+i] convention: col*6 + row]
        
        for i in range(bs):
            for k in range(bs):
                for j in range(bs):
                    col = col_start + i * bs + k * bs + j
                    u_idx = (a_row0 + i) * 6 + (a_col0 + k)
                    v_idx = (b_row0 + k) * 6 + (b_col0 + j)
                    w_idx = (c_col0 + j) * 6 + (c_row0 + i)
                    
                    U6[u_idx, col] = 1.0
                    V6[v_idx, col] = 1.0
                    W6[w_idx, col] = 1.0
    
    # Verify the decomposition
    if time.time() < deadline:
        try:
            # Check: sum_r U[i*6+j, r] * V[j*6+k, r] * W[k*6+i, r] should be 1 for all i,j,k
            # This is equivalent to checking that the tensor reconstruction gives ones
            U3g = U6.reshape(6, 6, -1)
            V3g = V6.reshape(6, 6, -1)
            W3g = W6.reshape(6, 6, -1)
            
            # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
            # Should equal 1 for all i,j,k (since C[i,k] = sum_j A[i,j]*B[j,k])
            reconstructed = np.einsum('ijr,jkr,kir->ijk', U3g, V3g, W3g)
            target = np.ones((6, 6, 6))
            
            if np.allclose(reconstructed, target, atol=1e-9):
                return {"U": U6.tolist(), "V": V6.tolist(), "W": W6.tolist()}
            else:
                # Debug: check what went wrong
                max_err = np.max(np.abs(reconstructed - target))
                if max_err > 1e-9:
                    # Fall through to fallback
                    pass
        except Exception:
            pass
    
    # Fallback: full schoolbook rank 216
    rank = 216
    U_f = np.zeros((36, rank))
    V_f = np.zeros((36, rank))
    W_f = np.zeros((36, rank))
    for i in range(6):
        for j in range(6):
            for k in range(6):
                idx = i * 6 + j
                jk = j * 6 + k
                ki = k * 6 + i
                col = i * 36 + j * 6 + k
                U_f[idx, col] = 1.0
                V_f[jk, col] = 1.0
                W_f[ki, col] = 1.0
    return {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}

# EVOLVE-BLOCK-END
