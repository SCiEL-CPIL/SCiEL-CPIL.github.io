"""Rewritten program with optimized tensor decomposition for 3x5x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Helper to create identity-like or specific small decompositions
    
    # 1. Rank-7 decomposition for 2x2x2 (Strassen)
    # U: 4x7 (A entries), V: 4x7 (B entries), W: 4x7 (C entries)
    # A indices: 0:(1,1), 1:(1,2), 2:(2,1), 3:(2,2)
    # B indices: 0:(1,1), 1:(1,2), 2:(2,1), 3:(2,2)
    # C indices: 0:(1,1), 1:(1,2), 2:(2,1), 3:(2,2)
    
    U_222 = np.zeros((4, 7))
    V_222 = np.zeros((4, 7))
    W_222 = np.zeros((4, 7))
    
    # M1
    U_222[0, 0] = 1; U_222[3, 0] = 1
    V_222[0, 0] = 1; V_222[3, 0] = 1
    W_222[0, 0] = 1; W_222[3, 0] = 1 # c11 += M1, c22 += M1
    
    # M2
    U_222[2, 1] = 1; U_222[3, 1] = 1
    V_222[0, 1] = 1
    W_222[2, 1] = 1; W_222[3, 1] = -1 # c21 += M2, c22 -= M2
    
    # M3
    U_222[0, 2] = 1
    V_222[1, 2] = 1; V_222[3, 2] = -1
    W_222[1, 2] = 1; W_222[3, 2] = 1 # c12 += M3, c22 += M3
    
    # M4
    U_222[3, 3] = 1
    V_222[2, 3] = 1; V_222[0, 3] = -1
    W_222[0, 3] = 1; W_222[2, 3] = 1 # c11 += M4, c21 += M4
    
    # M5
    U_222[0, 4] = 1; U_222[1, 4] = 1
    V_222[3, 4] = 1
    W_222[0, 4] = -1; W_222[1, 4] = 1 # c11 -= M5, c12 += M5
    
    # M6
    U_222[2, 5] = 1; U_222[0, 5] = -1
    V_222[0, 5] = 1; V_222[1, 5] = 1
    W_222[3, 5] = 1 # c22 += M6
    
    # M7
    U_222[1, 6] = 1; U_222[3, 6] = -1
    V_222[2, 6] = 1; V_222[3, 6] = 1
    W_222[0, 6] = 1 # c11 += M7

    # 2. Schoolbook for 2x1x2
    # U: 2x4, V: 1x4, W: 4x4? No, W is p*n = 4? No, C is 2x2, so 4 entries.
    # Wait, 2x1x2 means A(2x1), B(1x2), C(2x2).
    # Rank 4.
    U_212 = np.zeros((2, 4))
    V_212 = np.zeros((1, 4))
    W_212 = np.zeros((4, 4)) # C has 2*2=4 entries
    for k in range(2): # k is the single dimension of B (size 1), but B is 1x2, so col index?
        # B is 1x2. Let B = [b0, b1].
        # A is 2x1. Let A = [a0, a1]^T.
        # C = A B = [[a0 b0, a0 b1], [a1 b0, a1 b1]]
        # Term 1: a0 b0 -> c00, c10? No.
        # c00 = a0 b0. U[0]=1, V[0]=1, W[0]=1
        # c01 = a0 b1. U[0]=1, V[1]=1, W[1]=1
        # c10 = a1 b0. U[1]=1, V[0]=1, W[2]=1
        # c11 = a1 b1. U[1]=1, V[1]=1, W[3]=1
        
        # Let's index V by its 2 columns.
        pass
    
    U_212 = np.array([[1, 1, 0, 0],
                      [0, 0, 1, 1]])
    V_212 = np.array([[1, 0, 1, 0],
                      [0, 1, 0, 1]])
    # W maps to C entries (row-major): 0:c00, 1:c01, 2:c10, 3:c11
    W_212 = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])

    # 3. Schoolbook for 1x2x2
    # A(1x2), B(2x2), C(1x2). Rank 4.
    U_122 = np.zeros((2, 4))
    V_122 = np.zeros((4, 4))
    W_122 = np.zeros((2, 4))
    # A=[a0, a1], B=[[b00, b01],[b10, b11]], C=[c0, c1]
    # c0 = a0 b00 + a1 b10
    # c1 = a0 b01 + a1 b11
    U_122 = np.array([[1, 0, 1, 0],
                      [0, 1, 0, 1]])
    V_122 = np.array([[1, 0, 0, 0],
                      [0, 0, 1, 0],
                      [0, 1, 0, 0],
                      [0, 0, 0, 1]])
    W_122 = np.array([[1, 0, 0, 0],
                      [0, 0, 1, 0],
                      [0, 1, 0, 0],
                      [0, 0, 0, 1]]) # Wait, W shape is (p*n, rank). p=2, n=1? No.
    # For 1x2x2: n=1, m=2, p=2.
    # U shape (1*2, 4) = (2,4). Correct.
    # V shape (2*2, 4) = (4,4). Correct.
    # W shape (2*1, 4) = (2,4). Correct.
    # C is 1x2. Entries c0, c1.
    # Term 1: a0 b00 -> c0. U[0]=1, V[0]=1, W[0]=1
    # Term 2: a1 b10 -> c0. U[1]=1, V[2]=1, W[0]=1
    # Term 3: a0 b01 -> c1. U[0]=1, V[1]=1, W[1]=1
    # Term 4: a1 b11 -> c1. U[1]=1, V[3]=1, W[1]=1
    W_122 = np.array([[1, 1, 0, 0],
                      [0, 0, 1, 1]])

    # 4. Schoolbook for 1x1x2
    # A(1x1), B(1x2), C(1x2). Rank 2.
    U_112 = np.array([[1],
                      [1]]) # Wait, U is (1*1, 2) = (1,2).
    U_112 = np.array([[1, 1]])
    V_112 = np.array([[1, 0],
                      [0, 1]]) # V is (1*2, 2) = (2,2)
    W_112 = np.array([[1, 0],
                      [0, 1]]) # W is (2*1, 2) = (2,2)
    
    # Now assemble
    # n=3, m=5, p=6
    # Split n: [2, 1]
    # Split m: [2, 2, 1]
    # Split p: [2, 2, 2]
    
    # Global U size: (3*5, 84) = (15, 84)
    # Global V size: (5*6, 84) = (30, 84)
    # Global W size: (6*3, 84) = (18, 84)
    
    total_rank = 84
    U = np.zeros((15, total_rank))
    V = np.zeros((30, total_rank))
    W = np.zeros((18, total_rank))
    
    col_offset = 0
    
    # Block (i, j, k) where i is n-block, j is m-block, k is p-block
    # n_blocks: [0:2, 2:3] (sizes 2, 1)
    # m_blocks: [0:2, 2:4, 4:5] (sizes 2, 2, 1)
    # p_blocks: [0:2, 2:4, 4:6] (sizes 2, 2, 2)
    
    n_splits = [(0, 2), (2, 3)]
    m_splits = [(0, 2), (2, 4), (4, 5)]
    p_splits = [(0, 2), (2, 4), (4, 6)]
    
    for ni in range(2):
        n_start, n_end = n_splits[ni]
        n_size = n_end - n_start
        
        for mi in range(3):
            m_start, m_end = m_splits[mi]
            m_size = m_end - m_start
            
            for pi in range(3):
                p_start, p_end = p_splits[pi]
                p_size = p_end - p_start
                
                # Determine block type
                if n_size == 2 and m_size == 2 and p_size == 2:
                    block_rank = 7
                    u_block = U_222
                    v_block = V_222
                    w_block = W_222
                elif n_size == 2 and m_size == 1 and p_size == 2:
                    block_rank = 4
                    u_block = U_212
                    v_block = V_212
                    w_block = W_212
                elif n_size == 1 and m_size == 2 and p_size == 2:
                    block_rank = 4
                    u_block = U_122
                    v_block = V_122
                    w_block = W_122
                elif n_size == 1 and m_size == 1 and p_size == 2:
                    block_rank = 2
                    u_block = U_112
                    v_block = V_112
                    w_block = W_112
                else:
                    # Should not happen with these splits
                    continue

                # Map local indices to global indices
                # U local: (n_size * m_size, block_rank)
                # Global U row index: (n_global * m_global)
                # n_global = n_start + i, m_global = m_start + j
                # u_row_global = (n_start + i) * m + (m_start + j)
                
                # V local: (m_size * p_size, block_rank)
                # Global V row index: (m_global * p_global)
                # m_global = m_start + j, p_global = p_start + k
                # v_row_global = (m_start + j) * p + (p_start + k)
                
                # W local: (p_size * n_size, block_rank)
                # Global W row index: (p_global * n_global)
                # p_global = p_start + k, n_global = n_start + i
                # w_row_global = (p_start + k) * n + (n_start + i)
                
                for l in range(block_rank):
                    for i in range(n_size):
                        for j in range(m_size):
                            u_row = (n_start + i) * m + (m_start + j)
                            U[u_row, col_offset + l] = u_block[i * m_size + j, l]
                    
                    for j in range(m_size):
                        for k in range(p_size):
                            v_row = (m_start + j) * p + (p_start + k)
                            V[v_row, col_offset + l] = v_block[j * p_size + k, l]
                    
                    for k in range(p_size):
                        for i in range(n_size):
                            w_row = (p_start + k) * n + (n_start + i)
                            W[w_row, col_offset + l] = w_block[k * n_size + i, l]
                
                col_offset += block_rank
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
