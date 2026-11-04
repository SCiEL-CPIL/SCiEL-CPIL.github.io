"""
Strategy: Use Strassen's algorithm for 2x2 blocks and rank-1 decomposition for 2x1x2 blocks.
Partition n=4 into [2, 2].
Partition m=5 into [2, 2, 1].
Partition p=6 into [2, 2, 2].
This creates disjoint sub-tensors:
- 12 blocks of size 2x2x2 (Rank 7 each)
- 6 blocks of size 2x1x2 (Rank 1 each)
Total Rank = 12*7 + 6*1 = 90.
"""

import numpy as np
import time

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np

def _strassen_2x2():
    """
    Returns U, V, W for 2x2x2 matrix multiplication using Strassen's algorithm.
    Shapes: U (4, 7), V (4, 7), W (4, 7).
    Indices:
    U: i*2 + j (i,j in 0..1)
    V: j*2 + k (j,k in 0..1)
    W: k*2 + i (k,i in 0..1)
    """
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    
    # Strassen's 7 products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We need to define U, V, W such that C = sum U*V*W.
    # The entries of U, V, W are coefficients for the linear combinations of A and B.
    
    # r=0: M1
    U[0, 0] = 1; U[3, 0] = 1 # A00, A11
    V[0, 0] = 1; V[3, 0] = 1 # B00, B11
    W[0, 0] = 1; W[3, 0] = 1 # C00, C11 (Wait, C00 gets +1, C11 gets +1)
    
    # r=1: M2
    U[2, 1] = 1; U[3, 1] = 1 # A10, A11
    V[0, 1] = 1 # B00
    W[2, 1] = 1; W[3, 1] = -1 # C10 gets +1, C11 gets -1
    
    # r=2: M3
    U[0, 2] = 1 # A00
    V[2, 2] = 1; V[3, 2] = -1 # B10, -B11
    W[1, 2] = 1; W[3, 2] = 1 # C01 gets +1, C11 gets +1
    
    # r=3: M4
    U[3, 3] = 1 # A11
    V[1, 3] = 1; V[0, 3] = -1 # B01, -B00
    W[0, 3] = 1; W[2, 3] = 1 # C00 gets +1, C10 gets +1
    
    # r=4: M5
    U[0, 4] = 1; U[1, 4] = 1 # A00, A01
    V[3, 4] = 1 # B11
    W[0, 4] = -1; W[1, 4] = 1 # C00 gets -1, C01 gets +1
    
    # r=5: M6
    U[2, 5] = 1; U[0, 5] = -1 # A10, -A00
    V[0, 5] = 1; V[1, 5] = 1 # B00, B01
    W[3, 5] = 1 # C11 gets +1
    
    # r=6: M7
    U[1, 6] = 1; U[3, 6] = -1 # A01, -A11
    V[2, 6] = 1; V[3, 6] = 1 # B10, B11
    W[0, 6] = 1 # C00 gets +1
    
    return U, V, W

def _rank1_2x1x2():
    """
    Returns U, V, W for 2x1x2 matrix multiplication.
    Shapes: U (2, 1), V (1, 1), W (4, 1).
    Indices:
    U: i (i in 0..1)
    V: j=0
    W: k*2 + i
    """
    U = np.ones((2, 1))
    V = np.ones((1, 1))
    W = np.ones((4, 1))
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Partitions
    # n = 4 -> [2, 2]
    n_splits = [2, 2]
    # m = 5 -> [2, 2, 1]
    m_splits = [2, 2, 1]
    # p = 6 -> [2, 2, 2]
    p_splits = [2, 2, 2]
    
    # Calculate total rank
    total_rank = 0
    blocks_info = []
    
    n_offset = 0
    for n_size in n_splits:
        m_offset = 0
        for m_size in m_splits:
            p_offset = 0
            for p_size in p_splits:
                if n_size == 2 and m_size == 2 and p_size == 2:
                    rank = 7
                    type_id = 's' # strassen
                elif n_size == 2 and m_size == 1 and p_size == 2:
                    rank = 1
                    type_id = 'r' # rank1
                else:
                    # Fallback to schoolbook if unexpected size
                    rank = n_size * m_size * p_size
                    type_id = 's' # dummy
                total_rank += rank
                blocks_info.append((n_offset, m_offset, p_offset, n_size, m_size, p_size, rank, type_id))
                p_offset += p_size
            m_offset += m_size
        n_offset += n_size
        
    # Allocate matrices
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col_offset = 0
    for n_off, m_off, p_off, n_sz, m_sz, p_sz, rank, type_id in blocks_info:
        if time.time() > deadline:
            break
            
        if type_id == 's' and n_sz == 2 and m_sz == 2 and p_sz == 2:
            u_loc, v_loc, w_loc = _strassen_2x2()
        elif type_id == 'r' and n_sz == 2 and m_sz == 1 and p_sz == 2:
            u_loc, v_loc, w_loc = _rank1_2x1x2()
        else:
            # Schoolbook fallback
            u_loc = np.zeros((n_sz * m_sz, rank))
            v_loc = np.zeros((m_sz * p_sz, rank))
            w_loc = np.zeros((p_sz * n_sz, rank))
            c = 0
            for i in range(n_sz):
                for j in range(m_sz):
                    for k in range(p_sz):
                        u_loc[i * m_sz + j, c] = 1.0
                        v_loc[j * p_sz + k, c] = 1.0
                        w_loc[k * n_sz + i, c] = 1.0
                        c += 1
        
        # Copy into global matrices with offsets
        # U index: (n_off + i) * m + (m_off + j)
        # V index: (m_off + j) * p + (p_off + k)
        # W index: (p_off + k) * n + (n_off + i)
        
        for r in range(rank):
            for i in range(n_sz):
                for j in range(m_sz):
                    u_val = u_loc[i * m_sz + j, r]
                    if u_val != 0:
                        U[(n_off + i) * m + (m_off + j), col_offset + r] = u_val
                        
            for j in range(m_sz):
                for k in range(p_sz):
                    v_val = v_loc[j * p_sz + k, r]
                    if v_val != 0:
                        V[(m_off + j) * p + (p_off + k), col_offset + r] = v_val
                        
            for k in range(p_sz):
                for i in range(n_sz):
                    w_val = w_loc[k * n_sz + i, r]
                    if w_val != 0:
                        W[(p_off + k) * n + (n_off + i), col_offset + r] = w_val
                        
        col_offset += rank
        
    def to_list(mat):
        return mat.tolist()
        
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
# EVOLVE-BLOCK-END
