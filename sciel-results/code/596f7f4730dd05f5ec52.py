"""Tensor decomposition for 3x5x6 matrix multiplication."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _verify(U, V, W, n, m, p):
    """Verify exact tensor decomposition."""
    U = np.array(U, dtype=object)
    V = np.array(V, dtype=object)
    W = np.array(W, dtype=object)
    rank = U.shape[1]
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(rank))
                if total != 1:
                    return False
    return True

def _strassen_2x2x2():
    """
    Strassen decomposition for 2x2x2 matrix multiplication.
    Returns U, V, W of shape (4, 7) each with 0/1 entries.
    U[i*2+j, c] for i,j in {0,1}
    V[j*2+k, c] for j,k in {0,1}
    W[k*2+i, c] for k,i in {0,1}
    """
    U = [
        [1, 1, 0, 0, 1, 0, -1],  # A[0,0]
        [0, 0, 1, 0, 1, 0, -1],  # A[0,1]
        [0, 1, 0, 0, 0, 1, 0],   # A[1,0]
        [1, 1, 0, 1, 0, 0, 1],   # A[1,1]
    ]
    V = [
        [1, 0, -1, 1, 0, 1, 1],  # B[0,0] (j=0,k=0)
        [0, 0, 0, -1, 0, 1, 0],  # B[0,1] (j=0,k=1)
        [1, 1, 1, 0, 1, 0, 1],   # B[1,0] (j=1,k=0)
        [1, 0, -1, 0, 1, 0, 1],  # B[1,1] (j=1,k=1)
    ]
    W = [
        [1, 0, 0, 0, 0, 1, 0],   # k=0,i=0
        [0, 0, 1, 0, 0, 0, 1],   # k=0,i=1
        [0, 1, 0, 0, 1, 0, 0],   # k=1,i=0
        [0, -1, 1, 1, -1, -1, 0] # k=1,i=1
    ]
    
    return U, V, W

def _schoolbook_block(n_block, m_block, p_block):
    """Schoolbook decomposition for n_block x m_block x p_block."""
    rank = n_block * m_block * p_block
    U = [[0]*rank for _ in range(n_block * m_block)]
    V = [[0]*rank for _ in range(m_block * p_block)]
    W = [[0]*rank for _ in range(p_block * n_block)]
    col = 0
    for i in range(n_block):
        for j in range(m_block):
            for k in range(p_block):
                U[i*m_block+j][col] = 1
                V[j*p_block+k][col] = 1
                W[k*n_block+i][col] = 1
                col += 1
    return U, V, W

def _combine_blocks(blocks, n, m, p):
    """
    Combine multiple block decompositions into a single global decomposition.
    
    Each block has:
    - U_block: (n_b * m_b) x rank_b
    - V_block: (m_b * p_b) x rank_b
    - W_block: (p_b * n_b) x rank_b
    - i_offset, j_offset, k_offset: where this block sits in the full matrix
    
    The full matrices are:
    - U_full: (n*m) x total_rank
    - V_full: (m*p) x total_rank
    - W_full: (p*n) x total_rank
    """
    total_rank = sum(b['rank'] for b in blocks)
    U_full = [[0]*total_rank for _ in range(n * m)]
    V_full = [[0]*total_rank for _ in range(m * p)]
    W_full = [[0]*total_rank for _ in range(p * n)]
    
    col_offset = 0
    for b in blocks:
        i_off = b['i_offset']
        j_off = b['j_offset']
        k_off = b['k_offset']
        U_b = b['U']
        V_b = b['V']
        W_b = b['W']
        rank_b = b['rank']
        n_b = b['n']
        m_b = b['m']
        p_b = b['p']
        
        for li in range(n_b * m_b):
            i_local = li // m_b
            j_local = li % m_b
            for c in range(rank_b):
                U_full[(i_off + i_local) * m + (j_off + j_local)][col_offset + c] += U_b[li][c]
        
        for lj in range(m_b * p_b):
            j_local = lj // p_b
            k_local = lj % p_b
            for c in range(rank_b):
                V_full[(j_off + j_local) * p + (k_off + k_local)][col_offset + c] += V_b[lj][c]
        
        for lk in range(p_b * n_b):
            k_local = lk // n_b
            i_local = lk % n_b
            for c in range(rank_b):
                W_full[(k_off + k_local) * n + (i_off + i_local)][col_offset + c] += W_b[lk][c]
        
        col_offset += rank_b
    
    return U_full, V_full, W_full

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    n, m, p = 3, 5, 6
    
    # Define blocks:
    # 1. I={0,1,2}, J={0,1,2}, K={0,1,2} : 3x3x3
    # 2. I={0,1,2}, J={0,1,2}, K={3,4,5} : 3x3x3
    # 3. I={0,1}, J={3,4}, K={0,1} : 2x2x2
    # 4. I={0,1}, J={3,4}, K={2,3} : 2x2x2
    # 5. I={0,1}, J={3,4}, K={4,5} : 2x2x2
    # 6. I={2}, J={3,4}, K={0,1} : 1x2x2
    # 7. I={2}, J={3,4}, K={2,3} : 1x2x2
    # 8. I={2}, J={3,4}, K={4,5} : 1x2x2
    
    blocks = []
    
    # Block 1: 3x3x3 at i=0-2, j=0-2, k=0-2
    U1, V1, W1 = _schoolbook_block(3, 3, 3)
    blocks.append({
        'U': U1, 'V': V1, 'W': W1,
        'rank': 27, 'n': 3, 'm': 3, 'p': 3,
        'i_offset': 0, 'j_offset': 0, 'k_offset': 0
    })
    
    # Block 2: 3x3x3 at i=0-2, j=0-2, k=3-5
    U2, V2, W2 = _schoolbook_block(3, 3, 3)
    blocks.append({
        'U': U2, 'V': V2, 'W': W2,
        'rank': 27, 'n': 3, 'm': 3, 'p': 3,
        'i_offset': 0, 'j_offset': 0, 'k_offset': 3
    })
    
    # Block 3: 2x2x2 at i=0-1, j=3-4, k=0-1
    U3, V3, W3 = _strassen_2x2x2()
    blocks.append({
        'U': U3, 'V': V3, 'W': W3,
        'rank': 7, 'n': 2, 'm': 2, 'p': 2,
        'i_offset': 0, 'j_offset': 3, 'k_offset': 0
    })
    
    # Block 4: 2x2x2 at i=0-1, j=3-4, k=2-3
    U4, V4, W4 = _strassen_2x2x2()
    blocks.append({
        'U': U4, 'V': V4, 'W': W4,
        'rank': 7, 'n': 2, 'm': 2, 'p': 2,
        'i_offset': 0, 'j_offset': 3, 'k_offset': 2
    })
    
    # Block 5: 2x2x2 at i=0-1, j=3-4, k=4-5
    U5, V5, W5 = _strassen_2x2x2()
    blocks.append({
        'U': U5, 'V': V5, 'W': W5,
        'rank': 7, 'n': 2, 'm': 2, 'p': 2,
        'i_offset': 0, 'j_offset': 3, 'k_offset': 4
    })
    
    # Block 6: 1x2x2 at i=2, j=3-4, k=0-1 (schoolbook, rank 4)
    U6, V6, W6 = _schoolbook_block(1, 2, 2)
    blocks.append({
        'U': U6, 'V': V6, 'W': W6,
        'rank': 4, 'n': 1, 'm': 2, 'p': 2,
        'i_offset': 2, 'j_offset': 3, 'k_offset': 0
    })
    
    # Block 7: 1x2x2 at i=2, j=3-4, k=2-3
    U7, V7, W7 = _schoolbook_block(1, 2, 2)
    blocks.append({
        'U': U7, 'V': V7, 'W': W7,
        'rank': 4, 'n': 1, 'm': 2, 'p': 2,
        'i_offset': 2, 'j_offset': 3, 'k_offset': 2
    })
    
    # Block 8: 1x2x2 at i=2, j=3-4, k=4-5
    U8, V8, W8 = _schoolbook_block(1, 2, 2)
    blocks.append({
        'U': U8, 'V': V8, 'W': W8,
        'rank': 4, 'n': 1, 'm': 2, 'p': 2,
        'i_offset': 2, 'j_offset': 3, 'k_offset': 4
    })
    
    total_rank = sum(b['rank'] for b in blocks)
    # Total rank = 27+27+7+7+7+4+4+4 = 87
    
    if time.time() > deadline:
        # Fallback to schoolbook
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][col] = 1
                    V_fb[j * p + k][col] = 1
                    W_fb[k * n + i][col] = 1
                    col += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    U_full, V_full, W_full = _combine_blocks(blocks, n, m, p)
    
    # Verify
    if _verify(U_full, V_full, W_full, n, m, p):
        return {"U": U_full, "V": V_full, "W": W_full}
    
    # Fallback to schoolbook if verification fails
    rank_fb = n * m * p
    U_fb = [[0] * rank_fb for _ in range(n * m)]
    V_fb = [[0] * rank_fb for _ in range(m * p)]
    W_fb = [[0] * rank_fb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    return {"U": U_fb, "V": V_fb, "W": W_fb}
# EVOLVE-BLOCK-END
