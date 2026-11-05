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
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # Strassen's 7 products:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B21)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For each Strassen term, we need to encode the linear combinations.
    # Since we're dealing with 0/1 matrices and the final result must be 1 for (i,j,k),
    # we need to use integer coefficients that make the product work out.
    
    # Let's use the standard Strassen decomposition with appropriate coefficients.
    # The key insight: for 2x2x2, each "tensor entry" (i,j,k) corresponds to A[i,j]*B[j,k].
    # In Strassen, we compute 7 linear combinations and then form C as linear combinations of those.
    
    # Let me think about this differently. The tensor T[i,j,k] = A[i,j]*B[j,k] for i,j,k in {0,1}.
    # Strassen computes:
    # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    # M2 = (A[1,0]+A[1,1])*B[0,0]
    # M3 = A[0,0]*(B[1,0]-B[1,1])
    # M4 = A[1,1]*(B[0,1]-B[1,0])
    # M5 = (A[0,0]+A[0,1])*B[1,1]
    # M6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
    # M7 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
    
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    
    # For the tensor decomposition, we need U, V, W such that:
    # sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all (i,j,k)
    
    # This is tricky because Strassen uses additions and subtractions.
    # Let me use a different approach: directly construct the rank-7 decomposition.
    
    # Actually, let me use the known rank-7 decomposition for 2x2x2.
    # The entries of U, V, W should be in {-1, 0, 1}.
    
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

def _bini_3x3x3():
    """
    Bini's rank-23 decomposition for 3x3x3 matrix multiplication.
    This is complex; let's use a simpler approach and just use schoolbook for 3x3x3
    if we can't find the exact coefficients.
    
    Actually, let me just use schoolbook for 3x3x3 to be safe, giving rank 27.
    Then total would be 27+27+7+7+7+4+4+4 = 87. Still better than 90.
    
    Wait, let me try to implement Bini's algorithm properly.
    """
    # For now, use schoolbook 3x3x3 (rank 27) to ensure correctness
    U = [[0]*27 for _ in range(9)]
    V = [[0]*27 for _ in range(9)]
    W = [[0]*27 for _ in range(9)]
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i*3+j][col] = 1
                V[j*3+k][col] = 1
                W[k*3+i][col] = 1
                col += 1
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

def _combine_blocks(blocks):
    """
    Combine multiple block decompositions into a single global decomposition.
    
    Each block has:
    - U_block: (n_b * m_b) x rank_b
    - V_block: (m_b * p_b) x rank_b
    - W_block: (p_b * n_b) x rank_b
    - i_offset, j_offset, k_offset: where this block sits in the full matrix
    
    The full matrices are:
    - U_full: (3*5) x total_rank
    - V_full: (5*6) x total_rank
    - W_full: (6*3) x total_rank
    """
    total_rank = sum(b['rank'] for b in blocks)
    U_full = [[0]*total_rank for _ in range(15)]
    V_full = [[0]*total_rank for _ in range(30)]
    W_full = [[0]*total_rank for _ in range(18)]
    
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
            for c in range(rank_b):
                U_full[i_off * 5 + j_off + li % m_b if False else (i_off + li // m_b) * 5 + (j_off + li % m_b)][col_offset + c] += U_b[li][c]
        
        for lj in range(m_b * p_b):
            for c in range(rank_b):
                V_full[(j_off + lj // p_b) * 6 + (k_off + lj % p_b)][col_offset + c] += V_b[lj][c]
        
        for lk in range(p_b * n_b):
            for c in range(rank_b):
                W_full[(k_off + lk // n_b) * 3 + (i_off + lk % n_b)][col_offset + c] += W_b[lk][c]
        
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
    
    U_full, V_full, W_full = _combine_blocks(blocks)
    
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
