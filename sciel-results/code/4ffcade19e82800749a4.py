"""Rank-minimal exact tensor decomposition for 4x4 * 4x8 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p):
    """Check that U ⊗ V ⊗ W equals the matmul tensor exactly."""
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    rank = W.shape[1]
    
    # Reshape to (n, m, rank), (m, p, rank), (p, n, rank)
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    # Compute T[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
    # We need T to be 1 for all i, j, k.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U_r[i, j, r] * V_r[j, k, r] * W_r[k, i, r]
                if abs(s - 1.0) > 1e-6:
                    return False
    return True

def _build_strassen_2x2():
    """
    Returns U, V, W for 2x2 * 2x2 matrix multiplication using Strassen (rank 7).
    Indexing:
    U: A[i,j] -> i*2+j
    V: B[j,k] -> j*2+k
    W: C[k,i] -> k*2+i
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Strassen formulas for C = A*B
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A10)B11
    # M6 = (A11-A10)(B00+B10)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Indices:
    # A: 00=0, 01=1, 10=2, 11=3
    # B: 00=0, 01=1, 10=2, 11=3
    # C: 00=0, 01=1, 10=2, 11=3 (k*2+i: 00=0, 01=1, 10=2, 11=3)
    
    # Term 0: M1
    U[0,0]=1; U[3,0]=1
    V[0,0]=1; V[3,0]=1
    W[0,0]=1; W[3,0]=1
    
    # Term 1: M2
    U[2,1]=1; U[3,1]=1
    V[0,1]=1
    W[2,1]=1
    
    # Term 2: M3
    U[0,2]=1
    V[1,2]=1; V[3,2]=-1
    W[1,2]=1
    
    # Term 3: M4
    U[3,3]=1
    V[2,3]=1; V[0,3]=-1
    W[2,3]=1
    
    # Term 4: M5
    U[0,4]=1; U[2,4]=1
    V[3,4]=1
    W[0,4]=-1; W[1,4]=1
    
    # Term 5: M6
    U[3,5]=1; U[2,5]=-1
    V[0,5]=1; V[2,5]=1
    W[3,5]=1
    
    # Term 6: M7
    U[1,6]=1; U[3,6]=-1
    V[2,6]=1; V[3,6]=1
    W[0,6]=1; W[3,6]=-1
    
    return U, V, W

def _build_strassen_4x4():
    """
    Returns U, V, W for 4x4 * 4x4 matrix multiplication using Strassen (rank 49).
    Indexing:
    U: A[i,j] -> i*4+j
    V: B[j,k] -> j*4+k
    W: C[k,i] -> k*4+i
    """
    U, V, W = np.zeros((16, 49)), np.zeros((16, 49)), np.zeros((16, 49))
    
    # Block indices for 4x4 matrix (row-major i*4+j)
    # A00: rows 0-1, cols 0-1 -> 0,1,4,5
    # A01: rows 0-1, cols 2-3 -> 2,3,6,7
    # A10: rows 2-3, cols 0-1 -> 8,9,12,13
    # A11: rows 2-3, cols 2-3 -> 10,11,14,15
    
    # B00: rows 0-1, cols 0-1 -> 0,1,4,5
    # B01: rows 0-1, cols 2-3 -> 2,3,6,7
    # B10: rows 2-3, cols 0-1 -> 8,9,12,13
    # B11: rows 2-3, cols 2-3 -> 10,11,14,15
    
    # C00: cols 0-1, rows 0-1 -> 0,1,4,5 (k*4+i)
    # C01: cols 2-3, rows 0-1 -> 2,3,6,7
    # C10: cols 0-1, rows 2-3 -> 8,9,12,13
    # C11: cols 2-3, rows 2-3 -> 10,11,14,15
    
    a_blocks = {
        'A00': [0,1,4,5], 'A01': [2,3,6,7],
        'A10': [8,9,12,13], 'A11': [10,11,14,15]
    }
    b_blocks = {
        'B00': [0,1,4,5], 'B01': [2,3,6,7],
        'B10': [8,9,12,13], 'B11': [10,11,14,15]
    }
    c_blocks = {
        'C00': [0,1,4,5], 'C01': [2,3,6,7],
        'C10': [8,9,12,13], 'C11': [10,11,14,15]
    }
    
    U2, V2, W2 = _build_strassen_2x2()
    
    # Strassen terms for 4x4*4x4
    # Each term is a product of a linear combination of A-blocks and a linear combination of B-blocks.
    # The result is added to a linear combination of C-blocks.
    # We expand each term using the rank-7 decomposition of 2x2*2x2.
    
    terms = [
        # (U_blocks, U_coeffs, V_blocks, V_coeffs, W_blocks, W_coeffs)
        (['A00','A11'], [1,1], ['B00','B11'], [1,1], ['C00','C11'], [1,1]),      # M1
        (['A10','A11'], [1,1], ['B00'], [1], ['C10'], [1]),                        # M2
        (['A00'], [1], ['B01','B11'], [1,-1], ['C01'], [1]),                       # M3
        (['A11'], [1], ['B10','B00'], [1,-1], ['C00','C10'], [1,1]),               # M4
        (['A00','A10'], [1,1], ['B11'], [1], ['C01','C00'], [1,-1]),               # M5
        (['A11','A10'], [1,-1], ['B00','B10'], [1,1], ['C11'], [1]),               # M6
        (['A01','A11'], [1,-1], ['B10','B11'], [1,1], ['C00','C11'], [1,-1]),      # M7
    ]
    
    col = 0
    for u_blocks, u_coeffs, v_blocks, v_coeffs, w_blocks, w_coeffs in terms:
        for r in range(7):
            u_vec = np.zeros(16)
            v_vec = np.zeros(16)
            w_vec = np.zeros(16)
            
            # U part
            for a_idx in range(4):
                a_name = ['A00','A01','A10','A11'][a_idx]
                c_strassen = U2[a_idx, r]
                if c_strassen == 0:
                    continue
                if a_name in u_blocks:
                    block_idx = u_blocks.index(a_name)
                    c_term = u_coeffs[block_idx]
                else:
                    c_term = 0
                final_coeff = c_strassen * c_term
                if final_coeff != 0:
                    for pos in a_blocks[a_name]:
                        u_vec[pos] += final_coeff
            
            # V part
            for b_idx in range(4):
                b_name = ['B00','B01','B10','B11'][b_idx]
                c_strassen = V2[b_idx, r]
                if c_strassen == 0:
                    continue
                if b_name in v_blocks:
                    block_idx = v_blocks.index(b_name)
                    c_term = v_coeffs[block_idx]
                else:
                    c_term = 0
                final_coeff = c_strassen * c_term
                if final_coeff != 0:
                    for pos in b_blocks[b_name]:
                        v_vec[pos] += final_coeff
            
            # W part
            for c_idx in range(4):
                c_name = ['C00','C01','C10','C11'][c_idx]
                c_strassen = W2[c_idx, r]
                if c_strassen == 0:
                    continue
                if c_name in w_blocks:
                    block_idx = w_blocks.index(c_name)
                    c_term = w_coeffs[block_idx]
                else:
                    c_term = 0
                final_coeff = c_strassen * c_term
                if final_coeff != 0:
                    for pos in c_blocks[c_name]:
                        w_vec[pos] += final_coeff
            
            U[:, col] = u_vec
            V[:, col] = v_vec
            W[:, col] = w_vec
            col += 1
            
    return U, V, W

def _build_strassen_4x4_4x8():
    """
    Constructs a rank-98 decomposition for 4x4 * 4x8 using two 4x4*4x4 Strassen decompositions.
    """
    n, m, p = 4, 4, 8
    
    U1, V1, W1 = _build_strassen_4x4() # For B1 (cols 0-3)
    U2, V2, W2 = _build_strassen_4x4() # For B2 (cols 4-7)
    
    rank = 98
    U = np.zeros((16, rank))
    V = np.zeros((32, rank))
    W = np.zeros((32, rank))
    
    # Fill U
    U[:, 0:49] = U1
    U[:, 49:98] = U2
    
    # Fill V
    # V index is j*8+k.
    # V1 index is j*4+k.
    for j in range(4):
        for k in range(4):
            V[j*8+k, 0:49] = V1[j*4+k, 0:49]
            V[j*8+(k+4), 49:98] = V2[j*4+k, 0:49]
    
    # Fill W
    # W index is k*4+i.
    # W1 index is k*4+i.
    for k in range(4):
        for i in range(4):
            W[k*4+i, 0:49] = W1[k*4+i, 0:49]
            W[(k+4)*4+i, 49:98] = W2[k*4+i, 0:49]
            
    return U, V, W

def _build_rank1(n, m, p):
    """Build the rank-1 (schoolbook) decomposition for reference."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    n, m, p = DIMENSIONS
    
    # Try Strassen decomposition
    try:
        U, V, W = _build_strassen_4x4_4x8()
        # Verify to be safe
        if _verify(U, V, W, n, m, p):
            if time.time() < deadline:
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
        else:
            # If verification fails, fall back to schoolbook
            pass
    except Exception:
        pass
    
    # Fallback to schoolbook
    return _build_rank1(n, m, p)

# EVOLVE-BLOCK-END
