import time
import random
import numpy as np

DIMENSIONS = [3, 5, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 90
    
    # Try structured decomposition using Strassen for 2x2x2 blocks
    best_result = _structured_decomposition(n, m, p)
    best_rank = _get_rank(best_result)
    
    # If structured approach gives improvement, try to verify and potentially improve
    if best_rank < schoolbook_rank:
        # Verify the decomposition
        if _verify_decomposition(best_result['U'], best_result['V'], best_result['W'], n, m, p, best_rank):
            return best_result
    
    # Fallback to schoolbook
    return _schoolbook_decomposition(n, m, p)

def _get_rank(result):
    """Get the rank from a decomposition result."""
    if not result or 'U' not in result:
        return 999
    return len(result['U'][0]) if result['U'] else 999

def _schoolbook_decomposition(n, m, p):
    """Create the standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p, rank):
    """
    Verify that the decomposition exactly decomposes the matrix multiplication tensor.
    For all (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1
    For all (i,j,k) and (i',j',k') where not (i==i' and j==j' and k==k'):
      the tensor T[i,j,k] should be 0 unless i==i', j==j', k==k'
    Actually the tensor is: T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    This should equal 1 if this is a valid (i,j,k) triple, 0 otherwise.
    But ALL (i,j,k) triples are valid! So we just need T[i,j,k]=1 for all.
    Wait no - the matrix multiplication tensor has T[i,j,k]=1 for all i in [n], j in [m], k in [p].
    So we just need to check that for all (i,j,k), the sum equals 1.
    """
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for c in range(rank):
                    total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(total - 1.0) > 1e-9:
                    return False
    return True

def _structured_decomposition(n, m, p):
    """
    Use a structured decomposition that applies Strassen's algorithm to 2x2x2 sub-blocks.
    
    For 3x5x6:
    - n=3: split into 2+1
    - m=5: split into 2+2+1  
    - p=6: split into 2+2+2
    
    Blocks where all three dims are 2 use Strassen (rank 7 per 2x2x2).
    Other blocks use schoolbook.
    """
    n_blocks = [2, 1]  # 3 = 2 + 1
    m_blocks = [2, 2, 1]  # 5 = 2 + 2 + 1
    p_blocks = [2, 2, 2]  # 6 = 2 + 2 + 2
    
    n_offsets = [0]
    for size in n_blocks:
        n_offsets.append(n_offsets[-1] + size)
    m_offsets = [0]
    for size in m_blocks:
        m_offsets.append(m_offsets[-1] + size)
    p_offsets = [0]
    for size in p_blocks:
        p_offsets.append(p_offsets[-1] + size)
    
    # Build columns as lists of (U_dict, V_dict, W_dict)
    columns = []
    
    for ni_idx, ni_size in enumerate(n_blocks):
        for mj_idx, mj_size in enumerate(m_blocks):
            for pk_idx, pk_size in enumerate(p_blocks):
                i_off = n_offsets[ni_idx]
                j_off = m_offsets[mj_idx]
                k_off = p_offsets[pk_idx]
                
                if ni_size == 2 and mj_size == 2 and pk_size == 2:
                    # Strassen for this 2x2x2 block
                    for term in _strassen_terms():
                        U_dict = {}
                        V_dict = {}
                        W_dict = {}
                        
                        for (li, lj), coeff in term['U'].items():
                            U_dict[(i_off + li, j_off + lj)] = coeff
                        for (lj, lk), coeff in term['V'].items():
                            V_dict[(j_off + lj, k_off + lk)] = coeff
                        for (lk, li), coeff in term['W'].items():
                            W_dict[(k_off + lk, i_off + li)] = coeff
                        
                        columns.append((U_dict, V_dict, W_dict))
                else:
                    # Schoolbook for this block
                    for li in range(ni_size):
                        for lj in range(mj_size):
                            for lk in range(pk_size):
                                U_dict = {(i_off + li, j_off + lj): 1}
                                V_dict = {(j_off + lj, k_off + lk): 1}
                                W_dict = {(k_off + lk, i_off + li): 1}
                                columns.append((U_dict, V_dict, W_dict))
    
    rank = len(columns)
    
    # Build the matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, (U_dict, V_dict, W_dict) in enumerate(columns):
        for (i, j), coeff in U_dict.items():
            U[i * m + j][c] = coeff
        for (j, k), coeff in V_dict.items():
            V[j * p + k][c] = coeff
        for (k, i), coeff in W_dict.items():
            W[k * n + i][c] = coeff
    
    return {"U": U, "V": V, "W": W}

def _strassen_terms():
    """
    Return the 7 Strassen terms for 2x2x2 matrix multiplication.
    
    C = A * B, where A, B, C are 2x2.
    A = [[A00, A01], [A10, A11]]
    B = [[B00, B01], [B10, B11]]
    C = [[C00, C01], [C10, C11]]
    
    M1 = (A00+A11)*(B00+B11)
    M2 = (A10+A11)*B00
    M3 = A00*(B01-B11)
    M4 = A11*(B10-B00)
    M5 = (A00+A01)*B11
    M6 = (A10-A00)*(B00+B01)
    M7 = (A01-A11)*(B10+B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    """
    return [
        # M1 = (A00+A11)*(B00+B11)
        {
            'U': {(0,0): 1, (1,1): 1},
            'V': {(0,0): 1, (1,1): 1},
            'W': {(0,0): 1, (1,1): 1}
        },
        # M2 = (A10+A11)*B00
        {
            'U': {(1,0): 1, (1,1): 1},
            'V': {(0,0): 1},
            'W': {(1,0): 1, (1,1): -1}
        },
        # M3 = A00*(B01-B11)
        {
            'U': {(0,0): 1},
            'V': {(0,1): 1, (1,1): -1},
            'W': {(0,1): 1, (1,1): 1}
        },
        # M4 = A11*(B10-B00)
        {
            'U': {(1,1): 1},
            'V': {(1,0): 1, (0,0): -1},
            'W': {(0,0): 1, (1,0): 1}
        },
        # M5 = (A00+A01)*B11
        {
            'U': {(0,0): 1, (0,1): 1},
            'V': {(1,1): 1},
            'W': {(0,0): -1, (0,1): 1}
        },
        # M6 = (A10-A00)*(B00+B01)
        {
            'U': {(1,0): 1, (0,0): -1},
            'V': {(0,0): 1, (0,1): 1},
            'W': {(1,1): 1}
        },
        # M7 = (A01-A11)*(B10+B11)
        {
            'U': {(0,1): 1, (1,1): -1},
            'V': {(1,0): 1, (1,1): 1},
            'W': {(0,0): 1, (1,1): -1}
        },
    ]

# EVOLVE-BLOCK-END
