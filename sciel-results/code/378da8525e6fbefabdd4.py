"""Program for 3x4x5 matrix multiplication tensor decomposition."""
import numpy as np

DIMENSIONS = [3, 4, 5]

# Strassen's 2x2x2 rank-7 decomposition factors
# U has shape (n*m, rank) = (4, 7)
# V has shape (m*p, rank) = (4, 7)
# W has shape (p*n, rank) = (4, 7)
# Convention: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]

STRASSEN_U = np.array([
    [1, 0, 0, 1, 0, 0, 0],  # A11 (i=0,j=0)
    [0, 1, 1, 0, 1, 0, 1],  # A12 (i=0,j=1)
    [1, 0, 0, 0, 0, 1, -1], # A21 (i=1,j=0)
    [0, 1, 0, 0, 1, 1, 0],  # A22 (i=1,j=1)
], dtype=float)

STRASSEN_V = np.array([
    [1, 1, 0, 1, 0, 1, 1],  # B11 (j=0,k=0)
    [0, 0, 1, 0, -1, 0, 0], # B12 (j=0,k=1)
    [1, 1, -1, 0, 0, 0, 0], # B21 (j=1,k=0)
    [1, 0, -1, 0, 0, 1, 1], # B22 (j=1,k=1)
], dtype=float)

STRASSEN_W = np.array([
    [1, 0, 0, 1, -1, 0, 1],  # C11 (k=0,i=0)
    [0, 0, 1, 1, 0, 1, 0],   # C12 (k=1,i=0)
    [0, 1, 0, 1, 1, 0, 0],   # C21 (k=0,i=1)
    [1, -1, 1, 0, 0, 1, 0],  # C22 (k=1,i=1)
], dtype=float)


def _build_factors(n, m, p, sub_blocks):
    """
    Build U, V, W factors for a decomposition where the middle dimension m
    is split into sub_blocks. Each sub_block is a dict with:
      - 'n_sub': number of rows in A sub-block
      - 'm_sub': number of cols in A / rows in B sub-block  
      - 'p_sub': number of cols in B
      - 'type': 'strassen' (for 2x2x2) or 'schoolbook'
      - 'i_offset': row offset in A
      - 'j_offset': col offset in A / row offset in B
      - 'k_offset': col offset in B
    
    Returns U (n*m x rank), V (m*p x rank), W (p*n x rank)
    """
    total_rank = 0
    for sb in sub_blocks:
        n_s = sb['n_sub']
        m_s = sb['m_sub']
        p_s = sb['p_sub']
        if sb['type'] == 'strassen':
            total_rank += 7
        else:
            total_rank += n_s * m_s * p_s
    
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    col = 0
    for sb in sub_blocks:
        n_s = sb['n_sub']
        m_s = sb['m_sub']
        p_s = sb['p_sub']
        i_off = sb['i_offset']
        j_off = sb['j_offset']
        k_off = sb['k_offset']
        
        if sb['type'] == 'strassen':
            # 2x2x2 Strassen
            assert n_s == 2 and m_s == 2 and p_s == 2
            for r in range(7):
                for i in range(2):
                    for j in range(2):
                        U[(i_off + i) * m + (j_off + j), col + r] = STRASSEN_U[i * 2 + j, r]
                for j in range(2):
                    for k in range(2):
                        V[(j_off + j) * p + (k_off + k), col + r] = STRASSEN_V[j * 2 + k, r]
                for k in range(2):
                    for i in range(2):
                        W[(k_off + k) * n + (i_off + i), col + r] = STRASSEN_W[k * 2 + i, r]
            col += 7
        else:
            # Schoolbook
            for i in range(n_s):
                for j in range(m_s):
                    for k in range(p_s):
                        U[(i_off + i) * m + (j_off + j), col] = 1
                        V[(j_off + j) * p + (k_off + k), col] = 1
                        W[(k_off + k) * n + (i_off + i), col] = 1
                        col += 1
    
    return U, V, W


def construct():
    n, m, p = DIMENSIONS  # 3, 4, 5
    
    # Split middle dimension m=4 into two parts of size 2
    # Block 1: j in [0,1], Block 2: j in [2,3]
    
    # For each block, we have A_sub (3x2) * B_sub (2x5)
    # Split A_sub rows into [0,1] and [2], B_sub cols into [0,1], [2,3], [4]
    # Sub-problems for each block:
    #   (2x2x2) Strassen, (2x2x2) Strassen, (2x2x1) schoolbook,
    #   (1x2x2) schoolbook, (1x2x2) schoolbook, (1x2x1) schoolbook
    
    sub_blocks = []
    
    for block_idx, j_off in enumerate([0, 2]):
        # Strassen 2x2x2 blocks
        # (2x2x2) for A rows 0-1, B cols 0-1
        sub_blocks.append({'n_sub': 2, 'm_sub': 2, 'p_sub': 2, 'type': 'strassen',
                           'i_offset': 0, 'j_offset': j_off, 'k_offset': 0})
        # (2x2x2) for A rows 0-1, B cols 2-3
        sub_blocks.append({'n_sub': 2, 'm_sub': 2, 'p_sub': 2, 'type': 'strassen',
                           'i_offset': 0, 'j_offset': j_off, 'k_offset': 2})
        # (2x2x1) for A rows 0-1, B col 4
        sub_blocks.append({'n_sub': 2, 'm_sub': 2, 'p_sub': 1, 'type': 'schoolbook',
                           'i_offset': 0, 'j_offset': j_off, 'k_offset': 4})
        # (1x2x2) for A row 2, B cols 0-1
        sub_blocks.append({'n_sub': 1, 'm_sub': 2, 'p_sub': 2, 'type': 'schoolbook',
                           'i_offset': 2, 'j_offset': j_off, 'k_offset': 0})
        # (1x2x2) for A row 2, B cols 2-3
        sub_blocks.append({'n_sub': 1, 'm_sub': 2, 'p_sub': 2, 'type': 'schoolbook',
                           'i_offset': 2, 'j_offset': j_off, 'k_offset': 2})
        # (1x2x1) for A row 2, B col 4
        sub_blocks.append({'n_sub': 1, 'm_sub': 2, 'p_sub': 1, 'type': 'schoolbook',
                           'i_offset': 2, 'j_offset': j_off, 'k_offset': 4})
    
    U, V, W = _build_factors(n, m, p, sub_blocks)
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
