"""
Tensor decomposition for 5x5x5 matrix multiplication using Strassen's algorithm
embedded in a block structure.

Partition 5 = 2+2+1. Apply Strassen (rank 7) to 2x2*2x2 block products,
schoolbook to remaining block products.
"""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def _strassen_2x2():
    """
    Return the 7 Strassen bilinear forms for 2x2*2x2 matrix multiplication.
    
    Strassen:
    M1 = (A11+A22)(B11+B22)
    M2 = (A21+A22)B11
    M3 = A11(B21-B22)
    M4 = A22(B12-B11)
    M5 = (A11+A12)B22
    M6 = (A21-A11)(B11+B12)
    M7 = (A12-A22)(B21+B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    Returns (U_list, V_list, W_list) where each is a list of (index, coeff) pairs
    for the 4 entries of C, and U/V for the 4 entries of A/B.
    """
    # A indices: A11=(0,0), A12=(0,1), A21=(1,0), A22=(1,1)
    # B indices: B11=(0,0), B12=(0,1), B21=(1,0), B22=(1,1)
    # C indices: C11=(0,0), C12=(0,1), C21=(1,0), C22=(1,1)
    
    # Each term: (U_coeff_dict, V_coeff_dict, W_coeff_dict)
    # U_coeff_dict maps (row, col) -> coefficient
    # V_coeff_dict maps (row, col) -> coefficient  
    # W_coeff_dict maps (row, col) -> coefficient
    
    terms = []
    
    # M1 = (A11+A22)(B11+B22)
    terms.append({
        'U': {(0,0): 1, (1,1): 1},
        'V': {(0,0): 1, (1,1): 1},
        'W': {(0,0): 1, (1,1): 1}
    })
    # M2 = (A21+A22)B11
    terms.append({
        'U': {(1,0): 1, (1,1): 1},
        'V': {(0,0): 1},
        'W': {(1,0): 1}
    })
    # M3 = A11(B21-B22)
    terms.append({
        'U': {(0,0): 1},
        'V': {(1,0): 1, (1,1): -1},
        'W': {(0,1): 1}
    })
    # M4 = A22(B12-B11)
    terms.append({
        'U': {(1,1): 1},
        'V': {(0,1): 1, (0,0): -1},
        'W': {(0,0): 1, (1,0): 1}
    })
    # M5 = (A11+A12)B22
    terms.append({
        'U': {(0,0): 1, (0,1): 1},
        'V': {(1,1): 1},
        'W': {(0,0): -1, (0,1): 1}
    })
    # M6 = (A21-A11)(B11+B12)
    terms.append({
        'U': {(1,0): 1, (0,0): -1},
        'V': {(0,0): 1, (0,1): 1},
        'W': {(1,1): 1}
    })
    # M7 = (A12-A22)(B21+B22)
    terms.append({
        'U': {(0,1): 1, (1,1): -1},
        'V': {(1,0): 1, (1,1): 1},
        'W': {(0,0): 1}
    })
    
    return terms

def _schoolbook_block(m, n, p):
    """
    Return schoolbook terms for (m x n) * (n x p) block product.
    Returns list of dicts with 'U', 'V', 'W' keys.
    """
    terms = []
    for i in range(m):
        for j in range(n):
            for k in range(p):
                terms.append({
                    'U': {(i, j): 1},
                    'V': {(j, k): 1},
                    'W': {(i, k): 1}
                })
    return terms

def construct():
    """
    Returns a tensor decomposition of 5x5x5 matrix multiplication.
    
    Strategy: Partition 5 = 2+2+1. For each block product:
    - If both blocks are 2x2, use Strassen (rank 7)
    - Otherwise, use schoolbook
    
    Block structure:
    A = [[A11(2x2), A12(2x2), A13(2x1)],
         [A21(2x2), A22(2x2), A23(2x1)],
         [A31(1x2), A32(1x2), A33(1x1)]]
    
    B = [[B11(2x2), B12(2x2), B13(2x1)],
         [B21(2x2), B22(2x2), B23(2x1)],
         [B31(1x2), B32(1x2), B33(1x1)]]
    
    C = A*B has 9 blocks:
    C11(2x2) = A11*B11 + A12*B21 + A13*B31
    C12(2x2) = A11*B12 + A12*B22 + A13*B32
    C13(2x1) = A11*B13 + A12*B23 + A13*B33
    C21(2x2) = A21*B11 + A22*B21 + A23*B31
    C22(2x2) = A21*B12 + A22*B22 + A23*B32
    C23(2x1) = A21*B13 + A22*B23 + A23*B33
    C31(1x2) = A31*B11 + A32*B21 + A33*B31
    C32(1x2) = A31*B12 + A32*B22 + A33*B32
    C33(1x1) = A31*B13 + A32*B23 + A33*B33
    """
    start_time = time.time()
    deadline = start_time + 100
    
    # Block sizes: rows/cols partition [0,1], [2,3], [4]
    # A block (i,j): rows i, cols j
    # Block row ranges: [0,2), [2,4), [4,5)
    # Block col ranges: [0,2), [2,4), [4,5)
    block_sizes = [(0, 2), (2, 4), (4, 5)]
    
    # Collect all bilinear terms
    # Each term: (U_dict, V_dict, W_dict) where dicts map (global_row, global_col) -> coeff
    all_terms = []
    
    # C11 (2x2) = A11*B11 + A12*B21 + A13*B31
    # A11*B11: both 2x2, use Strassen
    for t in _strassen_2x2():
        U_g = {}
        for (r, c), v in t['U'].items():
            U_g[(block_sizes[0][0] + r, block_sizes[0][0] + c)] = v
        V_g = {}
        for (r, c), v in t['V'].items():
            V_g[(block_sizes[0][0] + r, block_sizes[0][0] + c)] = v
        W_g = {}
        for (r, c), v in t['W'].items():
            W_g[(block_sizes[0][0] + r, block_sizes[0][0] + c)] = v
        all_terms.append((U_g, V_g, W_g))
    
    # A12*B21: A12 is 2x2 (rows 0-1, cols 2-3), B21 is 2x2 (rows 2-3, cols 0-1)
    # Product is 2x2 (rows 0-1, cols 0-1)
    # Use Strassen with appropriate index mapping
    for t in _strassen_2x2():
        U_g = {}
        for (r, c), v in t['U'].items():
            U_g[(block_sizes[0][0] + r, block_sizes[1][0] + c)] = v
        V_g = {}
        for (r, c), v in t['V'].items():
            V_g[(block_sizes[1][0] + r, block_sizes[0][0] + c)] = v
        W_g = {}
        for (r, c), v in t['W'].items():
            W_g[(block_sizes[0][0] + r, block_sizes[0][0] + c)] = v
        all_terms.append((U_g, V_g, W_g))
    
    # A13*B31: A13 is 2x1 (rows 0-1, col 4), B31 is 1x2 (row 4, cols 0-1)
    # Product is 2x2 (rows 0-1, cols 0-1), schoolbook rank = 2*1*2 = 4
    for t in _schoolbook_block(2, 1, 2):
        U_g = {}
        for (r, c), v in t['U'].items():
            U_g[(block_sizes[0][0] + r, block_sizes[2][0] + c)] = v
        V_g = {}
        for (r, c), v in t['V'].items():
            V_g[(block_sizes[2][0] + r, block_sizes[0][0] + c)] = v
        W_g = {}
        for (r, c), v in t['W'].items():
            W_g[(block_sizes[0][0] + r, block_sizes[0][0] + c)] = v
        all_terms.append((U_g, V_g, W_g))
    
    # C12 (2x2) = A11*B12 + A12*B22 + A13*B32
    # A11*B12: A11 2x2 (rows 0-1, cols 0-1), B12 2x2 (rows 0-1, cols 2-3)
    for t in _strassen_2x2():
        U_g = {}
        for (r, c), v in t['U'].items():
            U_g[(block_sizes[0][0] + r, block_sizes[0][0] + c)] = v
        V_g = {}
        for (r, c), v in t['V'].items():
            V_g[(block_sizes[0][0] + r, block_sizes[1][0] + c)] = v
        W_g = {}
        for (r, c), v in t['W'].items():
            W_g[(block_sizes[0][0] + r, block_sizes[1][0] + c)] = v
        all_terms.append((U_g, V_g, W_g))
    
    # A12*B22: A12 2x2 (rows 0-1, cols 2-3), B22 2x2 (rows 2-3, cols 2-3)
    for t in _strassen_2x2():
        U_g = {}
        for (r, c), v in t['U'].items():
            U_g[(block_sizes[0][0] + r, block_sizes[1][0] + c)] = v
        V_g = {}
        for (r, c), v in t['V'].items():
            V_g[(block_sizes[1][0] + r, block_sizes[1][0] + c)] = v
        W_g = {}
        for (r, c), v in t['W'].items():
            W_g[(block_sizes[0][0] + r, block_sizes[1][0] + c)] = v
        all_terms.append((U_g, V_g, W_g))
    
    # A13*B32: A13 2x1 (rows 0-1, col 4), B32 1x2 (row 4, cols 2-3)
    for t in _schoolbook_block(2, 1, 2):
        U_g = {}
        for (r, c), v in t['U'].items():
            U_g[(block_sizes[0][0] + r, block_sizes[2][0] + c)] = v
        V_g = {}
        for (r, c), v in t['V'].items():
            V_g[(block_sizes[2][0] + r, block_sizes[1][0] + c)] = v
        W_g = {}
        for (r, c), v in t['W'].items():
            W_g[(block_sizes[0][0] + r, block_sizes[1][0] + c)] = v
        all_terms.append((U_g, V_g, W_g))
    
    # C13 (2x1) = A11*B13 + A12*B23 + A13*B33
    # A11*B13: A11 2x2 (rows 0-1, cols 0-1), B13 2x1 (rows 0-1, col 4)
    for t in _schoolbook_block(2, 2, 1):
        U_g = {}
        for (r, c), v in t['U'].items():
            U_g[(block_sizes[0][0] + r, block_sizes[0][0] + c)] = v
        V_g = {}
        for (r, c), v in t['V'].items():
            V_g[(block_sizes[0][0] + r, block_sizes[2][0] + c)] = v
        W_g = {}
        for (r, c), v in t['W'].items():
            W_g[(block_sizes[0][0] + r, block_sizes[2][0] + c)] = v
        all_terms.append((U_g, V_g, W_g))
    
    # A12*B23: A12 2x2 (rows 0-1, cols 2-3), B23 2x1 (rows 2-3, col 4)
    for t in _schoolbook_block(2, 2, 1):
        U_g = {}
        for (r, c), v in t['U'].items():
            U_g[(block_sizes[0][0] + r, block_sizes[1][0] + c)] = v
        V_g = {}
        for (r, c), v in t['V'].items():
            V_g[(block_sizes[1][0] + r, block_sizes[2][0] + c)] = v
        W_g = {}
        for (r, c), v in t['W'].items():
            W_g[(block_sizes[0][0] + r, block_sizes[2][0] + c)] = v
        all_terms.append((U_g, V_g, W_g))
    
    # A13*B33: A13 2x1 (rows 0-1, col 4), B33 1x1 (row 4, col 4)
    for t in _schoolbook_block(2, 1, 1):
        U_g = {}
        for (r, c), v in t['U'].items():
            U_g[(block_sizes[0][0] + r, block_sizes[2][0] + c)] = v
        V_g = {}
        for (r, c), v in t['V'].items():
            V_g[(block_sizes[2][0] + r, block_sizes[2][0] + c)] = v
        W_g = {}
        for (r, c), v in t['W'].items():
            W_g[(block_sizes[0][0] + r, block_sizes[2][0] + c)] = v
        all_terms.append((U_g, V_g, W_g))
    
    # C21 (2x2) = A21*B11 + A22*B21 + A23*B31
    # A21*B11: A21 2x2 (rows 2-3, cols 0-1), B11 2x2 (rows 0-1, cols 0-1)
    for t in _strassen_2x2():
        U_g = {}
        for (r, c), v in t['U'].items():
            U_g[(block_sizes[1][0] + r, block_sizes[0][0] + c)] = v
        V_g = {}
        for (r, c), v in t['V'].items():
            V_g[(block_sizes[0][0] + r, block_sizes[0][0] + c)] = v
        W_g = {}
        for (r, c), v in t['W'].items():
            W_g[(block_sizes[1][0] + r, block_sizes[0][0] + c)] = v
        all_terms.append((U_g, V_g, W_g))
    
    # A22*B21: A22 2x2 (rows 2-3, cols 2-3), B21 2x2 (rows 2-3, cols 0-1)
    for t in _strassen_2x2():
        U_g = {}
        for (r, c), v in t['U'].items():
            U_g[(block_sizes[1][0] + r, block_sizes[1][0] + c)] = v
        V_g = {}
        for (r, c), v in t['V'].items():
            V_g[(block_sizes[1][0] + r, block_sizes[0][0] + c)] = v
        W_g = {}
        for (r, c), v in t['W'].items():
            W_g[(block_sizes[1][0] + r, block_sizes[0][0] + c)] = v
        all_terms.append((U_g, V_g, W_g))
    
    # A23*B31: A23 2x1 (rows 2-3, col 4), B31 1x2 (row 4, cols 0-1)
    for t in _schoolbook_block(2, 1, 2):
        U_g = {}
        for (r, c), v in t['U'].items():
            U_g[(block_sizes[1][0] + r, block_sizes[2][0] + c)] = v
        V_g = {}
        for (r, c), v in t['V'].items():
            V_g[(block_sizes[2][0] + r, block_sizes[0][0] + c)] = v
        W_g = {}
        for (r, c), v in t['W'].items():
            W_g[(block_sizes[1][0] + r, block_sizes[0][0] + c)] = v
        all_terms.append((U_g, V_g, W_g))
    
    # C22 (2x2) = A21*B12 + A22*B22 + A23*B32
    # A21*B12: A21 2x2 (rows 2-3, cols 0-1), B12 2x2 (rows 0-1, cols 2-3)
    for t in _strassen_2x2():
        U_g = {}
        for (r, c), v in t['U'].items():
            U_g[(block_sizes[1][0] + r, block_sizes[0][0] + c)] = v
        V_g = {}
        for (r, c), v in t['V'].items():
            V_g[(block_sizes[0][0] + r, block_sizes[1][0] + c)] = v
        W_g = {}
        for (r, c), v in t['W'].items():
            W_g[(block_sizes[1][0] + r, block_sizes[1][0] + c)] = v
        all_terms.append((U_g, V_g, W_g))
    
    # A22*B22: A22 2x2 (rows 2-3, cols 2-3), B22 2x2 (rows 2-3, cols 2-3)
    for t in _strassen_2x2():
        U_g = {}
        for (r, c), v in t['U'].items():
            U_g[(block_sizes[1][0] + r, block_sizes[1][0] + c)] = v
        V_g = {}
        for (r, c), v in t['V'].items():
            V_g[(block_sizes[1][0] + r, block_sizes[1][0] + c)] = v
        W_g = {}
        for (r, c), v in t['W'].items():
            W_g[(block_sizes[1][0] + r, block_sizes[1][0] + c)] = v
        all_terms.append((U_g, V_g, W_g))
    
    # A23*B32: A23 2x1 (rows 2-3, col 4), B32 1x2 (row 4, cols 2-3)
    for t in _schoolbook_block(2, 1, 2):
        U_g = {}
        for (r, c), v in t['U'].items():
            U_g[(block_sizes[1][0] + r, block_sizes[2][0] + c)] = v
        V_g = {}
        for (r, c), v in t['V'].items():
            V_g[(block_sizes[2][0] + r, block_sizes[1][0] + c)] = v
        W_g = {}
        for (r, c), v in t['W'].items():
            W_g[(block_sizes[1][0] + r, block_sizes[1][0] + c)] = v
        all_terms.append((U_g, V_g, W_g))
    
    # C23 (2x1) = A21*B13 + A22*B23 + A23*B33
    # A21*B13: A21 2x2 (rows 2-3, cols 0-1), B13 2x1 (rows 0-1, col 4)
    for t in _schoolbook_block(2, 2, 1):
        U_g = {}
        for (r, c), v in t['U'].items():
            U_g[(block_sizes[1][0] + r, block_sizes[0][0] + c)] = v
        V_g = {}
        for (r, c), v in t['V'].items():
            V_g[(block_sizes[0][0] + r, block_sizes[2][0] + c)] = v
        W_g = {}
        for (r, c), v in t['W'].items():
            W_g[(block_sizes[1][0] + r, block_sizes[2][0] + c)] = v
        all_terms.append((U_g, V_g, W_g))
    
    # A22*B23: A22 2x2 (rows 2-3, cols 2-3), B23 2x1 (rows 2-3, col 4)
    for t in _schoolbook_block(2, 2, 1):
        U_g = {}
        for (r, c), v in t['U'].items():
            U_g[(block_sizes[1][0] + r, block_sizes[1][0] + c)] = v
        V_g = {}
        for (r, c), v in t['V'].items():
            V_g[(block_sizes[1][0] + r, block_sizes[2][0] + c)] = v
        W_g = {}
        for (r, c), v in t['W'].items():
            W_g[(block_sizes[1][0] + r, block_sizes[2][0] + c)] = v
        all_terms.append((U_g, V_g, W_g))
    
    # A23*B33: A23 2x1 (rows 2-3, col 4), B33 1x1 (row 4, col 4)
    for t in _schoolbook_block(2, 1, 1):
        U_g = {}
        for (r, c), v in t['U'].items():
            U_g[(block_sizes[1][0] + r, block_sizes[2][0] + c)] = v
        V_g = {}
        for (r, c), v in t['V'].items():
            V_g[(block_sizes[2][0] + r, block_sizes[2][0] + c)] = v
        W_g = {}
        for (r, c), v in t['W'].items():
            W_g[(block_sizes[1][0] + r, block_sizes[2][0] + c)] = v
        all_terms.append((U_g, V_g, W_g))
    
    # C31 (1x2) = A31*B11 + A32*B21 + A33*B31
    # A31*B11: A31 1x2 (row 4, cols 0-1), B11 2x2 (rows 0-1, cols 0-1)
    for t in _schoolbook_block(1, 2, 2):
        U_g = {}
        for (r, c), v in t['U'].items():
            U_g[(block_sizes[2][0] + r, block_sizes[0][0] + c)] = v
        V_g = {}
        for (r, c), v in t['V'].items():
            V_g[(block_sizes[0][0] + r, block_sizes[0][0] + c)] = v
        W_g = {}
        for (r, c), v in t['W'].items():
            W_g[(block_sizes[2][0] + r, block_sizes[0][0] + c)] = v
        all_terms.append((U_g, V_g, W_g))
    
    # A32*B21: A32 1x2 (row 4, cols 2-3), B21 2x2 (rows 2-3, cols 0-1)
    for t in _schoolbook_block(1, 2, 2):
        U_g = {}
        for (r, c), v in t['U'].items():
            U_g[(block_sizes[2][0] + r, block_sizes[1][0] + c)] = v
        V_g = {}
        for (r, c), v in t['V'].items():
            V_g[(block_sizes[1][0] + r, block_sizes[0][0] + c)] = v
        W_g = {}
        for (r, c), v in t['W'].items():
            W_g[(block_sizes[2][0] + r, block_sizes[0][0] + c)] = v
        all_terms.append((U_g, V_g, W_g))
    
    # A33*B31: A33 1x1 (row 4, col 4), B31 1x2 (row 4, cols 0-1)
    for t in _schoolbook_block(1, 1, 2):
        U_g = {}
        for (r, c), v in t['U'].items():
            U_g[(block_sizes[2][0] + r, block_sizes[2][0] + c)] = v
        V_g = {}
        for (r, c), v in t['V'].items():
            V_g[(block_sizes[2][0] + r, block_sizes[0][0] + c)] = v
        W_g = {}
        for (r, c), v in t['W'].items():
            W_g[(block_sizes[2][0] + r, block_sizes[0][0] + c)] = v
        all_terms.append((U_g, V_g, W_g))
    
    # C32 (1x2) = A31*B12 + A32*B22 + A33*B32
    # A31*B12: A31 1x2 (row 4, cols 0-1), B12 2x2 (rows 0-1, cols 2-3)
    for t in _schoolbook_block(1, 2, 2):
        U_g = {}
        for (r, c), v in t['U'].items():
            U_g[(block_sizes[2][0] + r, block_sizes[0][0] + c)] = v
        V_g = {}
        for (r, c), v in t['V'].items():
            V_g[(block_sizes[0][0] + r, block_sizes[1][0] + c)] = v
        W_g = {}
        for (r, c), v in t['W'].items():
            W_g[(block_sizes[2][0] + r, block_sizes[1][0] + c)] = v
        all_terms.append((U_g, V_g, W_g))
    
    # A32*B22: A32 1x2 (row 4, cols 2-3), B22 2x2 (rows 2-3, cols 2-3)
    for t in _schoolbook_block(1, 2, 2):
        U_g = {}
        for (r, c), v in t['U'].items():
            U_g[(block_sizes[2][0] + r, block_sizes[1][0] + c)] = v
        V_g = {}
        for (r, c), v in t['V'].items():
            V_g[(block_sizes[1][0] + r, block_sizes[1][0] + c)] = v
        W_g = {}
        for (r, c), v in t['W'].items():
            W_g[(block_sizes[2][0] + r, block_sizes[1][0] + c)] = v
        all_terms.append((U_g, V_g, W_g))
    
    # A33*B32: A33 1x1 (row 4, col 4), B32 1x2 (row 4, cols 2-3)
    for t in _schoolbook_block(1, 1, 2):
        U_g = {}
        for (r, c), v in t['U'].items():
            U_g[(block_sizes[2][0] + r, block_sizes[2][0] + c)] = v
        V_g = {}
        for (r, c), v in t['V'].items():
            V_g[(block_sizes[2][0] + r, block_sizes[1][0] + c)] = v
        W_g = {}
        for (r, c), v in t['W'].items():
            W_g[(block_sizes[2][0] + r, block_sizes[1][0] + c)] = v
        all_terms.append((U_g, V_g, W_g))
    
    # C33 (1x1) = A31*B13 + A32*B23 + A33*B33
    # A31*B13: A31 1x2 (row 4, cols 0-1), B13 2x1 (rows 0-1, col 4)
    for t in _schoolbook_block(1, 2, 1):
        U_g = {}
        for (r, c), v in t['U'].items():
            U_g[(block_sizes[2][0] + r, block_sizes[0][0] + c)] = v
        V_g = {}
        for (r, c), v in t['V'].items():
            V_g[(block_sizes[0][0] + r, block_sizes[2][0] + c)] = v
        W_g = {}
        for (r, c), v in t['W'].items():
            W_g[(block_sizes[2][0] + r, block_sizes[2][0] + c)] = v
        all_terms.append((U_g, V_g, W_g))
    
    # A32*B23: A32 1x2 (row 4, cols 2-3), B23 2x1 (rows 2-3, col 4)
    for t in _schoolbook_block(1, 2, 1):
        U_g = {}
        for (r, c), v in t['U'].items():
            U_g[(block_sizes[2][0] + r, block_sizes[1][0] + c)] = v
        V_g = {}
        for (r, c), v in t['V'].items():
            V_g[(block_sizes[1][0] + r, block_sizes[2][0] + c)] = v
        W_g = {}
        for (r, c), v in t['W'].items():
            W_g[(block_sizes[2][0] + r, block_sizes[2][0] + c)] = v
        all_terms.append((U_g, V_g, W_g))
    
    # A33*B33: A33 1x1 (row 4, col 4), B33 1x1 (row 4, col 4)
    for t in _schoolbook_block(1, 1, 1):
        U_g = {}
        for (r, c), v in t['U'].items():
            U_g[(block_sizes[2][0] + r, block_sizes[2][0] + c)] = v
        V_g = {}
        for (r, c), v in t['V'].items():
            V_g[(block_sizes[2][0] + r, block_sizes[2][0] + c)] = v
        W_g = {}
        for (r, c), v in t['W'].items():
            W_g[(block_sizes[2][0] + r, block_sizes[2][0] + c)] = v
        all_terms.append((U_g, V_g, W_g))
    
    # Now construct U, V, W matrices
    n, m, p = DIMENSIONS
    rank = len(all_terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col, (U_dict, V_dict, W_dict) in enumerate(all_terms):
        for (r, c), val in U_dict.items():
            U[r * m + c][col] += val
        for (r, c), val in V_dict.items():
            V[r * p + c][col] += val
        for (r, c), val in W_dict.items():
            W[r * n + c][col] += val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
