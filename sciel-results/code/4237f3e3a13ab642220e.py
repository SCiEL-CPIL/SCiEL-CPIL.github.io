import numpy as np

def construct():
    """
    Returns a rank-49 exact tensor decomposition for 4x4x4 matrix multiplication.
    This is achieved by applying Strassen's algorithm (rank 7) to the 2x2 block
    structure of the 4x4 matrices.
    
    Interface:
    U: (16, 49)
    V: (16, 49)
    W: (16, 49)
    
    Indices:
    A index a = i*4 + j
    B index b = j*4 + k
    C index c = k*4 + i (Trace convention)
    """
    n, m, p = 4, 4, 4
    rank = 49
    
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W = np.zeros((16, rank))
    
    # Strassen 2x2x2 decomposition terms
    # Each term is a tuple (U_dict, V_dict, W_dict)
    # Keys are (row, col) within the 2x2 block.
    
    # M1 = (A11 + A22) * (B11 + B22) -> C11 += M1, C22 += M1
    s1 = (
        {(0,0): 1, (1,1): 1}, 
        {(0,0): 1, (1,1): 1}, 
        {(0,0): 1, (1,1): 1}
    )
    # M2 = A22 * (B21 - B11) -> C21 += M2, C22 -= M2
    s2 = (
        {(1,1): 1}, 
        {(1,0): 1, (0,0): -1}, 
        {(1,0): 1, (1,1): -1}
    )
    # M3 = (A11 - A22) * B22 -> C11 += M3, C21 += M3
    s3 = (
        {(0,0): 1, (1,1): -1}, 
        {(1,1): 1}, 
        {(0,0): 1, (1,0): 1}
    )
    # M4 = A11 * (B12 - B22) -> C12 += M4, C22 -= M4
    s4 = (
        {(0,0): 1}, 
        {(0,1): 1, (1,1): -1}, 
        {(0,1): 1, (1,1): -1}
    )
    # M5 = (A12 + A22) * B11 -> C12 += M5, C11 -= M5
    s5 = (
        {(0,1): 1, (1,1): 1}, 
        {(0,0): 1}, 
        {(0,1): 1, (0,0): -1}
    )
    # M6 = (A11 - A12) * (B12 + B22) -> C22 -= M6
    s6 = (
        {(0,0): 1, (0,1): -1}, 
        {(0,1): 1, (1,1): 1}, 
        {(1,1): -1}
    )
    # M7 = (A21 + A22) * (B11 - B21) -> C11 += M7
    s7 = (
        {(1,0): 1, (1,1): 1}, 
        {(0,0): 1, (1,0): -1}, 
        {(0,0): 1}
    )
    
    strassen_terms = [s1, s2, s3, s4, s5, s6, s7]
    
    # Outer Strassen 4x4 terms
    # Each term is a tuple (A_blocks, B_blocks, C_blocks)
    # A_blocks: list of ((block_row, block_col), coeff)
    
    # M1 = (A11 + A22) * (B11 + B22)
    o1 = (
        [((0,0), 1), ((1,1), 1)], 
        [((0,0), 1), ((1,1), 1)], 
        [((0,0), 1), ((1,1), 1)]
    )
    # M2 = A22 * (B21 - B11)
    o2 = (
        [((1,1), 1)], 
        [((1,0), 1), ((0,0), -1)], 
        [((1,0), 1), ((1,1), -1)]
    )
    # M3 = (A11 - A22) * B22
    o3 = (
        [((0,0), 1), ((1,1), -1)], 
        [((1,1), 1)], 
        [((0,0), 1), ((1,0), 1)]
    )
    # M4 = A11 * (B12 - B22)
    o4 = (
        [((0,0), 1)], 
        [((0,1), 1), ((1,1), -1)], 
        [((0,1), 1), ((1,1), -1)]
    )
    # M5 = (A12 + A22) * B11
    o5 = (
        [((0,1), 1), ((1,1), 1)], 
        [((0,0), 1)], 
        [((0,1), 1), ((0,0), -1)]
    )
    # M6 = (A11 - A12) * (B12 + B22)
    o6 = (
        [((0,0), 1), ((0,1), -1)], 
        [((0,1), 1), ((1,1), 1)], 
        [((1,1), -1)]
    )
    # M7 = (A21 + A22) * (B11 - B21)
    o7 = (
        [((1,0), 1), ((1,1), 1)], 
        [((0,0), 1), ((1,0), -1)], 
        [((0,0), 1)]
    )
    
    outer_terms = [o1, o2, o3, o4, o5, o6, o7]
    
    col = 0
    for ot in outer_terms:
        a_blocks, b_blocks, c_blocks = ot
        for st in strassen_terms:
            u_dict, v_dict, w_dict = st
            
            # Fill U
            for (br, bc), ca in a_blocks:
                for (lr, lc), v in u_dict.items():
                    a_idx = (br * 2 + lr) * 4 + (bc * 2 + lc)
                    U[a_idx, col] += ca * v
                    
            # Fill V
            for (br, bc), cb in b_blocks:
                for (lr, lc), v in v_dict.items():
                    b_idx = (br * 2 + lr) * 4 + (bc * 2 + lc)
                    V[b_idx, col] += cb * v
                    
            # Fill W
            for (br, bc), cc in c_blocks:
                for (lr, lc), v in w_dict.items():
                    c_idx = (br * 2 + lr) * 4 + (bc * 2 + lc)
                    W[c_idx, col] += cc * v
                    
            col += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
