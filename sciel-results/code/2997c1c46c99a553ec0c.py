import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-49 decomposition for 4x4 matrix multiplication
    using recursive Strassen (rank 7 for 2x2, composed twice).
    
    The 4x4 multiplication is viewed as 2x2 block multiplication,
    where each block is 2x2. Strassen's algorithm gives 7 block products,
    each of which is a 2x2 matrix multiplication decomposed with rank 7.
    Total rank: 7 * 7 = 49.
    """
    n, m, p = DIMENSIONS  # 4, 4, 4
    
    # Block indices in the 4x4 matrix (row-major, index = i*4+j)
    def block_indices(r, c):
        """Return the 4 indices of the 2x2 block at top-left (r,c) in a 4x4 matrix."""
        return [(r+dr)*4 + (c+dc) for dr in range(2) for dc in range(2)]
    
    A00_idx = block_indices(0, 0)  # [0, 1, 4, 5]
    A01_idx = block_indices(0, 2)  # [2, 3, 6, 7]
    A10_idx = block_indices(2, 0)  # [8, 9, 12, 13]
    A11_idx = block_indices(2, 2)  # [10, 11, 14, 15]
    
    B00_idx = block_indices(0, 0)  # [0, 1, 4, 5]
    B01_idx = block_indices(0, 2)  # [2, 3, 6, 7]
    B10_idx = block_indices(2, 0)  # [8, 9, 12, 13]
    B11_idx = block_indices(2, 2)  # [10, 11, 14, 15]
    
    C00_idx = block_indices(0, 0)  # [0, 1, 4, 5]
    C01_idx = block_indices(0, 2)  # [2, 3, 6, 7]
    C10_idx = block_indices(2, 0)  # [8, 9, 12, 13]
    C11_idx = block_indices(2, 2)  # [10, 11, 14, 15]
    
    def add_dict(d1, d2):
        result = dict(d1)
        for k, v in d2.items():
            result[k] = result.get(k, 0) + v
        return result
    
    # Define the 7 outer Strassen products.
    # For each: x_coeffs (list of 4 dicts for X entries), y_coeffs (list of 4 dicts for Y entries),
    #           z_to_c (list of 4 dicts mapping Z entries to C indices)
    
    # P1 = (A00+A11)(B00+B11)
    x1 = [add_dict({A00_idx[i]: 1}, {A11_idx[i]: 1}) for i in range(4)]
    y1 = [add_dict({B00_idx[i]: 1}, {B11_idx[i]: 1}) for i in range(4)]
    z1_to_c = [
        {C00_idx[0]: 1, C11_idx[0]: 1},
        {C00_idx[1]: 1, C11_idx[1]: 1},
        {C00_idx[2]: 1, C11_idx[2]: 1},
        {C00_idx[3]: 1, C11_idx[3]: 1},
    ]
    
    # P2 = (A10+A11)B00
    x2 = [add_dict({A10_idx[i]: 1}, {A11_idx[i]: 1}) for i in range(4)]
    y2 = [{B00_idx[i]: 1 for i in range(4)} for _ in range(4)]
    z2_to_c = [
        {C10_idx[0]: 1},
        {C10_idx[1]: 1},
        {C10_idx[2]: 1},
        {C10_idx[3]: 1},
    ]
    
    # P3 = A00(B01-B11)
    x3 = [{A00_idx[i]: 1 for i in range(4)} for _ in range(4)]
    y3 = [add_dict({B01_idx[i]: 1}, {B11_idx[i]: -1}) for i in range(4)]
    z3_to_c = [
        {C01_idx[0]: 1},
        {C01_idx[1]: 1},
        {C01_idx[2]: 1},
        {C01_idx[3]: 1},
    ]
    
    # P4 = A11(B10-B00)
    x4 = [{A11_idx[i]: 1 for i in range(4)} for _ in range(4)]
    y4 = [add_dict({B10_idx[i]: 1}, {B00_idx[i]: -1}) for i in range(4)]
    z4_to_c = [
        {C00_idx[0]: 1, C10_idx[0]: 1},
        {C00_idx[1]: 1, C10_idx[1]: 1},
        {C00_idx[2]: 1, C10_idx[2]: 1},
        {C00_idx[3]: 1, C10_idx[3]: 1},
    ]
    
    # P5 = (A00+A01)B11
    x5 = [add_dict({A00_idx[i]: 1}, {A01_idx[i]: 1}) for i in range(4)]
    y5 = [{B11_idx[i]: 1 for i in range(4)} for _ in range(4)]
    z5_to_c = [
        {C00_idx[0]: -1, C01_idx[0]: 1},
        {C00_idx[1]: -1, C01_idx[1]: 1},
        {C00_idx[2]: -1, C01_idx[2]: 1},
        {C00_idx[3]: -1, C01_idx[3]: 1},
    ]
    
    # P6 = (A10-A00)(B00+B01)
    x6 = [add_dict({A10_idx[i]: 1}, {A00_idx[i]: -1}) for i in range(4)]
    y6 = [add_dict({B00_idx[i]: 1}, {B01_idx[i]: 1}) for i in range(4)]
    z6_to_c = [
        {C11_idx[0]: 1},
        {C11_idx[1]: 1},
        {C11_idx[2]: 1},
        {C11_idx[3]: 1},
    ]
    
    # P7 = (A01-A11)(B10+B11)
    x7 = [add_dict({A01_idx[i]: 1}, {A11_idx[i]: -1}) for i in range(4)]
    y7 = [add_dict({B10_idx[i]: 1}, {B11_idx[i]: 1}) for i in range(4)]
    z7_to_c = [
        {C00_idx[0]: 1, C11_idx[0]: 1},
        {C00_idx[1]: 1, C11_idx[1]: 1},
        {C00_idx[2]: 1, C11_idx[2]: 1},
        {C00_idx[3]: 1, C11_idx[3]: 1},
    ]
    
    outer_products = [
        (x1, y1, z1_to_c),
        (x2, y2, z2_to_c),
        (x3, y3, z3_to_c),
        (x4, y4, z4_to_c),
        (x5, y5, z5_to_c),
        (x6, y6, z6_to_c),
        (x7, y7, z7_to_c),
    ]
    
    # Inner Strassen terms for 2x2 multiplication X*Y = Z.
    # Each term: (x_positions, y_positions, z_positions)
    # positions are lists of (entry_index, coefficient)
    inner_terms = [
        ([((0,1),(3,1))], [((0,1),(3,1))], [(0,1),(3,1)]),
        ([((2,1),(3,1))], [((0,1))], [(2,1),(3,-1)]),
        ([((0,1))], [((1,1),(3,-1))], [(1,1),(3,1)]),
        ([((3,1))], [((2,1),(0,-1))], [(0,1),(2,1)]),
        ([((0,1),(1,1))], [((3,1))], [(0,-1),(1,1)]),
        ([((2,1),(0,-1))], [((0,1),(1,1))], [(3,1)]),
        ([((1,1),(3,-1))], [((2,1),(3,1))], [(0,1)]),
    ]
    
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for outer_idx, (x_coeffs, y_coeffs, z_to_c) in enumerate(outer_products):
        for inner_idx, (x_pos, y_pos, z_pos) in enumerate(inner_terms):
            u_vec = {}
            for (x_entry, x_coeff) in x_pos:
                for a_idx, a_coeff in x_coeffs[x_entry].items():
                    u_vec[a_idx] = u_vec.get(a_idx, 0) + x_coeff * a_coeff
            
            v_vec = {}
            for (y_entry, y_coeff) in y_pos:
                for b_idx, b_coeff in y_coeffs[y_entry].items():
                    v_vec[b_idx] = v_vec.get(b_idx, 0) + y_coeff * b_coeff
            
            w_vec = {}
            for (z_entry, z_coeff) in z_pos:
                for c_idx, c_coeff in z_to_c[z_entry].items():
                    w_vec[c_idx] = w_vec.get(c_idx, 0) + z_coeff * c_coeff
            
            for a_idx, val in u_vec.items():
                U[a_idx][col] = val
            for b_idx, val in v_vec.items():
                V[b_idx][col] = val
            for c_idx, val in w_vec.items():
                i = c_idx // 4
                k = c_idx % 4
                w_row = k * n + i
                W[w_row][col] = val
            
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
