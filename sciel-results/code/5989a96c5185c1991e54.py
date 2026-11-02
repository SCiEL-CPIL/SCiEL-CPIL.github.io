"""Matrix multiplication 4x4 by 4x6 - Strassen-based decomposition."""
import time

DIMENSIONS = [4, 4, 6]

def construct():
    """
    Use Strassen's algorithm to decompose 4x4 * 4x6 matrix multiplication.
    Strassen reduces 2x2 matrix multiplication to 7 multiplications.
    We can apply this recursively or use it as a building block.
    
    For 4x4 * 4x6, we can think of this as:
    - Split A (4x4) into 2x2 blocks: A11, A12, A21, A22 (each 2x2)
    - Split B (4x6) into 2x3 blocks: B11, B12, B21, B22 (each 2x3)
    - Result C (4x6) has 2x3 blocks: C11, C12, C21, C22 (each 2x3)
    
    Using Strassen's formulas:
    C11 = (A11 + A22)(B11 + B22) - (A22 - A21)(B11 - B21)
    C12 = A12(B21 + B22) - (A11 + A12)B11
    C21 = (A11 + A21)B22 - (A11 - A22)(B11 + B22)  
    C22 = (A11 + A22)(B11 - B22) + (A11 - A22 + A21)(B22 - B21)
    
    This gives us 7 multiplications of 2x2 by 2x3 matrices.
    Each 2x2 * 2x3 multiplication can be done with 6 scalar multiplications (schoolbook).
    Total: 7 * 6 = 42 rank-1 terms.
    """
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # We'll build the decomposition using Strassen's algorithm
    # The key insight: each Strassen multiplication (2x2 * 2x3) needs 6 rank-1 terms
    # 7 such multiplications * 6 = 42 total rank-1 terms
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]  # 16 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 24 x 42
    
    # Column offset for each of the 7 Strassen multiplications
    col_base = [0, 6, 12, 18, 24, 30, 36]
    
    # For each Strassen multiplication, we need to set up U, V, W entries
    # Each 2x2 * 2x3 multiplication has 6 scalar multiplications
    
    # Strassen's 7 products:
    # M1 = (A11+A22) * (B11+B22)
    # M2 = (A11+A22) * (B21-B22)
    # M3 = (A11-A22) * (B11+B12)
    # M4 = (A11+A21) * (B22-B21)
    # M5 = (A22+A21) * (B11-B21)
    # M6 = (A12-A22) * (B11+B12)
    # M7 = (A12+A21) * (B21-B22)
    
    # Output blocks:
    # C11 = M1 - M2 + M4
    # C12 = M3 - M6
    # C21 = M5 + M6
    # C22 = M1 - M3 + M7
    
    # Now let's define the 6 scalar multiplications for each 2x2*2x3 product
    # A block is 2x2: [[a00, a01], [a10, a11]]
    # B block is 2x3: [[b00, b01, b02], [b10, b11, b12]]
    # Result is 2x3: [[r00, r01, r02], [r10, r11, r12]]
    
    # The 6 scalar multiplications for one 2x2*2x3:
    # r00 = a00*b00 + a01*b10
    # r01 = a00*b01 + a01*b11
    # r02 = a00*b02 + a01*b12
    # r10 = a10*b00 + a11*b10
    # r11 = a10*b01 + a11*b11
    # r12 = a10*b02 + a11*b12
    
    # For each of the 7 Strassen products, we need to express the 2x2 and 2x3
    # blocks as linear combinations of the original matrix entries.
    
    # Let's index the original matrices:
    # A is 4x4: A[i,j] for i,j in 0..3
    # B is 4x6: B[j,k] for j in 0..3, k in 0..5
    # C is 4x6: C[i,k] for i in 0..3, k in 0..5
    
    # Block decomposition:
    # A11 = A[0:2, 0:2], A12 = A[0:2, 2:4]
    # A21 = A[2:4, 0:2], A22 = A[2:4, 2:4]
    # B11 = B[0:2, 0:3], B12 = B[0:2, 3:6]
    # B21 = B[2:4, 0:3], B22 = B[2:4, 3:6]
    # C11 = C[0:2, 0:3], C12 = C[0:2, 3:6]
    # C21 = C[2:4, 0:3], C22 = C[2:4, 3:6]
    
    # For each Strassen product, define the coefficients for A and B blocks
    
    # Product definitions: (A_coeffs, B_coeffs, output_block, output_signs)
    # A_coeffs: how to form the 2x2 A block from A11, A12, A21, A22
    # B_coeffs: how to form the 2x3 B block from B11, B12, B21, B22
    
    strassen_products = [
        # M1 = (A11+A22) * (B11+B22)
        {'A': {('A11', 0, 0): 1, ('A11', 0, 1): 0, ('A11', 1, 0): 0, ('A11', 1, 1): 0,
               ('A12', 0, 0): 0, ('A12', 0, 1): 0, ('A12', 1, 0): 0, ('A12', 1, 1): 0,
               ('A21', 0, 0): 0, ('A21', 0, 1): 0, ('A21', 1, 0): 0, ('A21', 1, 1): 0,
               ('A22', 0, 0): 1, ('A22', 0, 1): 0, ('A22', 1, 0): 0, ('A22', 1, 1): 1},
         'B': {('B11', 0, 0): 1, ('B11', 0, 1): 1, ('B11', 0, 2): 1,
               ('B12', 0, 0): 0, ('B12', 0, 1): 0, ('B12', 0, 2): 0,
               ('B21', 0, 0): 0, ('B21', 0, 1): 0, ('B21', 0, 2): 0,
               ('B22', 0, 0): 1, ('B22', 0, 1): 1, ('B22', 0, 2): 1,
               ('B11', 1, 0): 0, ('B11', 1, 1): 0, ('B11', 1, 2): 0,
               ('B12', 1, 0): 0, ('B12', 1, 1): 0, ('B12', 1, 2): 0,
               ('B21', 1, 0): 0, ('B21', 1, 1): 0, ('B21', 1, 2): 0,
               ('B22', 1, 0): 1, ('B22', 1, 1): 1, ('B22', 1, 2): 1},
         'out': 'C11', 'sign': 1},
        
        # M2 = (A11+A22) * (B21-B22)
        {'A': {('A11', 0, 0): 1, ('A11', 0, 1): 0, ('A11', 1, 0): 0, ('A11', 1, 1): 0,
               ('A12', 0, 0): 0, ('A12', 0, 1): 0, ('A12', 1, 0): 0, ('A12', 1, 1): 0,
               ('A21', 0, 0): 0, ('A21', 0, 1): 0, ('A21', 1, 0): 0, ('A21', 1, 1): 0,
               ('A22', 0, 0): 1, ('A22', 0, 1): 0, ('A22', 1, 0): 0, ('A22', 1, 1): 1},
         'B': {('B11', 0, 0): 0, ('B11', 0, 1): 0, ('B11', 0, 2): 0,
               ('B12', 0, 0): 0, ('B12', 0, 1): 0, ('B12', 0, 2): 0,
               ('B21', 0, 0): 1, ('B21', 0, 1): 1, ('B21', 0, 2): 1,
               ('B22', 0, 0): -1, ('B22', 0, 1): -1, ('B22', 0, 2): -1,
               ('B11', 1, 0): 0, ('B11', 1, 1): 0, ('B11', 1, 2): 0,
               ('B12', 1, 0): 0, ('B12', 1, 1): 0, ('B12', 1, 2): 0,
               ('B21', 1, 0): 1, ('B21', 1, 1): 1, ('B21', 1, 2): 1,
               ('B22', 1, 0): -1, ('B22', 1, 1): -1, ('B22', 1, 2): -1},
         'out': 'C11', 'sign': -1},
        
        # M3 = (A11-A22) * (B11+B12)
        {'A': {('A11', 0, 0): 1, ('A11', 0, 1): 0, ('A11', 1, 0): 0, ('A11', 1, 1): 0,
               ('A12', 0, 0): 0, ('A12', 0, 1): 0, ('A12', 1, 0): 0, ('A12', 1, 1): 0,
               ('A21', 0, 0): 0, ('A21', 0, 1): 0, ('A21', 1, 0): 0, ('A21', 1, 1): 0,
               ('A22', 0, 0): -1, ('A22', 0, 1): 0, ('A22', 1, 0): 0, ('A22', 1, 1): -1},
         'B': {('B11', 0, 0): 1, ('B11', 0, 1): 1, ('B11', 0, 2): 1,
               ('B12', 0, 0): 1, ('B12', 0, 1): 1, ('B12', 0, 2): 1,
               ('B21', 0, 0): 0, ('B21', 0, 1): 0, ('B21', 0, 2): 0,
               ('B22', 0, 0): 0, ('B22', 0, 1): 0, ('B22', 0, 2): 0,
               ('B11', 1, 0): 0, ('B11', 1, 1): 0, ('B11', 1, 2): 0,
               ('B12', 1, 0): 0, ('B12', 1, 1): 0, ('B12', 1, 2): 0,
               ('B21', 1, 0): 0, ('B21', 1, 1): 0, ('B21', 1, 2): 0,
               ('B22', 1, 0): 0, ('B22', 1, 1): 0, ('B22', 1, 2): 0},
         'out': 'C12', 'sign': -1},
        
        # M4 = (A11+A21) * (B22-B21)
        {'A': {('A11', 0, 0): 1, ('A11', 0, 1): 0, ('A11', 1, 0): 0, ('A11', 1, 1): 0,
               ('A12', 0, 0): 0, ('A12', 0, 1): 0, ('A12', 1, 0): 0, ('A12', 1, 1): 0,
               ('A21', 0, 0): 1, ('A21', 0, 1): 0, ('A21', 1, 0): 0, ('A21', 1, 1): 1,
               ('A22', 0, 0): 0, ('A22', 0, 1): 0, ('A22', 1, 0): 0, ('A22', 1, 1): 0},
         'B': {('B11', 0, 0): 0, ('B11', 0, 1): 0, ('B11', 0, 2): 0,
               ('B12', 0, 0): 0, ('B12', 0, 1): 0, ('B12', 0, 2): 0,
               ('B21', 0, 0): -1, ('B21', 0, 1): -1, ('B21', 0, 2): -1,
               ('B22', 0, 0): 1, ('B22', 0, 1): 1, ('B22', 0, 2): 1,
               ('B11', 1, 0): 0, ('B11', 1, 1): 0, ('B11', 1, 2): 0,
               ('B12', 1, 0): 0, ('B12', 1, 1): 0, ('B12', 1, 2): 0,
               ('B21', 1, 0): -1, ('B21', 1, 1): -1, ('B21', 1, 2): -1,
               ('B22', 1, 0): 1, ('B22', 1, 1): 1, ('B22', 1, 2): 1},
         'out': 'C11', 'sign': 1},
        
        # M5 = (A22+A21) * (B11-B21)
        {'A': {('A11', 0, 0): 0, ('A11', 0, 1): 0, ('A11', 1, 0): 0, ('A11', 1, 1): 0,
               ('A12', 0, 0): 0, ('A12', 0, 1): 0, ('A12', 1, 0): 0, ('A12', 1, 1): 0,
               ('A21', 0, 0): 1, ('A21', 0, 1): 0, ('A21', 1, 0): 0, ('A21', 1, 1): 1,
               ('A22', 0, 0): 1, ('A22', 0, 1): 0, ('A22', 1, 0): 0, ('A22', 1, 1): 1},
         'B': {('B11', 0, 0): 1, ('B11', 0, 1): 1, ('B11', 0, 2): 1,
               ('B12', 0, 0): 0, ('B12', 0, 1): 0, ('B12', 0, 2): 0,
               ('B21', 0, 0): -1, ('B21', 0, 1): -1, ('B21', 0, 2): -1,
               ('B22', 0, 0): 0, ('B22', 0, 1): 0, ('B22', 0, 2): 0,
               ('B11', 1, 0): 0, ('B11', 1, 1): 0, ('B11', 1, 2): 0,
               ('B12', 1, 0): 0, ('B12', 1, 1): 0, ('B12', 1, 2): 0,
               ('B21', 1, 0): -1, ('B21', 1, 1): -1, ('B21', 1, 2): -1,
               ('B22', 1, 0): 0, ('B22', 1, 1): 0, ('B22', 1, 2): 0},
         'out': 'C21', 'sign': 1},
        
        # M6 = (A12-A22) * (B11+B12)
        {'A': {('A11', 0, 0): 0, ('A11', 0, 1): 0, ('A11', 1, 0): 0, ('A11', 1, 1): 0,
               ('A12', 0, 0): 1, ('A12', 0, 1): 0, ('A12', 1, 0): 0, ('A12', 1, 1): 1,
               ('A21', 0, 0): 0, ('A21', 0, 1): 0, ('A21', 1, 0): 0, ('A21', 1, 1): 0,
               ('A22', 0, 0): -1, ('A22', 0, 1): 0, ('A22', 1, 0): 0, ('A22', 1, 1): -1},
         'B': {('B11', 0, 0): 1, ('B11', 0, 1): 1, ('B11', 0, 2): 1,
               ('B12', 0, 0): 1, ('B12', 0, 1): 1, ('B12', 0, 2): 1,
               ('B21', 0, 0): 0, ('B21', 0, 1): 0, ('B21', 0, 2): 0,
               ('B22', 0, 0): 0, ('B22', 0, 1): 0, ('B22', 0, 2): 0,
               ('B11', 1, 0): 0, ('B11', 1, 1): 0, ('B11', 1, 2): 0,
               ('B12', 1, 0): 0, ('B12', 1, 1): 0, ('B12', 1, 2): 0,
               ('B21', 1, 0): 0, ('B21', 1, 1): 0, ('B21', 1, 2): 0,
               ('B22', 1, 0): 0, ('B22', 1, 1): 0, ('B22', 1, 2): 0},
         'out': 'C21', 'sign': 1},
        
        # M7 = (A12+A21) * (B21-B22)
        {'A': {('A11', 0, 0): 0, ('A11', 0, 1): 0, ('A11', 1, 0): 0, ('A11', 1, 1): 0,
               ('A12', 0, 0): 1, ('A12', 0, 1): 0, ('A12', 1, 0): 0, ('A12', 1, 1): 1,
               ('A21', 0, 0): 1, ('A21', 0, 1): 0, ('A21', 1, 0): 0, ('A21', 1, 1): 1,
               ('A22', 0, 0): 0, ('A22', 0, 1): 0, ('A22', 1, 0): 0, ('A22', 1, 1): 0},
         'B': {('B11', 0, 0): 0, ('B11', 0, 1): 0, ('B11', 0, 2): 0,
               ('B12', 0, 0): 0, ('B12', 0, 1): 0, ('B12', 0, 2): 0,
               ('B21', 0, 0): 1, ('B21', 0, 1): 1, ('B21', 0, 2): 1,
               ('B22', 0, 0): -1, ('B22', 0, 1): -1, ('B22', 0, 2): -1,
               ('B11', 1, 0): 0, ('B11', 1, 1): 0, ('B11', 1, 2): 0,
               ('B12', 1, 0): 0, ('B12', 1, 1): 0, ('B12', 1, 2): 0,
               ('B21', 1, 0): 1, ('B21', 1, 1): 1, ('B21', 1, 2): 1,
               ('B22', 1, 0): -1, ('B22', 1, 1): -1, ('B22', 1, 2): -1},
         'out': 'C22', 'sign': 1},
    ]
    
    # For each product and each of its 6 scalar multiplications, build U, V, W entries
    
    # The 6 scalar multiplications in a 2x2 * 2x3 product:
    # (a_row, a_col, b_row, b_col, out_row, out_col)
    # a is 2x2, b is 2x3, result is 2x3
    scalar_mults = [
        (0, 0, 0, 0, 0, 0),  # r00 += a00*b00
        (0, 1, 1, 0, 0, 0),  # r00 += a01*b10
        (0, 0, 0, 1, 0, 1),  # r01 += a00*b01
        (0, 1, 1, 1, 0, 1),  # r01 += a01*b11
        (0, 0, 0, 2, 0, 2),  # r02 += a00*b02
        (0, 1, 1, 2, 0, 2),  # r02 += a01*b12
        (1, 0, 0, 0, 1, 0),  # r10 += a10*b00
        (1, 1, 1, 0, 1, 0),  # r10 += a11*b10
        (1, 0, 0, 1, 1, 1),  # r11 += a10*b01
        (1, 1, 1, 1, 1, 1),  # r11 += a11*b11
        (1, 0, 0, 2, 1, 2),  # r12 += a10*b02
        (1, 1, 1, 2, 1, 2),  # r12 += a11*b12
    ]
    
    # Map block names to actual indices in the original matrices
    # A blocks (2x2 each):
    # A11: rows 0-1, cols 0-1
    # A12: rows 0-1, cols 2-3
    # A21: rows 2-3, cols 0-1
    # A22: rows 2-3, cols 2-3
    
    # B blocks (2x3 each):
    # B11: rows 0-1, cols 0-2
    # B12: rows 0-1, cols 3-5
    # B21: rows 2-3, cols 0-2
    # B22: rows 2-3, cols 3-5
    
    # C blocks (2x3 each):
    # C11: rows 0-1, cols 0-2
    # C12: rows 0-1, cols 3-5
    # C21: rows 2-3, cols 0-2
    # C22: rows 2-3, cols 3-5
    
    def block_A_idx(block_name, r, c):
        """Map block name and local (r,c) to global row,col in A."""
        if block_name == 'A11':
            return (r, c)
        elif block_name == 'A12':
            return (r, c + 2)
        elif block_name == 'A21':
            return (r + 2, c)
        elif block_name == 'A22':
            return (r + 2, c + 2)
    
    def block_B_idx(block_name, r, c):
        """Map block name and local (r,c) to global row,col in B."""
        if block_name == 'B11':
            return (r, c)
        elif block_name == 'B12':
            return (r, c + 3)
        elif block_name == 'B21':
            return (r + 2, c)
        elif block_name == 'B22':
            return (r + 2, c + 3)
    
    def block_C_idx(block_name, r, c):
        """Map block name and local (r,c) to global row,col in C."""
        if block_name == 'C11':
            return (r, c)
        elif block_name == 'C12':
            return (r, c + 3)
        elif block_name == 'C21':
            return (r + 2, c)
        elif block_name == 'C22':
            return (r + 2, c + 3)
    
    # Now build the decomposition
    for prod_idx, prod in enumerate(strassen_products):
        base_col = col_base[prod_idx]
        out_block = prod['out']
        out_sign = prod['sign']
        
        # For each of the 12 scalar multiplications
        for mult_idx, (ar, ac, br, bc, orow, ocol) in enumerate(scalar_mults):
            col = base_col + mult_idx
            
            # U part: coefficient of A entries
            # The local A block has entries at positions determined by the A_coeffs
            # We need to sum contributions from all blocks that contribute to this local position
            
            a_coeff = 0
            for block_name in ['A11', 'A12', 'A21', 'A22']:
                coeff = prod['A'].get((block_name, ar, ac), 0)
                if coeff != 0:
                    gr, gc = block_A_idx(block_name, ar, ac)
                    U[gr * m + gc][col] += coeff
            
            # V part: coefficient of B entries
            b_coeff = 0
            for block_name in ['B11', 'B12', 'B21', 'B22']:
                coeff = prod['B'].get((block_name, br, bc), 0)
                if coeff != 0:
                    gr, gc = block_B_idx(block_name, br, bc)
                    V[gr * p + gc][col] += coeff
            
            # W part: output entry
            gr, gc = block_C_idx(out_block, orow, ocol)
            W[gc * n + gr][col] += out_sign
    
    return {"U": U, "V": V, "W": W}
