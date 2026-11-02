"""Matrix multiplication 4x4x7 - using Block Strassen for rank 98."""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START

def _build_block_strassen_4x4x4():
    """
    Build U, V, W for 4x4x4 matrix multiplication using Block Strassen.
    
    We view A as 4x4, B as 4x4, C as 4x4.
    Split into 2x2 blocks of size 2x2:
    A = [[A11, A12],
         [A21, A22]]
    B = [[B11, B12],
         [B21, B22]]
    
    Strassen 7 products (each is 2x2 * 2x2, which uses 7 scalar mults):
    M1 = (A11 + A22)(B11 + B22)
    M2 = (A21 + A22)B11
    M3 = A11(B12 - B22)
    M4 = A22(B21 - B11)
    M5 = (A11 + A12)B22
    M6 = (A21 - A11)(B11 + B12)
    M7 = (A12 - A22)(B21 + B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    Each 2x2*2x2 uses the standard Strassen 7-product formula on 2x2 blocks,
    but since the blocks are 2x2, each "scalar" product in Strassen is actually
    a 2x2*2x2 multiplication, which we further decompose.
    
    Actually, let's think more carefully. We want scalar tensor decomposition.
    A 4x4*4x4 -> 4x4 product has 16*16*16 = 4096 terms in the tensor.
    We want to express it as sum of rank-1 tensors.
    
    Block Strassen: 7 block products, each block is 2x2.
    Each 2x2*2x2 -> 2x2 uses 7 scalar products (Strassen).
    Total: 7 * 7 = 49 scalar products.
    
    So we need to write out all 49 (U,V,W) triples.
    """
    n = m = p = 4
    rank = 49
    U = np.zeros((n*m, rank))  # 16 x 49
    V = np.zeros((m*p, rank))  # 16 x 49
    W = np.zeros((p*n, rank))  # 16 x 49
    
    # Index conventions:
    # U[i*m + j, c] = coefficient for A[i,j] in the c-th scalar product
    # V[j*p + k, c] = coefficient for B[j,k] in the c-th scalar product
    # W[k*n + i, c] = coefficient for C[i,k] in the c-th scalar product
    # (W uses the trace convention W[k*n + i, r])
    
    # Strassen 7 products for 2x2*2x2 -> 2x2:
    # Given P = X * Y where X, Y, P are 2x2
    # X = [[x00, x01], [x10, x11]]
    # Y = [[y00, y01], [y10, y11]]
    # 
    # S1 = (x00+x11)(y00+y11)
    # S2 = (x10+x11)y00
    # S3 = x00(y01-y11)
    # S4 = x11(y10-y00)
    # S5 = (x00+x01)y11
    # S6 = (x10-x00)(y00+y01)
    # S7 = (x01-x11)(y10+y11)
    #
    # p00 = S1 + S4 - S5 + S7
    # p01 = S3 + S5
    # p10 = S2 + S4
    # p11 = S1 - S2 + S3 + S6
    
    # For each of the 7 block-level Strassen products, we have a 2x2*2x2 multiplication
    # which decomposes into 7 scalar products.
    
    # Block-level Strassen:
    # Block indices: (bi, bj) for A block, (bk, bl) for B block
    # A blocks: A11 = rows 0-1, cols 0-1; A12 = rows 0-1, cols 2-3; A21 = rows 2-3, cols 0-1; A22 = rows 2-3, cols 2-3
    # B blocks: B11 = rows 0-1, cols 0-1; B12 = rows 0-1, cols 2-3; B21 = rows 2-3, cols 0-1; B22 = rows 2-3, cols 2-3
    # C blocks: C11 = rows 0-1, cols 0-1; C12 = rows 0-1, cols 2-3; C21 = rows 2-3, cols 0-1; C22 = rows 2-3, cols 2-3
    
    # For a 2x2*2x2 -> 2x2 with input X at (row_offset_x, col_offset_x), Y at (row_offset_y, col_offset_y),
    # output P at (row_offset_p, col_offset_p):
    # X indices: (row_offset_x + dx, col_offset_x + dy) for dx,dy in {0,1}
    # Y indices: (row_offset_y + dx, col_offset_y + dy) for dx,dy in {0,1}
    # P indices: (row_offset_p + dx, col_offset_p + dy) for dx,dy in {0,1}
    
    # The 7 scalar Strassen products for 2x2*2x2:
    # Each scalar product s has:
    # - X coefficients: dict mapping (dx,dy) to coeff
    # - Y coefficients: dict mapping (dx,dy) to coeff
    # - P output: for each (out_dx, out_dy), a coefficient
    
    # Let's define the 7 scalar Strassen products explicitly.
    # s1: X = x00+x11, Y = y00+y11
    # s2: X = x10+x11, Y = y00
    # s3: X = x00, Y = y01-y11
    # s4: X = x11, Y = y10-y00
    # s5: X = x00+x01, Y = y11
    # s6: X = x10-x00, Y = y00+y01
    # s7: X = x01-x11, Y = y10+y11
    
    # P00 = s1 + s4 - s5 + s7
    # P01 = s3 + s5
    # P10 = s2 + s4
    # P11 = s1 - s2 + s3 + s6
    
    # Define the 7 products with their X and Y coefficient dicts and P output dict
    strassen2 = [
        # (X_coeffs, Y_coeffs, P_coeffs)
        # s1
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # s2
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1}),
        # s3
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1}),
        # s4
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        # s5
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        # s6
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # s7
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1, (1,1):-1}),
    ]
    
    # Block-level Strassen: 7 block products
    # M1 = (A11+A22)(B11+B22) -> contributes to C11 (+1) and C22 (+1)
    # M2 = (A21+A22)B11 -> contributes to C21 (+1) and C22 (-1)
    # M3 = A11(B12-B22) -> contributes to C12 (+1) and C22 (+1)
    # M4 = A22(B21-B11) -> contributes to C11 (+1) and C21 (+1)
    # M5 = (A11+A12)B22 -> contributes to C11 (-1) and C12 (+1)
    # M6 = (A21-A11)(B11+B12) -> contributes to C22 (+1)
    # M7 = (A12-A22)(B21+B22) -> contributes to C11 (+1)
    
    # For each block product, we need to specify:
    # - Which A blocks are summed (with signs) to form the left operand
    # - Which B blocks are summed (with signs) to form the right operand
    # - Which C blocks receive the result (with signs)
    
    # Block positions:
    # Block (bi, bj): rows bi*2 to bi*2+1, cols bj*2 to bj*2+1
    # A blocks: A11=(0,0), A12=(0,1), A21=(1,0), A22=(1,1)
    # B blocks: B11=(0,0), B12=(0,1), B21=(1,0), B22=(1,1)
    # C blocks: C11=(0,0), C12=(0,1), C21=(1,0), C22=(1,1)
    
    # For each block product, the left operand is a sum of A blocks, right is sum of B blocks.
    # Then we apply the 2x2*2x2 Strassen to get 7 scalar products.
    # Each scalar product then contributes to specific C blocks.
    
    # Let's define the block products:
    # (left_A_blocks, right_B_blocks, C_contributions)
    # left_A_blocks: list of (block_idx, sign) where block_idx is (bi,bj)
    # right_B_blocks: list of (block_idx, sign)
    # C_contributions: list of (C_block_idx, sign)
    
    block_products = [
        # M1 = (A11+A22)(B11+B22)
        {
            'left': [((0,0),1), ((1,1),1)],
            'right': [((0,0),1), ((1,1),1)],
            'c_out': [((0,0),1), ((1,1),1)]
        },
        # M2 = (A21+A22)B11
        {
            'left': [((1,0),1), ((1,1),1)],
            'right': [((0,0),1)],
            'c_out': [((1,0),1), ((1,1),-1)]
        },
        # M3 = A11(B12-B22)
        {
            'left': [((0,0),1)],
            'right': [((0,1),1), ((1,1),-1)],
            'c_out': [((0,1),1), ((1,1),1)]
        },
        # M4 = A22(B21-B11)
        {
            'left': [((1,1),1)],
            'right': [((1,0),1), ((0,0),-1)],
            'c_out': [((0,0),1), ((1,0),1)]
        },
        # M5 = (A11+A12)B22
        {
            'left': [((0,0),1), ((0,1),1)],
            'right': [((1,1),1)],
            'c_out': [((0,0),-1), ((0,1),1)]
        },
        # M6 = (A21-A11)(B11+B12)
        {
            'left': [((1,0),1), ((0,0),-1)],
            'right': [((0,0),1), ((0,1),1)],
            'c_out': [((1,1),1)]
        },
        # M7 = (A12-A22)(B21+B22)
        {
            'left': [((0,1),1), ((1,1),-1)],
            'right': [((1,0),1), ((1,1),1)],
            'c_out': [((0,0),1)]
        },
    ]
    
    # Now build U, V, W
    col = 0
    for bp in block_products:
        left_blocks = bp['left']   # list of ((bi,bj), sign)
        right_blocks = bp['right'] # list of ((bi,bj), sign)
        c_out = bp['c_out']        # list of ((ci,cj), sign)
        
        for s_idx in range(7):
            x_coeffs, y_coeffs, p_coeffs = strassen2[s_idx]
            # x_coeffs: dict (dx,dy) -> coeff for left 2x2 block
            # y_coeffs: dict (dx,dy) -> coeff for right 2x2 block
            # p_coeffs: dict (dx,dy) -> coeff for output 2x2 block
            
            # For U: sum over all left blocks
            for (bi, bj), block_sign in left_blocks:
                for (dx, dy), x_coeff in x_coeffs.items():
                    # A element at (bi*2 + dx, bj*2 + dy)
                    i = bi * 2 + dx
                    j = bj * 2 + dy
                    U[i * m + j, col] += block_sign * x_coeff
            
            # For V: sum over all right blocks
            for (bi, bj), block_sign in right_blocks:
                for (dx, dy), y_coeff in y_coeffs.items():
                    # B element at (bi*2 + dx, bj*2 + dy)
                    j = bi * 2 + dx
                    k = bj * 2 + dy
                    V[j * p + k, col] += block_sign * y_coeff
            
            # For W: sum over all output C blocks
            for (ci, cj), c_sign in c_out:
                for (dx, dy), p_coeff in p_coeffs.items():
                    # C element at (ci*2 + dx, cj*2 + dy)
                    i_out = ci * 2 + dx
                    k_out = cj * 2 + dy
                    # W[k*n + i, r] convention
                    W[k_out * n + i_out, col] += c_sign * p_coeff
            
            col += 1
    
    return U, V, W

def _build_4x4x7_decomposition():
    """
    Build U, V, W for 4x4*4x7 -> 4x7 using two Block Strassen 4x4x4 decompositions.
    First handles columns 0-3 of B (and output), second handles columns 4-6 (padded to 4x4).
    Total rank: 49 + 49 = 98.
    """
    n, m, p = 4, 4, 7
    rank = 98
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # First 49 columns: handle B columns 0-3, output columns 0-3
    U1, V1, W1 = _build_block_strassen_4x4x4()
    U[:, 0:49] = U1
    V[:, 0:49] = V1
    W[:, 0:49] = W1
    
    # Second 49 columns: handle B columns 4-6 (padded to 0-3 in local 4x4), output columns 4-6
    U2, V2, W2 = _build_block_strassen_4x4x4()
    U[:, 49:98] = U2
    
    # For V: map local column k_local (0-3) to global column k_global
    # k_global = k_local + 4 for k_local in 0,1,2; k_local=3 is padding (ignore)
    for k_local in range(4):
        if k_local < 3:  # valid columns 4,5,6
            k_global = k_local + 4
            V[:, 49 + k_local] = V2[:, k_local]  # This is wrong, need to map properly
        # else: padding, leave as 0
    
    # Actually, V2 has shape (16, 49) where column index in V is j*p + k = j*4 + k_local
    # We need to remap: for each local k, the V entries at positions j*4 + k_local should go to j*4 + k_global
    # Let's redo this properly.
    
    # Reset V and W for the second block
    V[:, 49:98] = 0
    W[:, 49:98] = 0
    
    # V2[j*4 + k_local, col] should become V[j*4 + k_global, col] for valid k_local
    for j in range(4):
        for k_local in range(3):  # 0,1,2 -> global 4,5,6
            k_global = k_local + 4
            V[j*4 + k_global, 49:98] = V2[j*4 + k_local, 49:98]
    # k_local=3 is padding, so V[j*4+3, 49:98] remains 0 (which is correct since there's no column 7)
    
    # W2[k*4 + i, col] should become W[k*4 + i, col] but with k offset
    # W2 has output for columns 0-3 (local), we need columns 4-6 (global)
    # W2[k_local*4 + i, col] -> W[(k_local+4)*4 + i, col] for k_local in 0,1,2
    for k_local in range(3):
        k_global = k_local + 4
        W[k_global*4: (k_global+1)*4, 49:98] = W2[k_local*4: (k_local+1)*4, 49:98]
    # k_local=3 padding, so W[12:16, 49:98] remains 0
    
    return U, V, W

def construct():
    deadline = time.time() + 120
    U, V, W = _build_4x4x7_decomposition()
    
    # Convert to lists of lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
