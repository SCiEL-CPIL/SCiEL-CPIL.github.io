import numpy as np

DIMENSIONS = [4, 5, 5]

def _get_strassen_2x2():
    """
    Returns U, V, W matrices for 2x2x2 Strassen decomposition.
    Shapes: (4, 7)
    Indices for 2x2 matrix: 00->0, 01->1, 10->2, 11->3
    """
    # Strassen's 7 multiplications for 2x2
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # U, V, W are 4x7
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # Column 0: M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    # Contribution to C:
    # C00 += M1, C01 -= M1, C10 -= M1, C11 += M1 (Wait, check formulas)
    # C00 = M1 - M4 + M7 - M6
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M5 + M6 + M7
    W[0, 0] = 1  # C00
    W[1, 0] = -1 # C01
    W[2, 0] = -1 # C10
    W[3, 0] = 1  # C11

    # Column 1: M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1  # C10

    # Column 2: M3
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1  # C01

    # Column 3: M4
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = -1 # C00
    W[2, 3] = 1  # C10

    # Column 4: M5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[1, 4] = 1  # C01
    W[3, 4] = -1 # C11

    # Column 5: M6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[0, 5] = -1 # C00
    W[3, 5] = 1  # C11

    # Column 6: M7
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1  # C00
    W[3, 6] = 1  # C11

    return U, V, W

def _generate_strassen_4x4_terms():
    """
    Generates U, V, W lists for 4x4x4 Strassen decomposition (rank 49).
    Returns list of tuples (u_vec, v_vec, w_vec) where vecs are length 16.
    """
    U2, V2, W2 = _get_strassen_2x2()
    
    # Map 4x4 indices to 2x2 block indices
    # Block (br, bc) of size 2x2.
    # Global index i = 2*br + ir, j = 2*bc + jc
    # For 4x4, br, bc in {0,1}.
    
    # The 7 outer Strassen steps operate on 2x2 blocks of A, B.
    # Let A_blocks be 2x2 grid of 2x2 matrices.
    # M1 = (A00 + A11) * (B00 + B11)
    # ...
    
    # We need to expand each outer step (which is a 2x2 matrix multiplication of blocks)
    # into 7 inner scalar multiplications.
    
    # The outer Strassen steps are identical to the 2x2 case, but entries are 2x2 matrices.
    # So we can reuse the U2, V2, W2 structure for the "block" level.
    
    # Outer U, V, W are 4x7 (for the 4 blocks).
    # Inner U, V, W are 4x7 (for the 4 scalars in a block).
    
    # Total terms: 7 outer * 7 inner = 49.
    
    terms = []
    
    # Outer coefficients
    # U2[o, k] is coeff for block o in outer step k.
    # V2[o, k] is coeff for block o in outer step k.
    # W2[o, k] is coeff for block o in outer step k.
    
    # Inner coefficients
    # U2[i, m] is coeff for scalar i in inner step m.
    
    # For each outer step k (0..6):
    #   For each inner step m (0..6):
    #     Construct u_vec, v_vec, w_vec of length 16.
    
    for k in range(7):
        # Get outer block coefficients for this step
        u_blocks = U2[:, k] # length 4
        v_blocks = V2[:, k] # length 4
        w_blocks = W2[:, k] # length 4
        
        for m in range(7):
            # Get inner scalar coefficients for this step
            u_scalars = U2[:, m] # length 4
            v_scalars = V2[:, m] # length 4
            w_scalars = W2[:, m] # length 4
            
            u_vec = np.zeros(16, dtype=int)
            v_vec = np.zeros(16, dtype=int)
            w_vec = np.zeros(16, dtype=int)
            
            # Map block index b (0..3) to 2x2 range
            # b=0 -> br=0, bc=0
            # b=1 -> br=0, bc=1
            # b=2 -> br=1, bc=0
            # b=3 -> br=1, bc=1
            
            for b in range(4):
                coeff_u = u_blocks[b]
                coeff_v = v_blocks[b]
                coeff_w = w_blocks[b]
                
                if coeff_u == 0 and coeff_v == 0 and coeff_w == 0:
                    continue
                    
                br = b // 2
                bc = b % 2
                
                # Inner scalars within this block
                for s in range(4):
                    # s=0 -> ir=0, ic=0
                    # s=1 -> ir=0, ic=1
                    # s=2 -> ir=1, ic=0
                    # s=3 -> ir=1, ic=1
                    ir = s // 2
                    ic = s % 2
                    
                    global_i = 2 * br + ir
                    global_j = 2 * bc + ic
                    
                    # For U (A matrix)
                    if coeff_u != 0:
                        u_scalar = u_scalars[s]
                        if u_scalar != 0:
                            u_vec[global_i * 4 + global_j] += coeff_u * u_scalar
                            
                    # For V (B matrix)
                    # B indices: row = bc, col = ?
                    # Wait, B is 4x4.
                    # Outer block index b corresponds to B block (bc, bk)?
                    # In Strassen, B blocks are also 2x2.
                    # b=0 -> B00, b=1 -> B01, b=2 -> B10, b=3 -> B11.
                    # So for B, block b has row block bc, col block bk.
                    # b=0: br=0, bc=0.
                    # b=1: br=0, bc=1.
                    # b=2: br=1, bc=0.
                    # b=3: br=1, bc=1.
                    
                    # Inner scalar s in B block:
                    # s=0 -> ir=0, ic=0.
                    # Global row = 2*bc + ir.
                    # Global col = 2*bk + ic.
                    
                    # Wait, for V (B), the indices are (row, col).
                    # Row index in B corresponds to 'j' in tensor.
                    # Col index in B corresponds to 'k' in tensor.
                    
                    # For block b (which is B_br, bc):
                    # Row block = bc (from b=2*bc+bk? No, b is block index).
                    # Let's define b = 2*br + bc for A.
                    # For B, the block index b corresponds to (row_block, col_block).
                    # In Strassen 2x2, B is split into B11, B12, B21, B22.
                    # My U2, V2, W2 use indices 0,1,2,3 for blocks.
                    # 0: 00, 1: 01, 2: 10, 3: 11.
                    # So for B block b:
                    # b=0 -> row 0, col 0.
                    # b=1 -> row 0, col 1.
                    # b=2 -> row 1, col 0.
                    # b=3 -> row 1, col 1.
                    
                    # So for B block b:
                    # row_block = b // 2
                    # col_block = b % 2
                    
                    # Inner scalar s:
                    # ir, ic.
                    # Global row = 2 * row_block + ir
                    # Global col = 2 * col_block + ic
                    
                    if coeff_v != 0:
                        v_scalar = v_scalars[s]
                        if v_scalar != 0:
                            b_row_block = b // 2
                            b_col_block = b % 2
                            global_row = 2 * b_row_block + ir
                            global_col = 2 * b_col_block + ic
                            v_vec[global_row * 4 + global_col] += coeff_v * v_scalar
                            
                    # For W (C matrix)
                    # C block b:
                    # b=0 -> row 0, col 0.
                    # b=1 -> row 0, col 1.
                    # b=2 -> row 1, col 0.
                    # b=3 -> row 1, col 1.
                    # So row_block = b // 2, col_block = b % 2.
                    # Inner scalar s: ir, ic.
                    # Global row = 2 * row_block + ir
                    # Global col = 2 * col_block + ic
                    
                    if coeff_w != 0:
                        w_scalar = w_scalars[s]
                        if w_scalar != 0:
                            w_row_block = b // 2
                            w_col_block = b % 2
                            global_row = 2 * w_row_block + ir
                            global_col = 2 * w_col_block + ic
                            w_vec[global_row * 4 + global_col] += coeff_w * w_scalar
            
            terms.append((u_vec, v_vec, w_vec))
            
    return terms

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=5, p=5
    
    # 1. Generate Strassen terms for 4x4x4 part (i in 0..3, j in 0..3, k in 0..3)
    strassen_terms = _generate_strassen_4x4_terms()
    
    U_list = []
    V_list = []
    W_list = []
    
    # Add Strassen terms
    for u, v, w in strassen_terms:
        # u, v, w are length 16 (4x4).
        # We need to expand them to length 20 (4x5) for U and V, and 20 (5x4) for W?
        # U is (n*m, rank) = (20, rank).
        # V is (m*p, rank) = (25, rank).
        # W is (p*n, rank) = (20, rank).
        
        # U indices: i*m + j. i in 0..3, j in 0..4.
        # Strassen u_vec is for j in 0..3.
        u_full = np.zeros(20, dtype=int)
        for idx, val in enumerate(u):
            i = idx // 4
            j = idx % 4
            u_full[i * 5 + j] = val
        U_list.append(u_full)
        
        # V indices: j*p + k. j in 0..4, k in 0..4.
        # Strassen v_vec is for j in 0..3, k in 0..3.
        v_full = np.zeros(25, dtype=int)
        for idx, val in enumerate(v):
            j = idx // 4
            k = idx % 4
            v_full[j * 5 + k] = val
        V_list.append(v_full)
        
        # W indices: k*n + i. k in 0..4, i in 0..3.
        # Strassen w_vec is for k in 0..3, i in 0..3.
        w_full = np.zeros(20, dtype=int)
        for idx, val in enumerate(w):
            k = idx // 4
            i = idx % 4
            w_full[k * 4 + i] = val
        W_list.append(w_full)
        
    # 2. Add schoolbook terms for remaining indices
    # Remaining: j=4 or k=4.
    # i in 0..3.
    for i in range(4):
        for j in range(5):
            for k in range(5):
                if j == 4 or k == 4:
                    u_full = np.zeros(20, dtype=int)
                    v_full = np.zeros(25, dtype=int)
                    w_full = np.zeros(20, dtype=int)
                    
                    u_full[i * 5 + j] = 1
                    v_full[j * 5 + k] = 1
                    w_full[k * 4 + i] = 1
                    
                    U_list.append(u_full)
                    V_list.append(v_full)
                    W_list.append(w_full)
                    
    rank = len(U_list)
    
    U = np.array(U_list).T # Shape (20, rank)
    V = np.array(V_list).T # Shape (25, rank)
    W = np.array(W_list).T # Shape (20, rank)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
