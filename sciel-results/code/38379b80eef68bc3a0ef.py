import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = DIMENSIONS
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    rank_444 = 49
    rank_441 = 16
    total_rank = rank_444 + rank_444 + rank_441
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # Strassen factors for 2x2x2
    # U2: 4x7, V2: 4x7, W2: 4x7
    U2 = np.array([
        [1, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, -1, -1]
    ])
    V2 = np.array([
        [1, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, -1, 1, 0, 1]
    ])
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ])
    
    # Define Strassen block combinations for 4x4x4
    # A blocks: A11 (0,0), A12 (0,1), A21 (1,0), A22 (1,1)
    # B blocks: B11 (0,0), B12 (0,1), B21 (1,0), B22 (1,1)
    # Each block is 2x2.
    
    # Strassen terms (X, Y) where X is from A, Y is from B
    # X and Y are linear combinations of blocks.
    # We represent a block combination as a dict: {(block_row, block_col): coeff}
    
    strassen_terms = [
        # 1: (A11+A22)(B11+B22)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # 2: (A21+A22)B11
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}),
        # 3: A11(B12-B22)
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}),
        # 4: A22(B21-B11)
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}),
        # 5: (A11+A12)B22
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # 6: (A21-A11)(B11+B12)
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}),
        # 7: (A12-A22)(B21+B22)
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1})
    ]
    
    # Function to add a 4x4x4 decomposition to U, V, W
    # offset_col: column offset in U, V, W
    # b_col_offset: offset for B columns (0 or 4)
    # c_col_offset: offset for C columns (0 or 4)
    def add_444_decomp(offset_col, b_col_offset, c_col_offset):
        for term_idx, (a_comb, b_comb) in enumerate(strassen_terms):
            # For this Strassen term, we need to compute X * Y
            # X is a 2x2 matrix (linear comb of A blocks)
            # Y is a 2x2 matrix (linear comb of B blocks)
            # We use the rank-7 decomposition for 2x2x2 to compute X*Y
            for s in range(7):
                # u_factor = U2[:, s] (size 4)
                # v_factor = V2[:, s] (size 4)
                # w_factor = W2[:, s] (size 4)
                
                # Construct U column for this rank-1 term
                u_col = np.zeros(n * m)
                v_col = np.zeros(m * p)
                w_col = np.zeros(p * n)
                
                # Map U2 indices to A block indices
                # U2 index 0,1 -> Row 0 of 2x2 block? No, U2 is flattened 2x2.
                # U2[i, s] corresponds to element (i//2, i%2) of the 2x2 matrix X.
                # X is composed of A blocks.
                # X[br, bc] where br, bc in {0,1} is a 2x2 block.
                # The element X[br, bc, r, c] (r,c in {0,1}) is from A block (br, bc) at (r,c).
                
                # For each element in U2 (4 elements), we add to U.
                for u_idx in range(4):
                    val = U2[u_idx, s]
                    if val == 0: continue
                    # u_idx maps to local (lr, lc) in 2x2 X
                    lr = u_idx // 2
                    lc = u_idx % 2
                    # X[lr, lc] is a linear combination of A blocks.
                    # Which A block?
                    # X is the "left" operand of the 2x2 block multiplication.
                    # X[lr, lc] corresponds to the block A[br, bc] where br=lr, bc=lc?
                    # No, X is formed by combining A blocks.
                    # In Strassen, X is a 2x2 matrix of blocks? No, X is a 2x2 matrix of scalars (which are blocks).
                    # Wait, the Strassen algorithm is applied to the 2x2 block structure.
                    # So X is a 2x2 matrix where each entry is a 2x2 block.
                    # So X[0,0] is a 2x2 block, X[0,1] is a 2x2 block, etc.
                    # The "scalar" multiplication in the 2x2 Strassen step is actually a 2x2 matrix multiplication.
                    # So we use the rank-7 decomp for the 2x2 matrix multiplication.
                    
                    # So, for the s-th rank-1 term of the 2x2 multiplication:
                    # The contribution to U comes from the "A" part of the 2x2 multiplication.
                    # The "A" part is the matrix X.
                    # X[br, bc] is the block at (br, bc).
                    # The value at X[br, bc, r, c] is sum_{(abr, abc) in a_comb} coeff * A[abr, abc, r, c].
                    
                    # So, for a given u_idx (which is an index in the 2x2 X matrix),
                    # we need to know which block (br, bc) of X it belongs to.
                    # X is 2x2. u_idx 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1).
                    br = u_idx // 2
                    bc = u_idx % 2
                    
                    # Now, X[br, bc] is a linear combination of A blocks.
                    # We look up a_comb for (br, bc).
                    coeff = a_comb.get((br, bc), 0)
                    if coeff == 0: continue
                    
                    # The element X[br, bc, r, c] is coeff * A[br, bc, r, c].
                    # A[br, bc, r, c] is at global index:
                    # Row: br*2 + r
                    # Col: bc*2 + c
                    # Global U index: (br*2 + r) * 4 + (bc*2 + c)
                    
                    # Wait, U2[u_idx, s] is the coefficient for the element at (u_idx//2, u_idx%2) in the 2x2 matrix.
                    # So we add U2[u_idx, s] * coeff to the corresponding A elements.
                    
                    # The element in A corresponding to X[br, bc, r, c] is A[br*2+r, bc*2+c].
                    # But we are summing over r,c? No, u_idx is a specific element.
                    # u_idx 0 -> (0,0). So r=0, c=0.
                    # u_idx 1 -> (0,1). So r=0, c=1.
                    # u_idx 2 -> (1,0). So r=1, c=0.
                    # u_idx 3 -> (1,1). So r=1, c=1.
                    
                    r_local = u_idx % 2 # Wait, u_idx 0 is (0,0). 1 is (0,1).
                    # So row = u_idx // 2, col = u_idx % 2.
                    r_loc = u_idx // 2
                    c_loc = u_idx % 2
                    
                    global_row = br * 2 + r_loc
                    global_col = bc * 2 + c_loc
                    global_u_idx = global_row * 4 + global_col
                    
                    u_col[global_u_idx] += val * coeff
                
                # Similarly for V
                for v_idx in range(4):
                    val = V2[v_idx, s]
                    if val == 0: continue
                    br = v_idx // 2
                    bc = v_idx % 2
                    coeff = b_comb.get((br, bc), 0)
                    if coeff == 0: continue
                    r_loc = v_idx // 2
                    c_loc = v_idx % 2
                    global_row = br * 2 + r_loc
                    global_col = bc * 2 + c_loc
                    # B global index: row * 9 + (col + b_col_offset)
                    global_v_idx = global_row * 9 + (global_col + b_col_offset)
                    v_col[global_v_idx] += val * coeff
                
                # Similarly for W
                for w_idx in range(4):
                    val = W2[w_idx, s]
                    if val == 0: continue
                    br = w_idx // 2
                    bc = w_idx % 2
                    # W is the output C.
                    # C[br, bc] is the block at (br, bc).
                    # C[br, bc, r, c] corresponds to global C index:
                    # Row: br*2 + r
                    # Col: bc*2 + c
                    # Global W index: (col + c_col_offset) * 4 + row
                    r_loc = w_idx // 2
                    c_loc = w_idx % 2
                    global_row = br * 2 + r_loc
                    global_col = bc * 2 + c_loc
                    global_w_idx = (global_col + c_col_offset) * 4 + global_row
                    w_col[global_w_idx] += val
                
                U[:, offset_col] = u_col
                V[:, offset_col] = v_col
                W[:, offset_col] = w_col
                offset_col += 1
        return offset_col
    
    # Add first 4x4x4 block (B cols 0-3, C cols 0-3)
    col = add_444_decomp(0, 0, 0)
    
    # Add second 4x4x4 block (B cols 4-7, C cols 4-7)
    col = add_444_decomp(col, 4, 4)
    
    # Add 4x4x1 block (B col 8, C col 8)
    for i in range(4):
        for j in range(4):
            u_idx = i * 4 + j
            v_idx = j * 9 + 8
            w_idx = 8 * 4 + i
            U[u_idx, col] = 1
            V[v_idx, col] = 1
            W[w_idx, col] = 1
            col += 1
    
    # Verify
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    # sum_r U[a,r] V[b,r] W[c,r] = 1
    
    # Quick check
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(U[a, :] * V[b, :], W[c, :])
                if abs(val - 1.0) > 1e-9:
                    # Fallback to schoolbook if error
                    rank_school = n * m * p
                    U_school = np.zeros((n * m, rank_school))
                    V_school = np.zeros((m * p, rank_school))
                    W_school = np.zeros((p * n, rank_school))
                    for col_s, (ii, jj, kk) in enumerate([(i2, j2, k2) for i2 in range(n) for j2 in range(m) for k2 in range(p)]):
                        U_school[ii * m + jj, col_s] = 1
                        V_school[jj * p + kk, col_s] = 1
                        W_school[kk * n + ii, col_s] = 1
                    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
