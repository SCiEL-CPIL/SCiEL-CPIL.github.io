"""Matrix multiplication 6x6x6 - Strassen-based decomposition of rank 189."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    n, m, p = 6, 6, 6
    
    # Strassen's algorithm for 2x2 block multiplication
    # Blocks are 3x3
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Each M_s is a 3x3 matrix product, which via schoolbook has 9 terms
    # Total rank = 7 * 9 = 63? No wait, 7 * (3*3*3) = 7*27 = 189
    
    rank = 189
    
    # U: shape (n*m, rank) = (36, 189)
    # V: shape (m*p, rank) = (36, 189)  
    # W: shape (p*n, rank) = (36, 189)
    # W follows trace convention W[k*n+i, r]
    
    U = np.zeros((n*m, rank), dtype=object)
    V = np.zeros((m*p, rank), dtype=object)
    W = np.zeros((p*n, rank), dtype=object)
    
    # For each of the 7 Strassen multiplications M_s = A^{(s)} * B^{(s)}
    # where A^{(s)} is a linear combination of A-blocks and B^{(s)} of B-blocks
    # A^{(s)}_{x,j} = sum_{block} coeff_A * A_{block, x+j, j}  (local coords)
    # B^{(s)}_{j,y} = sum_{block} coeff_B * B_{block, j, y+k}  (local coords)
    
    # Strassen steps: (A_coeffs, B_coeffs)
    # A_coeffs: dict from block_name to coefficient
    # B_coeffs: dict from block_name to coefficient
    # Block names: '11', '12', '21', '22'
    
    strassen_steps = [
        # M1 = (A11+A22)(B11+B22)
        ({'11': 1, '22': 1}, {'11': 1, '22': 1}),
        # M2 = (A21+A22)B11
        ({'21': 1, '22': 1}, {'11': 1}),
        # M3 = A11(B21-B22)
        ({'11': 1}, {'21': 1, '22': -1}),
        # M4 = A22(B12-B11)
        ({'22': 1}, {'12': 1, '11': -1}),
        # M5 = (A11+A12)B22
        ({'11': 1, '12': 1}, {'22': 1}),
        # M6 = (A21-A11)(B11+B12)
        ({'21': 1, '11': -1}, {'11': 1, '12': 1}),
        # M7 = (A12-A22)(B21+B22)
        ({'12': 1, '22': -1}, {'21': 1, '22': 1}),
    ]
    
    # C coefficients: C_block = sum_s coeff_C[s] * M_s
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    c_coeffs = {
        '11': [1, 0, 0, 1, -1, 0, 1],   # M1..M7
        '12': [0, 0, 1, 0, 1, 0, 0],
        '21': [0, 1, 0, 1, 0, 0, 0],
        '22': [1, -1, 1, 0, 0, 1, 0],
    }
    
    # Block to global offset mapping
    # Block 'ij' where i in {1,2} (row block), j in {1,2} (col block)
    # Row offset for block i: (i-1)*3
    # Col offset for block j: (j-1)*3
    
    block_row_offset = {'1': 0, '2': 3}
    block_col_offset = {'1': 0, '2': 3}
    
    col = 0
    for s in range(7):
        a_coeffs, b_coeffs = strassen_steps[s]
        
        # For each term in the schoolbook expansion of M_s = A^(s) * B^(s)
        # M_s[x, y] = sum_j A^(s)[x, j] * B^(s)[j, y]
        for x in range(3):
            for y in range(3):
                for j in range(3):
                    # U part: A^(s)[x, j] = sum_{block_i} a_coeffs[block_i] * A_{block_i}[x, j]
                    # A_{block_i}[x, j] is at global position ((block_row_offset[i]) + x) * m + (block_col_offset[i_col] + j)
                    # Wait, for A matrix: A is n x m = 6 x 6
                    # Block '11' is rows 0-2, cols 0-2
                    # Block '12' is rows 0-2, cols 3-5
                    # Block '21' is rows 3-5, cols 0-2
                    # Block '22' is rows 3-5, cols 3-5
                    # So for block with row-block r and col-block c:
                    #   A_{block}[x, j] is at global row (block_row_offset[r]+x), col (block_col_offset[c]+j)
                    #   U index = row * m + col
                    
                    # For each block in a_coeffs
                    for blk_name, coeff in a_coeffs.items():
                        r_blk, c_blk = blk_name[0], blk_name[1]
                        global_row = block_row_offset[r_blk] + x
                        global_col = block_col_offset[c_blk] + j
                        u_idx = global_row * m + global_col
                        U[u_idx, col] += coeff
                    
                    # V part: B^(s)[j, y] = sum_{block} b_coeffs[block] * B_{block}[j, y]
                    # B is m x p = 6 x 6
                    # Block '11' is rows 0-2, cols 0-2
                    # Block '12' is rows 0-2, cols 3-5
                    # Block '21' is rows 3-5, cols 0-2
                    # Block '22' is rows 3-5, cols 3-5
                    # B_{block}[j, y] is at global row (block_row_offset[r]+j), col (block_col_offset[c]+y)
                    # V index = row * p + col
                    for blk_name, coeff in b_coeffs.items():
                        r_blk, c_blk = blk_name[0], blk_name[1]
                        global_row = block_row_offset[r_blk] + j
                        global_col = block_col_offset[c_blk] + y
                        v_idx = global_row * p + global_col
                        V[v_idx, col] += coeff
                    
                    # W part: M_s[x, y] contributes to C blocks
                    # M_s is the product of two 3x3 matrices, so M_s[x, y] is a 3x3 block result
                    # This 3x3 block corresponds to C blocks based on which C_block it contributes to
                    # Actually, M_s[x,y] is a scalar that goes into the (x,y) position of the 3x3 result block
                    # The 3x3 result block for M_s corresponds to C blocks via c_coeffs
                    # For each C block that M_s contributes to, M_s[x,y] fills position (x,y) in that block
                    for c_blk_name, coeff in c_coeffs.items():
                        if coeff == 0:
                            continue
                        r_blk, c_blk = c_blk_name[0], c_blk_name[1]
                        # C is p x n = 6 x 6
                        # C_{block}[x, y] is at global row (block_row_offset[r]+x), col (block_col_offset[c]+y)
                        # W convention: W[k*n+i, r] where k is row in C (which is col in our notation?)
                        # Let me re-read: W[k*n+i, r]
                        # C = A*B, C is p x n = 6 x 6
                        # C[k, i] where k is row index of C (0..5), i is col index of C (0..5)
                        # W index = k*n + i
                        # In our block notation for C:
                        #   C_{block}[x, y] where x is local row (0..2), y is local col (0..2)
                        #   Global C row = block_row_offset[r] + x
                        #   Global C col = block_col_offset[c] + y
                        #   So k = block_row_offset[r] + x, i = block_col_offset[c] + y
                        #   W index = k*n + i
                        global_k = block_row_offset[r_blk] + x
                        global_i = block_col_offset[c_blk] + y
                        w_idx = global_k * n + global_i
                        W[w_idx, col] += coeff * 1  # The M_s[x,y] term has coefficient 1 in the schoolbook expansion
                    
                    col += 1
    
    # Convert to lists of lists (or keep as numpy arrays - the interface says factor matrices)
    # Let's return as numpy arrays since the task says "factor matrices of shapes (n*m, rank)" etc.
    # But the original returns lists of lists. Let me check - the original returns {"U": U, "V": V, "W": W}
    # where U, V, W are lists of lists. Let me convert.
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    # Convert numpy int/float to Python int where possible
    def convert_matrix(mat):
        result = []
        for row in mat:
            new_row = []
            for val in row:
                if isinstance(val, (np.integer, int)):
                    new_row.append(int(val))
                elif isinstance(val, (np.floating, float)):
                    if val == int(val):
                        new_row.append(int(val))
                    else:
                        new_row.append(val)
                else:
                    new_row.append(val)
            result.append(new_row)
        return result
    
    U_final = convert_matrix(U_list)
    V_final = convert_matrix(V_list)
    W_final = convert_matrix(W_list)
    
    elapsed = time.time() - start_time
    if elapsed > deadline - start_time:
        pass  # Should not happen given the simple construction
    
    return {"U": U_final, "V": V_final, "W": W_final}
# EVOLVE-BLOCK-END
