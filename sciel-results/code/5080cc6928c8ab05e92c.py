import numpy as np
import time

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START

def construct():
    """
    Construct a rank-88 tensor decomposition for 3x4 by 4x8 matrix multiplication.
    
    Strategy:
    1. Split the multiplication into 8 independent 3x2 by 2x2 multiplications.
       A (3x4) is split into A1 (3x2) and A2 (3x2).
       B (4x8) is split into 8 blocks of 2x2: B11, B12, B13, B14 (top 2 rows) and B21, B22, B23, B24 (bottom 2 rows).
       C = A1 * [B11 B12 B13 B14] + A2 * [B21 B22 B23 B24].
       Each term is a 3x2 by 2x2 multiplication.
       
    2. For each 3x2 by 2x2 multiplication, use a hybrid decomposition:
       - Split the 3x2 matrix into a 2x2 top part and a 1x2 bottom part.
       - The 2x2 by 2x2 part uses Strassen's algorithm (rank 7).
       - The 1x2 by 2x2 part uses schoolbook (rank 4).
       - Total rank per 3x2x2 block is 7 + 4 = 11.
       
    3. Total rank is 8 blocks * 11 rank/block = 88.
    """
    n, m, p = DIMENSIONS  # n=3, m=4, p=8
    
    # We will construct U, V, W matrices.
    # U: (n*m, rank) = (12, 88)
    # V: (m*p, rank) = (32, 88)
    # W: (p*n, rank) = (24, 88)
    
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    current_col = 0
    
    # Define the Strassen decomposition for 2x2x2
    # C = A * B
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # Strassen products:
    # p1 = (a+d)(e+h)
    # p2 = (c+d)e
    # p3 = a(f-h)
    # p4 = d(g-e)
    # p5 = (a+b)h
    # p6 = (c-a)(e+f)
    # p7 = (b-d)(g+h)
    # C = [[p1+p4-p5+p7, p3+p5], [p2+p4, p1-p2+p3+p6]]
    
    # We need to map these to U, V, W for the 2x2x2 tensor.
    # The tensor T[i, j, k] for 2x2x2.
    # U indices: i*2 + j (i is row of A, j is col of A)
    # V indices: j*2 + k (j is row of B, k is col of B)
    # W indices: k*2 + i (k is col of C, i is row of C)
    
    # Strassen terms for 2x2x2:
    # Each term c corresponds to a product U[i,j] * V[j,k] * W[k,i]
    # We need to define the coefficients for U, V, W for each of the 7 Strassen terms.
    
    # Let's define the 7 terms explicitly.
    # Term 0: (a+d)(e+h)
    # U: a=1, d=1 -> U[0,0]=1, U[1,1]=1
    # V: e=1, h=1 -> V[0,0]=1, V[1,1]=1
    # W: contributes to C[0,0] and C[1,1] with +1, and C[0,1], C[1,0] with 0?
    # Wait, the W matrix in the decomposition represents the linear combination in C.
    # For a rank-1 term, C = u * v * w.
    # Here, the "product" is the scalar multiplication of the transformed A and B.
    # The W matrix entries are the coefficients that sum up to form C.
    
    # Let's re-verify the decomposition format.
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i, j, k]
    # For 2x2x2, T[i, j, k] is 1 if the term A[i,j]*B[j,k] is part of C[i,k].
    # Actually, the tensor T is just 1 for all i,j,k in the schoolbook sense?
    # No, the tensor T[i, j, k] for matrix multiplication is 1 if (i, j, k) is a valid index.
    # The condition is that the sum reconstructs the multiplication.
    
    # For Strassen, we define 7 rank-1 tensors.
    # For each term t, we have vectors u_t, v_t, w_t.
    # The contribution to C[i,k] is sum_t u_t[i] * v_t[k] * w_t[i,k]? No.
    # The standard form is:
    # C = sum_t (u_t^T A v_t) (w_t^T ...)? No.
    
    # Let's stick to the U, V, W matrix definition.
    # For a specific rank-1 term c:
    # U[:, c] is a vector of length 4 (for 2x2 A).
    # V[:, c] is a vector of length 4 (for 2x2 B).
    # W[:, c] is a vector of length 4 (for 2x2 C).
    # The term contributes to C[i, k] the value:
    # (sum_{j} U[i,j] * ... )? No.
    # The contraction is:
    # C[i, k] += sum_j U[i*2+j, c] * V[j*2+k, c] * W[k*2+i, c]
    # This must equal the Strassen combination.
    
    # Let's define the 7 Strassen terms for 2x2x2.
    # We will create a helper to add a term to U, V, W for a 2x2x2 block.
    
    def add_strassen_term(offset_u, offset_v, offset_w, col_idx):
        # offset_u: index in U for the top-left of the 2x2 A block
        # offset_v: index in V for the top-left of the 2x2 B block
        # offset_w: index in W for the top-left of the 2x2 C block
        
        # Term 1: (a+d)(e+h)
        # U: a=1, d=1
        U[offset_u + 0][col_idx] += 1
        U[offset_u + 3][col_idx] += 1
        # V: e=1, h=1
        V[offset_v + 0][col_idx] += 1
        V[offset_v + 3][col_idx] += 1
        # W: C[0,0] += 1, C[1,1] += 1
        W[offset_w + 0][col_idx] += 1
        W[offset_w + 3][col_idx] += 1
        
        col_idx += 1
        
        # Term 2: (c+d)e
        U[offset_u + 2][col_idx] += 1
        U[offset_u + 3][col_idx] += 1
        V[offset_v + 0][col_idx] += 1
        # W: C[1,0] += 1, C[1,1] += 1
        W[offset_w + 2][col_idx] += 1
        W[offset_w + 3][col_idx] += 1
        
        col_idx += 1
        
        # Term 3: a(f-h)
        U[offset_u + 0][col_idx] += 1
        V[offset_v + 1][col_idx] += 1
        V[offset_v + 3][col_idx] += -1
        # W: C[0,1] += 1
        W[offset_w + 1][col_idx] += 1
        
        col_idx += 1
        
        # Term 4: d(g-e)
        U[offset_u + 3][col_idx] += 1
        V[offset_v + 2][col_idx] += 1
        V[offset_v + 0][col_idx] += -1
        # W: C[1,0] += 1, C[1,1] += 1
        W[offset_w + 2][col_idx] += 1
        W[offset_w + 3][col_idx] += 1
        
        col_idx += 1
        
        # Term 5: (a+b)h
        U[offset_u + 0][col_idx] += 1
        U[offset_u + 1][col_idx] += 1
        V[offset_v + 3][col_idx] += 1
        # W: C[0,0] -= 1, C[0,1] += 1
        W[offset_w + 0][col_idx] += -1
        W[offset_w + 1][col_idx] += 1
        
        col_idx += 1
        
        # Term 6: (c-a)(e+f)
        U[offset_u + 2][col_idx] += 1
        U[offset_u + 0][col_idx] += -1
        V[offset_v + 0][col_idx] += 1
        V[offset_v + 1][col_idx] += 1
        # W: C[1,1] += 1
        W[offset_w + 3][col_idx] += 1
        
        col_idx += 1
        
        # Term 7: (b-d)(g+h)
        U[offset_u + 1][col_idx] += 1
        U[offset_u + 3][col_idx] += -1
        V[offset_v + 2][col_idx] += 1
        V[offset_v + 3][col_idx] += 1
        # W: C[0,0] += 1
        W[offset_w + 0][col_idx] += 1
        
        col_idx += 1
        
        return col_idx

    def add_schoolbook_1x2_2x2_term(offset_u, offset_v, offset_w, col_idx):
        # This handles the 1x2 (bottom part of A) by 2x2 (B) multiplication.
        # A_row = [a, b] (1x2)
        # B = [[e, f], [g, h]] (2x2)
        # C_row = [ae+bg, af+bh] (1x2)
        # This is 4 schoolbook terms.
        
        # Term 1: a*e -> C[0]
        U[offset_u][col_idx] += 1
        V[offset_v + 0][col_idx] += 1
        W[offset_w + 0][col_idx] += 1
        col_idx += 1
        
        # Term 2: b*g -> C[0]
        U[offset_u + 1][col_idx] += 1
        V[offset_v + 2][col_idx] += 1
        W[offset_w + 0][col_idx] += 1
        col_idx += 1
        
        # Term 3: a*f -> C[1]
        U[offset_u][col_idx] += 1
        V[offset_v + 1][col_idx] += 1
        W[offset_w + 1][col_idx] += 1
        col_idx += 1
        
        # Term 4: b*h -> C[1]
        U[offset_u + 1][col_idx] += 1
        V[offset_v + 3][col_idx] += 1
        W[offset_w + 1][col_idx] += 1
        col_idx += 1
        
        return col_idx

    # Now, loop over the 8 blocks.
    # B is 4x8.
    # Blocks are 2x2.
    # B = [ B11 B12 B13 B14 ]
    #     [ B21 B22 B23 B24 ]
    # A is 3x4.
    # A = [ A1 A2 ] where A1 is 3x2, A2 is 3x2.
    
    # The multiplication is:
    # C = A1 * [B11 B12 B13 B14] + A2 * [B21 B22 B23 B24]
    
    # Let's process each of the 8 terms: A_part * B_block.
    # A_part is either A1 (rows 0-2, cols 0-1) or A2 (rows 0-2, cols 2-3).
    # B_block is a 2x2 block.
    
    # For A1 (cols 0,1 of A):
    # U offset for A1:
    # A1 is rows 0-2, cols 0-1.
    # In U (12x88), A1 corresponds to indices:
    # i in 0..2, j in 0..1.
    # U index = i*4 + j.
    # So U indices: 0, 1, 4, 5, 8, 9.
    
    # For A2 (cols 2,3 of A):
    # U indices: i*4 + 2, i*4 + 3.
    # So U indices: 2, 3, 6, 7, 10, 11.
    
    # B blocks:
    # B11: rows 0-1, cols 0-1. V indices: j*8 + k. j in 0-1, k in 0-1.
    # V indices: 0, 1, 8, 9.
    # B12: rows 0-1, cols 2-3. V indices: 2, 3, 10, 11.
    # B13: rows 0-1, cols 4-5. V indices: 4, 5, 12, 13.
    # B14: rows 0-1, cols 6-7. V indices: 6, 7, 14, 15.
    # B21: rows 2-3, cols 0-1. V indices: 16, 17, 24, 25.
    # B22: rows 2-3, cols 2-3. V indices: 18, 19, 26, 27.
    # B23: rows 2-3, cols 4-5. V indices: 20, 21, 28, 29.
    # B24: rows 2-3, cols 6-7. V indices: 22, 23, 30, 31.
    
    # C blocks:
    # C is 3x8.
    # W indices: k*3 + i. k in 0..7, i in 0..2.
    # For B11 (cols 0-1 of C): k in 0-1.
    # W indices: 0, 1, 3, 4, 6, 7. (k=0: 0,1,2; k=1: 3,4,5).
    # Wait, W index = k*n + i = k*3 + i.
    # k=0: 0, 1, 2.
    # k=1: 3, 4, 5.
    # So for cols 0-1, W indices are 0..5.
    
    # Let's map the 8 blocks to their offsets.
    
    blocks = [
        # (A_part, B_block_row, B_block_col, C_col_offset)
        # A1 is cols 0-1 of A. A2 is cols 2-3.
        # B_block_row 0 is rows 0-1 of B. B_block_row 1 is rows 2-3.
        # B_block_col 0 is cols 0-1 of B. ...
        
        # A1 * B11 -> C cols 0-1
        ('A1', 0, 0, 0),
        # A1 * B12 -> C cols 2-3
        ('A1', 0, 1, 2),
        # A1 * B13 -> C cols 4-5
        ('A1', 0, 2, 4),
        # A1 * B14 -> C cols 6-7
        ('A1', 0, 3, 6),
        # A2 * B21 -> C cols 0-1
        ('A2', 1, 0, 0),
        # A2 * B22 -> C cols 2-3
        ('A2', 1, 1, 2),
        # A2 * B23 -> C cols 4-5
        ('A2', 1, 2, 4),
        # A2 * B24 -> C cols 6-7
        ('A2', 1, 3, 6),
    ]
    
    for a_part, b_row, b_col, c_col in blocks:
        # Determine U offset for the 2x2 top part and 1x2 bottom part.
        # A is 3x4.
        # A1: cols 0,1. A2: cols 2,3.
        # A rows 0,1 are the "top" 2x2 part. Row 2 is the "bottom" 1x2 part.
        
        if a_part == 'A1':
            # Top 2x2: rows 0,1; cols 0,1.
            # U indices for top:
            # (0,0)->0, (0,1)->1, (1,0)->4, (1,1)->5.
            # We need to map these to a contiguous 2x2 block for the Strassen helper?
            # The helper assumes contiguous indices 0,1,2,3 for the 2x2 matrix.
            # But our U indices are not contiguous.
            # U[0], U[1] are row 0. U[4], U[5] are row 1.
            # We can modify the helper to take a list of indices or just inline it.
            
            # Let's inline the Strassen terms for this specific mapping.
            # Top part indices:
            u_a = 0 # A[0,0]
            u_b = 1 # A[0,1]
            u_c = 4 # A[1,0]
            u_d = 5 # A[1,1]
            
            # Bottom part indices:
            u_e = 8 # A[2,0]
            u_f = 9 # A[2,1]
            
        else: # A2
            # Top 2x2: rows 0,1; cols 2,3.
            u_a = 2 # A[0,2]
            u_b = 3 # A[0,3]
            u_c = 6 # A[1,2]
            u_d = 7 # A[1,3]
            
            # Bottom part indices:
            u_e = 10 # A[2,2]
            u_f = 11 # A[2,3]
        
        # Determine V offset for the 2x2 B block.
        # B is 4x8.
        # b_row 0: rows 0,1. b_row 1: rows 2,3.
        # b_col 0: cols 0,1. b_col 1: cols 2,3. etc.
        
        v_row_start = b_row * 2
        v_col_start = b_col * 2
        
        # V indices for 2x2 block:
        # (r, c) -> v_row_start*8 + v_col_start + c ? No.
        # V index = j*8 + k.
        # j is row of B, k is col of B.
        # j in {v_row_start, v_row_start+1}
        # k in {v_col_start, v_col_start+1}
        
        v_e = v_row_start * 8 + v_col_start # B[j0, k0]
        v_f = v_row_start * 8 + v_col_start + 1 # B[j0, k1]
        v_g = (v_row_start + 1) * 8 + v_col_start # B[j1, k0]
        v_h = (v_row_start + 1) * 8 + v_col_start + 1 # B[j1, k1]
        
        # Determine W offset for the 2x2 C block (actually 2 cols of C).
        # C is 3x8.
        # c_col is the start column of the 2x2 block in C.
        # k in {c_col, c_col+1}
        # i in {0, 1, 2}
        
        # W index = k*3 + i.
        
        # For the top 2x2 part of C (rows 0,1):
        w_00 = c_col * 3 + 0 # C[0, k0]
        w_01 = c_col * 3 + 1 # C[0, k1]
        w_10 = (c_col + 1) * 3 + 0 # C[1, k0]
        w_11 = (c_col + 1) * 3 + 1 # C[1, k1]
        
        # For the bottom 1x2 part of C (row 2):
        w_20 = c_col * 3 + 2 # C[2, k0]
        w_21 = (c_col + 1) * 3 + 2 # C[2, k1]
        
        # Now add Strassen terms for the top 2x2 part.
        # We need to map the standard Strassen indices to these specific indices.
        
        # Term 1: (a+d)(e+h)
        U[u_a][current_col] += 1
        U[u_d][current_col] += 1
        V[v_e][current_col] += 1
        V[v_h][current_col] += 1
        W[w_00][current_col] += 1
        W[w_11][current_col] += 1
        current_col += 1
        
        # Term 2: (c+d)e
        U[u_c][current_col] += 1
        U[u_d][current_col] += 1
        V[v_e][current_col] += 1
        W[w_10][current_col] += 1
        W[w_11][current_col] += 1
        current_col += 1
        
        # Term 3: a(f-h)
        U[u_a][current_col] += 1
        V[v_f][current_col] += 1
        V[v_h][current_col] += -1
        W[w_01][current_col] += 1
        current_col += 1
        
        # Term 4: d(g-e)
        U[u_d][current_col] += 1
        V[v_g][current_col] += 1
        V[v_e][current_col] += -1
        W[w_10][current_col] += 1
        W[w_11][current_col] += 1
        current_col += 1
        
        # Term 5: (a+b)h
        U[u_a][current_col] += 1
        U[u_b][current_col] += 1
        V[v_h][current_col] += 1
        W[w_00][current_col] += -1
        W[w_01][current_col] += 1
        current_col += 1
        
        # Term 6: (c-a)(e+f)
        U[u_c][current_col] += 1
        U[u_a][current_col] += -1
        V[v_e][current_col] += 1
        V[v_f][current_col] += 1
        W[w_11][current_col] += 1
        current_col += 1
        
        # Term 7: (b-d)(g+h)
        U[u_b][current_col] += 1
        U[u_d][current_col] += -1
        V[v_g][current_col] += 1
        V[v_h][current_col] += 1
        W[w_00][current_col] += 1
        current_col += 1
        
        # Now add Schoolbook terms for the bottom 1x2 part.
        # A_bottom = [e, f] (1x2)
        # B = [[e, f], [g, h]] (2x2) -- using different letters to avoid confusion
        # Let's call A_bottom [ae, af] and B [[be, bf], [bg, bh]]
        
        # Term 1: ae * be -> C[2, k0]
        U[u_e][current_col] += 1
        V[v_e][current_col] += 1
        W[w_20][current_col] += 1
        current_col += 1
        
        # Term 2: af * bg -> C[2, k0]
        U[u_f][current_col] += 1
        V[v_g][current_col] += 1
        W[w_20][current_col] += 1
        current_col += 1
        
        # Term 3: ae * bf -> C[2, k1]
        U[u_e][current_col] += 1
        V[v_f][current_col] += 1
        W[w_21][current_col] += 1
        current_col += 1
        
        # Term 4: af * bh -> C[2, k1]
        U[u_f][current_col] += 1
        V[v_h][current_col] += 1
        W[w_21][current_col] += 1
        current_col += 1

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
