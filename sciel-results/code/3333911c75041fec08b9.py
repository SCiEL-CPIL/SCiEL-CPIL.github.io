"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    # Decompose M_{2,3,7} into M_{2,3,3} (schoolbook, rank 18) + 2 * M_{2,3,2} (rank 11 each)
    # M_{2,3,2} is isomorphic to M_{2,2,3} = M_{2,2,2} (Strassen, rank 7) + M_{2,2,1} (schoolbook, rank 4)
    n, m, p = 2, 3, 7
    total_rank = 18 + 2 * 11
    U = [[0.0] * total_rank for _ in range(n * m)]  # 6 x total_rank
    V = [[0.0] * total_rank for _ in range(m * p)]  # 21 x total_rank
    W = [[0.0] * total_rank for _ in range(p * n)]  # 14 x total_rank
    
    col = 0
    
    # Part 1: M_{2,3,3} schoolbook (columns 0-2 of B)
    # A is 2x3, B is 3x3, C is 2x3
    # Indices: i in 0..1, j in 0..2, k in 0..2
    # U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1
    for i in range(n):
        for j in range(m):
            for k in range(3):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0  # k is 0..2, corresponds to cols 0-2 of B
                W[k * n + i][col] = 1.0
                col += 1
    
    # Part 2: Two blocks of M_{2,3,2} (columns 3-4 and 5-6 of B)
    # M_{2,3,2}: A is 2x3, B is 3x2, C is 2x2
    # Isomorphic to M_{2,2,3} by swapping dimensions appropriately.
    # Let's work directly with M_{2,3,2}.
    # A (2x3) * B (3x2) = C (2x2)
    # We decompose this into:
    #   - A 2x2 subproblem (rows 0-1, cols 0-1 of A? No, A is 2x3)
    #   Actually, let's split the 3 columns of A (which become 3 rows of B in the other orientation)
    #   Wait, M_{2,3,2} is (2x3) * (3x2).
    #   Let's split the 3 "inner" dimension into 2 + 1.
    #   So we have two sub-problems:
    #   1. A_sub1 (2x2) * B_sub1 (2x2) -> C_part1 (2x2). This is M_{2,2,2}, rank 7 (Strassen).
    #   2. A_sub2 (2x1) * B_sub2 (1x2) -> C_part2 (2x2). This is M_{2,1,2}, rank 4 (Schoolbook).
    #   Total rank = 7 + 4 = 11.
    
    def add_m232_block(offset_in_B, col_start):
        # Adds a decomposition for (2x3) * (3x2) -> (2x2)
        # The 3 rows of B correspond to the 3 columns of A in the standard multiplication A*B.
        # Here B is 3x2, so it has 3 rows and 2 columns.
        # The "inner" dimension is 3 (cols of A / rows of B).
        # Split inner dimension 3 into first 2 and last 1.
        
        # Sub-problem 1: M_{2,2,2} (Strassen)
        # A_sub is 2x2 (first 2 cols of A), B_sub is 2x2 (first 2 rows of B), C_sub is 2x2.
        # Strassen's 7 terms for 2x2 * 2x2:
        # Let A = [[a, b], [c, d]], B = [[e, f], [g, h]]
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-e)
        # M5 = (a+b)h
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        # C = [[M1+M4-M5+M7, M3+M5], [M2+M4, M1-M2+M3+M6]]
        
        strassen_coeffs = [
            # (U_row, U_val, V_row, V_val, W_row, W_val)
            # M1: (A00+A11) * (B00+B11)
            (0, 1.0, 0, 1.0, 0, 1.0), (1, 1.0, 0, 1.0, 0, 1.0),
            (0, 1.0, 1, 1.0, 0, 1.0), (1, 1.0, 1, 1.0, 0, 1.0),
            # M2: (A10+A11) * B00
            (0, 1.0, 0, 1.0, 1, 1.0), (1, 1.0, 0, 1.0, 1, 1.0),
            # M3: A00 * (B01-B11)
            (0, 1.0, 0, 1.0, 0, 1.0), (0, 1.0, 1, -1.0, 0, 1.0),
            # M4: A11 * (B10-B00)
            (1, 1.0, 1, 1.0, 1, 1.0), (1, 1.0, 0, -1.0, 1, 1.0),
            # M5: (A00+A01) * B11
            (0, 1.0, 1, 1.0, 0, -1.0), (1, 1.0, 1, 1.0, 0, -1.0),
            # M6: (A10-A00) * (B00+B01)
            (0, -1.0, 0, 1.0, 1, 1.0), (1, 1.0, 0, 1.0, 1, 1.0),
            (0, -1.0, 0, 1.0, 0, 1.0), (1, 1.0, 0, 1.0, 0, 1.0),
            # M7: (A01-A11) * (B10+B11)
            (0, 1.0, 1, 1.0, 0, 1.0), (1, -1.0, 1, 1.0, 0, 1.0),
            (0, 1.0, 1, 1.0, 1, 1.0), (1, -1.0, 1, 1.0, 1, 1.0),
        ]
        
        # Wait, the above Strassen coefficients are not in the standard 7-term format for tensor decomposition.
        # Tensor decomposition terms are of the form u * v * w.
        # Strassen computes C = sum_{r=1}^7 M_r * P_r, where M_r are linear combos of A, B and P_r are fixed matrices.
        # In tensor form, each term r contributes to U, V, W.
        # Let's use the standard Strassen tensor decomposition.
        
        # For M_{2,2,2}, the 7 terms (u, v, w) where u is 2x2 vec, v is 2x2 vec, w is 2x2 vec:
        # Indices: 0=A00, 1=A01, 2=A10, 3=A11
        #          0=B00, 1=B01, 2=B10, 3=B11
        #          0=C00, 1=C01, 2=C10, 3=C11
        
        s_terms = [
            # (u_vec, v_vec, w_vec)
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, -1]),  # M1
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 1]),    # M2
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),   # M3
            ([0, 0, 0, 1], [0, -1, 1, 0], [0, 0, 1, 0]),   # M4
            ([1, 1, 0, 0], [0, 0, 0, 1], [0, -1, 0, 1]),   # M5
            ([-1, 0, 1, 0], [1, 1, 0, 0], [1, 0, -1, 0]),  # M6
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, -1]),  # M7
        ]
        
        # Map to M_{2,3,2} indices.
        # U is 2x3 vec (index i*3+j). Here we use first 2 cols of A (j=0,1).
        # V is 3x2 vec (index k*2+l). Here we use first 2 rows of B (k=0,1).
        # W is 2x2 vec (index i*2+l).
        
        for u_vec, v_vec, w_vec in s_terms:
            # u_vec is 4 elements for 2x2. Map to U (2x3, first 2 cols).
            # u_vec[0]->U[0*3+0], u_vec[1]->U[0*3+1], u_vec[2]->U[1*3+0], u_vec[3]->U[1*3+1]
            u_map = {0: 0, 1: 1, 2: 3, 3: 4} # (i,j) -> i*3+j
            # v_vec is 4 elements for 2x2. Map to V (3x2, first 2 rows).
            # v_vec[0]->V[0*2+0], v_vec[1]->V[0*2+1], v_vec[2]->V[1*2+0], v_vec[3]->V[1*2+1]
            v_map = {0: 0, 1: 1, 2: 2, 3: 3} # (k,l) -> k*2+l
            # w_vec is 4 elements for 2x2. Map to W (2x2).
            w_map = {0: 0, 1: 1, 2: 2, 3: 3} # (i,l) -> i*2+l
            
            for idx_u, val_u in enumerate(u_vec):
                if val_u != 0:
                    U_row = u_map[idx_u]
                    U[U_row][col_start] += val_u
            for idx_v, val_v in enumerate(v_vec):
                if val_v != 0:
                    V_row = v_map[idx_v]
                    V[V_row][col_start] += val_v
            for idx_w, val_w in enumerate(w_vec):
                if val_w != 0:
                    W_row = w_map[idx_w]
                    W[W_row][col_start] += val_w
            
            col_start += 1
        
        # Sub-problem 2: M_{2,1,2} (Schoolbook, rank 4)
        # A_sub is 2x1 (last col of A, j=2), B_sub is 1x2 (last row of B, k=2), C_sub is 2x2.
        # C[i][l] = A[i][2] * B[2][l]
        # This is a rank 1 tensor? No, A[i][2] is a vector of size 2, B[2][l] is a vector of size 2.
        # The tensor is A_sub * B_sub * C_identity?
        # C = A_col * B_row.
        # C[i][l] = A[i][2] * B[2][l].
        # This is a single term in tensor decomposition?
        # U[i][col] = A[i][2] (which is 1 for the variable), V[2][col] = B[2][l] (which is 1 for the variable), W[i][l][col] = 1?
        # No, tensor term is U * V * W.
        # For a single multiplication term A[i][j] * B[j][l] -> C[i][l], the term is e_i in U, e_j in V, e_l in W? No.
        # The tensor T[i,j,l] = delta_{jl}.
        # The decomposition is sum_c U[i,c] V[j,c] W[l,c].
        # For schoolbook, c indexes (i,j,l) such that j is the inner index.
        # For M_{2,1,2}: i in 0..1, j=2 (only one), l in 0..1.
        # Terms:
        # c=0: i=0, j=2, l=0. U[0,0]=1, V[2,0]=1, W[0,0]=1.
        # c=1: i=0, j=2, l=1. U[0,1]=1, V[2,1]=1, W[1,1]=1.
        # c=2: i=1, j=2, l=0. U[1,2]=1, V[2,2]=1, W[2,2]=1.
        # c=3: i=1, j=2, l=1. U[1,3]=1, V[2,3]=1, W[3,3]=1.
        
        # Wait, U is 2x3 vec. Index for A[i][2] is i*3+2.
        # V is 3x2 vec. Index for B[2][l] is 2*2+l = 4+l.
        # W is 2x2 vec. Index for C[i][l] is i*2+l.
        
        m212_terms = [
            (0, 4, 0),  # i=0, l=0
            (0, 5, 1),  # i=0, l=1
            (1, 4, 2),  # i=1, l=0
            (1, 5, 3),  # i=1, l=1
        ]
        # Actually, U index is i*3+2. So 0->2, 1->5.
        # V index is 2*2+l. So l=0->4, l=1->5.
        # W index is i*2+l. So (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3.
        
        m212_corrected = [
            (2, 4, 0),
            (2, 5, 1),
            (5, 4, 2),
            (5, 5, 3),
        ]
        
        for u_idx, v_idx, w_idx in m212_corrected:
            U[u_idx][col_start] = 1.0
            V[v_idx][col_start] = 1.0
            W[w_idx][col_start] = 1.0
            col_start += 1
        
        # Adjust indices for the specific block offset in B
        # The above assumes B rows 0,1 for Strassen and row 2 for M_{2,1,2}.
        # But this block is for B columns 3-4 or 5-6.
        # The inner dimension of B (rows) is still 0,1,2.
        # So the indices in V are correct relative to the global V matrix.
        # However, the output W indices need to be offset?
        # No, W is the result C. The result C is 2x7.
        # Wait, the task says W[k*n+i, r].
        # My W matrix is (p*n) x rank. So index is k*n+i.
        # k is 0..6, i is 0..1.
        # In my M_{2,3,2} decomposition, the output C is 2x2.
        # But this C is a sub-block of the final 2x7 C.
        # Specifically, for the first block (B cols 3-4), C cols 0-1 are affected?
        # No. A (2x3) * B (3x7) = C (2x7).
        # If we split B into B1 (3x3) and B2 (3x4).
        # C = A*B1 + A*B2.
        # B2 is split into B2a (3x2) and B2b (3x2).
        # A*B2a is 2x2. This contributes to C cols 3-4.
        # A*B2b is 2x2. This contributes to C cols 5-6.
        
        # So the W indices for the first M_{2,3,2} block should be offset by 3*n = 6.
        # W index = (k+3)*n + i.
        # My W indices in the Strassen/M212 parts were 0..3.
        # I need to add 6 to the k part.
        # k is the first index of W (k*n+i).
        # In my local 2x2 C, k is 0 or 1.
        # So global k = local_k + 3.
        # Global W index = (local_k + 3)*n + i = local_k*n + i + 3*n.
        
        # I need to fix the W indices in the add_m232_block function.
        
    # Redo the function with correct W offsets
    
    def add_m232_block_corrected(offset_k, col_start):
        # offset_k is 3 for first block, 5 for second block.
        
        s_terms = [
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, -1]),
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 1]),
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),
            ([0, 0, 0, 1], [0, -1, 1, 0], [0, 0, 1, 0]),
            ([1, 1, 0, 0], [0, 0, 0, 1], [0, -1, 0, 1]),
            ([-1, 0, 1, 0], [1, 1, 0, 0], [1, 0, -1, 0]),
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, -1]),
        ]
        
        u_map = {0: 0, 1: 1, 2: 3, 3: 4}
        v_map = {0: 0, 1: 1, 2: 2, 3: 3}
        # w_map: local (k,l) -> global (k+offset_k, l)
        # local index 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        w_map = {
            0: (0 + offset_k) * n + 0,
            1: (0 + offset_k) * n + 1,
            2: (1 + offset_k) * n + 0,
            3: (1 + offset_k) * n + 1
        }
        
        for u_vec, v_vec, w_vec in s_terms:
            for idx_u, val_u in enumerate(u_vec):
                if val_u != 0:
                    U[u_map[idx_u]][col_start] += val_u
            for idx_v, val_v in enumerate(v_vec):
                if val_v != 0:
                    V[v_map[idx_v]][col_start] += val_v
            for idx_w, val_w in enumerate(w_vec):
                if val_w != 0:
                    W[w_map[idx_w]][col_start] += val_w
            col_start += 1
        
        # M_{2,1,2} part
        # U index: i*3+2. i=0->2, i=1->5.
        # V index: 2*2+l. l=0->4, l=1->5.
        # W index: (l + offset_k)*n + i? No, W index is k*n+i.
        # In C[i][l], l is the column index.
        # In the task convention W[k*n+i], k is the column index of C?
        # "output factor follows the official trace convention W[k*n+i,r]"
        # C is 2x7. C[i][k].
        # Tensor T[i,j,k].
        # W index is k*n+i.
        # So for C[i][l], the W index is l*n + i.
        # In my M_{2,3,2} block, the output columns are l=0,1.
        # Global columns are l+offset_k.
        # So W index = (l+offset_k)*n + i.
        
        m212_terms = [
            # (i, l)
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
        ]
        
        for i, l in m212_terms:
            u_idx = i * 3 + 2
            v_idx = 2 * 2 + l
            w_idx = (l + offset_k) * n + i
            U[u_idx][col_start] = 1.0
            V[v_idx][col_start] = 1.0
            W[w_idx][col_start] = 1.0
            col_start += 1
            
        return col_start
    
    # Call for first block (B cols 3-4, offset_k=3)
    col = add_m232_block_corrected(3, col)
    
    # Call for second block (B cols 5-6, offset_k=5)
    col = add_m232_block_corrected(5, col)
    
    # Verify rank
    assert col == total_rank, f"Rank mismatch: {col} vs {total_rank}"
    
    # Convert to lists of floats (or ints if possible)
    U_out = [[int(x) if x == int(x) else x for x in row] for row in U]
    V_out = [[int(x) if x == int(x) else x for x in row] for row in V]
    W_out = [[int(x) if x == int(x) else x for x in row] for row in W]
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
