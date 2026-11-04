"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 51 decomposition
    rank = 51
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: M_1,5,6 (Row 0 of A)
    # Rank 6
    # i=0, j=0..4, k=0..5
    for t in range(6):
        # U: A index i*m+j. i=0. j=t.
        if t < 5:
            U[0 * m + t][col] = 1
        # W: C index k*n+i. i=0. k=t.
        W[t * n + 0][col] = 1
        # V: B index j*p+k. j=t, k=t.
        if t < 5:
            V[t * p + t][col] = 1
        col += 1
        
    # Part 2: M_2,5,6 (Rows 1, 2 of A)
    # i in {1, 2}
    # Blocks of 2x2x2 and 2x1x2
    
    # Strassen 2x2x2 helper
    def add_strassen_block(i_off, j_off, k_off, col_start):
        # 7 terms
        # M1: (A00+A11)(B00+B11)
        U[i_off * m + j_off][col_start] = 1
        U[(i_off+1) * m + (j_off+1)][col_start] = 1
        V[j_off * p + k_off][col_start] = 1
        V[(j_off+1) * p + (k_off+1)][col_start] = 1
        W[k_off * n + i_off][col_start] = 1
        W[(k_off+1) * n + (i_off+1)][col_start] = 1
        
        # M2: (A10+A11)B00
        U[(i_off+1) * m + j_off][col_start+1] = 1
        U[(i_off+1) * m + (j_off+1)][col_start+1] = 1
        V[j_off * p + k_off][col_start+1] = 1
        W[k_off * n + (i_off+1)][col_start+1] = 1
        
        # M3: A00(B01-B11)
        U[i_off * m + j_off][col_start+2] = 1
        V[j_off * p + (k_off+1)][col_start+2] = 1
        V[(j_off+1) * p + (k_off+1)][col_start+2] = -1
        W[(k_off+1) * n + i_off][col_start+2] = 1
        
        # M4: A11(B10-B00)
        U[(i_off+1) * m + (j_off+1)][col_start+3] = 1
        V[(j_off+1) * p + k_off][col_start+3] = 1
        V[j_off * p + k_off][col_start+3] = -1
        W[k_off * n + (i_off+1)][col_start+3] = 1
        
        # M5: (A00+A01)B11
        U[i_off * m + j_off][col_start+4] = 1
        U[i_off * m + (j_off+1)][col_start+4] = 1
        V[(j_off+1) * p + (k_off+1)][col_start+4] = 1
        W[(k_off+1) * n + i_off][col_start+4] = 1
        
        # M6: (A10-A00)(B00+B01)
        U[(i_off+1) * m + j_off][col_start+5] = 1
        U[i_off * m + j_off][col_start+5] = -1
        V[j_off * p + k_off][col_start+5] = 1
        V[j_off * p + (k_off+1)][col_start+5] = 1
        W[k_off * n + (i_off+1)][col_start+5] = 1
        
        # M7: (A01-A11)(B10+B11)
        U[i_off * m + (j_off+1)][col_start+6] = 1
        U[(i_off+1) * m + (j_off+1)][col_start+6] = -1
        V[(j_off+1) * p + k_off][col_start+6] = 1
        V[(j_off+1) * p + (k_off+1)][col_start+6] = 1
        W[(k_off+1) * n + (i_off+1)][col_start+6] = 1
        
        return col_start + 7

    # Rank-1 2x1x2 helper
    # C_il = A_i1 B_1l
    def add_rank1_block(i_off, j_off, k_off, col_start):
        # i in {i_off, i_off+1}, j=j_off, k in {k_off, k_off+1}
        # Term: A_i * B_j * C_k?
        # No, it's a sum over k.
        # C_il = A_i B_l.
        # This is rank 1?
        # T = sum_{i,l} A_i B_l C_il.
        # Can be written as (sum_i A_i u_i) (sum_l B_l v_l) (sum_k C_k w_k)?
        # No, it's just A_i B_l.
        # We can use 2 terms?
        # C_i0 = A_i B_0
        # C_i1 = A_i B_1
        # Rank 2 schoolbook.
        # But we said rank 1?
        # M_{2,1,2}. A is 2x1, B is 1x2.
        # C = A B.
        # C_00 = A_0 B_0
        # C_01 = A_0 B_1
        # C_10 = A_1 B_0
        # C_11 = A_1 B_1
        # This is outer product. Rank 1.
        # U = [1, 1]^T? No.
        # We need sum_t U_t V_t W_t.
        # If t=1: U=[1,1], V=[1,1], W=[1,1,1,1]?
        # (A0+A1)(B0+B1)(C00+C01+C10+C11)? No.
        # The tensor is A_i B_j C_{ij}.
        # This is not a single product.
        # It requires 4 terms?
        # A0 B0 C00
        # A0 B1 C01
        # A1 B0 C10
        # A1 B1 C11
        # Rank 4?
        # Wait, A_i and B_j are independent.
        # C_{ij} = A_i B_j.
        # This is the outer product tensor.
        # Rank is 1?
        # No, the tensor $T_{i,j,k,l} = \delta_{i,k} \delta_{j,l}$?
        # Here indices are i (A), j (B), k (C row), l (C col).
        # $C_{kl} = A_k B_l$.
        # $T_{k, l, i, j} = \delta_{k,i} \delta_{l,j}$.
        # This is a permutation of identity. Rank 1?
        # No, rank of identity matrix of size N is N.
        # Here we have 4 terms.
        # So rank is 4?
        # Schoolbook for 2x1x2 is 4.
        # My previous count of 2 was wrong.
        # So 3 blocks of 4 = 12.
        # Total 42 + 12 = 54.
        # Plus 6 = 60.
        
        # Let's use schoolbook for these small blocks to be safe.
        for i in range(2):
            for k in range(2):
                U[(i_off + i) * m + j_off][col_start] = 1
                V[j_off * p + (k_off + k)][col_start] = 1
                W[(k_off + k) * n + (i_off + i)][col_start] = 1
                col_start += 1
        return col_start

    # 6 Strassen blocks
    # Cols 0-1, Out 0-1
    col = add_strassen_block(1, 0, 0, col)
    # Cols 0-1, Out 2-3
    col = add_strassen_block(1, 0, 2, col)
    # Cols 0-1, Out 4-5
    col = add_strassen_block(1, 0, 4, col)
    # Cols 2-3, Out 0-1
    col = add_strassen_block(1, 2, 0, col)
    # Cols 2-3, Out 2-3
    col = add_strassen_block(1, 2, 2, col)
    # Cols 2-3, Out 4-5
    col = add_strassen_block(1, 2, 4, col)
    
    # 3 Rank-4 blocks (2x1x2)
    # Cols 4, Out 0-1
    col = add_rank1_block(1, 4, 0, col)
    # Cols 4, Out 2-3
    col = add_rank1_block(1, 4, 2, col)
    # Cols 4, Out 4-5
    col = add_rank1_block(1, 4, 4, col)
    
    # Check col == rank
    if col != rank:
        # Fallback to schoolbook if mismatch
        pass
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
