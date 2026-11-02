"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Approach: Use bilinear complexity reduction via Kronecker canonical form.
    # For n=2, the multiplication tensor T_{i,j,k} = delta_{j,k} can be viewed
    # as a pair of commuting operators. We exploit the structure to find a
    # rank-minimal decomposition.
    #
    # The key insight: when n=2, we can write C = A*B where A is 2x4 and B is 4x5.
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    #
    # We can decompose this using the fact that the tensor has a special structure
    # for n=2. The bilinear complexity of 2x4 by 4x5 matrix multiplication can be
    # analyzed via the Kronecker canonical form of the associated linear maps.
    #
    # Strategy: Split B into blocks B1 (4x2) and B2 (4x3).
    # C = A*B1 + A*B2
    # 
    # For A*B1 (2x2 output): Use Strassen-like decomposition (rank 7)
    # For A*B2 (2x3 output): Use a custom decomposition
    #
    # Total: 7 + custom_rank
    #
    # Let's design the custom decomposition for 2x4 times 4x3.
    # C_{i,k} = sum_{j=0}^{3} A_{i,j} B_{j,k} for k=0,1,2
    #
    # We can use the following bilinear terms:
    # For each pair (i,k), C_{i,k} = A_{i,0}B_{0,k} + A_{i,1}B_{1,k} + A_{i,2}B_{2,k} + A_{i,3}B_{3,k}
    #
    # A rank-12 schoolbook would be 2*4*3 = 24 terms, but we can do better.
    # Using the structure, we can achieve rank 12 for the 2x4 by 4x3 case.
    #
    # Actually, let me reconsider. The bilinear complexity of m1 x m2 by m2 x m3
    # is at most m1*m2*m3 (schoolbook) and at least max(m1*m3, m1*m2, m2*m3).
    # For 2x4 by 4x3: lower bound is max(6, 8, 12) = 12.
    #
    # Let me try to achieve rank 12 for the 2x4 by 4x3 part.
    # 
    # Actually, for the full 2x4 by 4x5 problem:
    # Lower bound: max(2*5, 2*4, 4*5) = max(10, 8, 20) = 20
    #
    # Let me try a different approach. Use the following decomposition:
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    #
    # We can group the j indices and use bilinear forms.
    #
    # Let me try rank 20 first (the lower bound) and see if it's achievable.
    #
    # Actually, let me use a known result: the bilinear complexity of 2x4 by 4x5
    # matrix multiplication is 20 (achievable via the Kronecker canonical form approach).
    
    rank = 20
    U = [[0] * rank for _ in range(n * m)]  # 2*4 = 8 rows
    V = [[0] * rank for _ in range(m * p)]  # 4*5 = 20 rows
    W = [[0] * rank for _ in range(p * n)]  # 5*2 = 10 rows
    
    # For each bilinear term r, we define:
    # U[i*m + j, r] = coefficient of A_{i,j}
    # V[j*p + k, r] = coefficient of B_{j,k}
    # W[k*n + i, r] = coefficient for C_{i,k}
    #
    # C_{i,k} = sum_r W[k*n + i, r] * (sum_j U[i*m+j, r] * A_{i,j}) * (sum_{j',k'} V[j'*p+k', r] * B_{j',k'})
    #
    # For exact decomposition, we need:
    # sum_r W[k*n+i, r] * U[i*m+j, r] * V[j'*p+k', r] = delta_{j,j'} * delta_{k,k'}
    #
    # Let me construct rank 20 terms:
    # For each j in 0..3 and k in 0..4, we have 4*5 = 20 terms.
    # Term (j,k): U[i*m+j, r] = 1 for all i, V[j*p+k, r] = 1, W[k*n+i, r] = 1 for all i
    # But this gives C_{i,k} = sum_j A_{i,j} B_{j,k} which is correct!
    # Wait, that's just schoolbook with rank 2*4*5 = 40.
    #
    # Let me reconsider. The rank 20 decomposition:
    # For each (j, k) pair, we need to capture A_{i,j} * B_{j,k} for both i=0 and i=1.
    # 
    # Actually, I realize the schoolbook decomposition has rank n*m*p = 2*4*5 = 40.
    # The lower bound is 20, but achieving it requires a non-trivial decomposition.
    #
    # Let me try a different strategy: use the current approach but optimize.
    # Current: rank 36. Let me try to reduce it.
    
    # Reset
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen(a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset):
        nonlocal col
        u_idx_00 = a_row_offset * m + a_col_offset
        u_idx_01 = a_row_offset * m + a_col_offset + 1
        u_idx_10 = (a_row_offset + 1) * m + a_col_offset
        u_idx_11 = (a_row_offset + 1) * m + a_col_offset + 1
        
        v_idx_00 = b_row_offset * p + b_col_offset
        v_idx_01 = b_row_offset * p + b_col_offset + 1
        v_idx_10 = (b_row_offset + 1) * p + b_col_offset
        v_idx_11 = (b_row_offset + 1) * p + b_col_offset + 1
        
        w_idx_00 = c_col_offset * n + c_row_offset
        w_idx_01 = (c_col_offset + 1) * n + c_row_offset
        w_idx_10 = c_col_offset * n + c_row_offset + 1
        w_idx_11 = (c_col_offset + 1) * n + c_row_offset + 1
        
        # M1 = (a00+a11)(b00+b11)
        U[u_idx_00][col] += 1
        U[u_idx_11][col] += 1
        V[v_idx_00][col] += 1
        V[v_idx_11][col] += 1
        W[w_idx_00][col] += 1
        W[w_idx_11][col] += 1
        col += 1
        
        # M2 = (a10+a11)b00
        U[u_idx_10][col] += 1
        U[u_idx_11][col] += 1
        V[v_idx_00][col] += 1
        W[w_idx_10][col] += 1
        W[w_idx_11][col] -= 1
        col += 1
        
        # M3 = a00(b01-b11)
        U[u_idx_00][col] += 1
        V[v_idx_01][col] += 1
        V[v_idx_11][col] -= 1
        W[w_idx_01][col] += 1
        W[w_idx_11][col] += 1
        col += 1
        
        # M4 = a11(b10-b00)
        U[u_idx_11][col] += 1
        V[v_idx_10][col] += 1
        V[v_idx_00][col] -= 1
        W[w_idx_00][col] += 1
        W[w_idx_10][col] += 1
        col += 1
        
        # M5 = (a00+a01)b11
        U[u_idx_00][col] += 1
        U[u_idx_01][col] += 1
        V[v_idx_11][col] += 1
        W[w_idx_00][col] -= 1
        W[w_idx_01][col] += 1
        col += 1
        
        # M6 = (a10-a00)(b00+b01)
        U[u_idx_10][col] += 1
        U[u_idx_00][col] -= 1
        V[v_idx_00][col] += 1
        V[v_idx_01][col] += 1
        W[w_idx_11][col] += 1
        col += 1
        
        # M7 = (a01-a11)(b10+b11)
        U[u_idx_01][col] += 1
        U[u_idx_11][col] -= 1
        V[v_idx_10][col] += 1
        V[v_idx_11][col] += 1
        W[w_idx_00][col] += 1
        col += 1

    def add_schoolbook_2x1(a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset):
        nonlocal col
        u_idx_00 = a_row_offset * m + a_col_offset
        u_idx_01 = a_row_offset * m + a_col_offset + 1
        u_idx_10 = (a_row_offset + 1) * m + a_col_offset
        u_idx_11 = (a_row_offset + 1) * m + a_col_offset + 1
        
        v_idx_0 = b_row_offset * p + b_col_offset
        v_idx_1 = (b_row_offset + 1) * p + b_col_offset
        
        w_idx_0 = c_col_offset * n + c_row_offset
        w_idx_1 = c_col_offset * n + c_row_offset + 1
        
        U[u_idx_00][col] = 1
        V[v_idx_0][col] = 1
        W[w_idx_0][col] = 1
        col += 1
        
        U[u_idx_01][col] = 1
        V[v_idx_1][col] = 1
        W[w_idx_0][col] = 1
        col += 1
        
        U[u_idx_10][col] = 1
        V[v_idx_0][col] = 1
        W[w_idx_1][col] = 1
        col += 1
        
        U[u_idx_11][col] = 1
        V[v_idx_1][col] = 1
        W[w_idx_1][col] = 1
        col += 1

    # Part 1: C1 = A B1 (2x2), B1 = cols 0,1 of B
    add_strassen(0, 0, 0, 0, 0, 0)
    add_strassen(0, 2, 2, 0, 0, 0)
    
    # Part 2: C2 = A B2 (2x3), B2 = cols 2,3,4 of B
    # C21 = A B21 (2x1), B21 = col 2 of B
    add_schoolbook_2x1(0, 0, 0, 2, 0, 2)
    add_schoolbook_2x1(0, 2, 2, 2, 0, 2)
    
    # C22 = A B22 (2x2), B22 = cols 3,4 of B
    add_strassen(0, 0, 0, 3, 0, 3)
    add_strassen(0, 2, 2, 3, 0, 3)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
