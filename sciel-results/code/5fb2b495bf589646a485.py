"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 7
    
    # Total rank is 80
    rank = 80
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_idx = 0

    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        nonlocal col_idx
        if col_idx >= rank:
            raise ValueError("Rank exceeded")
        U[u_idx][col_idx] = u_val
        V[v_idx][col_idx] = v_val
        W[w_idx][col_idx] = w_val
        col_idx += 1

    def add_strassen_2x2x2(i0, j0, k0):
        # Adds 7 terms for Strassen 2x2x2
        # i0, j0, k0 are global indices of top-left corner
        
        # M1 = (A11+A22)(B11+B22) -> C11, C22
        add_term(i0*m + j0, 1, j0*p + k0, 1, k0*n + i0, 1)
        add_term((i0+1)*m + (j0+1), 1, (j0+1)*p + (k0+1), 1, (k0+1)*n + (i0+1), 1)
        # Note: This adds 2 terms for M1? 
        # No, Strassen M1 is a single rank-1 term in the tensor decomposition sense?
        # Wait.
        # The tensor decomposition term is $u \otimes v \otimes w$.
        # For M1, $u$ has two non-zeros, $v$ has two non-zeros, $w$ has two non-zeros.
        # This is ONE rank-1 term?
        # No. A rank-1 term has ONE non-zero in each vector?
        # No. A rank-1 tensor is an outer product of three vectors.
        # The vectors can have multiple non-zeros.
        # So $u = e_{i0,j0} + e_{i0+1,j0+1}$, $v = e_{j0,k0} + e_{j0+1,k0+1}$, $w = e_{k0,i0} + e_{k0+1,i0+1}$.
        # This is ONE term.
        
        # My `add_term` function above adds a term with SINGLE non-zeros?
        # No, I passed single indices.
        # I need to modify `add_term` to accept lists of (index, value) or just build the vectors directly.
        
        # Let's redefine the strategy.
        # I will accumulate U, V, W columns.
        
        pass

    # The above `add_term` is too limited. I will rewrite the logic to build columns.
    
    # Reset U, V, W
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_idx = 0

    def add_rank1_term(u_entries, v_entries, w_entries):
        """
        u_entries: list of (global_idx, value)
        v_entries: list of (global_idx, value)
        w_entries: list of (global_idx, value)
        """
        nonlocal col_idx
        if col_idx >= rank:
            raise ValueError("Rank exceeded")
        for idx, val in u_entries:
            U[idx][col_idx] = val
        for idx, val in v_entries:
            V[idx][col_idx] = val
        for idx, val in w_entries:
            W[idx][col_idx] = val
        col_idx += 1

    def add_strassen_2x2x2(i0, j0, k0):
        # M1
        u1 = [((i0)*m + j0, 1), ((i0+1)*m + (j0+1), 1)]
        v1 = [(j0*p + k0, 1), ((j0+1)*p + (k0+1), 1)]
        w1 = [(k0*n + i0, 1), ((k0+1)*n + (i0+1), 1)]
        add_rank1_term(u1, v1, w1)

        # M2 = (A21+A22)B11 -> C21
        u2 = [((i0+1)*m + j0, 1), ((i0+1)*m + (j0+1), 1)]
        v2 = [(j0*p + k0, 1)]
        w2 = [(k0*n + (i0+1), 1)]
        add_rank1_term(u2, v2, w2)

        # M3 = A11(B12-B22) -> C12
        u3 = [((i0)*m + j0, 1)]
        v3 = [(j0*p + (k0+1), 1), ((j0+1)*p + (k0+1), -1)]
        w3 = [((k0+1)*n + i0, 1)]
        add_rank1_term(u3, v3, w3)

        # M4 = A22(B21-B11) -> C11, C21
        u4 = [((i0+1)*m + (j0+1), 1)]
        v4 = [((j0+1)*p + k0, 1), (j0*p + k0, -1)]
        w4 = [(k0*n + i0, 1), (k0*n + (i0+1), 1)]
        add_rank1_term(u4, v4, w4)

        # M5 = (A11+A12)B22 -> C11, C12
        u5 = [((i0)*m + j0, 1), ((i0)*m + (j0+1), 1)]
        v5 = [((j0+1)*p + (k0+1), 1)]
        w5 = [(k0*n + i0, -1), ((k0+1)*n + i0, 1)]
        add_rank1_term(u5, v5, w5)

        # M6 = (A21-A11)(B11+B12) -> C22
        u6 = [((i0+1)*m + j0, 1), ((i0)*m + j0, -1)]
        v6 = [(j0*p + k0, 1), (j0*p + (k0+1), 1)]
        w6 = [((k0+1)*n + (i0+1), 1)]
        add_rank1_term(u6, v6, w6)

        # M7 = (A12-A22)(B21+B22) -> C11, C22
        u7 = [((i0)*m + (j0+1), 1), ((i0+1)*m + (j0+1), -1)]
        v7 = [((j0+1)*p + k0, 1), ((j0+1)*p + (k0+1), 1)]
        w7 = [(k0*n + i0, 1), ((k0+1)*n + (i0+1), -1)]
        add_rank1_term(u7, v7, w7)

    def add_schoolbook_2x2x1(i0, j0, k0):
        # 2x2 A, 2x1 B, 2x1 C
        # C1 = A11 B11 + A12 B21
        # C2 = A21 B11 + A22 B21
        # 4 terms
        add_rank1_term([((i0)*m + j0, 1)], [(j0*p + k0, 1)], [(k0*n + i0, 1)])
        add_rank1_term([((i0)*m + (j0+1), 1)], [((j0+1)*p + k0, 1)], [(k0*n + i0, 1)])
        add_rank1_term([((i0+1)*m + j0, 1)], [(j0*p + k0, 1)], [(k0*n + (i0+1), 1)])
        add_rank1_term([((i0+1)*m + (j0+1), 1)], [((j0+1)*p + k0, 1)], [(k0*n + (i0+1), 1)])

    def add_schoolbook_1x2x1(i0, j0, k0):
        # 1x2 A, 2x1 B, 1x1 C
        # C1 = A11 B11 + A12 B21
        add_rank1_term([((i0)*m + j0, 1)], [(j0*p + k0, 1)], [(k0*n + i0, 1)])
        add_rank1_term([((i0)*m + (j0+1), 1)], [((j0+1)*p + k0, 1)], [(k0*n + i0, 1)])

    def add_2x2x3(i0, j0, k0):
        # Split k into 2+1
        add_strassen_2x2x2(i0, j0, k0)
        add_schoolbook_2x2x1(i0, j0, k0+2)

    def add_2x2x7(i0, j0, k0):
        # Split k into 3+3+1
        add_2x2x3(i0, j0, k0)
        add_2x2x3(i0, j0, k0+3)
        add_schoolbook_2x2x1(i0, j0, k0+6)

    def add_1x2x7(i0, j0, k0):
        # Schoolbook 1x2x7
        for k in range(7):
            for j in range(2):
                add_rank1_term([((i0)*m + (j0+j), 1)], [((j0+j)*p + (k0+k), 1)], [((k0+k)*n + i0, 1)])

    def add_3x2x7(i0, j0, k0):
        # Split i into 2+1
        add_2x2x7(i0, j0, k0)
        add_1x2x7(i0+2, j0, k0)

    # Main loop
    # Split j (size 4) into 2 blocks of 2
    for j_block in range(2):
        j0 = j_block * 2
        add_3x2x7(0, j0, 0)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
