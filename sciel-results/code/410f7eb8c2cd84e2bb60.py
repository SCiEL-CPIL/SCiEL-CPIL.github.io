"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS # 3, 4, 4
    
    # Rank calculation:
    # 4 Strassen blocks (2x2 * 2x2) -> 4 * 7 = 28
    # 4 Schoolbook blocks (1x2 * 2x2) -> 4 * 4 = 16
    # Total Rank = 44
    R = 44
    
    U = [[0] * R for _ in range(n * m)] # 12 x 44
    V = [[0] * R for _ in range(m * p)] # 16 x 44
    W = [[0] * R for _ in range(p * n)] # 12 x 44
    
    col = 0
    
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        nonlocal col
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val
        col += 1

    def strassen_block(a_row_offset, b_col_offset, c_row_offset, c_col_offset):
        """
        Adds Strassen terms for a 2x2 * 2x2 multiplication.
        A block is at A[a_row_offset : a_row_offset+2, 0:2] (cols 0,1 for B1, 2,3 for B2? No, we split A by cols too)
        
        Wait, my decomposition was:
        A1 = A[:, 0:2], A2 = A[:, 2:4]
        B1 = B[0:2, :], B2 = B[2:4, :]
        
        Inside A1 (3x2):
        A1_top = A1[0:1, :] (1x2) -> Row 0 of A, Cols 0,1
        A1_bot = A1[1:3, :] (2x2) -> Rows 1,2 of A, Cols 0,1
        
        Inside B1 (2x4):
        B1_left = B1[:, 0:2] (2x2) -> Rows 0,1 of B, Cols 0,1
        B1_right = B1[:, 2:4] (2x2) -> Rows 0,1 of B, Cols 2,3
        
        So we have 4 sub-products for C1 (from A1*B1):
        1. A1_bot * B1_left (2x2 * 2x2) -> C rows 1,2, cols 0,1
        2. A1_bot * B1_right (2x2 * 2x2) -> C rows 1,2, cols 2,3
        3. A1_top * B1_left (1x2 * 2x2) -> C row 0, cols 0,1
        4. A1_top * B1_right (1x2 * 2x2) -> C row 0, cols 2,3
        
        Similarly for C2 (from A2*B2):
        A2_top = A[0:1, 2:4] (1x2)
        A2_bot = A[1:3, 2:4] (2x2)
        B2_left = B[2:4, 0:2] (2x2)
        B2_right = B[2:4, 2:4] (2x2)
        
        5. A2_bot * B2_left (2x2 * 2x2) -> C rows 1,2, cols 0,1
        6. A2_bot * B2_right (2x2 * 2x2) -> C rows 1,2, cols 2,3
        7. A2_top * B2_left (1x2 * 2x2) -> C row 0, cols 0,1
        8. A2_top * B2_right (1x2 * 2x2) -> C row 0, cols 2,3
        """
        pass

    # Helper to get U index for A[i, j]
    def u_idx(i, j):
        return i * m + j
        
    # Helper to get V index for B[j, k]
    def v_idx(j, k):
        return j * p + k
        
    # Helper to get W index for C[k, i]
    def w_idx(k, i):
        return k * n + i

    # --- Strassen Implementation ---
    def add_strassen(a_i0, a_i1, a_j0, a_j1, b_j0, b_j1, b_k0, b_k1, c_k0, c_k1, c_i0, c_i1):
        """
        Adds 7 terms for multiplying A[a_i0:a_i0+2, a_j0:a_j0+2] * B[b_j0:b_j0+2, b_k0:b_k0+2]
        into C[c_k0:c_k0+2, c_i0:c_i0+2].
        
        Note: A is 3x4, B is 4x4, C is 4x3.
        A indices: i (0..2), j (0..3).
        B indices: j (0..3), k (0..3).
        C indices: k (0..3), i (0..2).
        
        A block rows: a_i0, a_i0+1. Cols: a_j0, a_j0+1.
        B block rows: b_j0, b_j0+1. Cols: b_k0, b_k0+1.
        C block rows: c_k0, c_k0+1. Cols: c_i0, c_i0+1.
        
        Local 2x2 mapping:
        X[0,0] = A[a_i0, a_j0]
        X[0,1] = A[a_i0, a_j0+1]
        X[1,0] = A[a_i0+1, a_j0]
        X[1,1] = A[a_i0+1, a_j0+1]
        
        Y[0,0] = B[b_j0, b_k0]
        Y[0,1] = B[b_j0, b_k0+1]
        Y[1,0] = B[b_j0+1, b_k0]
        Y[1,1] = B[b_j0+1, b_k0+1]
        
        Z[0,0] = C[c_k0, c_i0]
        Z[0,1] = C[c_k0, c_i0+1]
        Z[1,0] = C[c_k0+1, c_i0]
        Z[1,1] = C[c_k0+1, c_i0+1]
        """
        # Pre-compute indices
        u00 = u_idx(a_i0, a_j0)
        u01 = u_idx(a_i0, a_j0+1)
        u10 = u_idx(a_i0+1, a_j0)
        u11 = u_idx(a_i0+1, a_j0+1)
        
        v00 = v_idx(b_j0, b_k0)
        v01 = v_idx(b_j0, b_k0+1)
        v10 = v_idx(b_j0+1, b_k0)
        v11 = v_idx(b_j0+1, b_k0+1)
        
        w00 = w_idx(c_k0, c_i0)
        w01 = w_idx(c_k0, c_i0+1)
        w10 = w_idx(c_k0+1, c_i0)
        w11 = w_idx(c_k0+1, c_i0+1)
        
        # m0 = (x00+x11)(y00+y11)
        # z00 += m0, z11 += m0
        add_term(u00, v00, w00, 1, 1, 1)
        add_term(u00, v00, w11, 1, 1, 1) # Wait, m0 is one term. It contributes to z00 and z11?
        # No, Strassen computes m0 once, then adds/subtracts it.
        # My add_term adds a single rank-1 term.
        # So for m0, I need to create a term that is (x00+x11) * (y00+y11).
        # This is not a single rank-1 term in the original space, it's a sum of 4.
        # BUT, in the tensor decomposition, each rank-1 term corresponds to a specific choice of coefficients in U, V, W.
        # A term in the decomposition is $u \cdot v \cdot w$.
        # If I want to compute $(x00+x11)(y00+y11)$, I cannot do it in 1 rank.
        # Strassen works by changing the basis.
        # The rank 7 decomposition means there are 7 terms $T_1..T_7$.
        # Each $T_r$ is a product of a linear combo of A entries, a linear combo of B entries, and a linear combo of C entries.
        # So, for m0:
        # U vector has 1s at u00 and u11.
        # V vector has 1s at v00 and v11.
        # W vector has 1s at w00 and w11 (with appropriate signs).
        
        # Let's re-verify how to add a Strassen term.
        # Term r:
        # U has non-zeros at specific A indices.
        # V has non-zeros at specific B indices.
        # W has non-zeros at specific C indices.
        
        # m0:
        # U: u00=1, u11=1. Others 0.
        # V: v00=1, v11=1. Others 0.
        # W: w00=1, w11=1. Others 0.
        # This single term contributes:
        # (A00+A11)(B00+B11) * (C00 + C11)??
        # No. The tensor decomposition is $\sum_r U_r \otimes V_r \otimes W_r$.
        # The coefficient of $C_{kl}$ in the product is $\sum_r W_{kl,r} ( \sum_i U_{i,r} A_i ) ( \sum_j V_{j,r} B_j )$.
        # Wait, the standard form is $C_{kl} = \sum_r W_{kl,r} U_{i,r} V_{j,r}$? No.
        # $C = \sum_r W_r (U_r^T A) (V_r^T B)$? No.
        
        # Let's look at the definition:
        # $C_{kl} = \sum_{i,j} A_{ij} B_{jk}$.
        # Decomposition: $A_{ij} B_{jk} = \sum_r U_{ij,r} V_{jk,r} W_{kl,r}$.
        # So for a fixed r, the term is $W_{kl,r} \times (\sum_i U_{ij,r} A_{ij}) \times (\sum_j V_{jk,r} B_{jk})$.
        
        # So for Strassen m0:
        # We want the term to be proportional to $(A_{00}+A_{11})(B_{00}+B_{11})$.
        # And it should contribute to $C_{00}$ and $C_{11}$ (in local coords).
        # Specifically $Z_{00} += m_0$ and $Z_{11} += m_0$.
        # So for this rank-1 term r:
        # U part: $A_{00} + A_{11}$. So $U_{00,r}=1, U_{11,r}=1$.
        # V part: $B_{00} + B_{11}$. So $V_{00,r}=1, V_{11,r}=1$.
        # W part: $C_{00} + C_{11}$. So $W_{00,r}=1, W_{11,r}=1$.
        
        # Let's check if this works for $Z_{00}$.
        # Contribution to $Z_{00}$ from this term is $W_{00,r} \times (A_{00}+A_{11}) \times (B_{00}+B_{11}) = 1 \cdot m_0$. Correct.
        # Contribution to $Z_{11}$ from this term is $W_{11,r} \times (A_{00}+A_{11}) \times (B_{00}+B_{11}) = 1 \cdot m_0$. Correct.
        
        # So I can implement each Strassen multiplier as ONE rank-1 term in my list, 
        # where U, V, W vectors contain the linear combination coefficients.
        
        # Let's re-write add_strassen to add 7 such terms.
        
        # Term 0: m0
        U[u00][col] += 1; U[u11][col] += 1
        V[v00][col] += 1; V[v11][col] += 1
        W[w00][col] += 1; W[w11][col] += 1
        col += 1
        
        # Term 1: m1 = (x10+x11)y00
        # z10 += m1, z11 += m1
        U[u10][col] += 1; U[u11][col] += 1
        V[v00][col] += 1
        W[w10][col] += 1; W[w11][col] += 1
        col += 1
        
        # Term 2: m2 = x00(y01-y11)
        # z01 += m2, z11 -= m2
        U[u00][col] += 1
        V[v01][col] += 1; V[v11][col] += -1
        W[w01][col] += 1; W[w11][col] += -1
        col += 1
        
        # Term 3: m3 = x11(y10-y00)
        # z00 -= m3, z10 += m3
        U[u11][col] += 1
        V[v10][col] += 1; V[v00][col] += -1
        W[w00][col] += -1; W[w10][col] += 1
        col += 1
        
        # Term 4: m4 = (x00+x01)y11
        # z01 += m4, z11 -= m4
        U[u00][col] += 1; U[u01][col] += 1
        V[v11][col] += 1
        W[w01][col] += 1; W[w11][col] += -1
        col += 1
        
        # Term 5: m5 = (x10-x00)(y00+y01)
        # z00 += m5
        U[u10][col] += 1; U[u00][col] += -1
        V[v00][col] += 1; V[v01][col] += 1
        W[w00][col] += 1
        col += 1
        
        # Term 6: m6 = (x01-x11)(y10+y11)
        # z00 -= m6
        U[u01][col] += 1; U[u11][col] += -1
        V[v10][col] += 1; V[v11][col] += 1
        W[w00][col] += -1
        col += 1

    # --- Schoolbook 1x2 * 2x2 Implementation ---
    def add_schoolbook_1x2(a_i, a_j0, a_j1, b_j0, b_j1, b_k0, b_k1, c_k0, c_k1, c_i):
        """
        Adds 4 terms for A[i, a_j0:a_j1+1] * B[b_j0:b_j1+1, b_k0:b_k1+1]
        into C[c_k0:c_k1+1, c_i].
        
        A is 1x2. B is 2x2. C is 1x2.
        Z[0] = X[0]Y[0,0] + X[1]Y[1,0]
        Z[1] = X[0]Y[0,1] + X[1]Y[1,1]
        
        X[0] = A[i, a_j0]
        X[1] = A[i, a_j1]
        
        Y[0,0] = B[b_j0, b_k0]
        Y[0,1] = B[b_j0, b_k1]
        Y[1,0] = B[b_j1, b_k0]
        Y[1,1] = B[b_j1, b_k1]
        
        Z[0] = C[c_k0, c_i]
        Z[1] = C[c_k1, c_i]
        """
        u0 = u_idx(a_i, a_j0)
        u1 = u_idx(a_i, a_j1)
        
        v00 = v_idx(b_j0, b_k0)
        v01 = v_idx(b_j0, b_k1)
        v10 = v_idx(b_j1, b_k0)
        v11 = v_idx(b_j1, b_k1)
        
        w0 = w_idx(c_k0, c_i)
        w1 = w_idx(c_k1, c_i)
        
        # Term 1: x0 y00 -> z0
        U[u0][col] += 1
        V[v00][col] += 1
        W[w0][col] += 1
        col += 1
        
        # Term 2: x1 y10 -> z0
        U[u1][col] += 1
        V[v10][col] += 1
        W[w0][col] += 1
        col += 1
        
        # Term 3: x0 y01 -> z1
        U[u0][col] += 1
        V[v01][col] += 1
        W[w1][col] += 1
        col += 1
        
        # Term 4: x1 y11 -> z1
        U[u1][col] += 1
        V[v11][col] += 1
        W[w1][col] += 1
        col += 1

    # --- Call the blocks ---
    
    # C1 from A1 (A[:, 0:2]) * B1 (B[0:2, :])
    
    # 1. A1_bot (rows 1,2; cols 0,1) * B1_left (rows 0,1; cols 0,1) -> C (rows 0,1; cols 1,2)
    # a_i0=1, a_i1=2, a_j0=0, a_j1=1
    # b_j0=0, b_j1=1, b_k0=0, b_k1=1
    # c_k0=0, c_k1=1, c_i0=1, c_i1=2
    add_strassen(1, 2, 0, 1, 0, 1, 0, 1, 0, 1, 1, 2)
    
    # 2. A1_bot (rows 1,2; cols 0,1) * B1_right (rows 0,1; cols 2,3) -> C (rows 2,3; cols 1,2)
    # a_i0=1, a_i1=2, a_j0=0, a_j1=1
    # b_j0=0, b_j1=1, b_k0=2, b_k1=3
    # c_k0=2, c_k1=3, c_i0=1, c_i1=2
    add_strassen(1, 2, 0, 1, 0, 1, 2, 3, 2, 3, 1, 2)
    
    # 3. A1_top (row 0; cols 0,1) * B1_left (rows 0,1; cols 0,1) -> C (rows 0,1; col 0)
    # a_i=0, a_j0=0, a_j1=1
    # b_j0=0, b_j1=1, b_k0=0, b_k1=1
    # c_k0=0, c_k1=1, c_i=0
    add_schoolbook_1x2(0, 0, 1, 0, 1, 0, 1, 0, 1, 0)
    
    # 4. A1_top (row 0; cols 0,1) * B1_right (rows 0,1; cols 2,3) -> C (rows 2,3; col 0)
    # a_i=0, a_j0=0, a_j1=1
    # b_j0=0, b_j1=1, b_k0=2, b_k1=3
    # c_k0=2, c_k1=3, c_i=0
    add_schoolbook_1x2(0, 0, 1, 0, 1, 2, 3, 2, 3, 0)
    
    # C2 from A2 (A[:, 2:4]) * B2 (B[2:4, :])
    
    # 5. A2_bot (rows 1,2; cols 2,3) * B2_left (rows 2,3; cols 0,1) -> C (rows 0,1; cols 1,2)
    # a_i0=1, a_i1=2, a_j0=2, a_j1=3
    # b_j0=2, b_j1=3, b_k0=0, b_k1=1
    # c_k0=0, c_k1=1, c_i0=1, c_i1=2
    add_strassen(1, 2, 2, 3, 2, 3, 0, 1, 0, 1, 1, 2)
    
    # 6. A2_bot (rows 1,2; cols 2,3) * B2_right (rows 2,3; cols 2,3) -> C (rows 2,3; cols 1,2)
    # a_i0=1, a_i1=2, a_j0=2, a_j1=3
    # b_j0=2, b_j1=3, b_k0=2, b_k1=3
    # c_k0=2, c_k1=3, c_i0=1, c_i1=2
    add_strassen(1, 2, 2, 3, 2, 3, 2, 3, 2, 3, 1, 2)
    
    # 7. A2_top (row 0; cols 2,3) * B2_left (rows 2,3; cols 0,1) -> C (rows 0,1; col 0)
    # a_i=0, a_j0=2, a_j1=3
    # b_j0=2, b_j1=3, b_k0=0, b_k1=1
    # c_k0=0, c_k1=1, c_i=0
    add_schoolbook_1x2(0, 2, 3, 2, 3, 0, 1, 0, 1, 0)
    
    # 8. A2_top (row 0; cols 2,3) * B2_right (rows 2,3; cols 2,3) -> C (rows 2,3; col 0)
    # a_i=0, a_j0=2, a_j1=3
    # b_j0=2, b_j1=3, b_k0=2, b_k1=3
    # c_k0=2, c_k1=3, c_i=0
    add_schoolbook_1x2(0, 2, 3, 2, 3, 2, 3, 2, 3, 0)
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
