DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 26  # Based on the block decomposition analysis
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, coeff):
        nonlocal col
        U[u_idx][col] += coeff
        V[v_idx][col] += coeff
        W[w_idx][col] += coeff

    # Mapping functions for local indices to global indices
    # Block 0: indices 0, 1. Block 1: index 2.
    
    # Sub-problem 1: A00 * B00 -> C00 (Strassen 2x2)
    # Local A: i,j in 0..1. Global A: i*3+j
    # Local B: j,k in 0..1. Global B: j*3+k
    # Local C: i,k in 0..1. Global C: k*3+i (W index)
    
    # Strassen Coefficients
    # M_k = (sum alpha A) * (sum beta B)
    # C_pq = sum gamma M_k
    
    # alpha[k][i][j], beta[k][j][l], gamma[k][p][q]
    alpha = [
        [[1, 0], [0, 1]],  # M1: A00 + A11
        [[0, 0], [1, 1]],  # M2: A10 + A11
        [[1, 0], [0, 0]],  # M3: A00
        [[0, 0], [0, 1]],  # M4: A11
        [[1, 1], [0, 0]],  # M5: A00 + A01
        [[-1, 0], [1, 0]], # M6: A10 - A00
        [[0, 1], [0, -1]]  # M7: A01 - A11
    ]
    beta = [
        [[1, 0], [0, 1]],  # M1: B00 + B11
        [[1, 0], [0, 0]],  # M2: B00
        [[0, 0], [1, -1]], # M3: B10 - B11
        [[-1, 1], [0, 0]], # M4: B01 - B00
        [[0, 0], [0, 1]],  # M5: B11
        [[1, 1], [0, 0]],  # M6: B00 + B01
        [[0, 0], [1, 1]]   # M7: B10 + B11
    ]
    gamma = [
        [[1, 0], [0, 0]],  # M1 -> C00, C11
        [[0, 0], [0, -1]], # M2 -> C10, C11
        [[0, 1], [0, 1]],  # M3 -> C01, C11
        [[1, 0], [1, 0]],  # M4 -> C00, C10
        [[-1, 1], [0, 0]], # M5 -> C00, C01
        [[0, 0], [0, 1]],  # M6 -> C11
        [[1, 0], [0, 0]]   # M7 -> C00
    ]
    # Correction on Gamma:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let's re-verify Gamma matrix: gamma[k][p][q]
    # k=0 (M1): C00=1, C11=1
    # k=1 (M2): C10=1, C11=-1
    # k=2 (M3): C01=1, C11=1
    # k=3 (M4): C00=1, C10=1
    # k=4 (M5): C00=-1, C01=1
    # k=5 (M6): C11=1
    # k=6 (M7): C00=1
    
    gamma = [
        [[1, 0], [0, 1]],  # M1
        [[0, 0], [1, -1]], # M2
        [[0, 1], [0, 1]],  # M3
        [[1, 0], [1, 0]],  # M4
        [[-1, 1], [0, 0]], # M5
        [[0, 0], [0, 1]],  # M6
        [[1, 0], [0, 0]]   # M7
    ]
    
    for k in range(7):
        for i in range(2):
            for j in range(2):
                a_coeff = alpha[k][i][j]
                if a_coeff == 0: continue
                for l in range(2):
                    b_coeff = beta[k][j][l]
                    if b_coeff == 0: continue
                    for p in range(2):
                        for q in range(2):
                            g_coeff = gamma[k][p][q]
                            if g_coeff == 0: continue
                            coeff = a_coeff * b_coeff * g_coeff
                            # Global indices
                            u_idx = i * 3 + j
                            v_idx = j * 3 + l
                            w_idx = q * 3 + p
                            add_term(u_idx, v_idx, w_idx, coeff)
        col += 1

    # Sub-problem 2: A01 * B10 -> C00 (Outer Product)
    # A01: i in 0..1, j=2. Global A: i*3+2
    # B10: j=2, k in 0..1. Global B: 2*3+k
    # C00: p in 0..1, q in 0..1. Global C: q*3+p
    # C_pq = A_i2 * B_2k
    for i in range(2):
        for k in range(2):
            u_idx = i * 3 + 2
            v_idx = 2 * 3 + k
            w_idx = k * 3 + i
            add_term(u_idx, v_idx, w_idx, 1)
    col += 1 # Wait, rank 4. We added 4 terms.
    # The loop above adds 4 terms.
    # We need to increment col for each term?
    # No, add_term uses the current col.
    # So we need to call add_term, then col += 1.
    
    # Let's refactor add_term to return new col or just manage col outside.
    # My add_term uses nonlocal col.
    # So I must increment col manually after each add_term call if it's a new rank-1 term.
    # In the Strassen part, I added multiple terms to the SAME col?
    # No, Strassen terms are linear combinations.
    # The rank-1 decomposition of the Strassen algorithm sums 7 rank-7 tensors?
    # No, each M_k is a product of two linear forms.
    # M_k = x_k y_k.
    # x_k = sum A_ij alpha.
    # y_k = sum B_jl beta.
    # M_k = sum A_ij B_jl alpha beta.
    # This is a sum of rank-1 tensors.
    # So for each k, we are adding a set of rank-1 tensors to the SAME column?
    # No, the column represents a single rank-1 tensor in the final decomposition.
    # If we sum multiple rank-1 tensors into one column, it's not a valid rank-1 decomposition.
    # A column in U, V, W defines ONE rank-1 tensor.
    # U[:, col] is a vector u. V[:, col] is v. W[:, col] is w.
    # The tensor is u \otimes v \otimes w.
    # If we add multiple (i,j,l) terms to the same col, we are creating a sum of rank-1 tensors in that column.
    # That is NOT allowed. Each column must be a single rank-1 tensor.
    # So we need one column per (k, i, j, l) combination?
    # No, that would explode the rank.
    
    # The Strassen algorithm has rank 7.
    # This means there are 7 terms M_k.
    # Each M_k is a scalar.
    # The contribution to C is sum gamma M_k.
    # This means the total tensor is sum_k gamma_k (x_k \otimes y_k \otimes z_k)?
    # No, C is a matrix.
    # C_pq = sum_k gamma_kpq M_k.
    # M_k = x_k y_k.
    # So C_pq = sum_k gamma_kpq x_k y_k.
    # This is a sum of rank-1 tensors in the C-space?
    # No, the tensor is T(A,B,C^*).
    # T = sum_{pq} C_pq e_pq^*.
    # T = sum_{pq} (sum_k gamma_kpq x_k y_k) e_pq^*.
    # T = sum_k (sum_{pq} gamma_kpq e_pq^*) \otimes x_k \otimes y_k.
    # Let w_k = sum_{pq} gamma_kpq e_pq^*.
    # Then T = sum_k w_k \otimes x_k \otimes y_k.
    # This is a sum of 7 rank-1 tensors.
    # So we need 7 columns.
    # For each k, we have one column.
    # U[:, col] corresponds to x_k.
    # V[:, col] corresponds to y_k.
    # W[:, col] corresponds to w_k.
    
    # So in the Strassen part:
    # For each k:
    #   Initialize U_col, V_col, W_col as zeros.
    #   For each i,j: U_col[i*3+j] += alpha[k][i][j]
    #   For each j,l: V_col[j*3+l] += beta[k][j][l]
    #   For each p,q: W_col[q*3+p] += gamma[k][p][q]
    #   Assign to U, V, W at col.
    #   col += 1.

    # Let's rewrite the Strassen part correctly.
    col = 0
    for k in range(7):
        for i in range(2):
            for j in range(2):
                if alpha[k][i][j] != 0:
                    U[i * 3 + j][col] += alpha[k][i][j]
        for j in range(2):
            for l in range(2):
                if beta[k][j][l] != 0:
                    V[j * 3 + l][col] += beta[k][j][l]
        for p in range(2):
            for q in range(2):
                if gamma[k][p][q] != 0:
                    W[q * 3 + p][col] += gamma[k][p][q]
        col += 1

    # Sub-problem 2: A01 * B10 -> C00
    # Rank 4. 4 columns.
    for i in range(2):
        for k in range(2):
            U[i * 3 + 2][col] = 1
            V[2 * 3 + k][col] = 1
            W[k * 3 + i][col] = 1
            col += 1

    # Sub-problem 3: A00 * B01 -> C01
    # A00: i,j in 0..1. B01: j in 0..1, k=2.
    # C01: p in 0..1, q=2.
    # C_p2 = sum_j A_pj B_j2
    # Rank 4.
    for p in range(2):
        for j in range(2):
            U[p * 3 + j][col] = 1
            V[j * 3 + 2][col] = 1
            W[2 * 3 + p][col] = 1
            col += 1

    # Sub-problem 4: A01 * B11 -> C01
    # A01: i in 0..1, j=2. B11: j=2, k=2.
    # C01: p in 0..1, q=2.
    # C_p2 = A_p2 B_22
    # Rank 2.
    for p in range(2):
        U[p * 3 + 2][col] = 1
        V[2 * 3 + 2][col] = 1
        W[2 * 3 + p][col] = 1
        col += 1

    # Sub-problem 5: A10 * B00 -> C10
    # A10: i=2, j in 0..1. B00: j in 0..1, k in 0..1.
    # C10: p=2, q in 0..1.
    # C_2q = sum_j A_2j B_jq
    # Rank 4.
    for q in range(2):
        for j in range(2):
            U[2 * 3 + j][col] = 1
            V[j * 3 + q][col] = 1
            W[q * 3 + 2][col] = 1
            col += 1

    # Sub-problem 6: A11 * B10 -> C10
    # A11: i=2, j=2. B10: j=2, k in 0..1.
    # C10: p=2, q in 0..1.
    # C_2q = A_22 B_2q
    # Rank 2.
    for q in range(2):
        U[2 * 3 + 2][col] = 1
        V[2 * 3 + q][col] = 1
        W[q * 3 + 2][col] = 1
        col += 1

    # Sub-problem 7: A10 * B01 -> C11
    # A10: i=2, j in 0..1. B01: j in 0..1, k=2.
    # C11: p=2, q=2.
    # C_22 = sum_j A_2j B_j2
    # Rank 2.
    for j in range(2):
        U[2 * 3 + j][col] = 1
        V[j * 3 + 2][col] = 1
        W[2 * 3 + 2][col] = 1
        col += 1

    # Sub-problem 8: A11 * B11 -> C11
    # A11: i=2, j=2. B11: j=2, k=2.
    # C11: p=2, q=2.
    # C_22 = A_22 B_22
    # Rank 1.
    U[2 * 3 + 2][col] = 1
    V[2 * 3 + 2][col] = 1
    W[2 * 3 + 2][col] = 1
    col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
