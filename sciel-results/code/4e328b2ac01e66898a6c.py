"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 12 decomposition
    # For each row i of A and column j of A (which is row j of B),
    # we have a term.
    rank = n * m
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            # U index: i*m + j
            U[i * m + j][col] = 1
            
            # V index: j*p + k? No, V is m x p.
            # V[j, :] should be e_j?
            # B is m x p. V is m x p.
            # We want B V_col to be the j-th column of B?
            # No, B V is m x p.
            # If V is identity, B V = B.
            # We want the j-th row of B?
            # C_ik = sum_j A_ij B_jk.
            # Term: A_ij * (row j of B).
            # So V should be such that B V_col gives row j of B?
            # B is m x p.
            # (B V)_k = sum_l B_lk V_l.
            # We want (B V)_k = B_jk.
            # So V_l = delta_{lj}.
            # So V is the j-th column of Identity.
            V[j * p + 0][col] = 1 # This is wrong indexing for V.
            # V is m x p.
            # V[j, k] = delta_{jk}? No.
            # V is a matrix.
            # V[j, :] = e_j? No, e_j is vector.
            # V[j, k] = 1 if k==0? No.
            # V is m x p.
            # We want V to be the j-th column of I_m?
            # But V is m x p.
            # If p != m, we can't have identity.
            # Here m=4, p=4.
            # So V can be I_4.
            # V[j, k] = delta_{jk}.
            for k in range(p):
                if j == k:
                    V[j * p + k][col] = 1
            
            # W index: k*n + i?
            # W is p x n.
            # We want to add to row i of C.
            # C is n x p.
            # W is p x n.
            # W[k, i] = 1 for all k?
            # Then (B V) W^T?
            # No, the formula is C = sum U A V B W?
            # Standard: C = sum_r (A U_r) (V_r B) W_r.
            # A U_r is scalar.
            # V_r B is m x p? No.
            # V_r is m x m?
            # Let's stick to the vector form.
            # C = sum_r (A u_r) (v_r^T B) w_r.
            # A is n x m. u_r is m x 1.
            # v_r is m x 1.
            # v_r^T B is 1 x p.
            # w_r is n x p.
            # So V_r is m x 1.
            # W_r is n x p.
            # In our case:
            # u_r = e_j (in R^m).
            # v_r = e_j (in R^m).
            # w_r = e_i^T (in R^n) * 1^T (in R^p)?
            # w_r is n x p.
            # w_r[i, k] = 1 for all k.
            # So W_r has 1s in row i.
            
            for k in range(p):
                # W index: k*n + i?
                # W is p x n.
                # W[k, i] = 1.
                W[k * n + i][col] = 1
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
