    U_mat = U.reshape(n, m)
    V_mat = V.reshape(m, p)
    C_calc = np.zeros((n, p))
    for r in range(rank):
        a_comb = U_mat[:, r] @ A   # (n,)
        b_comb = V_mat[:, r] @ B   # (p,)
        for i in range(n):
            for k in range(p):
                C_calc[i, k] += a_comb[i] * b_comb[k] * W[k*n+i, r]
