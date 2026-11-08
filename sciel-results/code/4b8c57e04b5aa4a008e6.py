def add_term(u_vals, v_vals, w_vals):
    U = np.zeros(n * m, dtype=float)
    V = np.zeros(m * p, dtype=float)
    W = np.zeros(p * n, dtype=float)
    for (i, j), val in u_vals.items():
        U[i * m + j] = val
    for (j, k), val in v_vals.items():
        V[j * p + k] = val
    for (k, i), val in w_vals.items():
        W[k * n + i] = val
