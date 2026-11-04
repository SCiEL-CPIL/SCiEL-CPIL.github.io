    # Fallback
    U50, V50, W50 = get_rank50()
    T_full = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_full[i * m + j, j * p + k, k * n + i] = 1.0
    if verify_decomp(U50, V50, W50, T_full):
        return {"U": U50.tolist(), "V": V50.tolist(), "W": W50.tolist()}
    # If even rank 50 fails (should not happen), return schoolbook
    U_sc, V_sc, W_sc = get_schoolbook()
    return {"U": U_sc.tolist(), "V": V_sc.tolist(), "W": W_sc.tolist()}
