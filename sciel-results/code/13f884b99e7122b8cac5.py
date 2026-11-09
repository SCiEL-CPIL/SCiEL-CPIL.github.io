<<<<<<< SEARCH
    if time.time() > deadline:
        pass

    return {"U": U, "V": V, "W": W}
=======
    # Verify correctness
    if time.time() < deadline - 2:
        a_idx, b_idx, c_idx = np.indices((n*m, m*p, p*n))
        i_a = a_idx // m
        j_a = a_idx % m
        j_b = b_idx // p
        k_b = b_idx % p
        k_c = c_idx // n
        i_c = c_idx % n
        T_expected = ((j_a == j_b) & (i_a == i_c) & (k_b == k_c)).astype(float)
        
        T_decomp = np.zeros((n*m, m*p, p*n))
        for r in range(rank):
            T_decomp += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        
        if not np.allclose(T_decomp, T_expected, atol=1e-10):
            # Fall back to schoolbook
            rank = n * m * p
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            r = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j, r] = 1
                        V[j * p + k, r] = 1
                        W[k * n + i, r] = 1
                        r += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
>>>>>>> REPLACE
