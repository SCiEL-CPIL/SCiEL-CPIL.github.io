def _verify_decomposition(U, V, W, n, m, p):
    # U: (nm, R), V: (mp, R), W: (pn, R)
    # Check all i, j, k
    for i in range(n * m):
        r = i // m
        c = i % m
        for j in range(m * p):
            c_j = j // p
            k_j = j % p
            for k in range(p * n):
                k_k = k // n
                r_k = k % n
                
                expected = 0.0
                if c == c_j and r == r_k and k_j == k_k:
                    expected = 1.0
                else:
                    expected = 0.0
                
                actual = 0.0
                # Sum over l
                # U[i, l] V[j, l] W[k, l]
                # This is expensive if done naively for 1024 entries.
                # But R is small (27).
                # 1024 * 27 = 27648 operations.
                # This is fine.
                for l in range(len(U[0])):
                    actual += U[i][l] * V[j][l] * W[k][l]
                
                if abs(actual - expected) > 1e-6:
                    return False
    return True
