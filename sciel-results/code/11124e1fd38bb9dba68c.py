    def build(n, m, p, i_start, j_start, k_start, N, M, P, U, V, W, col):
        # i in [i_start, i_start+n)
        # j in [j_start, j_start+m)
        # k in [k_start, k_start+p)
        
        # Global I = i * M + j
        # Global J = j * P + k
        # Global K = k * N + i
        
        if n == 2 and m == 2 and p == 2:
            # Strassen
            # 7 terms
            # ...
            return col + 7
            
        if n == 1:
            # m * p terms
            for j in range(m):
                for k in range(p):
                    i = i_start
                    j_g = j_start + j
                    k_g = k_start + k
                    I = i * M + j_g
                    J = j_g * P + k_g
                    K = k_g * N + i
                    U[I, col] = 1
                    V[J, col] = 1
                    W[K, col] = 1
                    col += 1
            return col
            
        # Choose split
        if m >= 2:
            m1 = m // 2
            m2 = m - m1
            col = build(n, m1, p, i_start, j_start, k_start, N, M, P, U, V, W, col)
            col = build(n, m2, p, i_start, j_start + m1, k_start, N, M, P, U, V, W, col)
            return col
        elif p >= 2:
            p1 = p // 2
            p2 = p - p1
            col = build(n, m, p1, i_start, j_start, k_start, N, M, P, U, V, W, col)
            col = build(n, m, p2, i_start, j_start, k_start + p1, N, M, P, U, V, W, col)
            return col
        elif n >= 2:
            n1 = n // 2
            n2 = n - n1
            col = build(n1, m, p, i_start, j_start, k_start, N, M, P, U, V, W, col)
            col = build(n2, m, p, i_start + n1, j_start, k_start, N, M, P, U, V, W, col)
            return col
    