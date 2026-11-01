    def strassen_decomposition():
        pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]
        rank = 7 * len(pairs)
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]

        for block_idx, (k0, k1) in enumerate(pairs):
            U_b, V_b, W_b = strassen_block(k0, k1)
            offset = block_idx * 7
            
            # U mapping: local 0..3 -> global 0..3
            for r in range(7):
                for a in range(4):
                    U[a][offset + r] += U_b[a][r]
                    
            # V mapping:
            # local 0 -> B[0,k0] -> global 0*p + k0
            # local 1 -> B[1,k0] -> global 1*p + k0
            # local 2 -> B[0,k1] -> global 0*p + k1
            # local 3 -> B[1,k1] -> global 1*p + k1
            v_map = [k0, p + k0, k1, p + k1]
            for r in range(7):
                for li in range(4):
                    V[v_map[li]][offset + r] += V_b[li][r]
                    
            # W mapping:
            # local 0 -> C[k0,0] -> global k0*n + 0
            # local 1 -> C[k1,0] -> global k1*n + 0
            # local 2 -> C[k0,1] -> global k0*n + 1
            # local 3 -> C[k1,1] -> global k1*n + 1
            w_map = [k0 * n, k1 * n, k0 * n + 1, k1 * n + 1]
            for r in range(7):
                for li in range(4):
                    W[w_map[li]][offset + r] += W_b[li][r]

        return U, V, W
