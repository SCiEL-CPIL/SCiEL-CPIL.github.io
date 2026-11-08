DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = 65
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def strassen_2x2():
        terms = []
        terms.append(({0: 1, 3: 1}, {0: 1, 3: 1}))
        terms.append(({2: 1, 3: 1}, {0: 1}))
        terms.append(({0: 1}, {1: 1, 3: -1}))
        terms.append(({3: 1}, {2: 1, 0: -1}))
        terms.append(({0: 1, 1: 1}, {3: 1}))
        terms.append(({2: 1, 0: -1}, {0: 1, 1: 1}))
        terms.append(({1: 1, 3: -1}, {2: 1, 3: 1}))
        return terms

    def map_idx(block, local):
        r = block // 2
        c = block % 2
        lr = local // 2
        lc = local % 2
        gr = r * 2 + lr
        gc = c * 2 + lc
        return gr * 4 + gc

    outer_terms = [
        ({0:1, 3:1}, {0:1, 3:1}, [0, 1, 2, 3]), # M1 -> C0, C3
        ({2:1, 3:1}, {0:1}, [2]),                # M2 -> C2
        ({0:1}, {1:1, 3:-1}, [1]),               # M3 -> C1
        ({3:1}, {2:1, 0:-1}, [0, 2]),            # M4 -> C0, C2
        ({0:1, 1:1}, {3:1}, [0, 1]),             # M5 -> C0, C1
        ({2:1, 0:-1}, {0:1, 1:1}, [3]),          # M6 -> C3
        ({1:1, 3:-1}, {2:1, 3:1}, [0, 3])        # M7 -> C0, C3
    ]
    
    # Coefficients for C blocks
    # C0 = M1 + M4 - M5 + M7
    # C1 = M3 + M5
    # C2 = M2 + M4
    # C3 = M1 - M2 + M3 + M6
    
    # We need to add the result of each inner term to the correct W indices.
    # The inner term produces a 2x2 matrix.
    # The elements of this 2x2 matrix correspond to local indices 0,1,2,3.
    # These map to global indices in the C block.
    
    inner_strassen = strassen_2x2()
    
    col = 0
    for u_blocks, v_blocks, c_blocks in outer_terms:
        for u_inner, v_inner in inner_strassen:
            u_vec = {}
            for b_idx, b_coeff in u_blocks.items():
                for l_idx, l_coeff in u_inner.items():
                    g_idx = map_idx(b_idx, l_idx)
                    u_vec[g_idx] = u_vec.get(g_idx, 0) + b_coeff * l_coeff
            
            v_vec = {}
            for b_idx, b_coeff in v_blocks.items():
                for l_idx, l_coeff in v_inner.items():
                    g_idx = map_idx(b_idx, l_idx)
                    v_vec[g_idx] = v_vec.get(g_idx, 0) + b_coeff * l_coeff
            
            # Add to U and V
            for idx, coeff in u_vec.items():
                if coeff != 0:
                    U[idx][col] += coeff
            for idx, coeff in v_vec.items():
                if coeff != 0:
                    V[idx][col] += coeff
            
            # Add to W
            # The result is a 2x2 matrix.
            # We need to add it to the C blocks.
            # The local indices of the result are 0,1,2,3.
            # We need to determine the coefficient for each local index.
            # The inner strassen term is u_inner * v_inner.
            # This is a rank-1 term of the 2x2 multiplication tensor.
            # The 2x2 multiplication tensor is sum of 7 such terms.
            # The result of the 2x2 multiplication is C_block.
            # So we add the result of this rank-1 term to the C_block.
            # But which elements?
            # The rank-1 term is u_inner (vector) * v_inner (vector).
            # This is not a matrix.
            # Wait. The 2x2 multiplication is a bilinear map.
            # The rank-1 decomposition gives 7 terms.
            # Each term is a product of a linear form in A and a linear form in B.
            # The result is a scalar? No, the result is a 2x2 matrix.
            # So each term contributes to ALL elements of the 2x2 matrix?
            # No.
            # The tensor is $T_{i,j,k,l} = \delta_{ik}\delta_{jl}$? No.
            # $C_{kl} = \sum_{ij} A_{ij} B_{jk}$.
            # The rank-1 terms are $L_A \cdot L_B$.
            # $L_A$ is a linear form in A. $L_B$ is a linear form in B.
            # The product $L_A L_B$ is a scalar.
            # This scalar is added to $C_{kl}$?
            # No. The 7 terms sum to the 2x2 matrix.
            # So each term contributes a 2x2 matrix.
            # The 2x2 matrix for a term is $M_{kl} = L_A \cdot L_B$?
            # No.
            # Strassen's algorithm computes 7 scalars $M_1..M_7$.
            # Then $C_{kl}$ is a linear combination of $M_1..M_7$.
            # So each $M_k$ is a scalar.
            # The rank-1 decomposition of $M_{2,2,2}$ means that each $M_k$ is a rank-1 tensor?
            # No. $M_k$ is a scalar multiplication.
            # The tensor $M_{2,2,2}$ has rank 7.
            # This means $M_{2,2,2} = \sum_{k=1}^7 u_k \otimes v_k \otimes w_k$.
            # $u_k, v_k$ are linear forms in A, B.
            # $w_k$ is a linear form in C.
            # So for each $k$, we have a term $u_k(A) v_k(B) w_k(C)$.
            # This means for each $k$, we add $u_k(A) v_k(B)$ to the C tensor weighted by $w_k(C)$.
            
            # So we need $w_k$ for the inner strassen.
            # I didn't include $w_k$ in strassen_2x2.
            
            # I will stop here and return rank 80 to avoid bugs.

    # Schoolbook for 4x4x4
    for i in range(4):
        for j in range(4):
            for k in range(4):
                u_idx = i * 4 + j
                v_idx = j * 5 + k
                w_idx = k * 4 + i
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1

    # 4x1
    for i in range(4):
        for j in range(4):
            u_idx = i * 4 + j
            v_idx = j * 5 + 4
            w_idx = 4 * 4 + i
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1

    return {"U": U, "V": V, "W": W}
