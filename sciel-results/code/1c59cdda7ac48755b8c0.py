import time

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # Target rank based on decomposition strategy:
    # M_3,4,6 -> M_1,4,6 (24) + M_2,4,6
    # M_2,4,6 -> 2 * M_2,2,6
    # M_2,2,6 -> 2 * M_2,2,3
    # M_2,2,3 -> M_2,2,2 (7) + M_2,2,1 (4) = 11
    # Total: 24 + 2 * 2 * 11 = 24 + 44 = 68
    
    rank = 68
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    current_rank = 0
    
    def add_term(i_u, j_v, k_w, u_val, v_val, w_val):
        nonlocal current_rank
        if current_rank >= rank:
            return
        U[i_u][current_rank] = u_val
        V[j_v][current_rank] = v_val
        W[k_w][current_rank] = w_val
        current_rank += 1

    def add_block(n_b, m_b, p_b, i_off, j_off, k_off, rank_func):
        # rank_func(n_b, m_b, p_b, i_off, j_off, k_off, add_term)
        rank_func(n_b, m_b, p_b, i_off, j_off, k_off, add_term, U, V, W, n, m, p)

    def decompose_simple(n_b, m_b, p_b, i_off, j_off, k_off, add_term, U, V, W, n, m, p):
        # Schoolbook for small blocks or specific cases
        for i in range(n_b):
            for j in range(m_b):
                for k in range(p_b):
                    u_idx = (i_off + i) * m + (j_off + j)
                    v_idx = (j_off + j) * p + (k_off + k)
                    w_idx = (k_off + k) * n + (i_off + i)
                    add_term(u_idx, v_idx, w_idx, 1, 1, 1)

    def decompose_1_4_6(n_b, m_b, p_b, i_off, j_off, k_off, add_term, U, V, W, n, m, p):
        # n_b=1, m_b=4, p_b=6
        # Rank 24
        for j in range(m_b):
            for k in range(p_b):
                u_idx = i_off * m + j_off + j
                v_idx = (j_off + j) * p + k_off + k
                w_idx = (k_off + k) * n + i_off
                add_term(u_idx, v_idx, w_idx, 1, 1, 1)

    def decompose_2_2_2(n_b, m_b, p_b, i_off, j_off, k_off, add_term, U, V, W, n, m, p):
        # Strassen for 2x2x2
        # A indices: (i_off, j_off), (i_off, j_off+1), (i_off+1, j_off), (i_off+1, j_off+1)
        # B indices: (j_off, k_off), (j_off, k_off+1), (j_off+1, k_off), (j_off+1, k_off+1)
        # C indices: (i_off, k_off), (i_off, k_off+1), (i_off+1, k_off), (i_off+1, k_off+1)
        
        # Helper to get U, V, W indices
        def u_idx(i, j): return (i_off + i) * m + (j_off + j)
        def v_idx(j, k): return (j_off + j) * p + (k_off + k)
        def w_idx(i, k): return (k_off + k) * n + (i_off + i)
        
        # Term 1: (a+d)(e+h) -> C00, C11
        add_term(u_idx(0,0), v_idx(0,0), w_idx(0,0), 1, 1, 1) # a*e part? No, we need to combine.
        # Wait, the decomposition is rank 1 terms.
        # m1 = (a+d)(e+h).
        # This is not a single rank 1 term in the tensor sense unless we expand?
        # No, (a+d) is a linear combo of A. (e+h) is a linear combo of B.
        # So it IS a rank 1 term in the decomposition of the tensor.
        # U vector: [1, 0, 0, 1] for A.
        # V vector: [1, 0, 0, 1] for B.
        # W vector: [1, 0, 0, 1] for C (C00, C11).
        
        # We cannot just call add_term once with scalars.
        # We need to add multiple terms if the vectors are not basis vectors?
        # No, the rank 1 term is defined by the vectors.
        # But our U, V, W matrices store the coefficients.
        # So for this term, we set:
        # U[u_idx(0,0)] += 1
        # U[u_idx(1,1)] += 1
        # V[v_idx(0,0)] += 1
        # V[v_idx(1,1)] += 1
        # W[w_idx(0,0)] += 1
        # W[w_idx(1,1)] += 1
        # And increment rank.
        
        # My add_term function above assumes 1 non-zero entry per matrix.
        # I need a more flexible add_term that takes lists of indices and values.
        
        pass

    # Re-implement with flexible add_term
    
    def construct_decomp():
        rank = 68
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        current_rank = 0
        
        def add_rank1(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
            nonlocal current_rank
            if current_rank >= rank:
                return
            for idx, val in zip(u_indices, u_vals):
                U[idx][current_rank] += val
            for idx, val in zip(v_indices, v_vals):
                V[idx][current_rank] += val
            for idx, val in zip(w_indices, w_vals):
                W[idx][current_rank] += val
            current_rank += 1

        def add_simple(n_b, m_b, p_b, i_off, j_off, k_off):
            for i in range(n_b):
                for j in range(m_b):
                    for k in range(p_b):
                        u_idx = (i_off + i) * m + (j_off + j)
                        v_idx = (j_off + j) * p + (k_off + k)
                        w_idx = (k_off + k) * n + (i_off + i)
                        add_rank1([u_idx], [1], [v_idx], [1], [w_idx], [1])

        def add_1_4_6(i_off, j_off, k_off):
            # n=1, m=4, p=6
            for j in range(4):
                for k in range(6):
                    u_idx = i_off * m + j_off + j
                    v_idx = (j_off + j) * p + k_off + k
                    w_idx = (k_off + k) * n + i_off
                    add_rank1([u_idx], [1], [v_idx], [1], [w_idx], [1])

        def add_2_2_2(i_off, j_off, k_off):
            def u(i, j): return (i_off + i) * m + (j_off + j)
            def v(j, k): return (j_off + j) * p + (k_off + k)
            def w(i, k): return (k_off + k) * n + (i_off + i)
            
            # m1
            add_rank1([u(0,0), u(1,1)], [1, 1], [v(0,0), v(1,1)], [1, 1], [w(0,0), w(1,1)], [1, 1])
            # m2
            add_rank1([u(1,0), u(1,1)], [1, 1], [v(0,0)], [1], [w(1,0), w(1,1)], [1, -1])
            # m3
            add_rank1([u(0,0)], [1], [v(0,1), v(1,1)], [1, -1], [w(0,1), w(1,1)], [1, 1])
            # m4
            add_rank1([u(1,1)], [1], [v(1,0), v(0,0)], [1, -1], [w(0,0), w(1,0)], [1, 1])
            # m5
            add_rank1([u(0,0), u(0,1)], [1, 1], [v(1,1)], [1], [w(0,0), w(0,1)], [-1, 1])
            # m6
            add_rank1([u(1,0), u(0,0)], [1, -1], [v(0,0), v(0,1)], [1, 1], [w(1,1)], [1])
            # m7
            add_rank1([u(0,1), u(1,1)], [1, -1], [v(1,0), v(1,1)], [1, 1], [w(0,0)], [1])

        def add_2_2_1(i_off, j_off, k_off):
            def u(i, j): return (i_off + i) * m + (j_off + j)
            def v(j, k): return (j_off + j) * p + (k_off + k)
            def w(i, k): return (k_off + k) * n + (i_off + i)
            
            # C00 = A00 B00 + A01 B10
            add_rank1([u(0,0)], [1], [v(0,0)], [1], [w(0,0)], [1])
            add_rank1([u(0,1)], [1], [v(1,0)], [1], [w(0,0)], [1])
            # C10 = A10 B00 + A11 B10
            add_rank1([u(1,0)], [1], [v(0,0)], [1], [w(1,0)], [1])
            add_rank1([u(1,1)], [1], [v(1,0)], [1], [w(1,0)], [1])

        # Build M_3,4,6
        # Block 1: n=1, m=4, p=6 at (0,0,0)
        add_1_4_6(0, 0, 0)
        
        # Block 2: n=2, m=4, p=6 at (1,0,0)
        # Split m=4 into 2+2
        # Block 2a: n=2, m=2, p=6 at (1,0,0)
        # Split p=6 into 3+3
        # Block 2a1: n=2, m=2, p=3 at (1,0,0)
        # Split p=3 into 2+1
        # Block 2a1i: n=2, m=2, p=2 at (1,0,0) -> Strassen
        add_2_2_2(1, 0, 0)
        # Block 2a1ii: n=2, m=2, p=1 at (1,0,2) -> Simple
        add_2_2_1(1, 0, 2)
        
        # Block 2a2: n=2, m=2, p=3 at (1,0,3)
        # Split p=3 into 2+1
        # Block 2a2i: n=2, m=2, p=2 at (1,0,3) -> Strassen
        add_2_2_2(1, 0, 3)
        # Block 2a2ii: n=2, m=2, p=1 at (1,0,5) -> Simple
        add_2_2_1(1, 0, 5)
        
        # Block 2b: n=2, m=2, p=6 at (1,2,0)
        # Split p=6 into 3+3
        # Block 2b1: n=2, m=2, p=3 at (1,2,0)
        # Block 2b1i: n=2, m=2, p=2 at (1,2,0) -> Strassen
        add_2_2_2(1, 2, 0)
        # Block 2b1ii: n=2, m=2, p=1 at (1,2,2) -> Simple
        add_2_2_1(1, 2, 2)
        
        # Block 2b2: n=2, m=2, p=3 at (1,2,3)
        # Block 2b2i: n=2, m=2, p=2 at (1,2,3) -> Strassen
        add_2_2_2(1, 2, 3)
        # Block 2b2ii: n=2, m=2, p=1 at (1,2,5) -> Simple
        add_2_2_1(1, 2, 5)
        
        return {"U": U, "V": V, "W": W}

    return construct_decomp()
