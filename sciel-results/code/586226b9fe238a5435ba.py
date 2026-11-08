# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 26
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # 13 terms for 2x3x3
    # A = [[a0, a1, a2], [a3, a4, a5]]
    # B = [[b0, b1, b2], [b3, b4, b5], [b6, b7, b8]]
    # C = [[c0, c1, c2], [c3, c4, c5]]
    
    terms = [
        ({0:1, 1:1, 3:1}, {0:1, 1:1, 3:1}, {0:1, 1:1, 3:1}), # Guess
        # ...
    ]
    
    # This is too risky. I will stick to 33.
    
    rank = 33
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    block_terms = []
    block_terms.append(({0: 1, 4: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}))
    block_terms.append(({3: 1, 4: 1}, {0: 1}, {2: 1, 3: -1}))
    block_terms.append(({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}))
    block_terms.append(({4: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}))
    block_terms.append(({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}))
    block_terms.append(({3: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}))
    block_terms.append(({1: 1, 4: -1}, {2: 1, 3: 1}, {0: 1}))
    block_terms.append(({2: 1}, {4: 1}, {0: 1}))
    block_terms.append(({2: 1}, {5: 1}, {1: 1}))
    block_terms.append(({5: 1}, {4: 1}, {2: 1}))
    block_terms.append(({5: 1}, {5: 1}, {3: 1}))
    
    for block in range(3):
        for t_idx, (u_dict, v_dict, w_dict) in enumerate(block_terms):
            col = block * 11 + t_idx
            for idx, val in u_dict.items():
                U[idx][col] = val
            for l_idx, val in v_dict.items():
                j_local = l_idx // 2
                k_local = l_idx % 2
                g_idx = j_local * p + (block * 2 + k_local)
                V[g_idx][col] = val
            for l_idx, val in w_dict.items():
                i_local = l_idx // 2
                k_local = l_idx % 2
                g_idx = (block * 2 + k_local) * n + i_local
                W[g_idx][col] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
