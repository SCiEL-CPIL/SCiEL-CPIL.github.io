"""Program for matrix multiplication 2,2,10 using block-diagonal Strassen decomposition."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Decompose 2x2 * 2x10 multiplication into five independent 2x2 * 2x2 blocks.
    Each 2x2 * 2x2 block uses the rank-7 Strassen decomposition.
    Total rank = 5 * 7 = 35.
    
    Returns U, V, W as lists of lists.
    U: (4, 35)
    V: (20, 35)
    W: (20, 35)
    """
    n, m, p = DIMENSIONS
    num_blocks = p // 2  # 5
    
    # Strassen rank-7 decomposition for 2x2 matrices
    # Indices: i,j for A (0..1), j,k for B (0..1), k,i for W (0..1)
    # Term structure: (U_dict, V_dict, W_dict)
    # U_dict: {(i,j): val}
    # V_dict: {(j,k): val}
    # W_dict: {(k,i): val}
    
    # M1 = (a+d)(e+h)
    t0 = ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1})
    # M2 = (c+d)e
    t1 = ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(0,1): 1, (1,1): -1})
    # M3 = a(f-h)
    t2 = ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1})
    # M4 = d(g-e)
    t3 = ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1})
    # M5 = (a+b)h
    t4 = ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (1,0): 1})
    # M6 = (c-a)(e+f)
    t5 = ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1})
    # M7 = (b-d)(g+h)
    t6 = ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1})
    
    strassen_terms = [t0, t1, t2, t3, t4, t5, t6]
    
    rank = num_blocks * 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(num_blocks):
        base_col = r * 7
        for t_idx in range(7):
            col = base_col + t_idx
            u_ent, v_ent, w_ent = strassen_terms[t_idx]
            
            for (i, j), val in u_ent.items():
                U[i * m + j][col] = val
            
            for (j, k), val in v_ent.items():
                V[j * p + (2 * r + k)][col] = val
            
            for (k, i), val in w_ent.items():
                W[(2 * r + k) * n + i][col] = val
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
