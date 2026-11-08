# EVOLVE-BLOCK-START
import time
import numpy as np

def _get_strassen_44_terms():
    """
    Generates the 49 terms for 4x4x4 Strassen decomposition.
    Returns list of (a_coefs, b_coefs, c_coefs)
    """
    terms = []
    
    # Strassen 2x2 block products
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)(B00)
    # M3 = (A00)(B10-B00)
    # M4 = (A11)(B01-B10)
    # M5 = (A00+A01)(B11)
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Block definitions: (a_blocks, b_blocks, c_map)
    # a_blocks: {(block_r, block_c): sign}
    # b_blocks: {(block_r, block_c): sign}
    # c_map: {p_pos: [(c_r, c_c, sign), ...]}
    
    block_products = [
        # M1
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1},
         {0: [(0,0,1),(2,2,1)], 1: [(0,1,1),(2,3,1)],
          2: [(1,0,1),(3,2,1)], 3: [(1,1,1),(3,3,1)]}),
        # M2
        ({(1,0):1, (1,1):1}, {(0,0):1},
         {0: [(2,0,1),(2,2,-1)], 1: [(2,1,1),(2,3,-1)],
          2: [(3,0,1),(3,2,-1)], 3: [(3,1,1),(3,3,-1)]}),
        # M3
        ({(0,0):1}, {(1,0):1, (0,0):-1},
         {0: [(0,0,-1),(0,2,1)], 1: [(0,1,-1),(0,3,1)],
          2: [(1,0,-1),(1,2,1)], 3: [(1,1,-1),(1,3,1)]}),
        # M4
        ({(1,1):1}, {(0,1):1, (1,0):-1},
         {0: [(0,0,-1),(2,0,1)], 1: [(0,1,-1),(2,1,1)],
          2: [(1,0,-1),(3,0,1)], 3: [(1,1,-1),(3,1,1)]}),
        # M5
        ({(0,0):1, (0,1):1}, {(1,1):1},
         {0: [(0,2,1),(2,2,1)], 1: [(0,3,1),(2,3,1)],
          2: [(1,2,1),(3,2,1)], 3: [(1,3,1),(3,3,1)]}),
        # M6
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1},
         {0: [(2,2,-1)], 1: [(2,3,-1)],
          2: [(3,2,-1)], 3: [(3,3,-1)]}),
        # M7
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1},
         {0: [(0,0,1)], 1: [(0,1,1)],
          2: [(1,0,1)], 3: [(1,1,1)]}),
    ]
    
    # Strassen 2x2 scalar products
    # m1=(X00+X11)(Y00+Y11)
    # m2=(X10+X11)(Y00)
    # m3=(X00)(Y10-Y00)
    # m4=(X11)(Y01-Y10)
    # m5=(X00+X01)(Y11)
    # m6=(X10-X00)(Y00+Y01)
    # m7=(X01-X11)(Y10+Y11)
    
    # c_contribs: for each mult k, {p_pos: sign}
    c_contribs = [
        {0: 1, 3: 1},      # m1
        {2: 1, 3: -1},     # m2
        {0: -1, 1: 1},     # m3
        {0: -1, 2: 1},     # m4
        {1: 1, 3: 1},      # m5
        {3: -1},           # m6
        {0: 1},            # m7
    ]
    
    def merge(*dicts):
        result = {}
        for d in dicts:
            for k, v in d.items():
                result[k] = result.get(k, 0) + v
        return {k: v for k, v in result.items() if v != 0}
        
    def negate(d):
        return {k: -v for k, v in d.items()}
        
    for a_blocks, b_blocks, c_map in block_products:
        # Build X and Y coefficient matrices (2x2 of dicts)
        X = [[{} for _ in range(2)] for _ in range(2)]
        for (br, bc), sign in a_blocks.items():
            for di in range(2):
                for dj in range(2):
                    idx = (br*2+di, bc*2+dj)
                    X[di][dj][idx] = sign
        
        Y = [[{} for _ in range(2)] for _ in range(2)]
        for (br, bc), sign in b_blocks.items():
            for di in range(2):
                for dj in range(2):
                    idx = (br*2+di, bc*2+dj)
                    Y[di][dj][idx] = sign
        
        scalar_mults = [
            (merge(X[0][0], X[1][1]), merge(Y[0][0], Y[1][1])),
            (merge(X[1][0], X[1][1]), merge(Y[0][0])),
            (merge(X[0][0]), merge(Y[1][0], negate(Y[0][0]))),
            (merge(X[1][1]), merge(Y[0][1], negate(Y[1][0]))),
            (merge(X[0][0], X[0][1]), merge(Y[1][1])),
            (merge(X[1][0], negate(X[0][0])), merge(Y[0][0], Y[0][1])),
            (merge(X[0][1], negate(X[1][1])), merge(Y[1][0], Y[1][1])),
        ]
        
        for k, (a_coefs, b_coefs) in enumerate(scalar_mults):
            if not a_coefs or not b_coefs:
                continue
            
            c_coefs = {}
            for p_pos, s_sign in c_contribs[k].items():
                pr, pc = divmod(p_pos, 2)
                for (c_row, c_col, c_sign) in c_map[p_pos]:
                    c_idx = (c_col, c_row)
                    total_sign = s_sign * c_sign
                    c_coefs[c_idx] = c_coefs.get(c_idx, 0) + total_sign
            
            c_coefs = {k: v for k, v in c_coefs.items() if v != 0}
            if c_coefs:
                terms.append((a_coefs, b_coefs, c_coefs))
                
    return terms

def construct():
    start_time = time.time()
    n, m, p = 4, 4, 9
    DEADLINE = start_time + 100
    
    strassen_terms = _get_strassen_44_terms()
    
    if len(strassen_terms) != 49:
        # Fallback to schoolbook
        rank_fb = n * m * p
        U_fb = np.zeros((n*m, rank_fb), dtype=int)
        V_fb = np.zeros((m*p, rank_fb), dtype=int)
        W_fb = np.zeros((p*n, rank_fb), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m + j, col] = 1
                    V_fb[j*p + k, col] = 1
                    W_fb[k*n + i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
        
    total_rank = 49 + 49 + 16
    U = np.zeros((n*m, total_rank), dtype=int)
    V = np.zeros((m*p, total_rank), dtype=int)
    W = np.zeros((p*n, total_rank), dtype=int)
    
    # Part 1: A * B1 (cols 0-3)
    for c, (a_coefs, b_coefs, c_coefs) in enumerate(strassen_terms):
        for (i, j), val in a_coefs.items():
            U[i*m + j, c] += val
        for (j, k), val in b_coefs.items():
            V[j*p + k, c] += val
        for (k, i), val in c_coefs.items():
            W[k*n + i, c] += val
            
    # Part 2: A * B2 (cols 4-7)
    offset = 49
    for c, (a_coefs, b_coefs, c_coefs) in enumerate(strassen_terms):
        for (i, j), val in a_coefs.items():
            U[i*m + j, offset + c] += val
        for (j, k), val in b_coefs.items():
            V[j*p + (k + 4), offset + c] += val
        for (k, i), val in c_coefs.items():
            W[k*n + i, offset + c] += val
            
    # Part 3: A * B3 (col 8)
    offset2 = 98
    for i in range(n):
        for j in range(m):
            col = offset2 + i*m + j
            U[i*m + j, col] = 1
            V[j*p + 8, col] = 1
            W[8*n + i, col] = 1
            
    # Verify
    if time.time() < DEADLINE:
        np.random.seed(42)
        A_test = np.random.randint(-3, 4, size=(n, m)).astype(float)
        B_test = np.random.randint(-3, 4, size=(m, p)).astype(float)
        C_expected = A_test @ B_test
        
        C_computed = np.zeros((n, p))
        for c in range(total_rank):
            u_vec = U[:, c]
            v_vec = V[:, c]
            w_vec = W[:, c]
            u_mat = u_vec.reshape(n, m)
            v_mat = v_vec.reshape(m, p)
            w_mat = w_vec.reshape(p, n)
            for i in range(n):
                for k in range(p):
                    C_computed[i, k] += np.dot(u_mat[i, :], v_mat[:, k] * w_mat[k, i])
                    
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            # Fallback
            rank_fb = n * m * p
            U_fb = np.zeros((n*m, rank_fb), dtype=int)
            V_fb = np.zeros((m*p, rank_fb), dtype=int)
            W_fb = np.zeros((p*n, rank_fb), dtype=int)
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_fb[i*m + j, col] = 1
                        V_fb[j*p + k, col] = 1
                        W_fb[k*n + i, col] = 1
                        col += 1
            return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
