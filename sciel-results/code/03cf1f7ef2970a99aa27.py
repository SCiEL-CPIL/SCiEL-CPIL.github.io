"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()
TIME_LIMIT = 140

def _test_strassen():
    """Verify Strassen formulas numerically."""
    import random
    for _ in range(100):
        a = random.randint(-5, 5)
        b = random.randint(-5, 5)
        c = random.randint(-5, 5)
        d = random.randint(-5, 5)
        e = random.randint(-5, 5)
        f = random.randint(-5, 5)
        g = random.randint(-5, 5)
        h = random.randint(-5, 5)
        
        # True product
        i_true = a*e + b*g
        j_true = a*f + b*h
        k_true = c*e + d*g
        l_true = c*f + d*h
        
        # Strassen
        m1 = (a + d) * (e + h)
        m2 = (c + d) * e
        m3 = a * (f - g)
        m4 = d * (h - g)
        m5 = (a + b) * h
        m6 = (c - a) * (e + g)
        m7 = (b - d) * (f + h)
        
        i_str = m1 + m4 - m5 + m7
        j_str = m3 + m5
        k_str = m2 + m4
        l_str = m1 - m2 + m3 + m6
        
        assert i_str == i_true, f"i: {i_str} != {i_true}"
        assert j_str == j_true, f"j: {j_str} != {j_true}"
        assert k_str == k_true, f"k: {k_str} != {k_true}"
        assert l_str == l_true, f"l: {l_str} != {l_true}"
    return True

def _strassen_2x2x2(rank_offset, U, V, W, i0, j0, k0, n, m, p):
    """Add Strassen 2x2x2. Returns new rank_offset."""
    # A block: rows i0,i0+1; cols j0,j0+1
    # B block: rows j0,j0+1; cols k0,k0+1
    # C block: rows i0,i0+1; cols k0,k0+1
    
    a_idx = i0 * m + j0
    b_idx = i0 * m + j0 + 1
    c_idx = (i0+1) * m + j0
    d_idx = (i0+1) * m + j0 + 1
    
    e_idx = j0 * p + k0
    f_idx = j0 * p + k0 + 1
    g_idx = (j0+1) * p + k0
    h_idx = (j0+1) * p + k0 + 1
    
    i_idx = k0 * n + i0
    j_idx = k0 * n + i0 + 1
    k_idx = (k0+1) * n + i0
    l_idx = (k0+1) * n + i0 + 1
    
    products = [
        # (U_coeffs, V_coeffs, W_coeffs)
        ({a_idx: 1, d_idx: 1}, {e_idx: 1, h_idx: 1}, {i_idx: 1, j_idx: 0, k_idx: 0, l_idx: 1}),
        ({c_idx: 1, d_idx: 1}, {e_idx: 1}, {k_idx: 1, l_idx: 0}),
        ({a_idx: 1}, {f_idx: 1, g_idx: -1}, {j_idx: 1, l_idx: 0}),
        ({d_idx: 1}, {h_idx: 1, g_idx: -1}, {i_idx: 1, k_idx: 0}),
        ({a_idx: 1, b_idx: 1}, {h_idx: 1}, {i_idx: -1, k_idx: 0}),
        ({c_idx: 1, a_idx: -1}, {e_idx: 1, g_idx: 1}, {l_idx: 1, j_idx: 0}),
        ({b_idx: 1, d_idx: -1}, {f_idx: 1, h_idx: 1}, {i_idx: 1, j_idx: 0}),
    ]
    
    for u_c, v_c, w_c in products:
        for idx, val in u_c.items():
            U[idx][rank_offset] += val
        for idx, val in v_c.items():
            V[idx][rank_offset] += val
        for idx, val in w_c.items():
            W[idx][rank_offset] += val
        rank_offset += 1
    return rank_offset

def _schoolbook_2x2xk(rank_offset, U, V, W, i0, j0, k0, k, n, m, p):
    """Schoolbook for 2x2 x k."""
    for di in range(2):
        for dj in range(2):
            for dk in range(k):
                u_idx = (i0+di) * m + (j0+dj)
                v_idx = (j0+dj) * p + (k0+dk)
                w_idx = (k0+dk) * n + (i0+di)
                U[u_idx][rank_offset] = 1
                V[v_idx][rank_offset] = 1
                W[w_idx][rank_offset] = 1
                rank_offset += 1
    return rank_offset

def construct():
    n, m, p = DIMENSIONS
    _test_strassen()
    
    # 4 blocks of A, each 2x2. For each, split B's 7 cols into 2+2+3.
    # 2x2x2: Strassen (7), 2x2x2: Strassen (7), 2x2x3: Strassen(2x2x2)+Schoolbook(2x2x1) = 7+4=11
    # Per block: 7+7+11 = 25. Total: 4*25 = 100.
    total_rank = 100
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    rank_offset = 0
    for i0 in [0, 2]:
        for j0 in [0, 2]:
            # B rows for this A block's columns
            # A block at (i0,j0) multiplies B rows j0,j0+1
            b_row = j0
            # Split 7 cols: 2+2+3
            k0 = 0
            rank_offset = _strassen_2x2x2(rank_offset, U, V, W, i0, j0, k0, n, m, p)
            k0 += 2
            rank_offset = _strassen_2x2x2(rank_offset, U, V, W, i0, j0, k0, n, m, p)
            k0 += 2
            rank_offset = _strassen_2x2x2(rank_offset, U, V, W, i0, j0, k0, n, m, p)
            k0 += 2
            rank_offset = _schoolbook_2x2xk(rank_offset, U, V, W, i0, j0, k0, 1, n, m, p)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
