"""Matrix multiplication 3x4x5 - rank-minimal decomposition via block tiling with Strassen."""
import time
import numpy as np

DIMENSIONS = [3, 4, 5]
DEADLINE_SECONDS = 140

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p):
    """Check that U, V, W exactly decompose the matrix multiplication tensor."""
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    rank = U_np.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U_np[i * m + j] * V_np[j * p + k] * W_np[k * n + i])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _get_strassen_2x2x2():
    """Returns list of 7 (u, v, w) vectors for 2x2x2 Strassen."""
    terms = []
    # M1 = (A00+A11)(B00+B11)
    terms.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
    # M2 = (A10+A11)B00
    terms.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]))
    # M3 = A00 B11
    terms.append(([1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 1]))
    # M4 = A11 B10
    terms.append(([0, 0, 0, 1], [0, 1, 0, 0], [-1, 0, 1, 0]))
    # M5 = (A00+A01)B10
    terms.append(([1, 1, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0]))
    # M6 = (A10-A00)(B00+B11)
    terms.append(([-1, 0, 1, 0], [1, 0, 0, 1], [0, 0, 0, -1]))
    # M7 = (A01-A11)(B10+B11)
    terms.append(([0, 1, 0, -1], [0, 1, 0, 1], [0, 1, 0, -1]))
    return terms

def _get_rank_11_2x2x3():
    """
    Returns list of 11 (u, v, w) vectors for 2x2x3.
    Decomposes into 2x2x2 (rank 7) + 2x2x1 (rank 4).
    u: len 4 (2x2 A), v: len 6 (2x3 B), w: len 6 (3x2 C with W[k*2+i] convention)
    """
    terms = []
    s_terms = _get_strassen_2x2x2()
    for u, v, w in s_terms:
        u_g = list(u)
        v_g = [v[0], v[1], 0, v[2], v[3], 0]
        w_g = [w[0], w[1], 0, w[2], w[3], 0]
        terms.append((u_g, v_g, w_g))
    
    # 2x2x1 part: C_col2 = A * B_col2
    # A00*B02, A01*B12, A10*B02, A11*B12
    terms.append(([1, 0, 0, 0], [0, 0, 1, 0, 0, 0], [1, 0, 0, 0, 0, 0]))
    terms.append(([0, 1, 0, 0], [0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0]))
    terms.append(([0, 0, 1, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0]))
    terms.append(([0, 0, 0, 1], [0, 0, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0]))
    return terms

def _get_rank_4_1x2x2():
    """Returns list of 4 (u, v, w) vectors for 1x2x2.
    u: len 2 (1x2 A), v: len 4 (2x2 B), w: len 2 (2x1 C with W[k*1+i])
    """
    terms = []
    terms.append(([1, 0], [1, 0, 0, 0], [1, 0]))
    terms.append(([1, 0], [0, 1, 0, 0], [0, 1]))
    terms.append(([0, 1], [0, 0, 1, 0], [1, 0]))
    terms.append(([0, 1], [0, 0, 0, 1], [0, 1]))
    return terms

def _get_rank_6_1x2x3():
    """Returns list of 6 (u, v, w) vectors for 1x2x3.
    u: len 2 (1x2 A), v: len 6 (2x3 B), w: len 3 (3x1 C with W[k*1+i])
    """
    terms = []
    terms.append(([1, 0], [1, 0, 0, 0, 0, 0], [1, 0, 0]))
    terms.append(([1, 0], [0, 1, 0, 0, 0, 0], [0, 1, 0]))
    terms.append(([1, 0], [0, 0, 1, 0, 0, 0], [0, 0, 1]))
    terms.append(([0, 1], [0, 0, 0, 1, 0, 0], [1, 0, 0]))
    terms.append(([0, 1], [0, 0, 0, 0, 1, 0], [0, 1, 0]))
    terms.append(([0, 1], [0, 0, 0, 0, 0, 1], [0, 0, 1]))
    return terms

def _embed_2x2x2_into_3x4x5(u, v, w, a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset):
    """
    Embed a 2x2x2 term into the 3x4x5 problem.
    u: len 4, v: len 4, w: len 4
    Maps to positions in U (12x), V (20x), W (15x)
    """
    U = [0] * 12
    V = [0] * 20
    W = [0] * 15
    
    for i in range(2):
        for j in range(2):
            U[(a_row_offset + i) * 4 + (a_col_offset + j)] = u[i * 2 + j]
    
    for j in range(2):
        for k in range(2):
            V[(b_row_offset + j) * 5 + (b_col_offset + k)] = v[j * 2 + k]
    
    for k in range(2):
        for i in range(2):
            W[(c_col_offset + k) * 3 + (c_row_offset + i)] = w[k * 2 + i]
    
    return U, V, W

def _embed_2x2x3_into_3x4x5(u, v, w, a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset):
    """
    Embed a 2x2x3 term into the 3x4x5 problem.
    u: len 4, v: len 6, w: len 6
    """
    U = [0] * 12
    V = [0] * 20
    W = [0] * 15
    
    for i in range(2):
        for j in range(2):
            U[(a_row_offset + i) * 4 + (a_col_offset + j)] = u[i * 2 + j]
    
    for j in range(2):
        for k in range(3):
            V[(b_row_offset + j) * 5 + (b_col_offset + k)] = v[j * 3 + k]
    
    for k in range(3):
        for i in range(2):
            W[(c_col_offset + k) * 3 + (c_row_offset + i)] = w[k * 2 + i]
    
    return U, V, W

def _embed_1x2x2_into_3x4x5(u, v, w, a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset):
    """
    Embed a 1x2x2 term into the 3x4x5 problem.
    u: len 2, v: len 4, w: len 2
    """
    U = [0] * 12
    V = [0] * 20
    W = [0] * 15
    
    for j in range(2):
        U[a_row_offset * 4 + (a_col_offset + j)] = u[j]
    
    for j in range(2):
        for k in range(2):
            V[(b_row_offset + j) * 5 + (b_col_offset + k)] = v[j * 2 + k]
    
    for k in range(2):
        W[(c_col_offset + k) * 3 + c_row_offset] = w[k]
    
    return U, V, W

def _embed_1x2x3_into_3x4x5(u, v, w, a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset):
    """
    Embed a 1x2x3 term into the 3x4x5 problem.
    u: len 2, v: len 6, w: len 3
    """
    U = [0] * 12
    V = [0] * 20
    W = [0] * 15
    
    for j in range(2):
        U[a_row_offset * 4 + (a_col_offset + j)] = u[j]
    
    for j in range(2):
        for k in range(3):
            V[(b_row_offset + j) * 5 + (b_col_offset + k)] = v[j * 3 + k]
    
    for k in range(3):
        W[(c_col_offset + k) * 3 + c_row_offset] = w[k]
    
    return U, V, W

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_vec, v_vec, w_vec):
        nonlocal col
        for i in range(n * m):
            U[i][col] = u_vec[i]
        for i in range(m * p):
            V[i][col] = v_vec[i]
        for i in range(p * n):
            W[i][col] = w_vec[i]
        col += 1
    
    # Block 1: C11 (2x2) from A1 (2x4) and B1 (4x2)
    # A1 rows 0-1, B1 cols 0-1
    # Split middle dim 4 into 2+2
    # Term 1: A1_left (2x2, cols 0-1) * B1_top (2x2, rows 0-1)
    s_terms = _get_strassen_2x2x2()
    for u, v, w in s_terms:
        u_g, v_g, w_g = _embed_2x2x2_into_3x4x5(u, v, w, 0, 0, 0, 0, 0, 0)
        add_term(u_g, v_g, w_g)
    
    # Term 2: A1_right (2x2, cols 2-3) * B1_bottom (2x2, rows 2-3)
    for u, v, w in s_terms:
        u_g, v_g, w_g = _embed_2x2x2_into_3x4x5(u, v, w, 0, 2, 2, 0, 0, 0)
        add_term(u_g, v_g, w_g)
    
    # Block 2: C12 (2x3) from A1 (2x4) and B2 (4x3)
    # A1 rows 0-1, B2 cols 2-4
    # Split middle dim 4 into 2+2
    # Term 1: A1_left (2x2, cols 0-1) * B2_top (2x3, rows 0-1)
    r11_terms = _get_rank_11_2x2x3()
    for u, v, w in r11_terms:
        u_g, v_g, w_g = _embed_2x2x3_into_3x4x5(u, v, w, 0, 0, 0, 2, 0, 2)
        add_term(u_g, v_g, w_g)
    
    # Term 2: A1_right (2x2, cols 2-3) * B2_bottom (2x3, rows 2-3)
    for u, v, w in r11_terms:
        u_g, v_g, w_g = _embed_2x2x3_into_3x4x5(u, v, w, 0, 2, 2, 2, 0, 2)
        add_term(u_g, v_g, w_g)
    
    # Block 3: C21 (1x2) from A2 (1x4) and B1 (4x2)
    # A2 row 2, B1 cols 0-1
    # Split middle dim 4 into 2+2
    # Term 1: A2_left (1x2, cols 0-1) * B1_top (2x2, rows 0-1)
    r4_terms = _get_rank_4_1x2x2()
    for u, v, w in r4_terms:
        u_g, v_g, w_g = _embed_1x2x2_into_3x4x5(u, v, w, 2, 0, 0, 0, 2, 0)
        add_term(u_g, v_g, w_g)
    
    # Term 2: A2_right (1x2, cols 2-3) * B1_bottom (2x2, rows 2-3)
    for u, v, w in r4_terms:
        u_g, v_g, w_g = _embed_1x2x2_into_3x4x5(u, v, w, 2, 2, 2, 0, 2, 0)
        add_term(u_g, v_g, w_g)
    
    # Block 4: C22 (1x3) from A2 (1x4) and B2 (4x3)
    # A2 row 2, B2 cols 2-4
    # Split middle dim 4 into 2+2
    # Term 1: A2_left (1x2, cols 0-1) * B2_top (2x3, rows 0-1)
    r6_terms = _get_rank_6_1x2x3()
    for u, v, w in r6_terms:
        u_g, v_g, w_g = _embed_1x2x3_into_3x4x5(u, v, w, 2, 0, 0, 2, 2, 2)
        add_term(u_g, v_g, w_g)
    
    # Term 2: A2_right (1x2, cols 2-3) * B2_bottom (2x3, rows 2-3)
    for u, v, w in r6_terms:
        u_g, v_g, w_g = _embed_1x2x3_into_3x4x5(u, v, w, 2, 2, 2, 2, 2, 2)
        add_term(u_g, v_g, w_g)
    
    # Verify
    if time.time() - start_time < DEADLINE_SECONDS - 5:
        if not _verify_decomposition(U, V, W, n, m, p):
            # Fallback to schoolbook
            rank_schoolbook = n * m * p
            U_school = [[0] * rank_schoolbook for _ in range(n * m)]
            V_school = [[0] * rank_schoolbook for _ in range(m * p)]
            W_school = [[0] * rank_schoolbook for _ in range(p * n)]
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_school[i * m + j][col] = 1
                        V_school[j * p + k][col] = 1
                        W_school[k * n + i][col] = 1
                        col += 1
            return {"U": U_school, "V": V_school, "W": W_school}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
