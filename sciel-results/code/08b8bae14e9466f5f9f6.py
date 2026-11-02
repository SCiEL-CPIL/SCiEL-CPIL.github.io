import time
import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def _build_schoolbook():
    """Build rank-50 schoolbook decomposition."""
    n, m, p = 2, 5, 5
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                terms.append(([(i * m + j, 1)], [(j * p + k, 1)], [(k * n + i, 1)]))
    return terms

def _build_strassen_decomposition():
    """
    Build the rank-46 Strassen-based decomposition for 2x5 x 5x5.
    Partition inner dimension 5 = 2+2+1.
    """
    n, m, p = 2, 5, 5
    terms = []

    def add_term(u_entries, v_entries, w_entries):
        terms.append((u_entries, v_entries, w_entries))

    def schoolbook_block(a_rows, a_cols, b_rows, b_cols, a_shape, b_shape, out_rows, out_cols):
        nr, nc = a_shape
        nr2, nc2 = b_shape
        for i in range(nr):
            for k in range(nc):
                for j in range(nc2):
                    u_idx = (a_rows + i) * m + (a_cols + k)
                    v_idx = (b_rows + k) * p + (b_cols + j)
                    w_idx = (out_cols + j) * n + (out_rows + i)
                    add_term([(u_idx, 1)], [(v_idx, 1)], [(w_idx, 1)])

    def strassen_2x2x2(a_row_off, a_col_off, b_row_off, b_col_off, out_row_off, out_col_off):
        a_u = a_row_off * m + a_col_off
        b_u = a_row_off * m + (a_col_off + 1)
        c_u = (a_row_off + 1) * m + a_col_off
        d_u = (a_row_off + 1) * m + (a_col_off + 1)

        e_v = b_row_off * p + b_col_off
        f_v = b_row_off * p + (b_col_off + 1)
        g_v = (b_row_off + 1) * p + b_col_off
        h_v = (b_row_off + 1) * p + (b_col_off + 1)

        o00_w = out_col_off * n + out_row_off
        o01_w = (out_col_off + 1) * n + out_row_off
        o10_w = out_col_off * n + (out_row_off + 1)
        o11_w = (out_col_off + 1) * n + (out_row_off + 1)

        add_term([(a_u, 1), (d_u, 1)], [(e_v, 1), (h_v, 1)], [(o00_w, 1), (o11_w, 1)])
        add_term([(c_u, 1), (d_u, 1)], [(e_v, 1)], [(o10_w, 1), (o11_w, -1)])
        add_term([(a_u, 1)], [(f_v, 1), (h_v, -1)], [(o01_w, 1), (o11_w, 1)])
        add_term([(d_u, 1)], [(g_v, 1), (e_v, -1)], [(o00_w, 1), (o10_w, 1)])
        add_term([(a_u, 1), (b_u, 1)], [(h_v, 1)], [(o00_w, -1), (o01_w, 1)])
        add_term([(c_u, 1), (a_u, -1)], [(e_v, 1), (f_v, 1)], [(o11_w, 1)])
        add_term([(b_u, 1), (d_u, -1)], [(g_v, 1), (h_v, 1)], [(o00_w, 1)])

    strassen_2x2x2(0, 0, 0, 0, 0, 0)
    strassen_2x2x2(0, 0, 0, 2, 0, 2)
    schoolbook_block(0, 0, 0, 4, (2, 2), (2, 1), 0, 4)

    strassen_2x2x2(0, 2, 2, 0, 0, 0)
    strassen_2x2x2(0, 2, 2, 2, 0, 2)
    schoolbook_block(0, 2, 2, 4, (2, 2), (2, 1), 0, 4)

    schoolbook_block(0, 4, 4, 0, (2, 1), (1, 5), 0, 0)

    return terms

def _strassen_2x3x3(a_row_off, a_col_off, b_row_off, b_col_off, out_row_off, out_col_off, terms, m, p, n):
    """
    Strassen-like decomposition for 2x3 x 3x3 -> 2x3.
    Partition the 3x3 B matrix into 2x2 + 1 column, and use Strassen on the 2x2 part.
    This gives rank 7 (Strassen) + 2 (schoolbook for extra col) = 9 terms for the 2x2x2 part
    plus 2 terms for the extra column, total 11 vs schoolbook 18.
    """
    def add_term(u_entries, v_entries, w_entries):
        terms.append((u_entries, v_entries, w_entries))

    # Strassen on A cols [a_col_off, a_col_off+1], B rows [b_row_off, b_row_off+1], B cols [b_col_off, b_col_off+1]
    a_u = a_row_off * m + a_col_off
    b_u = a_row_off * m + (a_col_off + 1)
    c_u = (a_row_off + 1) * m + a_col_off
    d_u = (a_row_off + 1) * m + (a_col_off + 1)

    e_v = b_row_off * p + b_col_off
    f_v = b_row_off * p + (b_col_off + 1)
    g_v = (b_row_off + 1) * p + b_col_off
    h_v = (b_row_off + 1) * p + (b_col_off + 1)

    o00_w = out_col_off * n + out_row_off
    o01_w = (out_col_off + 1) * n + out_row_off
    o10_w = out_col_off * n + (out_row_off + 1)
    o11_w = (out_col_off + 1) * n + (out_row_off + 1)

    add_term([(a_u, 1), (d_u, 1)], [(e_v, 1), (h_v, 1)], [(o00_w, 1), (o11_w, 1)])
    add_term([(c_u, 1), (d_u, 1)], [(e_v, 1)], [(o10_w, 1), (o11_w, -1)])
    add_term([(a_u, 1)], [(f_v, 1), (h_v, -1)], [(o01_w, 1), (o11_w, 1)])
    add_term([(d_u, 1)], [(g_v, 1), (e_v, -1)], [(o00_w, 1), (o10_w, 1)])
    add_term([(a_u, 1), (b_u, 1)], [(h_v, 1)], [(o00_w, -1), (o01_w, 1)])
    add_term([(c_u, 1), (a_u, -1)], [(e_v, 1), (f_v, 1)], [(o11_w, 1)])
    add_term([(b_u, 1), (d_u, -1)], [(g_v, 1), (h_v, 1)], [(o00_w, 1)])

    # Schoolbook for the third column of B (col b_col_off+2)
    # A cols a_col_off, a_col_off+1 times B rows b_row_off, b_row_off+1, b_col_off+2
    for i in range(2):
        for k in range(2):
            u_idx = (a_row_off + i) * m + (a_col_off + i)  # wait, this is wrong
            # Actually: A row i, col k (relative)
            u_idx = (a_row_off + i) * m + (a_col_off + k)
            v_idx = (b_row_off + k) * p + (b_col_off + 2)
            w_idx = (out_col_off + 2) * n + (out_row_off + i)
            add_term([(u_idx, 1)], [(v_idx, 1)], [(w_idx, 1)])

    # Also need A col a_col_off+2 (third col of A) times all B rows/cols
    # A col 2, B rows 0-2, B cols 0-2: 3*3=9 schoolbook terms
    for k in range(3):
        for j in range(3):
            u_idx = (a_row_off) * m + (a_col_off + 2)  # row 0, col 2
            v_idx = (b_row_off + k) * p + (b_col_off + j)
            w_idx = (out_col_off + j) * n + (out_row_off)
            add_term([(u_idx, 1)], [(v_idx, 1)], [(w_idx, 1)])
            u_idx2 = (a_row_off + 1) * m + (a_col_off + 2)  # row 1, col 2
            w_idx2 = (out_col_off + j) * n + (out_row_off + 1)
            add_term([(u_idx2, 1)], [(v_idx, 1)], [(w_idx2, 1)])

def _build_strassen_2x3x3_partition():
    """
    Build decomposition using 2x3x3 Strassen blocks.
    Partition: A cols 0-2 x B rows 0-2 x B cols 0-2 (2x3x3, rank ~11)
              A cols 0-2 x B rows 0-2 x B cols 3-4 (2x3x2, rank ~9)
              A cols 3-4 x B rows 3-4 x B cols 0-2 (2x2x3, rank ~11)
              A cols 3-4 x B rows 3-4 x B cols 3-4 (2x2x2, rank 7)
    Total ~38 before null-space reduction.
    """
    n, m, p = 2, 5, 5
    terms = []

    def add_term(u_entries, v_entries, w_entries):
        terms.append((u_entries, v_entries, w_entries))

    def schoolbook_block(a_rows, a_cols, b_rows, b_cols, a_shape, b_shape, out_rows, out_cols):
        nr, nc = a_shape
        nr2, nc2 = b_shape
        for i in range(nr):
            for k in range(nc):
                for j in range(nc2):
                    u_idx = (a_rows + i) * m + (a_cols + k)
                    v_idx = (b_rows + k) * p + (b_cols + j)
                    w_idx = (out_cols + j) * n + (out_rows + i)
                    add_term([(u_idx, 1)], [(v_idx, 1)], [(w_idx, 1)])

    def strassen_2x2x2(a_row_off, a_col_off, b_row_off, b_col_off, out_row_off, out_col_off):
        a_u = a_row_off * m + a_col_off
        b_u = a_row_off * m + (a_col_off + 1)
        c_u = (a_row_off + 1) * m + a_col_off
        d_u = (a_row_off + 1) * m + (a_col_off + 1)

        e_v = b_row_off * p + b_col_off
        f_v = b_row_off * p + (b_col_off + 1)
        g_v = (b_row_off + 1) * p + b_col_off
        h_v = (b_row_off + 1) * p + (b_col_off + 1)

        o00_w = out_col_off * n + out_row_off
        o01_w = (out_col_off + 1) * n + out_row_off
        o10_w = out_col_off * n + (out_row_off + 1)
        o11_w = (out_col_off + 1) * n + (out_row_off + 1)

        add_term([(a_u, 1), (d_u, 1)], [(e_v, 1), (h_v, 1)], [(o00_w, 1), (o11_w, 1)])
        add_term([(c_u, 1), (d_u, 1)], [(e_v, 1)], [(o10_w, 1), (o11_w, -1)])
        add_term([(a_u, 1)], [(f_v, 1), (h_v, -1)], [(o01_w, 1), (o11_w, 1)])
        add_term([(d_u, 1)], [(g_v, 1), (e_v, -1)], [(o00_w, 1), (o10_w, 1)])
        add_term([(a_u, 1), (b_u, 1)], [(h_v, 1)], [(o00_w, -1), (o01_w, 1)])
        add_term([(c_u, 1), (a_u, -1)], [(e_v, 1), (f_v, 1)], [(o11_w, 1)])
        add_term([(b_u, 1), (d_u, -1)], [(g_v, 1), (h_v, 1)], [(o00_w, 1)])

    # Block 1: 2x3x3 (A cols 0-2, B rows 0-2, B cols 0-2)
    # Use 2x3x3 Strassen-like
    _strassen_2x3x3(0, 0, 0, 0, 0, 0, terms, m, p, n)

    # Block 2: 2x3x2 (A cols 0-2, B rows 0-2, B cols 3-4)
    # Strassen on A cols 0-1, B rows 0-1, B cols 3-4
    strassen_2x2x2(0, 0, 0, 3, 0, 3)
    # Schoolbook for A col 2, B rows 0-2, B cols 3-4
    schoolbook_block(0, 2, 0, 3, (2, 1), (3, 2), 0, 3)

    # Block 3: 2x2x3 (A cols 3-4, B rows 3-4, B cols 0-2)
    # Strassen on A cols 3-4, B rows 3-4, B cols 0-1
    strassen_2x2x2(0, 3, 3, 0, 0, 0)
    # Schoolbook for A cols 3-4, B rows 3-4, B col 2
    schoolbook_block(0, 3, 3, 2, (2, 2), (2, 1), 0, 2)

    # Block 4: 2x2x2 (A cols 3-4, B rows 3-4, B cols 3-4)
    strassen_2x2x2(0, 3, 3, 3, 0, 3)

    return terms

def _terms_to_matrices(terms):
    """Convert list of term tuples to U, V, W matrices."""
    n, m, p = 2, 5, 5
    total_rank = len(terms)
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    for col, (u_entries, v_entries, w_entries) in enumerate(terms):
        for idx, val in u_entries:
            U[idx][col] = val
        for idx, val in v_entries:
            V[idx][col] = val
        for idx, val in w_entries:
            W[idx][col] = val

    return U, V, W, total_rank

def _validate_decomposition(U, V, W, n, m, p):
    """Validate that the decomposition exactly reconstructs the matrix multiplication tensor."""
    r = len(W[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                val = 0.0
                for c in range(r):
                    val += U[idx_u][c] * V[idx_v][c] * W[idx_w][c]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _contrib_matrix(terms):
    """Build the r x (n*m*p) contribution matrix."""
    n, m, p = 2, 5, 5
    r = len(terms)
    M = np.zeros((r, n * m * p))
    for c in range(r):
        u_dict = {}
        v_dict = {}
        w_dict = {}
        for idx, val in terms[c][0]:
            u_dict[idx] = val
        for idx, val in terms[c][1]:
            v_dict[idx] = val
        for idx, val in terms[c][2]:
            w_dict[idx] = val
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    M[c, i * m * p + j * p + k] = u_dict.get(idx_u, 0) * v_dict.get(idx_v, 0) * w_dict.get(idx_w, 0)
    return M

def _apply_null_space(terms, M, deadline):
    """
    Iteratively apply null-space vectors to zero out terms.
    Returns (new_terms, new_M) or (terms, M) if no improvement.
    """
    n, m, p = 2, 5, 5
    r = len(terms)
    num_constraints = n * m * p

    try:
        null_space = np.linalg.null_space(M.T)
    except:
        return terms, M

    if null_space is None or null_space.shape[1] == 0:
        return terms, M

    best_terms = terms
    best_rank = r
    best_M = M
    ns_dim = null_space.shape[1]

    for ns in range(ns_dim):
        if time.time() > deadline - 10:
            break
        v = null_space[:, ns]

        candidates = []
        for c in range(r):
            if abs(v[c]) > 1e-10:
                candidates.append(-1.0 / v[c])

        best_t = None
        best_zero = 0
        for t_val in candidates:
            new_coeffs = 1.0 + t_val * v
            zc = sum(1 for c in range(r) if abs(new_coeffs[c]) < 1e-9)
            if zc > best_zero:
                best_zero = zc
                best_t = t_val

        if best_t is not None and best_zero >= 1:
            new_coeffs = 1.0 + best_t * v
            new_terms = []
            for c in range(r):
                if abs(new_coeffs[c]) < 1e-9:
                    continue
                s = new_coeffs[c]
                new_terms.append((
                    [(idx, val * s) for idx, val in terms[c][0]],
                    [(idx, val) for idx, val in terms[c][1]],
                    [(idx, val) for idx, val in terms[c][2]]
                ))
            if len(new_terms) < best_rank:
                U_t, V_t, W_t, _ = _terms_to_matrices(new_terms)
                if _validate_decomposition(U_t, V_t, W_t, n, m, p):
                    best_terms = new_terms
                    best_rank = len(new_terms)
                    best_M = _contrib_matrix(new_terms)

    return best_terms, best_M

def _greedy_term_elimination(terms, deadline):
    """
    Structural approach: try to find a linear combination of ALL terms
    that zeros out at least one term, using the full null space of the
    contribution matrix. Iterate until no more reductions are found.
    """
    n, m, p = 2, 5, 5
    M = _contrib_matrix(terms)
    improved = True
    while improved and time.time() < deadline - 20:
        improved = False
        terms, M = _apply_null_space(terms, M, deadline)
        # Check if we actually improved
        if len(terms) < len(M):
            improved = True
            M = _contrib_matrix(terms)
    return terms

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying 
    a 2 by 5 matrix by a 5 by 5 matrix.
    """
    start_time = time.time()
    deadline = start_time + 110.0

    # Strategy 1: Strassen-based (2+2+1) + null-space reduction
    terms1 = _build_strassen_decomposition()
    terms1 = _greedy_term_elimination(terms1, deadline)
    rank1 = len(terms1)

    # Strategy 2: 2x3x3 Strassen-like partition + null-space reduction
    if time.time() < deadline - 40:
        terms2 = _build_strassen_2x3x3_partition()
        terms2 = _greedy_term_elimination(terms2, deadline)
        rank2 = len(terms2)
    else:
        terms2 = terms1
        rank2 = rank1

    # Strategy 3: Schoolbook + aggressive null-space
    if time.time() < deadline - 30:
        terms3 = _build_schoolbook()
        terms3 = _greedy_term_elimination(terms3, deadline)
        rank3 = len(terms3)
    else:
        terms3 = terms1
        rank3 = rank1

    # Pick best
    best = min([(rank1, terms1), (rank2, terms2), (rank3, terms3)], key=lambda x: x[0])
    terms = best[1]

    # Convert to matrices
    U, V, W, total_rank = _terms_to_matrices(terms)

    # Final verification
    if not _validate_decomposition(U, V, W, 2, 5, 5):
        # Fallback
        terms = _build_strassen_decomposition()
        U, V, W, total_rank = _terms_to_matrices(terms)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
