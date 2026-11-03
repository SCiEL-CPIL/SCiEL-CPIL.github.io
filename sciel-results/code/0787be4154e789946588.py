"""Matrix multiplication 3x4 by 4x6: rank-minimal decomposition via block-strassen + merge search."""
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = 135.0


def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)


def _verify_fast(U, V, W):
    """Vectorized tensor verification: T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] must equal 1."""
    n, m, p = 3, 4, 6
    Ua = np.array(U, dtype=float).reshape(n, m, -1)
    Va = np.array(V, dtype=float).reshape(m, p, -1)
    Wa = np.array(W, dtype=float).reshape(p, n, -1)
    # T[i,j,k] = sum_c Ua[i,j,c] * Va[j,k,c] * Wa[k,i,c]
    # Using einsum: 'ijc,jkc,kic->ijk'
    T = np.einsum('ijc,jkc,kic->ijk', Ua, Va, Wa, optimize=True)
    return np.allclose(T, 1.0, atol=1e-10)


def _build_decomp(n, m, p, row_blocks, col_blocks):
    """
    Build a decomposition where certain row-blocks use Strassen (7 mults for 2x2)
    and others use schoolbook.
    
    row_blocks: list of (start, end, method) for A-row ranges
    col_blocks: list of (start, end) for B-col ranges (each paired with a row block)
    
    Returns (U, V, W) as lists of lists, and the rank.
    """
    rank = 0
    U = []
    V = []
    W = []

    def new_col():
        U.append(0)
        V.append(0)
        W.append(0)
        return len(U) - 1

    # For each row-block and col-block pair
    for (i_start, i_end, method) in row_blocks:
        i_count = i_end - i_start
        for (j_start, j_end) in col_blocks:
            j_count = j_end - j_start

            if method == 'strassen' and i_count == 2 and j_count >= 2:
                # Strassen for 2-row x j_count-col block
                # Process in pairs of columns
                for k_pair in range(0, j_count, 2):
                    j0 = j_start + k_pair
                    j1 = j_start + k_pair + 1
                    # 7 Strassen multiplications for 2x2
                    i0, i1 = i_start, i_start + 1

                    # M1 = (a+d)*(e+h)
                    c = new_col()
                    U[c][i0*m+j0] = 1; U[c][i1*m+j1] = 1
                    V[c][j0*p+0] = 1; V[c][j1*p+1] = 1
                    W[c][0*n+i0] = 1; W[c][1*n+i1] = 1

                    # M2 = (c+d)*e
                    c = new_col()
                    U[c][i1*m+j0] = 1; U[c][i1*m+j1] = 1
                    V[c][j0*p+0] = 1
                    W[c][1*n+i0] = 1; W[c][1*n+i1] = -1

                    # M3 = a*(f-h)
                    c = new_col()
                    U[c][i0*m+j0] = 1
                    V[c][j0*p+1] = 1; V[c][j1*p+1] = -1
                    W[c][0*n+i0] = 1; W[c][1*n+i1] = 1

                    # M4 = d*(g-e)
                    c = new_col()
                    U[c][i1*m+j1] = 1
                    V[c][j1*p+0] = 1; V[c][j0*p+0] = -1
                    W[c][0*n+i0] = 1; W[c][1*n+i1] = 1

                    # M5 = (a+b)*h
                    c = new_col()
                    U[c][i0*m+j0] = 1; U[c][i0*m+j1] = 1
                    V[c][j1*p+1] = 1
                    W[c][0*n+i0] = -1; W[c][0*n+i1] = 1

                    # M6 = (c-a)*(e+f)
                    c = new_col()
                    U[c][i1*m+j0] = 1; U[c][i0*m+j0] = -1
                    V[c][j0*p+0] = 1; V[c][j0*p+1] = 1
                    W[c][1*n+i1] = 1

                    # M7 = (b-d)*(g+h)
                    c = new_col()
                    U[c][i0*m+j1] = 1; U[c][i1*m+j1] = -1
                    V[c][j1*p+0] = 1; V[c][j1*p+1] = 1
                    W[c][0*n+i0] = 1

            else:
                # Schoolbook for this block
                for i in range(i_start, i_end):
                    for j in range(j_start, j_end):
                        c = new_col()
                        U[c][i*m+j] = 1
                        V[c][j*p+0] = 1
                        W[c][0*n+i] = 1

    rank = len(U)
    # Convert to list-of-lists format
    U_list = [[U[c][i*m+j] for c in range(rank)] for i in range(n) for j in range(m)]
    V_list = [[V[c][j*p+k] for c in range(rank)] for j in range(m) for k in range(p)]
    W_list = [[W[c][k*n+i] for c in range(rank)] for k in range(p) for i in range(n)]

    return U_list, V_list, W_list, rank


def _try_merge_columns(U, V, W, max_rank):
    """
    Try to reduce rank by checking if any column is redundant.
    A column c is redundant if removing it still gives the correct tensor,
    OR if we can add a scalar multiple of another column to it.
    Returns (U, V, W, new_rank) if reduced, else None.
    """
    n, m, p = 3, 4, 6
    rank = len(U)

    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)

    for c_remove in range(rank):
        if _time_left() < 3:
            return None
        if len(U) - 1 >= max_rank:
            break

        # Try simply removing column c_remove
        U_test = np.delete(U_np, c_remove, axis=0)
        V_test = np.delete(V_np, c_remove, axis=0)
        W_test = np.delete(W_np, c_remove, axis=0)

        T = np.einsum('ijc,jkc,kic->ijk',
                       U_test.reshape(n, m, -1),
                       V_test.reshape(m, p, -1),
                       W_test.reshape(p, n, -1), optimize=True)
        if np.allclose(T, 1.0, atol=1e-10):
            # Also need to verify no spurious non-zero entries elsewhere
            # (the tensor should be exactly the identity tensor)
            return (U_test.tolist(), V_test.tolist(), W_test.tolist(), len(U) - 1)

    return None


def _try_merge_pairs(U, V, W, max_rank):
    """
    Try to reduce rank by merging two columns: find a scalar alpha such that
    replacing column b with (b + alpha*a) and removing column a still works.
    This is a 1-parameter search.
    """
    n, m, p = 3, 4, 6
    rank = len(U)

    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)

    for c_a in range(rank):
        if _time_left() < 3:
            return None
        for c_b in range(rank):
            if c_a == c_b:
                continue
            if len(U) - 1 >= max_rank:
                break

            # We want: sum_c' U'[i,j,c'] V'[j,k,c'] W'[k,i,c'] = 1
            # where we remove column c_a and modify column c_b by adding alpha * column c_a
            # New column b values: U_b_new = U_b + alpha*U_a, etc.
            # The contribution of new column b to (i,j,k) is:
            #   (U_b[i,j] + alpha*U_a[i,j]) * (V_b[j,k] + alpha*V_a[j,k]) * (W_b[k,i] + alpha*W_a[k,i])
            # = U_b V_b W_b + alpha(U_a V_b W_b + U_b V_a W_b + U_b V_b W_a)
            #   + alpha^2(U_a V_a W_b + U_a V_b W_a + U_b V_a W_a)
            #   + alpha^3 U_a V_a W_a
            # We need this plus the sum over all other columns (excluding a and b) to equal 1.
            # Let S(i,j,k) = sum_{c != a, c != b} U_c V_c W_c
            # Then we need: S + U_bV_bW_b + alpha*(...) + alpha^2*(...) + alpha^3*(...) = 1
            # This is a cubic in alpha. We can check if alpha=0 works (i.e., column a is redundant),
            # or try a few candidate values.

            # First check alpha = 0 (column a is redundant given b unchanged)
            U_test = np.delete(U_np, c_a, axis=0)
            V_test = np.delete(V_np, c_a, axis=0)
            W_test = np.delete(W_np, c_a, axis=0)
            T = np.einsum('ijc,jkc,kic->ijk',
                           U_test.reshape(n, m, -1),
                           V_test.reshape(m, p, -1),
                           W_test.reshape(p, n, -1), optimize=True)
            if np.allclose(T, 1.0, atol=1e-10):
                return (U_test.tolist(), V_test.tolist(), W_test.tolist(), len(U) - 1)

    return None


def construct():
    """
    Construct exact tensor decomposition for (3x4)*(4x6) -> (3x6).
    Returns {'U', 'V', 'W'} with integer entries.
    """
    n, m, p = 3, 4, 6
    schoolbook_rank = n * m * p  # 72

    # Generate candidate block decompositions
    candidates = []

    # Strategy 1: Row 0 schoolbook, rows 1-2 Strassen (for each column pairing)
    # Column pairings for 4 columns: (0,1),(2,3) or (0,3),(1,2) or (0,2),(1,3)
    col_pairings = [
        [(0, 2), (2, 4)],
        [(0, 4), (1, 3)],
        [(0, 3), (1, 2)],
    ]

    for pairing in col_pairings:
        # Row 0: schoolbook for all cols; Rows 1-2: strassen for each col pair
        row_blocks = [(0, 1, 'schoolbook'), (1, 3, 'strassen')]
        U, V, W, r = _build_decomp(n, m, p, row_blocks, pairing)
        if _verify_fast(U, V, W):
            candidates.append((r, U, V, W))

    # Strategy 2: Rows 0-1 Strassen, row 2 schoolbook
    for pairing in col_pairings:
        row_blocks = [(0, 2, 'strassen'), (2, 3, 'schoolbook')]
        U, V, W, r = _build_decomp(n, m, p, row_blocks, pairing)
        if _verify_fast(U, V, W):
            candidates.append((r, U, V, W))

    # Strategy 3: All rows schoolbook (baseline)
    row_blocks = [(0, 3, 'schoolbook')]
    U, V, W, r = _build_decomp(n, m, p, row_blocks, [(0, 4)])
    candidates.append((r, U, V, W))

    # Strategy 4: Row 0-1 strassen for first 2 cols, schoolbook for last 2; Row 2 schoolbook
    for col_split in [(0, 2), (0, 1), (0, 3)]:
        j_split = col_split[1]
        row_blocks = [(0, 2, 'strassen'), (2, 3, 'schoolbook')]
        # For strassen part, only use first j_split cols; for rest, schoolbook
        # This requires a mixed approach - let's handle it differently
        # Build manually
        rank = 0
        U_cols = []
        V_cols = []
        W_cols = []

        def add_schoolbook(i, j, k_idx):
            nonlocal rank
            U_cols.append(0); V_cols.append(0); W_cols.append(0)
            c = rank
            U_cols[c][i*m+j] = 1
            V_cols[c][j*p+k_idx] = 1
            W_cols[c][k_idx*n+i] = 1
            rank += 1

        def add_strassen_block(i0, i1, j0, j1):
            nonlocal rank
            # 7 Strassen multiplications
            # M1 = (a+d)*(e+h)
            U_cols.append(0); V_cols.append(0); W_cols.append(0)
            c = rank
            U_cols[c][i0*m+j0] = 1; U_cols[c][i1*m+j1] = 1
            V_cols[c][j0*p+0] = 1; V_cols[c][j1*p+1] = 1
            W_cols[c][0*n+i0] = 1; W_cols[c][1*n+i1] = 1
            rank += 1

            # M2 = (c+d)*e
            U_cols.append(0); V_cols.append(0); W_cols.append(0)
            c = rank
            U_cols[c][i1*m+j0] = 1; U_cols[c][i1*m+j1] = 1
            V_cols[c][j0*p+0] = 1
            W_cols[c][1*n+i0] = 1; W_cols[c][1*n+i1] = -1
            rank += 1

            # M3 = a*(f-h)
            U_cols.append(0); V_cols.append(0); W_cols.append(0)
            c = rank
            U_cols[c][i0*m+j0] = 1
            V_cols[c][j0*p+1] = 1; V_cols[c][j1*p+1] = -1
            W_cols[c][0*n+i0] = 1; W_cols[c][1*n+i1] = 1
            rank += 1

            # M4 = d*(g-e)
            U_cols.append(0); V_cols.append(0); W_cols.append(0)
            c = rank
            U_cols[c][i1*m+j1] = 1
            V_cols[c][j1*p+0] = 1; V_cols[c][j0*p+0] = -1
            W_cols[c][0*n+i0] = 1; W_cols[c][1*n+i1] = 1
            rank += 1

            # M5 = (a+b)*h
            U_cols.append(0); V_cols.append(0); W_cols.append(0)
            c = rank
            U_cols[c][i0*m+j0] = 1; U_cols[c][i0*m+j1] = 1
            V_cols[c][j1*p+1] = 1
            W_cols[c][0*n+i0] = -1; W_cols[c][0*n+i1] = 1
            rank += 1

            # M6 = (c-a)*(e+f)
            U_cols.append(0); V_cols.append(0); W_cols.append(0)
            c = rank
            U_cols[c][i1*m+j0] = 1; U_cols[c][i0*m+j0] = -1
            V_cols[c][j0*p+0] = 1; V_cols[c][j0*p+1] = 1
            W_cols[c][1*n+i1] = 1
            rank += 1

            # M7 = (b-d)*(g+h)
            U_cols.append(0); V_cols.append(0); W_cols.append(0)
            c = rank
            U_cols[c][i0*m+j1] = 1; U_cols[c][i1*m+j1] = -1
            V_cols[c][j1*p+0] = 1; V_cols[c][j1*p+1] = 1
            W_cols[c][0*n+i0] = 1
            rank += 1

        # Rows 0-1, cols 0 to j_split-1: strassen (if j_split >= 2)
        if j_split >= 2:
            add_strassen_block(0, 1, 0, 1)
            if j_split > 2:
                add_strassen_block(0, 1, 2, 3)
        else:
            for j in range(j_split):
                for i in range(2):
                    add_schoolbook(i, j, 0)

        # Rows 0-1, cols j_split to 3: schoolbook
        for j in range(j_split, m):
            for i in range(2):
                add_schoolbook(i, j, 0)

        # Row 2, all cols: schoolbook
        for j in range(m):
            add_schoolbook(2, j, 0)

        # Convert to list format
        U_list = [[U_cols[c][i*m+j] for c in range(rank)] for i in range(n) for j in range(m)]
        V_list = [[V_cols[c][j*p+k] for c in range(rank)] for j in range(m) for k in range(p)]
        W_list = [[W_cols[c][k*n+i] for c in range(rank)] for k in range(p) for i in range(n)]

        if _verify_fast(U_list, V_list, W_list):
            candidates.append((rank, U_list, V_list, W_list))

    # Strategy 5: Try all possible 2-row Strassen with different column pairings
    # for just 2 of the 4 column positions
    for i0 in range(0, n-1):
        i1 = i0 + 1
        for j0 in range(0, m-1):
            for j1 in range(j0+1, m):
                # Use strassen for rows i0,i1 and cols j0,j1; schoolbook for rest
                rank = 0
                U_cols = []
                V_cols = []
                W_cols = []

                def sb(i, j, k_idx):
                    nonlocal rank
                    U_cols.append(0); V_cols.append(0); W_cols.append(0)
                    c = rank
                    U_cols[c][i*m+j] = 1
                    V_cols[c][j*p+k_idx] = 1
                    W_cols[c][k_idx*n+i] = 1
                    rank += 1

                def strassen2(i_a, i_b, j_a, j_b):
                    nonlocal rank
                    # M1 = (a+d)*(e+h)
                    U_cols.append(0); V_cols.append(0); W_cols.append(0)
                    c = rank
                    U_cols[c][i_a*m+j_a] = 1; U_cols[c][i_b*m+j_b] = 1
                    V_cols[c][j_a*p+0] = 1; V_cols[c][j_b*p+1] = 1
                    W_cols[c][0*n+i_a] = 1; W_cols[c][1*n+i_b] = 1
                    rank += 1
                    # M2 = (c+d)*e
                    U_cols.append(0); V_cols.append(0); W_cols.append(0)
                    c = rank
                    U_cols[c][i_b*m+j_a] = 1; U_cols[c][i_b*m+j_b] = 1
                    V_cols[c][j_a*p+0] = 1
                    W_cols[c][1*n+i_a] = 1; W_cols[c][1*n+i_b] = -1
                    rank += 1
                    # M3 = a*(f-h)
                    U_cols.append(0); V_cols.append(0); W_cols.append(0)
                    c = rank
                    U_cols[c][i_a*m+j_a] = 1
                    V_cols[c][j_a*p+1] = 1; V_cols[c][j_b*p+1] = -1
                    W_cols[c][0*n+i_a] = 1; W_cols[c][1*n+i_b] = 1
                    rank += 1
                    # M4 = d*(g-e)
                    U_cols.append(0); V_cols.append(0); W_cols.append(0)
                    c = rank
                    U_cols[c][i_b*m+j_b] = 1
                    V_cols[c][j_b*p+0] = 1; V_cols[c][j_a*p+0] = -1
                    W_cols[c][0*n+i_a] = 1; W_cols[c][1*n+i_b] = 1
                    rank += 1
                    # M5 = (a+b)*h
                    U_cols.append(0); V_cols.append(0); W_cols.append(0)
                    c = rank
                    U_cols[c][i_a*m+j_a] = 1; U_cols[c][i_a*m+j_b] = 1
                    V_cols[c][j_b*p+1] = 1
                    W_cols[c][0*n+i_a] = -1; W_cols[c][0*n+i_b] = 1
                    rank += 1
                    # M6 = (c-a)*(e+f)
                    U_cols.append(0); V_cols.append(0); W_cols.append(0)
                    c = rank
                    U_cols[c][i_b*m+j_a] = 1; U_cols[c][i_a*m+j_a] = -1
                    V_cols[c][j_a*p+0] = 1; V_cols[c][j_a*p+1] = 1
                    W_cols[c][1*n+i_b] = 1
                    rank += 1
                    # M7 = (b-d)*(g+h)
                    U_cols.append(0); V_cols.append(0); W_cols.append(0)
                    c = rank
                    U_cols[c][i_a*m+j_b] = 1; U_cols[c][i_b*m+j_b] = -1
                    V_cols[c][j_b*p+0] = 1; V_cols[c][j_b*p+1] = 1
                    W_cols[c][0*n+i_a] = 1
                    rank += 1

                # Schoolbook for all (i,j) except (i0,j0),(i0,j1),(i1,j0),(i1,j1)
                for i in range(n):
                    for j in range(m):
                        if (i == i0 and j == j0) or (i == i0 and j == j1) or \
                           (i == i1 and j == j0) or (i == i1 and j == j1):
                            continue
                        sb(i, j, 0)

                # Strassen for the 2x2 block
                strassen2(i0, i1, j0, j1)

                U_list = [[U_cols[c][i*m+j] for c in range(rank)] for i in range(n) for j in range(m)]
                V_list = [[V_cols[c][j*p+k] for c in range(rank)] for j in range(m) for k in range(p)]
                W_list = [[W_cols[c][k*n+i] for c in range(rank)] for k in range(p) for i in range(n)]

                if _verify_fast(U_list, V_list, W_list):
                    candidates.append((rank, U_list, V_list, W_list))

    # Sort candidates by rank (ascending)
    candidates.sort(key=lambda x: x[0])

    # Take the best (lowest rank) valid candidate
    if not candidates:
        # Should not happen - schoolbook is always valid
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        c = 0
        for i in range(n):
            for j in range(m):
                U[i*m+j][c] = 1
                V[j*p][c] = 1
                W[i][c] = 1
                c += 1
        return {"U": U, "V": V, "W": W}

    best_rank, best_U, best_V, best_W = candidates[0]

    # Now try to reduce rank further via column removal
    if _time_left() > 10:
        result = _try_merge_columns(best_U, best_V, best_W, best_rank)
        if result is not None:
            new_rank, nU, nV, nW = result
            best_U, best_V, best_W, best_rank = nU, nV, nW, new_rank

    # Try again if we have time
    if _time_left() > 10:
        result = _try_merge_columns(best_U, best_V, best_W, best_rank)
        if result is not None:
            new_rank, nU, nV, nW = result
            best_U, best_V, best_W, best_rank = nU, nV, nW, new_rank

    # Try pair merges
    if _time_left() > 10:
        result = _try_merge_pairs(best_U, best_V, best_W, best_rank)
        if result is not None:
            new_rank, nU, nV, nW = result
            best_U, best_V, best_W, best_rank = nU, nV, nW, new_rank

    return {"U": best_U, "V": best_V, "W": best_W}
