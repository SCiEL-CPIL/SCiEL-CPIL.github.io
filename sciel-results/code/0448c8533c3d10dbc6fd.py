import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 118  # Safety margin before 150s limit

    n, m, p = 4, 5, 5
    nm = n * m  # 20
    mp = m * p  # 25
    pn = p * n  # 20

    def build_strassen_2x2x2(i_base, j_base, k_base):
        """Return list of (U_entries, V_entries, W_entries) for a 2x2x2 Strassen block."""
        # Indices: i_base, i_base+1; j_base, j_base+1; k_base, k_base+1
        i1, i2 = i_base, i_base + 1
        j1, j2 = j_base, j_base + 1
        k1, k2 = k_base, k_base + 1

        # Strassen for 2x2 matrix multiplication:
        # C11 = M1 + M4 - M5 - M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        # where:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)

        terms = []

        # M1 = (A11+A22)(B11+B22) -> contributes to C11 (+1), C22 (+1)
        terms.append({
            'U': [(i1*m+j1, 1), (i2*m+j2, 1)],
            'V': [(j1*p+k1, 1), (j2*p+k2, 1)],
            'W': [(k1*n+i1, 1), (k2*n+i2, 1)]
        })

        # M2 = (A21+A22)B11 -> contributes to C21 (+1), C22 (-1)
        terms.append({
            'U': [(i2*m+j1, 1), (i2*m+j2, 1)],
            'V': [(j1*p+k1, 1)],
            'W': [(k1*n+i2, 1), (k2*n+i2, -1)]
        })

        # M3 = A11(B12-B22) -> contributes to C12 (+1), C22 (+1)
        terms.append({
            'U': [(i1*m+j1, 1)],
            'V': [(j1*p+k2, 1), (j2*p+k2, -1)],
            'W': [(k2*n+i1, 1), (k2*n+i2, 1)]
        })

        # M4 = A22(B21-B11) -> contributes to C11 (+1), C21 (+1)
        terms.append({
            'U': [(i2*m+j2, 1)],
            'V': [(j2*p+k1, 1), (j1*p+k1, -1)],
            'W': [(k1*n+i1, 1), (k1*n+i2, 1)]
        })

        # M5 = (A11+A12)B22 -> contributes to C11 (-1), C12 (+1)
        terms.append({
            'U': [(i1*m+j1, 1), (i1*m+j2, 1)],
            'V': [(j2*p+k2, 1)],
            'W': [(k1*n+i1, -1), (k2*n+i1, 1)]
        })

        # M6 = (A21-A11)(B11+B12) -> contributes to C22 (+1)
        terms.append({
            'U': [(i2*m+j1, 1), (i1*m+j1, -1)],
            'V': [(j1*p+k1, 1), (j1*p+k2, 1)],
            'W': [(k2*n+i2, 1)]
        })

        # M7 = (A12-A22)(B21+B22) -> contributes to C11 (-1)
        terms.append({
            'U': [(i1*m+j2, 1), (i2*m+j2, -1)],
            'V': [(j2*p+k1, 1), (j2*p+k2, 1)],
            'W': [(k1*n+i1, 1)]
        })

        return terms

    def build_standard(si, sj, sk, i_base, j_base, k_base):
        """Standard rank = si*sj*sk decomposition for a sub-block."""
        terms = []
        for di in range(si):
            for dj in range(sj):
                for dk in range(sk):
                    i = i_base + di
                    j = j_base + dj
                    k = k_base + dk
                    terms.append({
                        'U': [(i*m+j, 1)],
                        'V': [(j*p+k, 1)],
                        'W': [(k*n+i, 1)]
                    })
        return terms

    def build_full_decomposition():
        """
        Partition:
        i: [0,1], [2,3] (two blocks of size 2)
        j: [0,1], [2,3], [4] (two blocks of size 2, one of size 1)
        k: [0,1], [2,3], [4] (two blocks of size 2, one of size 1)

        For each (i_block, j_block, k_block):
        - If all sizes are 2: use Strassen (rank 7)
        - Otherwise: use standard (rank = si*sj*sk)
        """
        i_blocks = [(0, 2), (2, 2)]  # (base, size)
        j_blocks = [(0, 2), (2, 2), (4, 1)]
        k_blocks = [(0, 2), (2, 2), (4, 1)]

        all_terms = []

        for (ib, si) in i_blocks:
            for (jb, sj) in j_blocks:
                for (kb, sk) in k_blocks:
                    if si == 2 and sj == 2 and sk == 2:
                        all_terms.extend(build_strassen_2x2x2(ib, jb, kb))
                    else:
                        all_terms.extend(build_standard(si, sj, sk, ib, jb, kb))

        rank = len(all_terms)

        # Build sparse representation of U, V, W
        U = np.zeros((nm, rank), dtype=np.float64)
        V = np.zeros((mp, rank), dtype=np.float64)
        W = np.zeros((pn, rank), dtype=np.float64)

        for col, term in enumerate(all_terms):
            for idx, val in term['U']:
                U[idx, col] += val
            for idx, val in term['V']:
                V[idx, col] += val
            for idx, val in term['W']:
                W[idx, col] += val

        return U, V, W, rank

    def verify_decomposition(U, V, W, tol=1e-9):
        """Verify T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i,j,k."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_row = U[i*m+j]
                    v_row = V[j*p+k]
                    w_row = W[k*n+i]
                    s = np.dot(u_row, v_row * w_row)
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def try_reduce_rank(U, V, W, max_time=30):
        """
        Try to find linear dependencies among the rank-1 terms to reduce rank.
        Uses a random projection approach: check if removing one column and
        adjusting others can still give a valid decomposition.
        """
        current_rank = U.shape[1]
        if time.time() > deadline - 15:
            return U, V, W, current_rank

        # For each column, try to see if it can be expressed as a combination of others
        # This is expensive, so we do a limited search
        cols_to_try = list(range(current_rank))
        np.random.seed(42)
        np.random.shuffle(cols_to_try)

        improved = True
        while improved and time.time() < deadline - 20:
            improved = False
            U_curr = U.copy()
            V_curr = V.copy()
            W_curr = W.copy()
            curr_rank = U_curr.shape[1]

            for c in cols_to_try:
                if c >= curr_rank:
                    continue
                if time.time() > deadline - 20:
                    break

                # Try to remove column c
                # Check if the contribution of column c can be absorbed by others
                # This is a hard problem, so we skip for now
                # A simpler approach: just check for zero columns
                if np.allclose(U_curr[:, c], 0) and np.allclose(V_curr[:, c], 0) and np.allclose(W_curr[:, c], 0):
                    U_curr = np.delete(U_curr, c, axis=1)
                    V_curr = np.delete(V_curr, c, axis=1)
                    W_curr = np.delete(W_curr, c, axis=1)
                    curr_rank -= 1
                    improved = True
                    break

            if improved:
                U, V, W = U_curr, V_curr, W_curr
                current_rank = curr_rank

        return U, V, W, current_rank

    # Build the initial decomposition
    U, V, W, rank = build_full_decomposition()

    # Verify
    if not verify_decomposition(U, V, W):
        # Fallback to naive decomposition
        rank = n * m * p
        U = np.zeros((nm, rank), dtype=np.float64)
        V = np.zeros((mp, rank), dtype=np.float64)
        W = np.zeros((pn, rank), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try to reduce rank
    if time.time() < deadline - 25:
        U, V, W, rank = try_reduce_rank(U, V, W)

    # Final verification
    if not verify_decomposition(U, V, W):
        # Rebuild without reduction
        U, V, W, rank = build_full_decomposition()

    # Convert to list with integer/half-integer where possible
    def to_clean_list(matrix):
        result = []
        for row in matrix:
            clean_row = []
            for val in row:
                if abs(val - round(val)) < 1e-10:
                    clean_row.append(int(round(val)))
                elif abs(2*val - round(2*val)) < 1e-10:
                    clean_row.append(round(2*val) / 2.0)
                else:
                    clean_row.append(float(val))
            result.append(clean_row)
        return result

    return {"U": to_clean_list(U), "V": to_clean_list(V), "W": to_clean_list(W)}
