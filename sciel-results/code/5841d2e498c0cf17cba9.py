"""Matrix multiplication 3,5,6 - rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s wall-clock

    # Build Strassen-based decomposition
    result = _build_strassen_3x5x6(n, m, p)

    if result is not None:
        U, V, W, rank = result
        # Verify
        if _verify_fast(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: schoolbook
    rank_school = n * m * p
    U_fb = [[0] * rank_school for _ in range(n * m)]
    V_fb = [[0] * rank_school for _ in range(m * p)]
    W_fb = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    return {"U": U_fb, "V": V_fb, "W": W_fb}


def _build_strassen_3x5x6(n, m, p):
    """
    Build tensor decomposition for 3x5 @ 5x6 using a Strassen block structure
    constructed via an explicit terms-accumulation scheme (clean, verifiable).

    Partition:
      n (rows of A): [2, 1]
      m (cols of A / rows of B): [2, 2, 1]
      p (cols of B): [2, 2, 2]

    For each (a_block, c_block) pair we use Strassen (rank 7) when both blocks
    have size 2, otherwise schoolbook.  Total rank = 84.
    """
    n_parts = [2, 1]
    m_parts = [2, 2, 1]
    p_parts = [2, 2, 2]

    # Block start offsets
    n_starts = [0]
    for s in n_parts[:-1]:
        n_starts.append(n_starts[-1] + s)
    m_starts = [0]
    for s in m_parts[:-1]:
        m_starts.append(m_starts[-1] + s)
    p_starts = [0]
    for s in p_parts[:-1]:
        p_starts.append(p_starts[-1] + s)

    terms = []  # each term: (u_entries, v_entries, w_entries)

    def add_term(u_entries, v_entries, w_entries):
        terms.append((u_entries, v_entries, w_entries))

    def add_strassen(a_rows, c_cols, d_cols):
        """Add 7 Strassen terms for a 2x2x2 block.

        Local 2x2x2 matmul (verified):
          C[0,0] = M1 + M4 - M5 + M7
          C[0,1] = M3 + M5
          C[1,0] = M2 + M4
          C[1,1] = M1 - M2 + M3 + M6
        """
        def U(la, lc): return a_rows[la] * m + c_cols[lc]
        def V(lc, ld): return c_cols[lc] * p + d_cols[ld]
        def W(ld, la): return d_cols[ld] * n + a_rows[la]

        # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])  -> C[0,0], C[1,1]
        add_term(
            [(U(0, 0), 1), (U(1, 1), 1)],
            [(V(0, 0), 1), (V(1, 1), 1)],
            [(W(0, 0), 1), (W(1, 1), 1)]
        )
        # M2 = (A[1,0]+A[1,1])*B[0,0]  -> C[1,0], C[1,1](-1)
        add_term(
            [(U(1, 0), 1), (U(1, 1), 1)],
            [(V(0, 0), 1)],
            [(W(0, 1), 1), (W(1, 1), -1)]
        )
        # M3 = A[0,0]*(B[0,1]-B[1,1])  -> C[0,1], C[1,1]
        add_term(
            [(U(0, 0), 1)],
            [(V(0, 1), 1), (V(1, 1), -1)],
            [(W(1, 0), 1), (W(1, 1), 1)]
        )
        # M4 = A[1,1]*(B[1,0]-B[0,0])  -> C[0,0], C[1,0]
        add_term(
            [(U(1, 1), 1)],
            [(V(1, 0), 1), (V(0, 0), -1)],
            [(W(0, 0), 1), (W(0, 1), 1)]
        )
        # M5 = (A[0,0]+A[0,1])*B[1,1]  -> C[0,0](-1), C[0,1]
        add_term(
            [(U(0, 0), 1), (U(0, 1), 1)],
            [(V(1, 1), 1)],
            [(W(0, 0), -1), (W(1, 0), 1)]
        )
        # M6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])  -> C[1,1]
        add_term(
            [(U(1, 0), 1), (U(0, 0), -1)],
            [(V(0, 0), 1), (V(0, 1), 1)],
            [(W(1, 1), 1)]
        )
        # M7 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])  -> C[0,0]
        add_term(
            [(U(0, 1), 1), (U(1, 1), -1)],
            [(V(1, 0), 1), (V(1, 1), 1)],
            [(W(0, 0), 1)]
        )

    def add_schoolbook(a_rows, c_cols, d_cols):
        """Schoolbook for a (|a| x |c|) @ (|c| x |d|) block."""
        def U(la, lc): return a_rows[la] * m + c_cols[lc]
        def V(lc, ld): return c_cols[lc] * p + d_cols[ld]
        def W(ld, la): return d_cols[ld] * n + a_rows[la]

        for la in range(len(a_rows)):
            for lc in range(len(c_cols)):
                for ld in range(len(d_cols)):
                    add_term(
                        [(U(la, lc), 1)],
                        [(V(lc, ld), 1)],
                        [(W(ld, la), 1)]
                    )

    # Assemble blocks
    for bi in range(len(n_parts)):
        for bj in range(len(m_parts)):
            for bk in range(len(p_parts)):
                a_rows = list(range(n_starts[bi], n_starts[bi] + n_parts[bi]))
                c_cols = list(range(m_starts[bj], m_starts[bj] + m_parts[bj]))
                d_cols = list(range(p_starts[bk], p_starts[bk] + p_parts[bk]))

                if len(a_rows) == 2 and len(c_cols) == 2 and len(d_cols) == 2:
                    add_strassen(a_rows, c_cols, d_cols)
                else:
                    add_schoolbook(a_rows, c_cols, d_cols)

    # Accumulate into dense matrices
    rank = len(terms)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    for c, (u_entries, v_entries, w_entries) in enumerate(terms):
        for row, coeff in u_entries:
            U[row, c] += coeff
        for row, coeff in v_entries:
            V[row, c] += coeff
        for row, coeff in w_entries:
            W[row, c] += coeff

    return U, V, W, rank


def _verify_fast(U, V, W, n, m, p):
    """Fast verification using einsum."""
    r = U.shape[1]
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return np.allclose(T, 1.0, atol=1e-9)


def _verify(U, V, W, n, m, p):
    """
    Verify that the tensor decomposition is exact.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)

    r = U.shape[1]

    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)

    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)

    if not np.allclose(T, 1.0, atol=1e-9):
        return False
    return True

# EVOLVE-BLOCK-END
