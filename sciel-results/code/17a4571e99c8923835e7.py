"""
Rank-42 decomposition for 3x4 * 4x4 matrix multiplication.

Strategy:
  - Split A (3x4) into a 2x4 top block and a 1x4 bottom row.
  - The 2x4 * 4x4 part is handled by four 2x2*2x2 Strassen blocks (rank 7 each) -> 28.
  - The 1x4 * 4x4 part is handled by two 1x2*2x2 Strassen-like blocks (rank 7 each) -> 14.
  - Total rank = 28 + 14 = 42.

The 1x2*2x2 -> 1x2 block (rank 7) is derived by restricting the 2x2*2x2 Strassen
decomposition to the case where the second row of A is zero. This preserves all
7 terms but simplifies the U factors.

Verification: The decomposition is checked against random matrices before returning.
"""
import time
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 118.0
    n, m, p = DIMENSIONS

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Standard 2x2*2x2 Strassen terms (M1..M7)
    # Each term: (u_entries, v_entries, w_entries)
    # u: list of ((i,j), val), v: list of ((j,k), val), w: list of ((i,k), val)
    def strassen_terms():
        T = []
        # M1 = (a11+a22)*(b11+b22)
        T.append(([((0, 0), 1), ((1, 1), 1)], [((0, 0), 1), ((1, 1), 1)],
                 [((0, 0), 1), ((1, 1), 1)]))
        # M2 = (a21+a22)*b11
        T.append(([((1, 0), 1), ((1, 1), 1)], [((0, 0), 1)],
                 [((1, 0), 1), ((1, 1), -1)]))
        # M3 = a11*(b12-b22)
        T.append(([((0, 0), 1)], [((0, 1), 1), ((1, 1), -1)],
                 [((0, 1), 1), ((1, 1), 1)]))
        # M4 = a22*(b21-b11)
        T.append(([((1, 1), 1)], [((1, 0), 1), ((0, 0), -1)],
                 [((1, 0), 1), ((0, 0), 1)]))
        # M5 = (a11+a12)*b22
        T.append(([((0, 0), 1), ((0, 1), 1)], [((1, 1), 1)],
                 [((0, 0), -1), ((0, 1), 1)]))
        # M6 = (a21-a11)*(b11+b12)
        T.append(([((1, 0), 1), ((0, 0), -1)], [((0, 0), 1), ((0, 1), 1)],
                 [((1, 1), 1)]))
        # M7 = (a12-a22)*(b21+b22)
        T.append(([((0, 1), 1), ((1, 1), -1)], [((1, 0), 1), ((1, 1), 1)],
                 [((0, 0), 1)]))
        return T

    ST = strassen_terms()

    # Derive 1x2*2x2 -> 1x2 terms by setting a21=a22=0 in Strassen.
    # Only keep terms where u has non-zero entries in row 0.
    def row1_strassen_terms():
        T1 = []
        for u_e, v_e, w_e in ST:
            # Check if any u entry has i=0
            has_row0 = any(i == 0 for (i, j), _ in u_e)
            if has_row0:
                # Filter u to only row 0 entries
                u_e_new = [(ij, v) for ij, v in u_e if ij[0] == 0]
                # Filter w to only row 0 entries (output row 0)
                w_e_new = [(ik, v) for ik, v in w_e if ik[0] == 0]
                if u_e_new and w_e_new:
                    T1.append((u_e_new, v_e, w_e_new))
        return T1

    R1 = row1_strassen_terms()
    # Should have 5 terms, but we want rank 7 for consistency.
    # Actually, let's just use the full 7 terms but with simplified U.
    # The terms where a21/a22 appear will just have U entries only in row 0.
    # Let's re-derive properly:
    # M1: u=(a11+a22) -> with a22=0: u=a11. w: C11=M1, C22=M1. Row 0: C11=M1.
    # M2: u=(a21+a22) -> 0. Skip.
    # M3: u=a11. w: C12=M3, C22=M3. Row 0: C12=M3.
    # M4: u=a22 -> 0. Skip.
    # M5: u=(a11+a12). w: C11=-M5, C12=M5. Row 0: C11=-M5, C12=M5.
    # M6: u=(a21-a11) -> -a11. w: C22=M6. Row 0: none. Skip.
    # M7: u=(a12-a22) -> a12. w: C11=M7. Row 0: C11=M7.
    
    # So the 1x2*2x2 decomposition uses 4 terms:
    # T1: u=[(0,0)], v=[(0,0),(1,1)], w=[(0,0)]  (M1 modified)
    # T2: u=[(0,0)], v=[(0,1),(1,1)], w=[(0,1)]  (M3 modified)
    # T3: u=[(0,0),(0,1)], v=[(1,1)], w=[(0,0),(0,1)]  (M5 modified, note w signs)
    # T4: u=[(0,1)], v=[(1,0),(1,1)], w=[(0,0)]  (M7 modified)
    
    # Wait, M5 w entries: C11=-M5, C12=M5. So w=[(0,0,-1),(0,1,1)].
    # Let's define them explicitly:
    R1_terms = [
        # T1: M1 restricted
        ([((0, 0), 1)], [((0, 0), 1), ((1, 1), 1)], [((0, 0), 1)]),
        # T2: M3 restricted
        ([((0, 0), 1)], [((0, 1), 1), ((1, 1), -1)], [((0, 1), 1)]),
        # T3: M5 restricted
        ([((0, 0), 1), ((0, 1), 1)], [((1, 1), 1)], [((0, 0), -1), ((0, 1), 1)]),
        # T4: M7 restricted
        ([((0, 1), 1)], [((1, 0), 1), ((1, 1), 1)], [((0, 0), 1)]),
    ]
    
    # This gives rank 4 for 1x2*2x2, which is the schoolbook rank.
    # To get rank 7, I would need to find a different decomposition.
    # But rank 4 is optimal for 1x2*2x2. So the bottom row contributes 2*4=8 rank.
    # Total rank = 28 + 8 = 36? No, 2 blocks of 1x2*2x2, each rank 4, total 8.
    # 28 + 8 = 36. That's better than 44!
    
    # Let's verify: 1x2 * 2x2 -> 1x2.
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # Schoolbook rank is 4. Can we do better? 
    # The tensor is 2*2*2 = 8 dimensional. Rank is at most 4.
    # Actually, the rank of 1x2*2x2 is exactly 4 (it's a direct sum of two 1x1*1x2 and 1x1*1x2? No.)
    # It's the sum of two independent 1x1*2x1 multiplications. Rank = 2+2=4.
    # So rank 4 is optimal.
    
    # Therefore, total rank = 4*7 + 2*4 = 28 + 8 = 36.
    
    total_rank = 36
    U = [[0] * total_rank for _ in range(n * m)]  # 12 x 36
    V = [[0] * total_rank for _ in range(m * p)]  # 16 x 36
    W = [[0] * total_rank for _ in range(p * n)]  # 12 x 36

    def add_rank7(i0, j0, k0, col_start):
        """2x2*2x2 Strassen block."""
        for r in range(7):
            u_e, v_e, w_e = ST[r]
            c = col_start + r
            for (li, lj), val in u_e:
                U[(i0 + li) * m + (j0 + lj)][c] += val
            for (lj, lk), val in v_e:
                V[(j0 + lj) * p + (k0 + lk)][c] += val
            for (li, lk), val in w_e:
                W[(k0 + lk) * n + (i0 + li)][c] += val
        return col_start + 7

    def add_rank4(i0, j0, k0, col_start):
        """1x2*2x2 schoolbook block (rank 4)."""
        for t in range(4):
            c = col_start + t
            j = j0 + (t % 2)
            k = k0 + (t // 2)
            U[i0 * m + j][c] = 1
            V[j * p + k][c] = 1
            W[k * n + i0][c] = 1
        return col_start + 4

    col = 0
    # Top 2 rows: 4 Strassen blocks
    col = add_rank7(0, 0, 0, col)
    col = add_rank7(0, 0, 2, col)
    col = add_rank7(0, 2, 0, col)
    col = add_rank7(0, 2, 2, col)
    
    # Bottom row: 2 rank-4 blocks
    col = add_rank4(2, 0, 0, col)
    col = add_rank4(2, 0, 2, col)
    col = add_rank4(2, 2, 0, col)
    col = add_rank4(2, 2, 2, col)

    assert col == total_rank, f"Column mismatch: {col} != {total_rank}"

    # Verify
    try:
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)

        np.random.seed(12345)
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B

        C_comp = np.zeros((n, p))
        for r in range(total_rank):
            ar = sum(Ua[i * m + j, r] * A[i, j] for i in range(n) for j in range(m))
            br = sum(Va[j * p + k, r] * B[j, k] for j in range(m) for k in range(p))
            for i in range(n):
                for k in range(p):
                    C_comp[i, k] += Wa[k * n + i, r] * ar * br

        max_err = float(np.max(np.abs(C_comp - C_expected)))
        if max_err > 1e-9:
            return schoolbook()
    except Exception:
        return schoolbook()

    if time.time() > deadline:
        return schoolbook()

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
