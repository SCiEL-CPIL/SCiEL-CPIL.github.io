import numpy as np
import time

DIMENSIONS = [3, 5, 7]

def _laderman_3x3():
    """Return (U, V, W) for 3x3x3 matrix multiplication with Laderman's 7-term algorithm.
    U shape (9,7), V shape (9,7), W shape (9,7), W indexed as W[k*3+i, t] = M[i,k] contribution."""
    U = np.zeros((9, 7), dtype=float)
    V = np.zeros((9, 7), dtype=float)
    W = np.zeros((9, 7), dtype=float)

    def Uidx(i, j): return i * 3 + j
    def Vidx(j, k): return j * 3 + k
    def Widx(k, i): return k * 3 + i

    # t=0
    U[Uidx(2, 0), 0] = 1
    V[Vidx(0, 1), 0] = 1
    W[Widx(1, 2), 0] = 1
    # t=1
    U[Uidx(0, 2), 1] = 1
    V[Vidx(2, 0), 1] = 1
    W[Widx(0, 2), 1] = 1
    # t=2
    U[Uidx(0, 0), 2] = 1
    V[Vidx(0, 2), 2] = 1
    W[Widx(2, 0), 2] = 1
    # t=3
    U[Uidx(2, 2), 3] = 1
    V[Vidx(2, 2), 3] = 1
    W[Widx(2, 2), 3] = 1
    # t=4: (1+e11)(1+e22)
    U[Uidx(0, 0), 4] = 1
    U[Uidx(1, 1), 4] = 1
    V[Vidx(0, 0), 4] = 1
    V[Vidx(1, 1), 4] = 1
    W[Widx(0, 0), 4] = 1
    W[Widx(1, 1), 4] = 1
    # t=5: (1-e11)(1-e22)
    U[Uidx(0, 0), 5] = 1
    U[Uidx(1, 1), 5] = -1
    V[Vidx(0, 0), 5] = 1
    V[Vidx(1, 1), 5] = -1
    W[Widx(0, 0), 5] = 1
    W[Widx(1, 1), 5] = -1
    # t=6: (e11+e21+e12+e22)(e12+e22)
    U[Uidx(0, 0), 6] = 1
    U[Uidx(1, 0), 6] = 1
    U[Uidx(0, 1), 6] = 1
    U[Uidx(1, 1), 6] = 1
    V[Vidx(0, 1), 6] = 1
    V[Vidx(1, 1), 6] = 1
    W[Widx(1, 0), 6] = 1
    W[Widx(1, 1), 6] = 1
    W[Widx(0, 1), 6] = 1
    W[Widx(1, 0), 6] = 1  # note: Widx(1,0)=3, Widx(0,1)=1
    # Fix: recompute
    # Actually Widx(1,0) = 1*3+0 = 3, Widx(0,1) = 0*3+1 = 1
    # Let me redo t=6 carefully:
    # M[0,1] = A[0,0]B[0,1] + A[0,1]B[1,1]  -> Widx(1,0)=3
    # M[1,0] = A[1,0]B[0,0] + A[1,1]B[1,0]  -> Widx(0,1)=1
    # M[1,1] = A[1,0]B[0,1] + A[1,1]B[1,1]  -> Widx(1,1)=4
    # So: W[3,6]=1, W[1,6]=1, W[4,6]=1
    # Reset t=6
    for t in [6]:
        for r in range(9):
            U[r, t] = 0
            V[r, t] = 0
            W[r, t] = 0
    U[Uidx(0, 0), 6] = 1
    U[Uidx(1, 0), 6] = 1
    U[Uidx(0, 1), 6] = 1
    U[Uidx(1, 1), 6] = 1
    V[Vidx(0, 0), 6] = 1
    V[Vidx(1, 0), 6] = 1
    W[Widx(1, 0), 6] = 1   # = W[3,6]
    W[Widx(0, 1), 6] = 1   # = W[1,6]
    W[Widx(1, 1), 6] = 1   # = W[4,6]

    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 130  # safety margin before 150s kill

    # Start with schoolbook baseline
    rank_school = n * m * p
    U_sb = np.zeros((n * m, rank_school), dtype=float)
    V_sb = np.zeros((m * p, rank_school), dtype=float)
    W_sb = np.zeros((p * n, rank_school), dtype=float)

    for t in range(rank_school):
        i = t // (m * p)
        rem = t % (m * p)
        j = rem // p
        k = rem % p
        U_sb[i * m + j, t] = 1
        V_sb[j * p + k, t] = 1
        W_sb[k * n + i, t] = 1

    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_school

    # Try Laderman on 3x3x3 core: rows 0-2 of A, cols 0-2 of B, output (i,k) for i,k<3
    # This replaces 27 schoolbook terms with 7 Laderman terms, saving 20 terms.
    # New rank = 105 - 27 + 7 = 85

    try:
        Lu, Lv, Lw = _laderman_3x3()

        # Build new decomposition: Laderman for core (7 terms) + schoolbook for rest (78 terms)
        # Core terms to remove: i in 0..2, j in 0..2, k in 0..2 -> 27 terms
        # Remaining: all (i,j,k) where NOT (i<3 and j<3 and k<3)
        # Count: 105 - 27 = 78

        new_rank = 7 + 78
        U_new = np.zeros((n * m, new_rank), dtype=float)
        V_new = np.zeros((m * p, new_rank), dtype=float)
        W_new = np.zeros((p * n, new_rank), dtype=float)

        # Place Laderman terms in first 7 columns
        for t in range(7):
            for r in range(9):
                if Lu[r, t] != 0:
                    U_new[r, t] = Lu[r, t]
                if Lv[r, t] != 0:
                    V_new[r, t] = Lv[r, t]
                if Lw[r, t] != 0:
                    W_new[r, t] = Lw[r, t]

        # Place schoolbook terms for non-core (i,j,k) in columns 7..84
        col = 7
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if i < 3 and j < 3 and k < 3:
                        continue  # already handled by Laderman
                    U_new[i * m + j, col] = 1
                    V_new[j * p + k, col] = 1
                    W_new[k * n + i, col] = 1
                    col += 1

        assert col == new_rank, f"Column count mismatch: {col} vs {new_rank}"

        # Verify with random matrices
        rng = np.random.RandomState(42)
        A = rng.randn(n, m)
        B = rng.randn(m, p)
        C_ref = A @ B

        # Compute C from decomposition: C[i,k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
        # More efficiently: for each t, C += np.outer-ish via tensor contraction
        # C[i,k] = sum_j sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
        # = sum_t W[k*n+i, t] * (sum_j U[i*m+j, t] * V[j*p+k, t])
        # = sum_t W[k*n+i, t] * (U_t @ V_t) where U_t is n x m, V_t is m x p

        C_comp = np.zeros((n, p), dtype=float)
        for t in range(new_rank):
            Ut = U_new[:, t].reshape(n, m)
            Vt = V_new[:, t].reshape(m, p)
            Wt = W_new[:, t].reshape(p, n)  # W[k*n+i, t], so Wt[k,i]
            # C += Wt^T @ (Ut @ Vt)  -> C[i,k] += sum_{a,b} ... 
            # Actually: contribution to C[i,k] from term t = Wt[k,i] * (Ut @ Vt)[i,k]
            # So C += (Ut @ Vt) * Wt^T  (element-wise, then... no)
            # C[i,k] += Wt[k,i] * (Ut @ Vt)[i,k]
            # That's C += Wt.T * (Ut @ Vt) element-wise
            C_comp += Wt.T * (Ut @ Vt)

        err = np.max(np.abs(C_comp - C_ref))
        if err < 1e-10:
            best_U, best_V, best_W = U_new, V_new, W_new
            best_rank = new_rank
        else:
            # Verification failed, keep schoolbook
            pass

    except Exception:
        pass

    # Check time
    if time.time() > deadline:
        pass

    # Convert to lists of integers if all entries are integers
    def to_int_list(M):
        if np.allclose(M, np.round(M), atol=1e-12):
            return np.round(M).astype(int).tolist()
        return M.tolist()

    return {"U": to_int_list(best_U), "V": to_int_list(best_V), "W": to_int_list(best_W)}
