import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-49 decomposition for 4x4 matrix multiplication via recursive Strassen,
    then reduced by simultaneous linear elimination (exact, integer-preserving).

    Convention:
        U[i*4+k, r]  -> coefficient of A[i,k] in term r
        V[k*4+j, r]  -> coefficient of B[k,j] in term r
        W[j*4+i, r]  -> coefficient of C[i,j] in term r  (trace convention W[k*n+i,r])
    """
    start_time = time.time()
    DEADLINE = 140  # safety margin before the 150s kill

    # ---- Build recursive Strassen (7 outer x 7 inner = 49 rank-1 terms) ----
    def _build_strassen49():
        U = np.zeros((16, 49), dtype=int)
        V = np.zeros((16, 49), dtype=int)
        W = np.zeros((16, 49), dtype=int)

        def u_idx(i, k): return i * 4 + k
        def v_idx(k, j): return k * 4 + j
        def w_idx(i, j): return j * 4 + i   # trace convention

        # Outer Strassen (4x4 -> 2x2 blocks): (A_blocks, B_blocks, C_blocks)
        outer_terms = [
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
            ({(1,0): 1, (1,1): 1}, {(0,0): 1},           {(1,0): 1, (1,1): -1}),
            ({(0,0): 1},           {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
            ({(1,1): 1},           {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
            ({(0,0): 1, (0,1): 1}, {(1,1): 1},           {(0,0): -1, (0,1): 1}),
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
        ]

        # Inner Strassen (2x2 -> scalar): (A_elems, B_elems, C_elems)
        inner_terms = [
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
            ({(1,0): 1, (1,1): 1}, {(0,0): 1},           {(1,0): 1, (1,1): -1}),
            ({(0,0): 1},           {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
            ({(1,1): 1},           {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
            ({(0,0): 1, (0,1): 1}, {(1,1): 1},           {(0,0): -1, (0,1): 1}),
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
        ]

        for t in range(7):
            a_blocks, b_blocks, c_blocks = outer_terms[t]
            for s in range(7):
                a_elems, b_elems, c_elems = inner_terms[s]
                r = t * 7 + s

                for (br, bc), bc_coef in a_blocks.items():
                    for (er, ec), ec_coef in a_elems.items():
                        row = 2 * br + er
                        col = 2 * bc + ec
                        U[u_idx(row, col), r] += bc_coef * ec_coef

                for (br, bc), bc_coef in b_blocks.items():
                    for (er, ec), ec_coef in b_elems.items():
                        row = 2 * br + er
                        col = 2 * bc + ec
                        V[v_idx(row, col), r] += bc_coef * ec_coef

                for (br, bc), bc_coef in c_blocks.items():
                    for (er, ec), ec_coef in c_elems.items():
                        row = 2 * br + er
                        col = 2 * bc + ec
                        W[w_idx(row, col), r] += bc_coef * ec_coef
        return U, V, W

    U, V, W = _build_strassen49()

    def _verify(Ua, Va, Wa):
        n = 4
        rank = Ua.shape[1]
        U4 = Ua.reshape(n, n, rank)   # [i, k, r]
        V4 = Va.reshape(n, n, rank)   # [k, j, r]
        W4 = Wa.reshape(n, n, rank)   # [j, i, r]
        rng = np.random.RandomState(12345)
        A = rng.randn(n, n)
        B = rng.randn(n, n)
        Cexp = A @ B
        u = np.einsum('ikr,ik->ir', U4, A)
        v = np.einsum('kjr,kj->jr', V4, B)
        Cc = np.einsum('jir,ir,jr->ij', W4, u, v)
        return float(np.max(np.abs(Cc - Cexp)))

    if _verify(U, V, W) > 1e-9:
        # Fallback: trivial rank-64 (guaranteed correct)
        Uf = np.zeros((16, 64), dtype=int)
        Vf = np.zeros((16, 64), dtype=int)
        Wf = np.zeros((16, 64), dtype=int)
        for i in range(4):
            for k in range(4):
                for j in range(4):
                    r = i * 16 + k * 4 + j
                    Uf[i*4+k, r] = 1
                    Vf[k*4+j, r] = 1
                    Wf[j*4+i, r] = 1
        return {"U": Uf.tolist(), "V": Vf.tolist(), "W": Wf.tolist()}

    best_U, best_V, best_W = U, V, W

    # ---- Simultaneous Gaussian elimination (exact integer arithmetic) ----
    # Column r is redundant if it lies in the column-span of columns 0..r-1 of U,
    # V, and W simultaneously. We eliminate it from the later columns, then drop it.
    def _reduce(Ua, Va, Wa):
        Ua = Ua.astype(object).copy()
        Va = Va.astype(object).copy()
        Wa = Wa.astype(object).copy()
        n_rows, R = Ua.shape
        keep = []
        for r in range(R):
            if time.time() > DEADLINE:
                break
            # find pivot column p < r with a nonzero entry in Ua
            p = None
            for c in range(r):
                if np.any(Ua[:, c] != 0):
                    p = c
                    break
            if p is None:
                keep.append(r)
                continue
            # pivot row q where Ua[:, p] != 0
            q = int(np.nonzero(Ua[:, p] != 0)[0][0])
            pv = Ua[q, p]
            # eliminate column r from all later columns
            for c in range(r + 1, R):
                factor = Ua[q, c]
                if factor != 0:
                    Ua[:, c] = Ua[:, c] * pv - factor * Ua[:, p]
                    Va[:, c] = Va[:, c] * pv - factor * Va[:, p]
                    Wa[:, c] = Wa[:, c] * pv - factor * Wa[:, p]
            # if column r now has no nonzero entry in Ua it is dependent -> drop
            if not np.any(Ua[:, r] != 0):
                continue
            keep.append(r)
        return Ua[:, keep], Va[:, keep], Wa[:, keep]

    U2, V2, W2 = _reduce(U, V, W)
    if _verify(U2, V2, W2) <= 1e-9 and U2.shape[1] < best_U.shape[1]:
        best_U, best_V, best_W = U2, V2, W2

    # ---- Optional: try random column permutations to expose more dependencies ----
    if time.time() < DEADLINE - 5:
        rng = np.random.RandomState(7)
        for _ in range(20):
            if time.time() > DEADLINE - 3:
                break
            perm = rng.permutation(49)
            Ur, Vr, Wr = _reduce(U[:, perm], V[:, perm], W[:, perm])
            if _verify(Ur, Vr, Wr) <= 1e-9 and Ur.shape[1] < best_U.shape[1]:
                best_U, best_V, best_W = Ur, Vr, Wr

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
