"""Combined program: structural rank-55 Strassen construction + local-search polish."""
import numpy as np
import time

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    deadline = time.time() + 135  # safety margin before 150s kill

    def build_tensor():
        T = np.zeros((nm, mp, pn))
        for i1 in range(n):
            for j1 in range(m):
                for k2 in range(p):
                    T[i1 * m + j1, j1 * p + k2, k2 * n + i1] = 1.0
        return T

    T = build_tensor()

    def schoolbook():
        r = n * m * p
        U = np.zeros((nm, r)); V = np.zeros((mp, r)); W = np.zeros((pn, r))
        c = 0
        for i1 in range(n):
            for j1 in range(m):
                for k2 in range(p):
                    U[i1 * m + j1, c] = 1.0
                    V[j1 * p + k2, c] = 1.0
                    W[k2 * n + i1, c] = 1.0
                    c += 1
        return U, V, W

    def strassen55():
        r = 55
        U = np.zeros((nm, r)); V = np.zeros((mp, r)); W = np.zeros((pn, r))
        c = 0
        # Part A: j=0 handled naively (2 x 10 = 20 multiplications)
        for i1 in range(n):
            for k2 in range(p):
                U[i1 * m + 0, c] = 1.0
                V[0 * p + k2, c] = 1.0
                W[k2 * n + i1, c] = 1.0
                c += 1
        # Part B: j in {1,2} as 2x2 x 2x10, split into 5 blocks of 2x2
        for b in range(5):
            k0 = 2 * b; k1 = 2 * b + 1
            u11 = 0 * m + 1; u12 = 0 * m + 2
            u21 = 1 * m + 1; u22 = 1 * m + 2
            v11 = 1 * p + k0; v12 = 1 * p + k1
            v21 = 2 * p + k0; v22 = 2 * p + k1
            w00 = k0 * n + 0; w01 = k1 * n + 0
            w10 = k0 * n + 1; w11 = k1 * n + 1
            # M1 = (a11+a22)(b11+b22)
            U[u11, c] += 1; U[u22, c] += 1
            V[v11, c] += 1; V[v22, c] += 1
            W[w00, c] += 1; W[w11, c] += 1
            c += 1
            # M2 = (a21+a22)b11
            U[u21, c] += 1; U[u22, c] += 1
            V[v11, c] += 1
            W[w10, c] += 1; W[w11, c] += -1
            c += 1
            # M3 = a11(b12-b22)
            U[u11, c] += 1
            V[v12, c] += 1; V[v22, c] += -1
            W[w01, c] += 1; W[w11, c] += 1
            c += 1
            # M4 = a22(b21-b11)
            U[u22, c] += 1
            V[v21, c] += 1; V[v11, c] += -1
            W[w00, c] += 1; W[w10, c] += 1
            c += 1
            # M5 = (a11+a12)b22
            U[u11, c] += 1; U[u12, c] += 1
            V[v22, c] += 1
            W[w01, c] += 1; W[w11, c] += -1
            c += 1
        return U, V, W

    def reconstruct(U, V, W):
        return np.einsum('ir,jr,kr->ijk', U, V, W)

    def max_err(U, V, W):
        return np.max(np.abs(reconstruct(U, V, W) - T))

    def round_half(U, V, W):
        Ur = np.round(U * 2) / 2
        Vr = np.round(V * 2) / 2
        Wr = np.round(W * 2) / 2
        return Ur, Vr, Wr

    def try_round(U, V, W):
        # half-integer rounding
        Ur, Vr, Wr = round_half(U, V, W)
        if max_err(Ur, Vr, Wr) < 1e-9:
            return Ur, Vr, Wr
        # integer rounding
        Ui, Vi, Wi = np.round(U), np.round(V), np.round(W)
        if max_err(Ui, Vi, Wi) < 1e-9:
            return Ui, Vi, Wi
        # very tight floating-point
        if max_err(U, V, W) < 1e-12:
            return U, V, W
        return None

    def solve_W(U, V):
        # T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
        M = np.einsum('ir,jr->ijr', U, V).reshape(nm * mp, U.shape[1])
        rhs = T.reshape(nm * mp, pn)
        W = np.linalg.lstsq(M, rhs, rcond=None)[0]
        return W.reshape(pn, U.shape[1])

    def solve_U(U, V, W):
        M = np.einsum('jr,kr->jkr', V, W).reshape(mp * pn, U.shape[1])
        rhs = T.reshape(nm, mp * pn)
        Un = np.linalg.lstsq(M.T, rhs.T, rcond=None)[0].T
        return Un.reshape(nm, U.shape[1])

    def solve_V(U, V, W):
        M = np.einsum('ir,kr->ikr', U, W).reshape(nm * pn, V.shape[1])
        rhs = T.reshape(mp, nm * pn)
        Vn = np.linalg.lstsq(M.T, rhs.T, rcond=None)[0].T
        return Vn.reshape(mp, V.shape[1])

    def als(U, V, W, max_iters=20, tol=1e-11):
        for _ in range(max_iters):
            Un = solve_U(U, V, W)
            Vn = solve_V(Un, V, W)
            Wn = solve_W(Un, Vn)
            if np.max(np.abs(Un - U)) < tol and np.max(np.abs(Vn - V)) < tol and np.max(np.abs(Wn - W)) < tol:
                return Un, Vn, Wn
            U, V, W = Un, Vn, Wn
        return U, V, W

    def sparsity(U, V, W, tol=1e-6):
        return (np.abs(U) > tol).sum() + (np.abs(V) > tol).sum() + (np.abs(W) > tol).sum()

    # ---- Build the structural rank-55 decomposition and verify ----
    U55, V55, W55 = strassen55()
    err55 = max_err(U55, V55, W55)

    # Fallback to schoolbook if structural construction is wrong
    if err55 > 1e-8:
        U, V, W = schoolbook()
        best_rank = n * m * p
    else:
        U, V, W = U55, V55, W55
        best_rank = 55

    # Rounding the structural decomposition
    rounded = try_round(U, V, W)
    if rounded is not None:
        U, V, W = rounded

    best_rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_sp = sparsity(best_U, best_V, best_W)

    # ---- Local search: perturb U,V columns (permute + linear combine), re-solve W, round ----
    def local_search(U0, V0, W0, r, n_iter=600):
        best_local = None
        U_c, V_c, W_c = U0.copy(), V0.copy(), W0.copy()
        for it in range(n_iter):
            if time.time() > deadline:
                break
            # Random invertible column mixing on a random subset of U and V
            k = np.random.randint(1, min(4, r) + 1)
            idx = np.random.choice(r, size=k, replace=False)
            G = np.eye(k) + 0.5 * np.random.randn(k, k)
            if np.abs(np.linalg.det(G)) < 1e-6:
                continue
            U_new = U_c.copy(); V_new = V_c.copy()
            U_new[:, idx] = U_new[:, idx] @ G
            V_new[:, idx] = V_new[:, idx] @ G
            # Re-solve W exactly for the new U,V
            W_new = solve_W(U_new, V_new)
            # Try rounding
            cand = try_round(U_new, V_new, W_new)
            if cand is not None:
                U_r, V_r, W_r = cand
                if best_local is None or sparsity(U_r, V_r, W_r) < best_local[0]:
                    best_local = (sparsity(U_r, V_r, W_r), U_r, V_r, W_r)
                    if best_local[0] == 0:
                        break
            # Occasionally accept the perturbed state to keep exploring
            if np.random.rand() < 0.4:
                U_c, V_c, W_c = U_new, V_new, W_new
        return best_local

    # Local search on the rank-55 structural decomposition
    if best_rank == 55:
        best_local = local_search(U, V, W, 55, n_iter=800)
        if best_local is not None:
            sp, U_r, V_r, W_r = best_local
            if sp < best_sp:
                best_sp = sp
                best_U, best_V, best_W = U_r, V_r, W_r

    # ---- Try to find a lower-rank decomposition (54, 53, ...) via ALS + rounding ----
    for r in range(best_rank - 1, max(nm, mp, pn) - 1, -1):
        if time.time() > deadline:
            break
        found = False
        for seed in range(10):
            if time.time() > deadline:
                break
            np.random.seed(1000 + r * 31 + seed)
            if seed < 3:
                U_c = U55[:, :r].copy()
                V_c = V55[:, :r].copy()
                W_c = W55[:, :r].copy()
            else:
                U_c = np.random.randn(nm, r)
                V_c = np.random.randn(mp, r)
                W_c = np.random.randn(pn, r)
            U_c, V_c, W_c = als(U_c, V_c, W_c, max_iters=15)
            cand = try_round(U_c, V_c, W_c)
            if cand is not None:
                U_r, V_r, W_r = cand
                # Local polish
                bl = local_search(U_r, V_r, W_r, r, n_iter=200)
                if bl is not None:
                    U_r, V_r, W_r = bl[1], bl[2], bl[3]
                best_rank = r
                best_U, best_V, best_W = U_r, V_r, W_r
                best_sp = sparsity(best_U, best_V, best_W)
                found = True
                break
        if found:
            if r <= 50:
                break

    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
