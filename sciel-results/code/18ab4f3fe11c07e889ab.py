import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """Find rank-minimal exact tensor decomposition for 2x3 * 3x6 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140  # safety margin before the 150 s kill

    n, m, p = DIMENSIONS  # 2, 3, 6
    schoolbook_rank = n * m * p  # 36

    def verify(U, V, W):
        """Exact check via vectorized contraction: C[i,k]=sum_j U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]."""
        C = np.zeros((n, p))
        for j in range(m):
            # (n, r) @ (r, p)
            T = np.einsum('ic,jc->ij', U[j * n:(j + 1) * n, :], V[j * p:(j + 1) * p, :])
            C += T * W.T  # W is (p*n, r); W[k*n+i,c] gives (n,p) after transpose
        max_err = float(np.max(np.abs(C - 1.0))) if C.size else 0.0
        return max_err < 1e-8, max_err

    def schoolbook():
        rank = schoolbook_rank
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def build_block2x2(g):
        """2x3 * 3x2 block (columns g*2, g*2+1). Returns (U(6,r),V(6,r),W(4,r))."""
        U = np.zeros((6, 12))
        V = np.zeros((6, 12))
        W = np.zeros((4, 12))
        col = 0
        for i in range(2):
            for j in range(3):
                for k in range(2):
                    U[i * 3 + j, col] = 1
                    V[j * 2 + k, col] = 1
                    W[k * 2 + i, col] = 1
                    col += 1
        return U, V, W

    def assemble_blocks(rank_per_block):
        U = np.zeros((n * m, rank_per_block * 3))
        V = np.zeros((m * p, rank_per_block * 3))
        W = np.zeros((p * n, rank_per_block * 3))
        for g in range(3):
            Ub, Vb, Wb = build_block2x2(g)
            c0 = g * rank_per_block
            U[:, c0:c0 + rank_per_block] = Ub
            V[:, c0:c0 + rank_per_block] = Vb
            W[:, c0:c0 + rank_per_block] = Wb
        return U, V, W

    def try_rank_direct(rank, max_trials=4, max_iters=40):
        """Alternating least-squares refinement (current-program machinery)."""
        n_eq = n * m * p
        n_u, n_v, n_w = n * m, m * p, p * n
        for trial in range(max_trials):
            if time.time() > deadline - 5:
                return None
            np.random.seed(42 + trial * 17)
            W = np.zeros((n_w, rank))
            for r in range(n_w):
                nz = min(rank, 2 + np.random.randint(2))
                idx = np.random.choice(rank, size=nz, replace=False)
                for c in idx:
                    W[r, c] = np.random.choice([-1.0, 1.0])
            U = np.random.randint(-1, 2, size=(n_u, rank)).astype(float)
            V = np.random.randint(-1, 2, size=(n_v, rank)).astype(float)
            for it in range(max_iters):
                if time.time() > deadline - 5:
                    return None
                Av = np.zeros((n_eq, n_v * rank)); bv = np.ones(n_eq)
                e = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            for c in range(rank):
                                wv = W[k * n + i, c]
                                if wv != 0:
                                    uv = U[i * m + j, c]
                                    if uv != 0:
                                        Av[e, (j * p + k) * rank + c] = uv * wv
                            e += 1
                Vs, *_ = np.linalg.lstsq(Av, bv, rcond=None)
                V = Vs.reshape(n_v, rank)
                Au = np.zeros((n_eq, n_u * rank)); bu = np.ones(n_eq)
                e = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            for c in range(rank):
                                wv = W[k * n + i, c]
                                if wv != 0:
                                    vv = V[j * p + k, c]
                                    if vv != 0:
                                        Au[e, (i * m + j) * rank + c] = vv * wv
                            e += 1
                Us, *_ = np.linalg.lstsq(Au, bu, rcond=None)
                U = Us.reshape(n_u, rank)
                if it % 5 == 0:
                    ok, _ = verify(U, V, W)
                    if ok:
                        return U, V, W
        return None

    best = None
    for r in [33, 32, 30, 28, 26]:
        if time.time() > deadline - 8:
            break
        res = try_rank_direct(r, max_trials=3, max_iters=30)
        if res is not None:
            best = res
            break

    if best is not None:
        U, V, W = best
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    U, V, W = assemble_blocks(12)
    ok, _ = verify(U, V, W)
    if ok:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    U, V, W = schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
