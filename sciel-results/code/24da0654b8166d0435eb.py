import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 130

    n, m, p = 2, 3, 9

    # Build the target tensor T[i,j,k]
    # i = a*m + am, j = b*p + am, k = b*n + a
    # T[i,j,k] = 1 iff am == (j // p)
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n * m):
        ai = i // m
        am = i % m
        for j in range(m * p):
            bj = j // p
            bm = j % p
            if am == bj:
                k = bm * n + ai
                T[i, j, k] = 1.0

    # Start with schoolbook decomposition (rank = n*m*p = 2*3*9 = 54)
    rank = 54
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    col = 0
    for a in range(n):
        for am in range(m):
            for k in range(p):
                i = a * m + am
                j = am * p + k
                w_idx = k * n + a
                U[i, col] = 1.0
                V[j, col] = 1.0
                W[w_idx, col] = 1.0
                col += 1

    # Verify the decomposition
    decomp = np.einsum('ir,jr,kr->ijk', U, V, W)
    if not np.allclose(decomp, T, atol=1e-10):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank
    current_U, current_V, current_W = U.copy(), V.copy(), W.copy()
    current_rank = rank

    def compute_residual(Um, Vm, Wm):
        d = np.einsum('ir,jr,kr->ijk', Um, Vm, Wm)
        return np.sum((d - T)**2)

    # Phase 1: Single column removal (greedy)
    while True:
        if time.time() > deadline - 30:
            break
        improved = False
        for col_to_remove in range(current_rank):
            if time.time() > deadline - 30:
                break
            mask = np.ones(current_rank, dtype=bool)
            mask[col_to_remove] = False
            U_new = current_U[:, mask]
            V_new = current_V[:, mask]
            W_new = current_W[:, mask]
            residual = compute_residual(U_new, V_new, W_new)
            if residual < 1e-8:
                current_U = U_new
                current_V = V_new
                current_W = W_new
                current_rank -= 1
                improved = True
                break
        if not improved:
            break

    if current_rank < best_rank:
        best_rank = current_rank
        best_U, best_V, best_W = current_U.copy(), V.copy(), W.copy()

    # Phase 2: Pair column removal
    while True:
        if time.time() > deadline - 15:
            break
        improved = False
        r = current_rank
        for i in range(r):
            if time.time() > deadline - 15:
                break
            for j in range(i + 1, r):
                if time.time() > deadline - 15:
                    break
                mask = np.ones(r, dtype=bool)
                mask[i] = False
                mask[j] = False
                U_new = current_U[:, mask]
                V_new = current_V[:, mask]
                W_new = current_W[:, mask]
                residual = compute_residual(U_new, V_new, W_new)
                if residual < 1e-8:
                    current_U = U_new
                    current_V = V_new
                    current_W = W_new
                    current_rank -= 2
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break

    if current_rank < best_rank:
        best_rank = current_rank
        best_U, best_V, best_W = current_U.copy(), V.copy(), W.copy()

    # Final: try single column removal one more time
    while True:
        if time.time() > deadline - 5:
            break
        improved = False
        for col_to_remove in range(current_rank):
            if time.time() > deadline - 5:
                break
            mask = np.ones(current_rank, dtype=bool)
            mask[col_to_remove] = False
            U_new = current_U[:, mask]
            V_new = current_V[:, mask]
            W_new = current_W[:, mask]
            residual = compute_residual(U_new, V_new, W_new)
            if residual < 1e-8:
                current_U = U_new
                current_V = V_new
                current_W = W_new
                current_rank -= 1
                improved = True
                break
        if not improved:
            break

    if current_rank < best_rank:
        best_rank = current_rank
        best_U, best_V, best_W = current_U.copy(), V.copy(), W.copy()

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
