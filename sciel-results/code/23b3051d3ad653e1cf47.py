"""Rewritten: search for low-rank exact tensor decomposition of 4x5x6 matrix mult."""
import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def _verify(U, V, W, n, m, p, tol=1e-8):
    """Check whether U,V,W exactly decompose the tensor."""
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    r = U.shape[1]
    Ur = U.reshape(n, m, r)
    Vr = V.reshape(m, p, r)
    Wr = W.reshape(p, n, r)
    T = np.einsum('ijr,jkr,kin->ijk', Ur, Vr, Wr)
    max_err = np.max(np.abs(T - 1.0))
    return max_err < tol, max_err

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 130.0

    # Schoolbook decomposition
    rank0 = n * m * p
    U0 = np.zeros((n * m, rank0))
    V0 = np.zeros((m * p, rank0))
    W0 = np.zeros((p * n, rank0))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j, col] = 1.0
                V0[j * p + k, col] = 1.0
                W0[k * n + i, col] = 1.0
                col += 1

    best_rank = rank0
    best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()

    # Try gradient descent from random initializations at various ranks
    np.random.seed(42)
    
    def run_gd(U_init, V_init, W_init, r, steps=100, lr_init=0.02, lr_decay=0.998):
        Ur = U_init.reshape(n, m, r).copy()
        Vr = V_init.reshape(m, p, r).copy()
        Wr = W_init.reshape(p, n, r).copy()
        lr = lr_init
        for step in range(steps):
            T = np.einsum('ijr,jkr,kin->ijk', Ur, Vr, Wr)
            residual = T - 1.0
            loss = 0.5 * np.sum(residual ** 2)
            if loss < 1e-12:
                break
            dUr = np.einsum('ijk,jkr,kin->ijr', residual, Vr, Wr)
            dVr = np.einsum('ijk,ijr,kin->jkr', residual, Ur, Wr)
            dWr = np.einsum('ijk,ijr,jkr->kin', residual, Ur, Vr)
            Ur -= lr * dUr
            Vr -= lr * dVr
            Wr -= lr * dWr
            lr *= lr_decay
            if time.time() > deadline - 5.0:
                break
        return Ur.reshape(n*m, r), Vr.reshape(m*p, r), Wr.reshape(p*n, r), loss

    # Try ranks from 50 to 110
    for r in range(50, 115, 5):
        if time.time() > deadline - 15.0:
            break
        if r >= best_rank:
            continue
        sigma = (1.0 / r) ** (1.0 / 6.0)
        for restart in range(2):
            if time.time() > deadline - 10.0:
                break
            U_rand = np.random.randn(n*m, r) * sigma
            V_rand = np.random.randn(m*p, r) * sigma
            W_rand = np.random.randn(p*n, r) * sigma
            U_gd, V_gd, W_gd, loss = run_gd(U_rand, V_rand, W_rand, r)
            ok, err = _verify(U_gd, V_gd, W_gd, n, m, p, tol=1e-5)
            if ok and r < best_rank:
                best_rank = r
                best_U, best_V, best_W = U_gd, V_gd, W_gd
                print(f"Found rank {r}, err={err:.2e}")

    if best_rank == rank0:
        return {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}
    else:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
