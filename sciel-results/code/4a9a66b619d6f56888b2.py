"""Matrix multiplication 3,3,7 tensor decomposition."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np
_START_TIME = time.time()
_DEADLINE = 145.0

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _build_tensor_3x3x3():
    T = np.zeros((9, 9, 9))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                T[i*3+j, j*3+k, k*3+i] = 1.0
    return T

def _try_rank_decomp(T, rank, max_iters=150, tol=1e-8, seed=42):
    np.random.seed(seed)
    n, m, p = T.shape
    U = np.random.randn(n, rank) * 0.5
    V = np.random.randn(m, rank) * 0.5
    W = np.random.randn(p, rank) * 0.5
    best_err = float('inf')
    best_UVW = None
    for it in range(max_iters):
        if _time_left() < 15.0:
            break
        for r in range(rank):
            Vr, Wr = V[:, r], W[:, r]
            denom = np.sum(Vr**2) * np.sum(Wr**2)
            if denom < 1e-12:
                U[:, r] = np.random.randn(n) * 0.1
                continue
            R = T - np.einsum('ir,jr,kr->ijk', np.delete(U, r, axis=1), np.delete(V, r, axis=1), np.delete(W, r, axis=1)) if rank > 1 else T.copy()
            U[:, r] = np.einsum('ijk,j,k->i', R, Vr, Wr) / denom
        for r in range(rank):
            Ur, Wr = U[:, r], W[:, r]
            denom = np.sum(Ur**2) * np.sum(Wr**2)
            if denom < 1e-12:
                V[:, r] = np.random.randn(m) * 0.1
                continue
            R = T - np.einsum('ir,jr,kr->ijk', np.delete(U, r, axis=1), np.delete(V, r, axis=1), np.delete(W, r, axis=1)) if rank > 1 else T.copy()
            V[:, r] = np.einsum('ijk,i,k->j', R, Ur, Wr) / denom
        for r in range(rank):
            Ur, Vr = U[:, r], V[:, r]
            denom = np.sum(Ur**2) * np.sum(Vr**2)
            if denom < 1e-12:
                W[:, r] = np.random.randn(p) * 0.1
                continue
            R = T - np.einsum('ir,jr,kr->ijk', np.delete(U, r, axis=1), np.delete(V, r, axis=1), np.delete(W, r, axis=1)) if rank > 1 else T.copy()
            W[:, r] = np.einsum('ijk,i,j->k', R, Ur, Vr) / denom
        if it % 5 == 0:
            recon = np.einsum('ir,jr,kr->ijk', U, V, W)
            err = np.linalg.norm(T - recon) / (np.linalg.norm(T) + 1e-15)
            if err < best_err:
                best_err = err
                best_UVW = (U.copy(), V.copy(), W.copy())
            if err < tol:
                return U, V, W
    if best_UVW is not None:
        return best_UVW
    return U, V, W

def _find_3x3x3():
    T = _build_tensor_3x3x3()
    for rank in [19, 20, 21, 22, 23]:
        if _time_left() < 25.0:
            break
        for seed in [42, 123, 999]:
            if _time_left() < 20.0:
                break
            U, V, W = _try_rank_decomp(T, rank, max_iters=100, seed=seed)
            recon = np.einsum('ir,jr,kr->ijk', U, V, W)
            err = np.linalg.norm(T - recon) / (np.linalg.norm(T) + 1e-15)
            if err < 1e-6:
                # Try rounding to half-integers
                U_h = np.round(U * 2) / 2.0
                V_h = np.round(V * 2) / 2.0
                W_h = np.round(W * 2) / 2.0
                recon_h = np.einsum('ir,jr,kr->ijk', U_h, V_h, W_h)
                err_h = np.linalg.norm(T - recon_h) / (np.linalg.norm(T) + 1e-15)
                if err_h < 1e-6:
                    return U_h, V_h, W_h, rank
                return U, V, W, rank
    # Schoolbook fallback
    U = np.zeros((9, 27))
    V = np.zeros((9, 27))
    W = np.zeros((9, 27))
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i*3+j, col] = 1
                V[j*3+k, col] = 1
                W[k*3+i, col] = 1
                col += 1
    return U, V, W, 27

def _schoolbook_3x3x1():
    U = np.zeros((9, 9))
    V = np.zeros((3, 9))
    W = np.zeros((3, 9))
    col = 0
    for i in range(3):
        for j in range(3):
            U[i*3+j, col] = 1
            V[j, col] = 1
            W[i, col] = 1
            col += 1
    return U, V, W

def _build_full(U33, V33, W33, rank33, U31, V31, W31):
    total_rank = rank33 + rank33 + 9
    U_full = np.zeros((9, total_rank))
    V_full = np.zeros((21, total_rank))
    W_full = np.zeros((21, total_rank))
    # Group 1: k=0,1,2
    U_full[:, 0:rank33] = U33
    for j in range(3):
        for k in range(3):
            V_full[j*7+k, 0:rank33] = V33[j*3+k, :]
    for i in range(3):
        for k in range(3):
            W_full[k*3+i, 0:rank33] = W33[k*3+i, :]
    # Group 2: k=3,4,5
    U_full[:, rank33:2*rank33] = U33
    for j in range(3):
        for k in range(3):
            V_full[j*7+(k+3), rank33:2*rank33] = V33[j*3+k, :]
    for i in range(3):
        for k in range(3):
            W_full[(k+3)*3+i, rank33:2*rank33] = W33[k*3+i, :]
    # Group 3: k=6
    U_full[:, 2*rank33:2*rank33+9] = U31
    for j in range(3):
        V_full[j*7+6, 2*rank33:2*rank33+9] = V31[j, :]
    for i in range(3):
        W_full[6*3+i, 2*rank33:2*rank33+9] = W31[i, :]
    return U_full, V_full, W_full

def _verify_full(U, V, W):
    # Build full 3x3*3x7 tensor
    T = np.zeros((9, 21, 21))
    for i in range(3):
        for j in range(3):
            for k in range(7):
                T[i*3+j, j*7+k, k*3+i] = 1.0
    recon = np.einsum('ir,jr,kr->ijk', U, V, W)
    return np.linalg.norm(T - recon) / (np.linalg.norm(T) + 1e-15)

def construct():
    n, m, p = DIMENSIONS
    if _time_left() < 30.0:
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    U33, V33, W33, rank33 = _find_3x3x3()
    U31, V31, W31 = _schoolbook_3x3x1()
    
    if _time_left() < 10.0:
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    U_full, V_full, W_full = _build_full(U33, V33, W33, rank33, U31, V31, W31)
    
    # Verify
    err = _verify_full(U_full, V_full, W_full)
    if err < 1e-6:
        return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
