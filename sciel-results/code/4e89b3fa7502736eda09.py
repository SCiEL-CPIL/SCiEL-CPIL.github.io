import numpy as np
import time
import itertools
import random

DIMENSIONS = [2, 3, 3]

def _build_rank17():
    n, m, p = DIMENSIONS
    U = np.zeros((n * m, 17))
    V = np.zeros((m * p, 17))
    W = np.zeros((p * n, 17))
    
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    U[0, 2] = 1
    V[1, 2] = 1; V[4, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    U[4, 3] = 1
    V[3, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    U[0, 4] = 1; U[1, 4] = 1
    V[4, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    U[1, 6] = 1; U[4, 6] = -1
    V[3, 6] = 1; V[4, 6] = 1
    W[0, 6] = 1
    
    U[2, 7] = 1
    V[6, 7] = 1
    W[0, 7] = 1
    
    U[2, 8] = 1
    V[7, 8] = 1
    W[2, 8] = 1
    
    U[5, 9] = 1
    V[6, 9] = 1
    W[1, 9] = 1
    
    U[5, 10] = 1
    V[7, 10] = 1
    W[3, 10] = 1
    
    U[0, 11] = 1
    V[2, 11] = 1
    W[4, 11] = 1
    
    U[1, 12] = 1
    V[5, 12] = 1
    W[4, 12] = 1
    
    U[2, 13] = 1
    V[8, 13] = 1
    W[4, 13] = 1
    
    U[3, 14] = 1
    V[2, 14] = 1
    W[5, 14] = 1
    
    U[4, 15] = 1
    V[5, 15] = 1
    W[5, 15] = 1
    
    U[5, 16] = 1
    V[8, 16] = 1
    W[5, 16] = 1
    
    return U, V, W

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Vectorized verification using numpy broadcasting."""
    rank = U.shape[1]
    total = np.zeros((n, m, p))
    for r in range(rank):
        Ur = U[:, r].reshape(n, m)
        Vr = V[:, r].reshape(m, p)
        Wr = W[:, r].reshape(p, n)
        total += Ur[:, :, np.newaxis] * Vr[np.newaxis, :, :] * Wr.T[np.newaxis, np.newaxis, :]
    return np.allclose(total, 1.0, atol=tol)

def _verify(U, V, W, n, m, p, tol=1e-9):
    return _verify_fast(U, V, W, n, m, p, tol)

def _try_remove_term(U, V, W, n, m, p, term_idx):
    rank = U.shape[1]
    mask = np.ones(rank, dtype=bool)
    mask[term_idx] = False
    U2 = U[:, mask]
    V2 = V[:, mask]
    W2 = W[:, mask]
    return _verify(U2, V2, W2, n, m, p)

def _search_rank16_by_perturbation(U, V, W, n, m, p, max_iters=5000):
    best = None
    for _ in range(max_iters):
        U2 = U.copy()
        V2 = V.copy()
        W2 = W.copy()
        idx = random.randint(0, 16)
        U2[:, idx] = 0
        V2[:, idx] = 0
        W2[:, idx] = 0
        
        if _verify(U2, V2, W2, n, m, p):
            mask = np.ones(17, dtype=bool)
            mask[idx] = False
            U3 = U2[:, mask]
            V3 = V2[:, mask]
            W3 = W2[:, mask]
            if _verify(U3, V3, W3, n, m, p):
                return U3, V3, W3
    
    return None

def construct():
    start_time = time.time()
    deadline = 145.0
    
    n, m, p = DIMENSIONS
    
    U, V, W = _build_rank17()
    
    if not _verify(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_rank = 17
    best_U, best_V, best_W = U, V, W
    
    for idx in range(17):
        if time.time() > deadline:
            break
        if _try_remove_term(U, V, W, n, m, p, idx):
            mask = np.ones(17, dtype=bool)
            mask[idx] = False
            U16 = U[:, mask]
            V16 = V[:, mask]
            W16 = W[:, mask]
            if _verify(U16, V16, W16, n, m, p):
                best_rank = 16
                best_U, best_V, best_W = U16, V16, W16
                break
    
    if best_rank > 16:
        if time.time() < deadline - 5:
            result = _search_rank16_by_perturbation(U, V, W, n, m, p, max_iters=100000)
            if result is not None:
                best_rank = 16
                best_U, best_V, best_W = result
    
    if best_rank > 15:
        if time.time() < deadline - 5:
            for idx1 in range(17):
                if time.time() > deadline:
                    break
                for idx2 in range(idx1 + 1, 17):
                    mask = np.ones(17, dtype=bool)
                    mask[idx1] = False
                    mask[idx2] = False
                    U15 = U[:, mask]
                    V15 = V[:, mask]
                    W15 = W[:, mask]
                    if _verify(U15, V15, W15, n, m, p):
                        best_rank = 15
                        best_U, best_V, best_W = U15, V15, W15
                        break
                if best_rank == 15:
                    break
    
    if not _verify(best_U, best_V, best_W, n, m, p):
        U, V, W = _build_rank17()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
