"""Search for a low-rank exact tensor decomposition of the 3x4 by 4x4 matmul."""
import time
import random
import itertools
import numpy as np

DIMENSIONS = [3, 4, 4]
DEADLINE = time.time() + 140.0  # safety margin under the 150s kill

def _verify(U, V, W):
    """Check whether factor matrices exactly decompose the matmul tensor.
    U: (n*m, r), V: (m*p, r), W: (p*n, r).
    Tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should be 1.
    """
    n, m, p = DIMENSIONS
    r = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # Build tensor via einsum: T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # Reshape U to (n,m,r), V to (m,p,r), W to (p,n,r)
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    # T[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    # Use einsum: 'ijc,jkc,kic->ijk'
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return np.allclose(T, 1.0, atol=1e-9)

def _schoolbook():
    """Return the schoolbook rank-48 decomposition."""
    n, m, p = DIMENSIONS
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

def _random_init(rank, n, m, p):
    """Random integer/half-integer initialization."""
    U = np.random.choice([-1, 0, 1], size=(n * m, rank))
    V = np.random.choice([-1, 0, 1], size=(m * p, rank))
    W = np.random.choice([-1, 0, 1], size=(p * n, rank))
    return U, V, W

def _residual(U, V, W, n, m, p):
    """Compute residual tensor: T - 1."""
    r = U.shape[1]
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return T - 1.0

def _gradient_step(U, V, W, lr, n, m, p):
    """One gradient descent step on the residual squared."""
    r = U.shape[1]
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    resid = T - 1.0
    # dL/dU3[i,j,c] = 2 * resid[i,j,k] * V3[j,k,c] * W3[k,i,c] summed over k
    dU3 = 2.0 * np.einsum('ijk,jkc,kic->ijc', resid, V3, W3)
    dV3 = 2.0 * np.einsum('ijk,ijc,kic->jkc', resid, U3, W3)
    dW3 = 2.0 * np.einsum('ijk,ijc,jkc->kic', resid, U3, V3)
    dU = dU3.reshape(n * m, r)
    dV = dV3.reshape(m * p, r)
    dW = dW3.reshape(p * n, r)
    U = U - lr * dU
    V = V - lr * dV
    W = W - lr * dW
    return U, V, W

def _round_to_half(X):
    """Round to nearest half-integer."""
    return np.round(X * 2.0) / 2.0

def _try_polish(U, V, W, n, m, p, max_iter=200, tol=1e-12):
    """Try to polish a near-exact decomposition using small perturbations."""
    U = U.astype(float).copy()
    V = V.astype(float).copy()
    W = W.astype(float).copy()
    r = U.shape[1]
    for it in range(max_iter):
        if time.time() > DEADLINE - 5:
            break
        resid = _residual(U, V, W, n, m, p)
        norm = np.linalg.norm(resid)
        if norm < tol:
            break
        # Gradient step
        lr = min(0.01, 1.0 / (norm + 1e-6))
        U, V, W = _gradient_step(U, V, W, lr, n, m, p)
    # Round to half-integers
    U_r = _round_to_half(U)
    V_r = _round_to_half(V)
    W_r = _round_to_half(W)
    if _verify(U_r.tolist(), V_r.tolist(), W_r.tolist()):
        return U_r.tolist(), V_r.tolist(), W_r.tolist()
    return None

def construct():
    n, m, p = DIMENSIONS
    best = _schoolbook()
    best_rank = n * m * p  # 48
    
    # Known lower bound: max(n*m, m*p, p*n) = max(12, 16, 12) = 16
    # Try ranks from 16 upward
    for rank in range(16, best_rank + 1):
        if time.time() > DEADLINE - 10:
            break
        attempts = 0
        max_attempts = 50 if rank <= 20 else 20
        found = False
        for attempt in range(max_attempts):
            if time.time() > DEADLINE - 5:
                break
            U, V, W = _random_init(rank, n, m, p)
            # Gradient descent
            for step in range(300):
                if time.time() > DEADLINE - 5:
                    break
                resid = _residual(U, V, W, n, m, p)
                norm = np.linalg.norm(resid)
                if norm < 1e-10:
                    break
                lr = min(0.05, 1.0 / (norm + 1e-6))
                U, V, W = _gradient_step(U, V, W, lr, n, m, p)
            # Polish
            result = _try_polish(U, V, W, n, m, p, max_iter=300)
            if result is not None:
                U_l, V_l, W_l = result
                if _verify(U_l, V_l, W_l):
                    best = {"U": U_l, "V": V_l, "W": W_l}
                    best_rank = rank
                    found = True
                    break
            attempts += 1
        if found:
            break
    return best

if __name__ == "__main__":
    result = construct()
    print("Rank:", len(result["U"][0]))
    print("Verified:", _verify(result["U"], result["V"], result["W"]))
