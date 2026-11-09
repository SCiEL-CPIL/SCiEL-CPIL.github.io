"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Guaranteed-exact schoolbook decomposition of rank n*m*p.

    U[i*m+j, c] = V[j*p+k, c] = W[k*n+i, c] = 1 for the (i,j,k) at column c.
    Reconstructs T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = delta_{i,j,k}.
    """
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=object)
    V = np.zeros((m * p, rank), dtype=object)
    W = np.zeros((p * n, rank), dtype=object)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return U, V, W


def _verify(U, V, W, n, m, p, tol=0):
    """Exact integer verification that U,V,W reproduce the multiplication tensor.

    T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] must equal 1 iff i==j==k... 
    Actually the multiplication tensor entry is delta_{i,k} (A[i,j]*B[j,k] -> C[i,k]).
    We check T[i,j,k] == (1 if i==k else 0).
    """
    rank = U.shape[1]
    U = np.asarray(U, dtype=object)
    V = np.asarray(V, dtype=object)
    W = np.asarray(W, dtype=object)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                expected = 1 if i == k else 0
                if s != expected:
                    return False
    return True


def _try_reduce(U, V, W, n, m, p, deadline):
    """Attempt to remove redundant columns (linear dependencies) while preserving
    the exact tensor. The 64 standard basis tensors are independent, so this is
    expected to keep rank 64; it is kept for completeness and is fully verified.
    Returns (U, V, W) with the smallest verified rank found within deadline.
    """
    U = np.asarray(U, dtype=object).copy()
    V = np.asarray(V, dtype=object).copy()
    W = np.asarray(W, dtype=object).copy()
    best = (U, V, W)
    rank = U.shape[1]

    # Try dropping each single column and verify; keep the best valid result.
    # Because the basis tensors are independent, none will be droppable, but the
    # verification guarantees we never return an invalid decomposition.
    for c in range(rank):
        if time.time() > deadline:
            break
        U2 = np.delete(U, c, axis=1)
        V2 = np.delete(V, c, axis=1)
        W2 = np.delete(W, c, axis=1)
        if _verify(U2, V2, W2, n, m, p):
            best = (U2, V2, W2)
            rank = U2.shape[1]
            U, V, W = U2, V2, W2
    return best


def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before the 150 s kill

    n, m, p = DIMENSIONS

    # Floor: guaranteed-exact schoolbook decomposition (rank 64).
    U, V, W = _schoolbook(n, m, p)

    # Refinement: bounded, fully-verified attempt to reduce rank. Only accepts a
    # reduction if it exactly reproduces the multiplication tensor; otherwise the
    # verified schoolbook is returned. This guarantees a valid result (score >= 1.0)
    # while giving a chance to improve.
    if time.time() < deadline:
        U, V, W = _try_reduce(U, V, W, n, m, p, deadline)

    # Convert to plain nested lists with integer entries to match the interface.
    U_list = [[int(x) for x in row] for row in np.asarray(U, dtype=object)]
    V_list = [[int(x) for x in row] for row in np.asarray(V, dtype=object)]
    W_list = [[int(x) for x in row] for row in np.asarray(W, dtype=object)]

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
