"""Matrix multiplication 2x4x4 rank search with deadline guard."""
import time
import random
import numpy as np

DIMENSIONS = [2, 4, 4]
_DEADLINE = None

def _elapsed():
    return time.time() - _DEADLINE[0] if _DEADLINE else 0.0

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
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

def _verify(U, V, W, tol=1e-6):
    # Check that sum_c U[:,c] V[:,c]^T W[:,c]^T == multiplication tensor
    n, m, p = DIMENSIONS
    # Build multiplication tensor T[i,j,k] = 1 for (i,j,k)
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    # Reconstruct: R[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    R = np.zeros((n, m, p))
    for c in range(U.shape[1]):
        u = U[:, c].reshape(n, m)
        v = V[:, c].reshape(m, p)
        w = W[:, c].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    R[i, j, k] += u[i, j] * v[j, k] * w[k, i]
    return np.max(np.abs(R - T)) < tol

def _try_reduce_rank(U, V, W, max_iter=400, tol=1e-7):
    # Try to find a linear relation among the rank-1 tensors.
    # A rank-1 tensor is T_c = U_c (m x p x n). We stack them into a vector.
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    # Vectorize each rank-1 tensor into a length n*m*p vector
    # T_c[i,j,k] = U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    vecs = []
    for c in range(rank):
        v = np.zeros(n * m * p)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    v[i * m * p + j * p + k] = U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
        vecs.append(v)
    M = np.stack(vecs, axis=1)  # shape (nmp, rank)
    # Find null space of M: vector a such that M @ a = 0
    # If null space is non-trivial, we have a relation.
    # Use SVD
    try:
        U_s, s, Vt = np.linalg.svd(M, full_matrices=False)
    except np.linalg.LinAlgError:
        return U, V, W
    # Rank of M
    r = np.sum(s > max(s) * 1e-10)
    null_dim = rank - r
    if null_dim <= 0:
        return U, V, W
    # Null space vectors are columns of Vt[r:]^T
    # For each null vector a, we have sum_c a_c T_c = 0
    # This means we can drop one component and adjust others.
    # We want to drop a component c0 where |a_c0| is large.
    for a in Vt[r:].T:
        # normalize
        a = a / np.max(np.abs(a))
        # find the component with largest |a_c| to drop
        c0 = int(np.argmax(np.abs(a)))
        if abs(a[c0]) < 1e-8:
            continue
        # New rank = rank - 1
        # For each remaining component c, we need to find new U_c', V_c', W_c'
        # such that sum_{c != c0} T_c' = T_c0
        # This is a rank-1 decomposition of the sum of (rank-1) rank-1 tensors... 
        # Actually, we can't just drop c0 and keep others unchanged because the relation
        # is sum a_c T_c = 0, so T_c0 = -sum_{c!=c0} (a_c/a_c0) T_c
        # But T_c0 is a rank-1 tensor, and the RHS is a sum of rank-1 tensors.
        # For the decomposition to still be valid with rank-1 factors, we need the RHS
        # to be representable as a single rank-1 tensor. This is not generally true.
        # So this approach only works if the null vector has only 2 nonzero entries
        # (i.e., two rank-1 tensors are linearly dependent).
        nz = np.where(np.abs(a) > 1e-8)[0]
        if len(nz) == 2:
            c1, c2 = nz
            # T_c1 = -a_c2/a_c1 * T_c2
            # We can drop c2 and scale c1 by 1/(-a_c2/a_c1) = -a_c1/a_c2
            scale = -a[c1] / a[c2]
            U_new = np.delete(U, c2, axis=1)
            V_new = np.delete(V, c2, axis=1)
            W_new = np.delete(W, c2, axis=1)
            U_new[:, c1] = U_new[:, c1] / scale
            V_new[:, c1] = V_new[:, c1] / scale
            W_new[:, c1] = W_new[:, c1] / scale
            if _verify(U_new, V_new, W_new):
                return U_new, V_new, W_new
    return U, V, W

def construct():
    global _DEADLINE
    start = time.time()
    _DEADLINE = [start]
    deadline = start + 140.0  # safety margin before 150s

    # Start from schoolbook
    U, V, W = _schoolbook()
    best_rank = U.shape[1]
    best = (U.copy(), V.copy(), W.copy())

    # Try to reduce rank via linear dependence
    if _elapsed() < 10:
        U2, V2, W2 = _try_reduce_rank(U, V, W)
        if U2.shape[1] < best_rank:
            best_rank = U2.shape[1]
            best = (U2.copy(), V2.copy(), W2.copy())

    # Additional search: try random linear combinations to find dependencies
    # This is a heuristic; we try to perturb and re-solve.
    # For now, return the best found.
    if _elapsed() > 145:
        pass  # just in case

    # Convert to lists of integer/half-integer entries if possible
    U_out, V_out, W_out = best
    def to_list(A):
        # Check if entries are close to integers or half-integers
        out = []
        for row in A:
            r = []
            for x in row:
                # Round to nearest half-integer
                y = round(x * 2) / 2
                if abs(x - y) < 1e-9:
                    r.append(y)
                else:
                    r.append(float(x))
            out.append(r)
        return out

    return {"U": to_list(U_out), "V": to_list(V_out), "W": to_list(W_out)}
