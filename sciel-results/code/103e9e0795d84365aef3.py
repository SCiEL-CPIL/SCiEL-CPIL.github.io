"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _matmul_tensor(n, m, p):
    # T[(i,j,k)] = 1 if i==j else 0 (tensor for C[i,k] = sum_j A[i,j] B[j,k])
    T = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        T[i, i, :] = 1.0
    return T

def _verify(U, V, W, n, m, p):
    # U: (n*m, r) indexed by (i,j); V: (m*p, r) indexed by (j,k); W: (p*n, r) indexed by (k,i)
    r = U.shape[1]
    T = np.zeros((n, m, p), dtype=float)
    for t in range(r):
        u = U[:, t].reshape(n, m)
        v = V[:, t].reshape(m, p)
        w = W[:, t].reshape(p, n)
        T += np.einsum('ij,jk,ki->ijk', u, v, w)
    expected = _matmul_tensor(n, m, p)
    return np.allclose(T, expected, atol=1e-9)

def _verify_fast(U, V, W, n, m, p, expected):
    """Vectorized verification using einsum over all rank terms at once."""
    r = U.shape[1]
    # Reshape to 4D: U -> (n, m, r), V -> (m, p, r), W -> (p, n, r)
    U4 = U.T.reshape(r, n, m)  # (r, n, m)
    V4 = V.T.reshape(r, m, p)  # (r, m, p)
    W4 = W.T.reshape(r, p, n)  # (r, p, n)
    # T[i,j,k] = sum_r U[r,i,j] * V[r,j,k] * W[r,k,i]
    T = np.einsum('rij,rjk,rki->ijk', U4, V4, W4)
    return np.allclose(T, expected, atol=1e-9)

def _prune_zero_columns(U, V, W, n, m, p):
    r = U.shape[1]
    keep = []
    for t in range(r):
        if np.any(U[:, t]) or np.any(V[:, t]) or np.any(W[:, t]):
            keep.append(t)
    keep = np.array(keep, dtype=int)
    U2, V2, W2 = U[:, keep], V[:, keep], W[:, keep]
    if _verify(U2, V2, W2, n, m, p):
        return U2, V2, W2
    return U, V, W

def _try_remove_column(U, V, W, t, n, m, p, expected):
    r = U.shape[1]
    idx = [x for x in range(r) if x != t]
    U2, V2, W2 = U[:, idx], V[:, idx], W[:, idx]
    if _verify_fast(U2, V2, W2, n, m, p, expected):
        return U2, V2, W2
    return None

def _try_remove_pair(U, V, W, t1, t2, n, m, p, expected):
    r = U.shape[1]
    idx = [x for x in range(r) if x != t1 and x != t2]
    U2, V2, W2 = U[:, idx], V[:, idx], W[:, idx]
    if _verify_fast(U2, V2, W2, n, m, p, expected):
        return U2, V2, W2
    return None

def construct():
    deadline = time.time() + 120 - 10  # safety margin
    n, m, p = DIMENSIONS  # 2, 4, 8

    # Rank 56 solution using Strassen decomposition on 2x2 blocks
    rank = 56
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    t = 0
    for a in range(2):  # j-group: 0 or 1, covering j=2a, 2a+1
        for b in range(4):  # k-group: 0..3, covering k=2b, 2b+1
            for s in range(7):  # Strassen step
                # U contributions (A[:, 2a:2a+2])
                if s == 0:  # X00 + X11
                    U[0 * m + 2 * a, t] += 1
                    U[1 * m + 2 * a + 1, t] += 1
                elif s == 1:  # X10 + X11
                    U[1 * m + 2 * a, t] += 1
                    U[1 * m + 2 * a + 1, t] += 1
                elif s == 2:  # X00
                    U[0 * m + 2 * a, t] += 1
                elif s == 3:  # X11
                    U[1 * m + 2 * a + 1, t] += 1
                elif s == 4:  # X00 + X01
                    U[0 * m + 2 * a, t] += 1
                    U[0 * m + 2 * a + 1, t] += 1
                elif s == 5:  # X10 - X00
                    U[1 * m + 2 * a, t] += 1
                    U[0 * m + 2 * a, t] += -1
                elif s == 6:  # X01 - X11
                    U[0 * m + 2 * a + 1, t] += 1
                    U[1 * m + 2 * a + 1, t] += -1

                # V contributions (B[2a:2a+2, 2b:2b+2])
                if s == 0:  # Y00 + Y11
                    V[2 * a * p + 2 * b, t] += 1
                    V[(2 * a + 1) * p + 2 * b + 1, t] += 1
                elif s == 1:  # Y00
                    V[2 * a * p + 2 * b, t] += 1
                elif s == 2:  # Y01 - Y11
                    V[2 * a * p + 2 * b + 1, t] += 1
                    V[(2 * a + 1) * p + 2 * b + 1, t] += -1
                elif s == 3:  # Y10 - Y00
                    V[(2 * a + 1) * p + 2 * b, t] += 1
                    V[2 * a * p + 2 * b, t] += -1
                elif s == 4:  # Y11
                    V[(2 * a + 1) * p + 2 * b + 1, t] += 1
                elif s == 5:  # Y00 + Y01
                    V[2 * a * p + 2 * b, t] += 1
                    V[2 * a * p + 2 * b + 1, t] += 1
                elif s == 6:  # Y10 + Y11
                    V[(2 * a + 1) * p + 2 * b, t] += 1
                    V[(2 * a + 1) * p + 2 * b + 1, t] += 1

                # W contributions (C[:, 2b:2b+2])
                if s == 0:  # M1: Z00+=1, Z11+=1
                    W[2 * b * n + 0, t] += 1
                    W[(2 * b + 1) * n + 1, t] += 1
                elif s == 1:  # M2: Z10+=1, Z11-=1
                    W[2 * b * n + 1, t] += 1
                    W[(2 * b + 1) * n + 1, t] += -1
                elif s == 2:  # M3: Z01+=1, Z11+=1
                    W[(2 * b + 1) * n + 0, t] += 1
                    W[(2 * b + 1) * n + 1, t] += 1
                elif s == 3:  # M4: Z00+=1, Z10+=1
                    W[2 * b * n + 0, t] += 1
                    W[2 * b * n + 1, t] += 1
                elif s == 4:  # M5: Z00-=1, Z01+=1
                    W[2 * b * n + 0, t] += -1
                    W[(2 * b + 1) * n + 0, t] += 1
                elif s == 5:  # M6: Z11+=1
                    W[(2 * b + 1) * n + 1, t] += 1
                elif s == 6:  # M7: Z00+=1
                    W[2 * b * n + 0, t] += 1

                t += 1

    # Verify baseline; if invalid, return as-is (should not happen)
    expected = _matmul_tensor(n, m, p)
    if not _verify(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Prune zero columns (exact)
    U, V, W = _prune_zero_columns(U, V, W, n, m, p)

    # Local rank reduction: try removing each column (bounded, deadline-guarded)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        r = U.shape[1]
        for t in range(r):
            if time.time() >= deadline:
                break
            res = _try_remove_column(U, V, W, t, n, m, p, expected)
            if res is not None:
                U, V, W = res
                improved = True
                break  # restart scan after any removal
        # prune again after removals
        if improved:
            U, V, W = _prune_zero_columns(U, V, W, n, m, p)

    # Try pair removals to break through single-removal plateau
    if time.time() < deadline:
        pair_improved = True
        while pair_improved and time.time() < deadline:
            pair_improved = False
            r = U.shape[1]
            for t1 in range(r):
                if time.time() >= deadline:
                    break
                for t2 in range(t1 + 1, r):
                    if time.time() >= deadline:
                        break
                    res = _try_remove_pair(U, V, W, t1, t2, n, m, p, expected)
                    if res is not None:
                        U, V, W = res
                        pair_improved = True
                        break
                if pair_improved:
                    break
            if pair_improved:
                U, V, W = _prune_zero_columns(U, V, W, n, m, p)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
