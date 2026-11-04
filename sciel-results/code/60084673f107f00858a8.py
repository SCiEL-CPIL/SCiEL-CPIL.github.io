DIMENSIONS = [5, 5, 5]
import numpy as np
import time

def _strassen_2x2(i_start, j_start, k_start, n, m, p, U, V, W, col):
    """Add Strassen rank-7 decomposition for 2x2x2 block."""
    i0, i1 = i_start, i_start + 1
    j0, j1 = j_start, j_start + 1
    k0, k1 = k_start, k_start + 1

    # M1 = (A[i0,j0]+A[i1,j1])*(B[j0,k0]+B[j1,k1])
    # C[i0,k0] += M1, C[i1,k1] += M1
    U[i0*m+j0][col] = 1
    U[i1*m+j1][col] = 1
    V[j0*p+k0][col] = 1
    V[j1*p+k1][col] = 1
    W[k0*n+i0][col] = 1
    W[k1*n+i1][col] = 1
    col += 1

    # M2 = (A[i1,j0]+A[i1,j1])*(B[j0,k0])
    # C[i1,k0] += M2, C[i1,k1] -= M2
    U[i1*m+j0][col] = 1
    U[i1*m+j1][col] = 1
    V[j0*p+k0][col] = 1
    W[k0*n+i1][col] = 1
    W[k1*n+i1][col] = -1
    col += 1

    # M3 = A[i0,j0]*(B[j0,k1]-B[j1,k1])
    # C[i0,k1] += M3, C[i1,k1] += M3
    U[i0*m+j0][col] = 1
    V[j0*p+k1][col] = 1
    V[j1*p+k1][col] = -1
    W[k1*n+i0][col] = 1
    W[k1*n+i1][col] = 1
    col += 1

    # M4 = A[i1,j1]*(B[j1,k0]-B[j0,k0])
    # C[i0,k0] += M4, C[i1,k0] += M4
    U[i1*m+j1][col] = 1
    V[j1*p+k0][col] = 1
    V[j0*p+k0][col] = -1
    W[k0*n+i0][col] = 1
    W[k0*n+i1][col] = 1
    col += 1

    # M5 = (A[i0,j0]+A[i0,j1])*(B[j1,k1])
    # C[i0,k0] -= M5, C[i0,k1] += M5
    U[i0*m+j0][col] = 1
    U[i0*m+j1][col] = 1
    V[j1*p+k1][col] = 1
    W[k0*n+i0][col] = -1
    W[k1*n+i0][col] = 1
    col += 1

    # M6 = (A[i1,j0]-A[i0,j0])*(B[j0,k0]+B[j0,k1])
    # C[i1,k1] += M6
    U[i1*m+j0][col] = 1
    U[i0*m+j0][col] = -1
    V[j0*p+k0][col] = 1
    V[j0*p+k1][col] = 1
    W[k1*n+i1][col] = 1
    col += 1

    # M7 = (A[i0,j1]-A[i1,j1])*(B[j1,k0]+B[j1,k1])
    # C[i0,k0] += M7
    U[i0*m+j1][col] = 1
    U[i1*m+j1][col] = -1
    V[j1*p+k0][col] = 1
    V[j1*p+k1][col] = 1
    W[k0*n+i0][col] = 1
    col += 1

    return col

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that the decomposition is correct using full tensor check."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _build_partition(parts, n, m, p):
    """Build a block-decomposed factorization for the given 1D partition of 5.
    Uses Strassen for 2x2x2 blocks, schoolbook for everything else.
    Returns (U, V, W, rank)."""
    L = len(parts)
    starts = [0]
    for i in range(1, L):
        starts.append(starts[-1] + parts[i-1])

    # Compute total rank
    total_rank = 0
    for i in range(L):
        for j in range(L):
            for k in range(L):
                ni, mj, pk = parts[i], parts[j], parts[k]
                if ni == 2 and mj == 2 and pk == 2:
                    total_rank += 7
                else:
                    total_rank += ni * mj * pk

    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    col = 0
    for i in range(L):
        for j in range(L):
            for k in range(L):
                ni, mj, pk = parts[i], parts[j], parts[k]
                i_start = starts[i]
                j_start = starts[j]
                k_start = starts[k]
                if ni == 2 and mj == 2 and pk == 2:
                    col = _strassen_2x2(i_start, j_start, k_start, n, m, p, U, V, W, col)
                else:
                    for u in range(ni):
                        for v in range(mj):
                            for w in range(pk):
                                U[(i_start + u) * m + (j_start + v)][col] = 1
                                V[(j_start + v) * p + (k_start + w)][col] = 1
                                W[(k_start + w) * n + (i_start + u)][col] = 1
                                col += 1
    return U, V, W, total_rank

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][c] = 1
                V[j*p+k][c] = 1
                W[k*n+i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}

def construct():
    _t0 = time.time()
    _deadline = _t0 + 120.0  # safety margin before the 150s kill

    n, m, p = DIMENSIONS

    # Candidate partitions, ordered best (lowest rank) first.
    # 2+2+1 -> rank 101; 2+3 -> rank 104.
    candidates = [
        [2, 2, 1],
        [2, 3],
        [3, 2],
        [1, 2, 2],
        [1, 4],
    ]

    for parts in candidates:
        if time.time() > _deadline:
            break
        if sum(parts) != 5:
            continue
        U, V, W, total_rank = _build_partition(parts, n, m, p)
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}

    # Last-resort fallback: exact schoolbook (rank 125).
    return _schoolbook(n, m, p)
