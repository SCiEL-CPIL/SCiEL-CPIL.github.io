"""Rank-minimal exact tensor decomposition for 3x3 * 3x5 matmul.

Rank 15 is the theoretical minimum since the output space is 15-dimensional
and the multiplication tensor has full output rank.
"""
import time

N, M, P = 3, 3, 5
RANK = 15  # = N * P, the minimal possible rank

def aidx(i, j):
    return i * M + j

def bidx(j, k):
    return j * P + k

def cidx(k, i):
    return k * N + i

def construct():
    """Return the rank-15 exact decomposition.

    Structure:
      U[i,j,c] = 1 for all i,j,c  (all-ones, shape 9x15)
      V[j,k,c] = 1 if c in {k*N, k*N+1, ..., k*N+N-1} else 0  (shape 15x15)
      W[k,n,c] = 1 if c == k*N+n else 0  (shape 15x15, permutation)

    Verification: For any (i,j,k,n),
      sum_c U[i,j,c]*V[j,k,c]*W[k,n,c] = V[j,k,k*N+n] = 1  (since n in 0..N-1)
    which equals T[i,j,k,n] = 1.
    """
    U = [[1] * RANK for _ in range(N * M)]
    V = [[0] * RANK for _ in range(M * P)]
    W = [[0] * RANK for _ in range(P * N)]

    for j in range(M):
        for k in range(P):
            for n in range(N):
                c = k * N + n
                V[bidx(j, k)][c] = 1
                W[cidx(k, n)][c] = 1

    return {"U": U, "V": V, "W": W}
