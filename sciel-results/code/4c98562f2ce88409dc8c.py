import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 10]

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    n, m, p = DIMENSIONS

    # T[i,j,k] = 1 for all i,j,k
    # We want: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k

    # Strategy: Try to find a rank-30 decomposition using structure
    # For each (j,k), the V factor V[j*p+k, c] is shared across i.
    # For each (i,j), the U factor U[i*m+j, c] is shared across k.
    # W[k*n+i, c] selects the output.

    # Let's try a systematic construction for rank 30:
    # Idea: Use the fact that for fixed j, we have 2*10 = 20 outputs that depend on A[:,j] and B[j,:]
    # Actually, let's try a different approach.

    # For rank 30, we can think of it as: 30 rank-1 terms.
    # Each term c contributes U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] to T[i,j,k].

    # Let's try to construct using a block structure.
    # Since m=3, p=10, n=2:
    # We have 3*10=30 (j,k) pairs for V, 2*3=6 (i,j) pairs for U, 10*2=20 (k,i) pairs for W.

    # One approach: For each (j,k), assign one rank-1 term, but that gives 30 terms.
    # But we need U[i*m+j,c] and W[k*n+i,c] to work out.

    # Let's try: for each (j,k), have one term c = j*10+k
    # V[j*p+k, c] = 1, V[j'*p+k', c] = 0 for (j',k') != (j,k)
    # Then we need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = U[i*m+j, j*10+k] * 1 * W[k*n+i, j*10+k] = 1
    # So we need U[i*m+j, j*10+k] * W[k*n+i, j*10+k] = 1 for all i,j,k.
    # This means for each (j,k), U[i*m+j, j*10+k] * W[k*n+i, j*10+k] = 1 for all i.
    # For fixed (j,k) and varying i: U[i*m+j, j*10+k] * W[k*n+i, j*10+k] = 1
    # For i=0: U[j, j*10+k] * W[k, j*10+k] = 1
    # For i=1: U[3+j, j*10+k] * W[10+k, j*10+k] = 1

    # We can set U[j, j*10+k] = 1, W[k, j*10+k] = 1 for all j,k
    # And U[3+j, j*10+k] = 1, W[10+k, j*10+k] = 1 for all j,k

    # But wait, this gives rank 30! Let me verify.

    rank = m * p  # 30
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    for j in range(m):
        for k in range(p):
            c = j * p + k
            V[j * p + k, c] = 1.0
            # For i=0
            U[0 * m + j, c] = 1.0
            W[k * n + 0, c] = 1.0
            # For i=1
            U[1 * m + j, c] = 1.0
            W[k * n + 1, c] = 1.0

    # Verify
    def verify(U_mat, V_mat, W_mat, n, m, p):
        U_mat = np.asarray(U_mat, dtype=float)
        V_mat = np.asarray(V_mat, dtype=float)
        W_mat = np.asarray(W_mat, dtype=float)
        r = U_mat.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(r):
                        val += U_mat[i*m+j, c] * V_mat[j*p+k, c] * W_mat[k*n+i, c]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    if verify(U, V, W, n, m, p):
        best_U, best_V, best_W, best_rank = U, V, W, rank
    else:
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        for c in range(rank):
            i = c // (m * p)
            j = (c // p) % m
            k = c % p
            U[i * m + j, c] = 1
            V[j * p + k, c] = 1
            W[k * n + i, c] = 1
        best_U, best_V, best_W, best_rank = U, V, W, rank

    # Convert to lists of integers or half-integers
    def to_exact(mat):
        mat_list = []
        for row in mat:
            r = []
            for val in row:
                v = round(val * 2) / 2.0
                if abs(v - round(v)) < 1e-9:
                    r.append(int(round(v)))
                else:
                    r.append(v)
            mat_list.append(r)
        return mat_list

    return {"U": to_exact(best_U), "V": to_exact(best_V), "W": to_exact(best_W)}
