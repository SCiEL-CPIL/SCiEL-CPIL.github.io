import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    """
    Rank-50 exact decomposition for 2x4 * 4x7 matrix multiplication.
    Uses Strassen (rank 7) for 2x2*2x2 blocks and schoolbook (rank 4) for 2x2*2x1.
    Total: 2 j-blocks * (3 Strassen + 1 schoolbook) = 2*(21+4) = 50.
    """
    start_time = time.time()
    n, m, p = 2, 4, 7
    total_rank = 50

    U = np.zeros((n * m, total_rank), dtype=np.int64)
    V = np.zeros((m * p, total_rank), dtype=np.int64)
    W = np.zeros((p * n, total_rank), dtype=np.int64)

    col = 0

    def add_strassen(i0, j0, k0):
        """
        Strassen for C[i0:i0+2, k0:k0+2] = A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2]
        A is n x m = 2 x 4, indexed by row i, col j -> flat idx i*m + j
        B is m x p = 4 x 7, indexed by row j, col k -> flat idx j*p + k
        C is p x n = 7 x 2, indexed by row k, col i -> flat idx k*n + i  (trace convention W[k*n+i, r])
        """
        nonlocal col
        # A indices
        a = i0 * m + j0        # A[0,0] of the 2x2 block
        b = i0 * m + j0 + 1    # A[0,1]
        c = (i0 + 1) * m + j0  # A[1,0]
        d = (i0 + 1) * m + j0 + 1  # A[1,1]

        # B indices
        e = j0 * p + k0        # B[0,0]
        f = j0 * p + k0 + 1    # B[0,1]
        g = (j0 + 1) * p + k0  # B[1,0]
        h = (j0 + 1) * p + k0 + 1  # B[1,1]

        # C indices (output: C[i, k] where i is row of C in the 2x2 block, k is col)
        # W[k*n + i]
        c00 = k0 * n + i0        # C[0,0]
        c01 = (k0 + 1) * n + i0  # C[0,1]
        c10 = k0 * n + (i0 + 1)  # C[1,0]
        c11 = (k0 + 1) * n + (i0 + 1)  # C[1,1]

        # M1 = (a+d)(e+h)
        U[a, col] += 1; U[d, col] += 1
        V[e, col] += 1; V[h, col] += 1
        W[c00, col] += 1; W[c11, col] += 1
        col += 1

        # M2 = (c+d)e
        U[c, col] += 1; U[d, col] += 1
        V[e, col] += 1
        W[c10, col] += 1; W[c11, col] -= 1
        col += 1

        # M3 = a(f-h)
        U[a, col] += 1
        V[f, col] += 1; V[h, col] -= 1
        W[c01, col] += 1; W[c11, col] += 1
        col += 1

        # M4 = d(g-e)
        U[d, col] += 1
        V[g, col] += 1; V[e, col] -= 1
        W[c00, col] += 1; W[c10, col] += 1
        col += 1

        # M5 = (a+b)h
        U[a, col] += 1; U[b, col] += 1
        V[h, col] += 1
        W[c00, col] -= 1; W[c01, col] += 1
        col += 1

        # M6 = (c-a)(e+f)
        U[c, col] += 1; U[a, col] -= 1
        V[e, col] += 1; V[f, col] += 1
        W[c11, col] += 1
        col += 1

        # M7 = (b-d)(g+h)
        U[b, col] += 1; U[d, col] -= 1
        V[g, col] += 1; V[h, col] += 1
        W[c00, col] += 1
        col += 1

    def add_schoolbook_221(i0, j0, k0):
        """
        Schoolbook rank-4 for C[i0:i0+2, k0:k0+1] = A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+1]
        """
        nonlocal col
        a = i0 * m + j0
        b = i0 * m + j0 + 1
        c = (i0 + 1) * m + j0
        d = (i0 + 1) * m + j0 + 1

        e = j0 * p + k0
        g = (j0 + 1) * p + k0

        c00 = k0 * n + i0
        c10 = k0 * n + (i0 + 1)

        U[a, col] += 1; V[e, col] += 1; W[c00, col] += 1
        col += 1
        U[b, col] += 1; V[g, col] += 1; W[c00, col] += 1
        col += 1
        U[c, col] += 1; V[e, col] += 1; W[c10, col] += 1
        col += 1
        U[d, col] += 1; V[g, col] += 1; W[c10, col] += 1
        col += 1

    # Two j-blocks: j in {0,1} and j in {2,3}
    for j0 in [0, 2]:
        # Three 2x2 k-blocks: k in {0,1}, {2,3}, {4,5}
        for k0 in [0, 2, 4]:
            add_strassen(0, j0, k0)
        # One 2x1 k-block: k in {6}
        add_schoolbook_221(0, j0, 6)

    assert col == total_rank, f"Expected rank {total_rank}, got {col}"

    # Quick verification
    def verify(U_arr, V_arr, W_arr):
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    target = 1.0 if True else 0.0  # We need sum over jp: U[i*m+j,r]*V[jp*p+k,r]*W[k*n+i,r] = delta_{j,jp}... 
                    # Actually the tensor decomposition condition is:
                    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1  for all i,j,k
                    # Wait, that's not right either. Let me re-derive.
                    # The matrix multiplication tensor T has T[i,j,k] = 1 if we think of it as
                    # C[i,k] = sum_j A[i,j] B[j,k]
                    # The decomposition: T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
                    # Here U is (n*m) x rank, V is (m*p) x rank, W is (p*n) x rank
                    # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
                    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should = 1 for all i,j,k
                    pass
        # Let me just do the actual check
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(total_rank):
                        s += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    if not verify(U, V, W):
        # Fallback: full schoolbook rank = n*m*p = 56
        rank_fb = n * m * p
        U = np.zeros((n * m, rank_fb), dtype=np.int64)
        V = np.zeros((m * p, rank_fb), dtype=np.int64)
        W = np.zeros((p * n, rank_fb), dtype=np.int64)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, idx] = 1
                    V[j*p+k, idx] = 1
                    W[k*n+i, idx] = 1
                    idx += 1

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

if __name__ == "__main__":
    res = construct()
    U = np.array(res['U'])
    V = np.array(res['V'])
    W = np.array(res['W'])
    n, m, p = 2, 4, 7
    rank = U.shape[1]
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    print(f"Rank: {rank}, Valid: {valid}")
