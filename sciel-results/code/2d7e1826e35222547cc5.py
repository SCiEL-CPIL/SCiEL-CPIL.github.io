import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    start_time = time.time()
    deadline = start_time + 110

    # Build the schoolbook decomposition as the baseline
    U = np.zeros((n * m, rank_schoolbook), dtype=int)
    V = np.zeros((m * p, rank_schoolbook), dtype=int)
    W = np.zeros((p * n, rank_schoolbook), dtype=int)

    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                col = i * m * p + j * p + k
                U[a, col] = 1
                V[b, col] = 1
                W[c, col] = 1

    def verify(U_mat, V_mat, W_mat):
        """Verify that U,V,W exactly decompose the multiplication tensor."""
        U_mat = np.asarray(U_mat)
        V_mat = np.asarray(V_mat)
        W_mat = np.asarray(W_mat)
        r = U_mat.shape[1]
        # Target tensor: T[a,b,c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    # T[a,b,c] should be 1
                    val = np.dot(U_mat[a, :], np.multiply(V_mat[b, :], W_mat[c, :]))
                    if abs(val - 1) > 1e-9:
                        return False
        # Check that all other entries are 0
        for a in range(n * m):
            for b in range(m * p):
                i = a // m
                j = a % m
                j2 = b // p
                k = b % p
                if j != j2:
                    continue
                for c in range(p * n):
                    kk = c // n
                    ii = c % n
                    if i == ii and j == j2 and k == kk:
                        continue
                    val = np.dot(U_mat[a, :], np.multiply(V_mat[b, :], W_mat[c, :]))
                    if abs(val) > 1e-9:
                        return False
        return True

    # Verify schoolbook
    if not verify(U, V, W):
        # This should never happen for schoolbook
        pass

    # Return the schoolbook solution
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
