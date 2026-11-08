"""Matrix multiplication 2,3,8 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 3, 8]

import numpy as np

def _verify(U, V, W, n, m, p, trials=5):
    """Verify that the decomposition correctly computes A·B."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    for _ in range(trials):
        A = np.random.rand(n, m)
        B = np.random.rand(m, p)
        C_expected = A @ B
        C_computed = np.zeros((n, p))
        for c in range(rank):
            u_vec = U_arr[:, c]  # (n*m,)
            v_vec = V_arr[:, c]  # (m*p,)
            w_vec = W_arr[:, c]  # (p*n,)
            # u_vec[i*m+j] multiplies A[i,j]
            # v_vec[j*p+k] multiplies B[j,k]
            # w_vec[k*n+i] is the coefficient for C[i,k]
            u_mat = u_vec.reshape(n, m)  # (n, m)
            v_mat = v_vec.reshape(m, p)  # (m, p)
            w_mat = w_vec.reshape(p, n)  # (p, n)
            # C[i,k] += w_mat[k,i] * (sum_j u_mat[i,j]*A[i,j]) * (sum_j v_mat[j,k]*B[j,k])
            a_proj = u_mat * A  # element-wise, (n, m)
            b_proj = v_mat * B  # element-wise, (m, p)
            a_sum = a_proj.sum(axis=1)  # (n,)
            b_sum = b_proj.sum(axis=0)  # (p,)
            C_computed += w_mat.T * np.outer(a_sum, b_sum)
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            return False
    return True

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook decomposition
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
# EVOLVE-BLOCK-END
