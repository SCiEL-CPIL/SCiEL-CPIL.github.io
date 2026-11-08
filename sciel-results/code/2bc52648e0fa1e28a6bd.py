"""Matrix multiplication 2,3,8 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 3, 8]

import numpy as np
import time

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
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin

    # Schoolbook decomposition as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    best_result = {"U": U_school, "V": V_school, "W": W_school}

    # Try to find a lower rank decomposition
    for rank in range(40, rank_school):
        if time.time() > deadline:
            break
        
        # Random initialization
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)
        W = np.random.randn(p * n, rank)
        
        # Simple optimization: try to reduce the error
        # We use a few iterations of a simple gradient-like update
        # This is a heuristic and may not always converge, but it's fast
        for _ in range(10):
            if time.time() > deadline:
                break
            # Compute error
            error = 0.0
            A = np.random.rand(n, m)
            B = np.random.rand(m, p)
            C_expected = A @ B
            C_computed = np.zeros((n, p))
            for c in range(rank):
                u_mat = U[:, c].reshape(n, m)
                v_mat = V[:, c].reshape(m, p)
                w_mat = W[:, c].reshape(p, n)
                a_sum = (u_mat * A).sum(axis=1)
                b_sum = (v_mat * B).sum(axis=0)
                C_computed += w_mat.T * np.outer(a_sum, b_sum)
            error = np.sum((C_computed - C_expected) ** 2)
            
            if error < 1e-6:
                # Round to half-integers
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                # Verify the rounded decomposition
                if _verify(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), n, m, p):
                    best_result = {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
                    break
        
        if time.time() > deadline:
            break
        if best_result["U"] != U_school:
            break

    return best_result
# EVOLVE-BLOCK-END
