import numpy as np

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p  # 72

    # Trivial exact tensor decomposition:
    # For each (i, j, k) triple, a rank-1 tensor E_i ⊗ E_j ⊗ E_k
    # where E_i is the i-th basis vector of R^n, etc.
    #
    # U[i*m + j, c] = 1  (selects row i of A, column j)
    # V[j*p + k, c] = 1  (selects row j of B, column k)
    # W[k*n + i, c] = 1  (selects output position (k, i) per trace convention)
    #
    # The sum over c of U[:,c] * V[:,c] * W[:,c] reconstructs C = A @ B
    # because C[i, k] = sum_j A[i,j] * B[j,k], and each term contributes
    # exactly to the (i, k) output position.

    i_vals, j_vals, k_vals = np.meshgrid(range(n), range(m), range(p), indexing='ij')
    i_flat = i_vals.flatten()
    j_flat = j_vals.flatten()
    k_flat = k_vals.flatten()

    u_rows = i_flat * m + j_flat
    v_rows = j_flat * p + k_flat
    w_rows = k_flat * n + i_flat

    U_np = np.zeros((n * m, rank), dtype=int)
    V_np = np.zeros((m * p, rank), dtype=int)
    W_np = np.zeros((p * n, rank), dtype=int)

    U_np[u_rows, np.arange(rank)] = 1
    V_np[v_rows, np.arange(rank)] = 1
    W_np[w_rows, np.arange(rank)] = 1

    return {"U": U_np.tolist(), "V": V_np.tolist(), "W": W_np.tolist()}
# EVOLVE-BLOCK-END
