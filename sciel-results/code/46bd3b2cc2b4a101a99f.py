"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Block-wise construction approach
    # We decompose the multiplication into independent terms for each (i, j, k)
    # but organized by blocks of the output matrix to demonstrate a different structural approach.
    
    rank = n * m * p
    
    # Pre-allocate matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_idx = 0
    # Iterate through the output matrix C (p x n)
    # For each output element C[k, i], we sum over j: sum_j A[i, j] B[j, k]
    # Each term A[i, j] B[j, k] is a separate rank-1 component in the tensor decomposition.
    for k in range(p):
        for i in range(n):
            for j in range(m):
                # This term contributes to the (i, j) entry of U, (j, k) entry of V, and (k, i) entry of W
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx][col_idx] = 1
                V[v_idx][col_idx] = 1
                W[w_idx][col_idx] = 1
                
                col_idx += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
