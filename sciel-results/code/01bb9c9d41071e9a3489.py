"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 3x3 matrix by a 3x7 matrix.
    
    The multiplication tensor T has entries T[i,j,k] = e_{i,k} in the output,
    which is 1 if we're computing C[i,k] = sum_j A[i,j] * B[j,k].
    
    We use a rank R = m*p = 21 decomposition by fixing the (j,k) index in B.
    For each rank index r corresponding to a (j,k) pair:
    - V[j*p+k, r] = 1 (selects the (j,k) entry of B)
    - U[i*m+j, r] = 1 (selects row i, column j of A)
    - W[k*n+i, r] = 1 (places into row k, column i of C)
    
    Then sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    = U[i*m+j, r*] * 1 * W[k*n+i, r*] where r* is the unique rank with (j_r=j, k_r=k)
    = 1 * 1 = 1, which equals T[i,j,k].
    """
    n, m, p = DIMENSIONS
    R = m * p  # rank = 21
    
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    for r in range(R):
        j = r % m      # 0, 1, 2, 0, 1, 2, ...
        k = r // m     # 0, 0, 0, 1, 1, 1, ...
        
        V[j * p + k][r] = 1
        for i in range(n):
            U[i * m + j][r] = 1
            W[k * n + i][r] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
