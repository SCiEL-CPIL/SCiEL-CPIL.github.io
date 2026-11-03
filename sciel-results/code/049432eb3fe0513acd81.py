"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [6, 6, 6]
_DEADLINE = 150.0
_START_TIME = time.time()

# EVOLVE-BLOCK-START
def _time_remaining():
    return _DEADLINE - (time.time() - _START_TIME)

def _schoolbook_3x3():
    """Returns U, V, W for 3x3 matrix multiplication with rank 27."""
    n = 3
    rank = n * n * n
    U = np.zeros((n * n, rank))
    V = np.zeros((n * n, rank))
    W = np.zeros((n * n, rank))
    
    c = 0
    for i in range(n):
        for k in range(n):
            for j in range(n):
                # U indexed by (i, k) of A
                U[i * n + k, c] = 1
                # V indexed by (k, j) of B
                V[k * n + j, c] = 1
                # W indexed by (j, i) of C (Trace convention W[k*n+i] where k is col, i is row)
                # Here col is j, row is i. So index is j * n + i
                W[j * n + i, c] = 1
                c += 1
    return U, V, W

def _strassen_6x6():
    """
    Constructs rank-189 decomposition for 6x6 matrix multiplication
    using Strassen's algorithm on 3x3 blocks.
    """
    n = 6
    block_size = 3
    U, V, W = _schoolbook_3x3()
    local_rank = 27
    
    # Global matrices
    U_global = np.zeros((n * n, 7 * local_rank))
    V_global = np.zeros((n * n, 7 * local_rank))
    W_global = np.zeros((n * n, 7 * local_rank))
    
    # Strassen coefficients for 7 products
    # Each entry: (alpha_dict, beta_dict, gamma_dict)
    # alpha: { (rb, cb): coeff } for A blocks
    # beta: { (rb, cb): coeff } for B blocks
    # gamma: { (rb, cb): coeff } for C blocks
    
    # M1 = (A00 + A11)(B00 + B11)
    m1 = (
        {(0,0): 1, (1,1): 1},
        {(0,0): 1, (1,1): 1},
        {(0,0): 1, (1,1): 1}
    )
    # M2 = (A10 + A11)B00
    m2 = (
        {(1,0): 1, (1,1): 1},
        {(0,0): 1},
        {(1,0): 1}
    )
    # M3 = A00(B01 - B11)
    m3 = (
        {(0,0): 1},
        {(0,1): 1, (1,1): -1},
        {(0,1): 1, (1,1): 1}
    )
    # M4 = A11(B10 - B00)
    m4 = (
        {(1,1): 1},
        {(1,0): 1, (0,0): -1},
        {(0,0): 1, (1,0): 1}
    )
    # M5 = (A00 + A01)B11
    m5 = (
        {(0,0): 1, (0,1): 1},
        {(1,1): 1},
        {(0,1): 1, (0,0): -1}
    )
    # M6 = (A10 - A00)(B00 + B01)
    m6 = (
        {(1,0): 1, (0,0): -1},
        {(0,0): 1, (0,1): 1},
        {(1,1): 1}
    )
    # M7 = (A01 - A11)(B10 + B11)
    m7 = (
        {(0,1): 1, (1,1): -1},
        {(1,0): 1, (1,1): 1},
        {(0,0): 1, (0,1): -1}
    )
    
    strassen_terms = [m1, m2, m3, m4, m5, m6, m7]
    
    for k, (alpha, beta, gamma) in enumerate(strassen_terms):
        base_col = k * local_rank
        
        # Iterate over local basis elements (i', k', j')
        for i_prime in range(block_size):
            for k_prime in range(block_size):
                for j_prime in range(block_size):
                    local_idx = i_prime * block_size + k_prime * block_size + j_prime 
                    # Wait, local U index is i'*n + k'. Local V is k'*n + j'. Local W is j'*n + i'.
                    # The local_rank index c_local corresponds to the triple (i', k', j').
                    # In _schoolbook_3x3, c was incremented in order i, k, j.
                    # So c_local = i_prime * 9 + k_prime * 3 + j_prime.
                    c_local = i_prime * (block_size * block_size) + k_prime * block_size + j_prime
                    
                    if U[i_prime * block_size + k_prime, c_local] != 1:
                        continue # Should always be 1 for schoolbook
                    
                    # Update Global U
                    # U_global[I, col] += alpha[rb, cb]
                    # I = rb * block_size + i_prime, cb * block_size + k_prime
                    for (rb, cb), coeff in alpha.items():
                        if coeff == 0: continue
                        I = rb * block_size + i_prime
                        J = cb * block_size + k_prime
                        U_global[I * n + J, base_col + c_local] += coeff
                    
                    # Update Global V
                    # V_global[J, col] += beta[rb, cb]
                    # J = rb * block_size + k_prime, cb * block_size + j_prime
                    for (rb, cb), coeff in beta.items():
                        if coeff == 0: continue
                        J = rb * block_size + k_prime
                        K = cb * block_size + j_prime
                        V_global[J * n + K, base_col + c_local] += coeff
                    
                    # Update Global W
                    # W_global[K, col] += gamma[rb, cb]
                    # W index is col * n + row.
                    # Global C index: row = rb * block_size + i_prime, col = cb * block_size + j_prime
                    # So W row index = (cb * block_size + j_prime) * n + (rb * block_size + i_prime)
                    for (rb, cb), coeff in gamma.items():
                        if coeff == 0: continue
                        row_c = rb * block_size + i_prime
                        col_c = cb * block_size + j_prime
                        W_idx = col_c * n + row_c
                        W_global[W_idx, base_col + c_local] += coeff

    return U_global, V_global, W_global

def construct():
    n, m, p = DIMENSIONS
    
    if _time_remaining() < 10:
        # Fallback to schoolbook if time is critical
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

    try:
        U, V, W = _strassen_6x6()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        # Fallback to schoolbook on error
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
