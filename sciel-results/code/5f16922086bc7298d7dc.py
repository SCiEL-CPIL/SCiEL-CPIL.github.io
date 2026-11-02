import numpy as np

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target tensor T[i, j, k] = 1 for all i, j, k
    # U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank)
    
    # Try to find a rank-24 decomposition
    rank = 24
    np.random.seed(42)
    
    # Initialize U and V randomly
    U = np.random.randn(n * m, rank)
    V = np.random.randn(m * p, rank)
    
    # We need to solve for W
    # Equation: sum_r U[r, i*m+j] * V[r, j*p+k] * W[r, k*n+i] = 1
    # For each (i, j, k), we have an equation.
    # Unknowns: W[r, k*n+i] for all r, i, k.
    # There are rank * p * n unknowns.
    # There are n * m * p equations.
    
    num_eqs = n * m * p
    num_unk = rank * p * n
    
    # Build the coefficient matrix A
    # A[eq_idx, unk_idx] = U[r, i*m+j] * V[r, j*p+k]
    # eq_idx = i * m * p + j * p + k
    # unk_idx = r * p * n + k * n + i
    
    A = np.zeros((num_eqs, num_unk))
    b = np.ones(num_eqs)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                eq_idx = i * m * p + j * p + k
                for r in range(rank):
                    # U index: i*m+j
                    # V index: j*p+k
                    # W index: k*n+i
                    u_val = U[r, i * m + j]
                    v_val = V[r, j * p + k]
                    unk_idx = r * p * n + k * n + i
                    A[eq_idx, unk_idx] = u_val * v_val
    
    # Solve A W_flat = b
    try:
        W_flat, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
        # Check if the solution is good
        if np.allclose(A @ W_flat, b, atol=1e-5):
            W = W_flat.reshape(rank, p * n).T # W shape (p*n, rank)
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except:
        pass
    
    # Fallback to schoolbook
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
