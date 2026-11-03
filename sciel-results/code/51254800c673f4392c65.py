"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that U, V, W form a valid rank-r decomposition of the matrix multiplication tensor.
    Condition: for all i, k, j1, j2:
    sum_c U[i*m+j1, c] * V[j2*p+k, c] * W[k*n+i, c] = delta_{j1, j2}
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Vectorized check for speed
    # U: (n*m, r)
    # V: (m*p, r)
    # W: (p*n, r)
    
    # We need to check if the tensor T[i, j1, j2, k] = delta_{j1, j2}
    # is equal to sum_c U[i, j1, c] V[j2, k, c] W[k, i, c]
    
    # Reshape U to (n, m, r)
    U_reshaped = U.reshape(n, m, -1)
    # Reshape V to (m, p, r)
    V_reshaped = V.reshape(m, p, -1)
    # Reshape W to (p, n, r)
    W_reshaped = W.reshape(p, n, -1)
    
    # Compute the contraction
    # Result should be (n, m, m, p)
    # Sum over r: U[n, m, r] * V[m, p, r] * W[p, n, r]
    
    # Use einsum
    # U[i, j1, c] V[j2, k, c] W[k, i, c]
    # Sum over c
    # Result T[i, j1, j2, k]
    
    # This might be memory intensive if r is large, but r=54 is small.
    # T = np.einsum('ijc,jkc,kic->ijjk', U_reshaped, V_reshaped, W_reshaped)
    
    # Let's do it in steps to avoid large intermediate arrays if possible, 
    # but for small dims, einsum is fine.
    
    # Intermediate 1: U[i, j1, c] * V[j2, k, c] -> A[i, j1, j2, k, c]
    # This is n*m*m*p*r = 2*3*3*9*54 = 8748 elements. Small.
    
    A = np.einsum('ijc,jkc->ijjk c', U_reshaped, V_reshaped)
    # Now contract with W[k, i, c]
    # B[i, j1, j2, k] = sum_c A[i, j1, j2, k, c] * W[k, i, c]
    # W is (p, n, r) -> W_reshaped[k, i, c]
    
    T = np.einsum('ijjk c,kic->ijjk', A, W_reshaped)
    
    # Check if T[i, j1, j2, k] == delta_{j1, j2}
    # Expected: 1 if j1 == j2 else 0. Independent of i, k.
    
    # Create expected tensor
    # delta[j1, j2]
    delta = np.eye(m)
    # Expand to (n, m, m, p)
    # Expected[i, j1, j2, k] = delta[j1, j2]
    
    # We can just check slices
    for i in range(n):
        for k in range(p):
            if not np.allclose(T[i, :, :, k], delta, atol=tol):
                return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Schoolbook decomposition (rank = n*m*p = 54)
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, c] = 1.0
                V_sb[j * p + k, c] = 1.0
                W_sb[k * n + i, c] = 1.0
                c += 1
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_sb
    
    # Attempt to find a lower rank decomposition
    # We know rank 54 is valid.
    # We try rank 53.
    # This is a hard search problem.
    # Given the time, we rely on the schoolbook solution.
    # If we had a known rank 53 algorithm, we would use it.
    
    # For now, return the schoolbook solution.
    # Convert to lists for the return interface
    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
