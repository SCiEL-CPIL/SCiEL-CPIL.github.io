"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    n, m, p = DIMENSIONS
    nm, mp, np_ = n * m, m * p, p * n  # Note: np is a module, use np_
    
    # Build the target tensor T of shape (nm, mp, np_)
    # T[(i*m+j), (j*p+k), (k*n+i)] = 1
    T = np.zeros((nm, mp, np_), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Start with schoolbook decomposition
    rank = n * m * p
    U = np.zeros((nm, rank), dtype=np.float64)
    V = np.zeros((mp, rank), dtype=np.float64)
    W = np.zeros((np_, rank), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    # Try to reduce rank by finding linear dependencies
    # For each column, check if it can be expressed as a combination of others
    # This is a greedy rank reduction
    
    def check_decomposition(U, V, W, T, tol=1e-9):
        """Check if the decomposition is valid."""
        # Reconstruct: for each (a,b), sum_t U[a,t]*V[b,t]*W[:,t] should match T[a,b,:]
        # But this is expensive. Instead, check a few random samples.
        # Actually, let's check the full thing but efficiently.
        # C[a,b,c] = sum_t U[a,t] * V[b,t] * W[c,t]
        # We can compute this as: for each t, outer(U[:,t], V[:,t]) * W[:,t]
        # Sum over t.
        # This is O(nm * mp * np_ * rank) which is 75*25*15*75 ~ 2M, fine.
        C = np.zeros((nm, mp, np_))
        for t in range(U.shape[1]):
            C += np.einsum('a,b,c->abc', U[:, t], V[:, t], W[:, t])
        return np.max(np.abs(C - T)) < tol
    
    # Greedy rank reduction
    current_rank = rank
    U_cur = U.copy()
    V_cur = V.copy()
    W_cur = W.copy()
    
    # Try to remove columns that are linearly dependent
    # For each column t, try to see if removing it still gives a valid decomposition
    # by adjusting the other columns. This is complex.
    
    # Alternative: use a simple approach - try all pairs of columns and see if
    # we can merge them.
    
    # Actually, let's try a different approach: use the fact that for matrix multiplication,
    # there are known rank reductions. For 3x5x5, let's try to find a good decomposition
    # using a structured approach.
    
    # For now, return the schoolbook decomposition as a baseline.
    # The rank is 75.
    
    # Convert to lists with integer/half-integer entries
    def to_list(mat):
        return [[float(mat[i, j]) for j in range(mat.shape[1])] for i in range(mat.shape[0])]
    
    return {"U": to_list(U_cur), "V": to_list(V_cur), "W": to_list(W_cur)}
# EVOLVE-BLOCK-END
