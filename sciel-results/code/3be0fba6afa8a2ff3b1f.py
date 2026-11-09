"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # safety margin before 120s limit

    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best_result = schoolbook()
    best_rank = n * m * p

    # Vectorized validation function
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # We need T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    def validate_vectorized(U, V, W, tol=1e-6):
        rank = U.shape[1]
        # Reshape U, V, W to align indices for einsum
        # U[i, j, r], V[j, k, r], W[k, i, r]
        # Note: U is (n*m, rank) -> reshape to (n, m, rank)
        # V is (m*p, rank) -> reshape to (m, p, rank)
        # W is (p*n, rank) -> reshape to (p, n, rank)
        
        U_3d = U.reshape(n, m, rank)
        V_3d = V.reshape(m, p, rank)
        W_3d = W.reshape(p, n, rank)
        
        # Compute T[i, j, k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # einsum: 'ijr,jkr,kr->ijk' ? No, W is (p,n,rank) so indices are (k, i, r)
        # So W_3d[k, i, r]
        # T[i, j, k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        
        try:
            T = np.einsum('ijr,jkr,kr->ijk', U_3d, V_3d, W_3d)
        except Exception:
            return False

        # Check if all entries are 1
        if not np.allclose(T, 1.0, atol=tol):
            return False
        
        # Check if entries are integer or half-integer
        # The problem states "integer/half-integer entries".
        # We should snap and re-check or just ensure the input was snapped.
        # Since we control the generation, we will generate snapped values.
        return True

    def snap_to_half_integer(arr):
        return np.round(arr * 2) / 2

    def try_rank(rank):
        """Try to find a valid decomposition with given rank using random search."""
        if rank >= best_rank:
            return None
        
        # We will try random initializations with values from a small set of half-integers
        # to satisfy the constraint directly.
        # Values: -1.5, -1, -0.5, 0, 0.5, 1, 1.5
        # Random search in this discrete space is hard, but we can try random floats and snap.
        
        for attempt in range(200):
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            # Scale to ensure products don't overflow or become too large
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            W = np.random.randn(p * n, rank)
            
            # Snap to half-integers
            U = snap_to_half_integer(U)
            V = snap_to_half_integer(V)
            W = snap_to_half_integer(W)
            
            # Validate
            if validate_vectorized(U, V, W):
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
        
        return None

    # Try to find lower ranks
    # The search space is vast, so we only try a few ranks close to the bound or known lower bounds
    # Lower bound is 30. Schoolbook is 90.
    # We try a few random samples from the range [30, 89]
    # But random search in high dimensions is inefficient.
    # We will just try a few specific ranks.
    
    ranks_to_try = [89, 88, 87, 85, 80, 75, 70, 65, 60, 55, 50]
    
    for rank in ranks_to_try:
        if time.time() > deadline - 10:
            break
        if rank >= best_rank:
            continue
            
        result = try_rank(rank)
        if result is not None:
            best_result = result
            best_rank = rank
            # If we found one, we might try to go lower, but it's hard.
            # We can break if we are confident, but let's continue if time permits.
            # However, finding one at 89 doesn't guarantee 88.
            # We will just return the best found.

    return best_result
# EVOLVE-BLOCK-END
