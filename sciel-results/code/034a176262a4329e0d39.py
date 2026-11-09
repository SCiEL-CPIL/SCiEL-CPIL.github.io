"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p
    deadline = time.time() + 110.0  # safety margin before 120s (and 150s kill)

    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W

    U_best, V_best, W_best = schoolbook()
    best_rank = rank_school

    # Verify a candidate decomposition exactly (with tolerance for float error)
    def verify(U, V, W, tol=1e-6):
        # Tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # For matrix multiplication, T[i,j,k] should be 1 if we're computing (A@B)[i,k] = sum_j A[i,j]*B[j,k]
        # The tensor for matmul is T[i,j,k] = 1 for all i,j,k (it's the identity in the middle index)
        # Actually, the matmul tensor has T[i,j,k] = 1 if we're looking at coefficient of A[i,j]*B[j,k] in C[i,k]
        # So T[i,j,k] = 1 for all valid i,j,k.
        # Let's verify: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1 for all i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(U.shape[1]):
                        s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    # Try to find lower rank decompositions using random search / alternating least squares
    # The matmul tensor for (n x m) @ (m x p) -> (n x p) has known rank bounds.
    # For 2x5 x 5x6, schoolbook is 60. Known lower bound is max(2*5, 5*6, 2*6) = 30.
    # There may be better decompositions. Let's try a simple iterative approach.
    
    # Since the problem is small, we can try to use a known construction or search.
    # For now, let's try a few random initializations with ALS if time permits.
    
    def try_als(rank, max_iters=50, seed=0):
        """Try alternating least squares to find a rank decomposition."""
        rng = np.random.RandomState(seed)
        U = rng.randn(n * m, rank)
        V = rng.randn(m * p, rank)
        W = rng.randn(p * n, rank)
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                return None, None, None
            
            # Update U: for each (i,j), U[i*m+j, :] should satisfy
            # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all k
            # This is a linear system in U[i*m+j, :]
            for i in range(n):
                for j in range(m):
                    # Build system: for each k, sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c]) = 1
                    A_mat = np.zeros((p, rank))
                    for k in range(p):
                        A_mat[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    b = np.ones(p)
                    # Solve least squares
                    try:
                        sol, residuals, rank_s, sv = np.linalg.lstsq(A_mat, b, rcond=None)
                        U[i * m + j, :] = sol
                    except:
                        pass
            
            # Update V: for each (j,k), V[j*p+k, :] should satisfy
            # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i
            for j in range(m):
                for k in range(p):
                    A_mat = np.zeros((n, rank))
                    for i in range(n):
                        A_mat[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    b = np.ones(n)
                    try:
                        sol, residuals, rank_s, sv = np.linalg.lstsq(A_mat, b, rcond=None)
                        V[j * p + k, :] = sol
                    except:
                        pass
            
            # Update W: for each (k,i), W[k*n+i, :] should satisfy
            # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all j
            for k in range(p):
                for i in range(n):
                    A_mat = np.zeros((m, rank))
                    for j in range(m):
                        A_mat[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    b = np.ones(m)
                    try:
                        sol, residuals, rank_s, sv = np.linalg.lstsq(A_mat, b, rcond=None)
                        W[k * n + i, :] = sol
                    except:
                        pass
        
        # Verify
        if verify(U, V, W, tol=1e-4):
            return U, V, W
        return None, None, W

    # Try ranks from 30 up to schoolbook rank
    # But ALS is expensive, so try a few seeds for each rank
    for rank in range(30, best_rank):
        if time.time() > deadline - 10:
            break
        found = False
        for seed in range(5):
            if time.time() > deadline - 5:
                break
            U_c, V_c, W_c = try_als(rank, max_iters=30, seed=seed + rank * 100)
            if U_c is not None:
                # Verify more strictly
                if verify(U_c, V_c, W_c, tol=1e-6):
                    U_best, V_best, W_best = U_c, V_c, W_c
                    best_rank = rank
                    found = True
                    break
        if found:
            # Try even lower? No, we're going up from 30. If we found one, we can stop at this rank
            # but let's continue to see if we can go lower (we're already going from 30 up)
            # Actually we're iterating from 30 upward, so first found is the minimum in this range
            break

    # Convert to lists as required
    U_list = U_best.tolist()
    V_list = V_best.tolist()
    W_list = W_best.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
