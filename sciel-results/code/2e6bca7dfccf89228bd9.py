"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def _schoolbook_decomposition(n, m, p):
    """Return the trivial rank n*m*p decomposition."""
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


def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly represents matrix multiplication."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = np.sum(U_arr[i * m + j, :] * V_arr[j * p + k, :] * W_arr[k * n + i, :])
                if abs(actual - expected) > tol:
                    return False
    return True

def _try_random_decomposition(n, m, p, target_rank, max_iters=100, seed=None):
    """
    Attempt to find a decomposition of rank 'target_rank' using random initialization 
    and alternating least squares (ALS) or similar heuristic.
    """
    if seed is not None:
        np.random.seed(seed)
    
    U = np.random.randn(n * m, target_rank)
    V = np.random.randn(m * p, target_rank)
    W = np.random.randn(p * n, target_rank)
    
    # We want to satisfy: sum_r U[i,j,r] V[j,k,r] W[i,k,r] = 1 for all i,j,k
    # This is a system of equations.
    # Let's define the error as the sum of squared differences.
    # We can't easily do gradient descent on this non-linear system directly with simple numpy without autograd.
    # However, we can try to solve for one matrix given the others.
    # Fix V and W. Solve for U.
    # The equation is: sum_r U[i,j,r] (V[j,k,r] * W[i,k,r]) = 1.
    # For a fixed i,k, let c_r = V[j,k,r] * W[i,k,r]. This depends on j.
    # This is not linear in U if we group by i,j.
    # Actually, for a fixed j, the terms are U[i,j,r] * (V[j,k,r] * W[i,k,r]).
    # Let's iterate.
    
    best_error = float('inf')
    best_U, best_V, best_W = U, V, W
    
    for _ in range(max_iters):
        # Update U
        for i in range(n):
            for j in range(m):
                # We need U[i,j,:] such that sum_r U[i,j,r] * (V[j,k,r] * W[i,k,r]) = 1 for all k
                # This is a linear system for U[i,j,:].
                # Let A_k = V[j,k,:] * W[i,k,:]
                # We want sum_r U[i,j,r] * A_k[r] = 1 for all k.
                # This is A @ U[i,j,:] = 1 (vector of ones of length p)
                # A is p x rank.
                A = np.array([V[j, k, :] * W[i, k, :] for k in range(p)])
                # Solve A @ x = ones(p)
                try:
                    sol, residuals, rank_a, s = np.linalg.lstsq(A, np.ones(p), rcond=None)
                    U[i * m + j, :] = sol
                except:
                    pass
        
        # Update V
        for j in range(m):
            for k in range(p):
                # We need V[j,k,:] such that sum_r V[j,k,r] * (U[i,j,r] * W[i,k,r]) = 1 for all i
                # Let B_i = U[i,j,:] * W[i,k,:]
                # We want sum_r V[j,k,r] * B_i[r] = 1 for all i.
                # B is n x rank.
                B = np.array([U[i * m + j, :] * W[i, k, :] for i in range(n)])
                try:
                    sol, residuals, rank_b, s = np.linalg.lstsq(B, np.ones(n), rcond=None)
                    V[j * p + k, :] = sol
                except:
                    pass
        
        # Update W
        for i in range(n):
            for k in range(p):
                # We need W[i,k,:] such that sum_r W[i,k,r] * (U[i,j,r] * V[j,k,r]) = 1 for all j
                # Let C_j = U[i,j,:] * V[j,k,:]
                # We want sum_r W[i,k,r] * C_j[r] = 1 for all j.
                # C is m x rank.
                C = np.array([U[i * m + j, :] * V[j * p + k, :] for j in range(m)])
                try:
                    sol, residuals, rank_c, s = np.linalg.lstsq(C, np.ones(m), rcond=None)
                    W[k * n + i, :] = sol # Note: W is indexed by k*n + i in the problem statement? 
                    # Problem statement: W[k*n+i, r]. Yes.
                except:
                    pass
        
        # Calculate error
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                    error += (val - 1.0) ** 2
        if error < best_error:
            best_error = error
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
        if best_error < 1e-8:
            break
            
    if best_error < 1e-6:
        return best_U.tolist(), best_V.tolist(), best_W.tolist()
    return None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # First, try to find a decomposition of rank 35
    # If found, return it.
    # Try a few random seeds.
    for rank in [35]:
        for seed in range(10):
            if time.time() > deadline:
                break
            res = _try_random_decomposition(n, m, p, rank, max_iters=20, seed=seed)
            if res is not None:
                U, V, W = res
                if _verify_decomposition(U, V, W, n, m, p):
                    return {"U": U, "V": V, "W": W}
        
    # If not found, return schoolbook
    result = _schoolbook_decomposition(n, m, p)
    return result
# EVOLVE-BLOCK-END
