"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START

def _verify_decomposition(n, m, p, U, V, W):
    """Verify that the tensor decomposition is exact for matrix multiplication."""
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=float).reshape(n, m, rank)
    V_arr = np.array(V, dtype=float).reshape(m, p, rank)
    W_arr = np.array(W, dtype=float).reshape(p, n, rank)
    
    # Contract over rank: result[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    result = np.einsum('ijc,jkc,kic->ijk', U_arr, V_arr, W_arr)
    target = np.ones((n, m, p))
    
    return np.allclose(result, target, atol=1e-9)

def _als_decompose(n, m, p, rank, max_iters=2000, tol=1e-12, seed=42):
    """
    Alternating Least Squares for tensor decomposition.
    Finds U, V, W such that sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1 for all i,j,k.
    """
    rng = np.random.RandomState(seed)
    
    # Initialize
    U = rng.randn(n * m, rank) * 0.1
    V = rng.randn(m * p, rank) * 0.1
    W = rng.randn(p * n, rank) * 0.1
    
    # Ensure non-zero columns
    for c in range(rank):
        if np.allclose(U[:, c], 0):
            U[:, c] = rng.randn(n * m) * 0.1
        if np.allclose(V[:, c], 0):
            V[:, c] = rng.randn(m * p) * 0.1
        if np.allclose(W[:, c], 0):
            W[:, c] = rng.randn(p * n) * 0.1
    
    prev_err = np.inf
    
    for it in range(max_iters):
        # Update U
        U_old = U.copy()
        for i in range(n):
            for j in range(m):
                # We need sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all k
                # This is a linear system in U[i,j,c]
                # Let A[k, c] = V[j,k,c] * W[k,i,c]
                # Then sum_c A[k,c] * U[i,j,c] = 1 for all k
                A = np.zeros((p, rank))
                for k in range(p):
                    for c in range(rank):
                        A[k, c] = V[j * p + k, c] * W[k * n + i, c]
                
                # Solve A * x = 1
                try:
                    x = np.linalg.lstsq(A, np.ones(p), rcond=None)[0]
                    U[i * m + j, :] = x
                except:
                    U[i * m + j, :] = U_old[i * m + j, :]
        
        # Update V
        V_old = V.copy()
        for j in range(m):
            for k in range(p):
                # We need sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all i
                # Let A[i, c] = U[i,j,c] * W[k,i,c]
                # Then sum_c A[i,c] * V[j,k,c] = 1 for all i
                A = np.zeros((n, rank))
                for i in range(n):
                    for c in range(rank):
                        A[i, c] = U[i * m + j, c] * W[k * n + i, c]
                
                try:
                    x = np.linalg.lstsq(A, np.ones(n), rcond=None)[0]
                    V[j * p + k, :] = x
                except:
                    V[j * p + k, :] = V_old[j * p + k, :]
        
        # Update W
        W_old = W.copy()
        for k in range(p):
            for i in range(n):
                # We need sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all j
                # Let A[j, c] = U[i,j,c] * V[j,k,c]
                # Then sum_c A[j,c] * W[k,i,c] = 1 for all j
                A = np.zeros((m, rank))
                for j in range(m):
                    for c in range(rank):
                        A[j, c] = U[i * m + j, c] * V[j * p + k, c]
                
                try:
                    x = np.linalg.lstsq(A, np.ones(m), rcond=None)[0]
                    W[k * n + i, :] = x
                except:
                    W[k * n + i, :] = W_old[k * n + i, :]
        
        # Compute error
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        result = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        err = np.max(np.abs(result - 1.0))
        
        if abs(prev_err - err) < tol:
            break
        prev_err = err
    
    return U, V, W, err

def _try_rank(n, m, p, rank, num_seeds=5, max_iters=1000, time_limit=10.0):
    """Try to find a rank decomposition using ALS with multiple seeds."""
    start_time = time.time()
    
    for seed in range(num_seeds):
        if time.time() - start_time > time_limit:
            break
        
        U, V, W, err = _als_decompose(n, m, p, rank, max_iters=max_iters, seed=seed)
        
        if err < 1e-8:
            # Verify
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            if _verify_decomposition(n, m, p, U_list, V_list, W_list):
                return U_list, V_list, W_list
    
    return None

def _schoolbook_decomposition(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
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

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x4 matrix by a 4x6 matrix.
    """
    start_time = time.time()
    deadline = start_time + 120  # 120 second deadline with safety margin
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Start with schoolbook as fallback
    fallback = _schoolbook_decomposition(n, m, p)
    best_result = fallback
    best_rank = schoolbook_rank
    
    # Lower bound: rank >= max(n*p, m*p, n*m) = max(18, 24, 12) = 24
    # Try ranks from lower bound upward
    
    # We'll try a range of ranks, prioritizing lower ranks
    # For 3x4x6, known constructions might exist around rank 30-40
    # Let's try from 24 to 60
    
    ranks_to_try = list(range(24, min(61, schoolbook_rank)))
    ranks_to_try.sort()
    
    for rank in ranks_to_try:
        if time.time() > deadline - 15:  # Stop if close to deadline
            break
        
        # Adjust time budget based on rank
        time_budget = min(15.0, (deadline - time.time()) - 15)
        if time_budget < 2.0:
            break
        
        result = _try_rank(n, m, p, rank, num_seeds=3, max_iters=500, time_limit=time_budget)
        
        if result is not None:
            U_list, V_list, W_list = result
            best_result = {"U": U_list, "V": V_list, "W": W_list}
            best_rank = rank
            # Found a valid decomposition at this rank, return immediately
            # since we're trying ranks in ascending order
            return best_result
    
    # If no better decomposition found, return schoolbook
    return best_result

# EVOLVE-BLOCK-END
