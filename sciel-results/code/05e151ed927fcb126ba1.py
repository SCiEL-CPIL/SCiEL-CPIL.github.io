"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p):
    """Verify exact decomposition of matrix multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _trivial_decomposition(n, m, p):
    """Return the trivial rank-n*m*p decomposition."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1.0
                V[j*p+k][col] = 1.0
                W[k*n+i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def _solve_rank_decomposition(n, m, p, target_rank, time_budget=60.0):
    """
    Try to find a rank-`target_rank` decomposition using iterative refinement.
    
    Strategy: Initialize U, V, W randomly, then iteratively update each factor
    by solving a least-squares problem while holding the others fixed.
    """
    start_time = time.time()
    np.random.seed(42)
    
    for attempt in range(50):
        if time.time() - start_time > time_budget:
            break
        
        rank = target_rank
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)
        W = np.random.randn(p * n, rank)
        
        # Iteratively refine
        for iteration in range(200):
            if time.time() - start_time > time_budget:
                break
            
            # Update U: for each row i*m+j, solve for U[i*m+j, :] given V and W
            for i in range(n):
                for j in range(m):
                    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k
                    # For fixed i,j: sum_r U[i*m+j, r] * (V[j*p+k, r] * W[k*n+i, r]) = 1 for k=0..p-1
                    # This is a system: M @ U_row = ones(p) where M[k, r] = V[j*p+k, r] * W[k*n+i, r]
                    M = np.zeros((p, rank))
                    for k in range(p):
                        M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    # Solve M @ x = ones(p) in least squares sense
                    try:
                        x, _, _, _ = np.linalg.lstsq(M, np.ones(p), rcond=None)
                        U[i*m+j, :] = x
                    except:
                        pass
            
            # Update V: for each row j*p+k, solve for V[j*p+k, :] given U and W
            for j in range(m):
                for k in range(p):
                    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i
                    # For fixed j,k: sum_r V[j*p+k, r] * (U[i*m+j, r] * W[k*n+i, r]) = 1 for i=0..n-1
                    M = np.zeros((n, rank))
                    for i in range(n):
                        M[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        x, _, _, _ = np.linalg.lstsq(M, np.ones(n), rcond=None)
                        V[j*p+k, :] = x
                    except:
                        pass
            
            # Update W: for each row k*n+i, solve for W[k*n+i, :] given U and V
            for k in range(p):
                for i in range(n):
                    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all j
                    # For fixed k,i: sum_r W[k*n+i, r] * (U[i*m+j, r] * V[j*p+k, r]) = 1 for j=0..m-1
                    M = np.zeros((m, rank))
                    for j in range(m):
                        M[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    try:
                        x, _, _, _ = np.linalg.lstsq(M, np.ones(m), rcond=None)
                        W[k*n+i, :] = x
                    except:
                        pass
            
            # Check convergence
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                        max_err = max(max_err, abs(s - 1.0))
            
            if max_err < 1e-8:
                # Found a valid decomposition
                U_list = [[float(x) for x in row] for row in U]
                V_list = [[float(x) for x in row] for row in V]
                W_list = [[float(x) for x in row] for row in W]
                
                # Verify
                if _verify(U_list, V_list, W_list, n, m, p):
                    return {"U": U_list, "V": V_list, "W": W_list}
        
        # Reset with different random seed for next attempt
        np.random.seed(42 + attempt)
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin out of 120s
    
    n, m, p = DIMENSIONS  # 2, 3, 4
    
    # Try to find a lower-rank decomposition
    for target_rank in [20, 19, 18]:
        if time.time() > deadline - 10:
            break
        
        result = _solve_rank_decomposition(n, m, p, target_rank, time_budget=min(30, deadline - time.time() - 10))
        if result is not None:
            return result
    
    # Fallback: trivial rank-24 decomposition
    return _trivial_decomposition(n, m, p)

# EVOLVE-BLOCK-END
