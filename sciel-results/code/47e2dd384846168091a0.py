"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from fractions import Fraction

def _validate(U, V, W, n, m, p):
    """Check if the decomposition exactly reproduces matrix multiplication."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=object)
    V_arr = np.array(V, dtype=object)
    W_arr = np.array(W, dtype=object)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                if s != 1:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def _try_find_low_rank(n, m, p, target_rank, max_time=60, seed=42):
    """
    Try to find a rank-`target_rank` decomposition using iterative refinement.
    
    Approach: Start with a random initialization of U, V, W with `target_rank` columns.
    Then use a gradient-like method or fixed-point iteration to satisfy the constraints:
    sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all (i,j,k).
    
    We use an alternating least-squares style approach:
    - Fix V, W: solve for U (linear in U)
    - Fix U, W: solve for V (linear in V)  
    - Fix U, V: solve for W (linear in W)
    """
    random.seed(seed)
    np.random.seed(seed)
    
    start_time = time.time()
    U = np.random.randn(n*m, target_rank) * 0.5
    V = np.random.randn(m*p, target_rank) * 0.5
    W = np.random.randn(p*n, target_rank) * 0.5
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for iteration in range(200):
        if time.time() - start_time > max_time:
            break
        
        # Fix V, W: for each (i,j), solve sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c]) = 1 for all k
        # This is a linear system: U[i*m+j,:] @ diag(V[j*p+k,:] * W[k*n+i,:]) = 1 for each k
        # More precisely: for fixed i,j, we need U[i*m+j,:] such that
        #   sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all k=0..p-1
        # This is: U[i*m+j,:] @ M_{i,j} = ones(p) where M_{i,j}[k,c] = V[j*p+k,c]*W[k*n+i,c]
        
        for i in range(n):
            for j in range(m):
                # Build the p x target_rank matrix M
                M = np.zeros((p, target_rank))
                for k in range(p):
                    M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                # Solve U[i*m+j,:] @ M = ones(p)
                # This is: M.T @ U[i*m+j,:].T = ones(p)
                try:
                    sol = np.linalg.lstsq(M.T, np.ones(p), rcond=None)[0]
                    U[i*m+j, :] = sol
                except:
                    pass
        
        # Fix U, W: for each (j,k), solve sum_c V[j*p+k,c] * (U[i*m+j,c] * W[k*n+i,c]) = 1 for all i
        for j in range(m):
            for k in range(p):
                M = np.zeros((n, target_rank))
                for i in range(n):
                    M[i, :] = U[i*m+j, :] * W[k*n+i, :]
                try:
                    sol = np.linalg.lstsq(M.T, np.ones(n), rcond=None)[0]
                    V[j*p+k, :] = sol
                except:
                    pass
        
        # Fix U, V: for each (k,i), solve sum_c W[k*n+i,c] * (U[i*m+j,c] * V[j*p+k,c]) = 1 for all j
        for k in range(p):
            for i in range(n):
                M = np.zeros((m, target_rank))
                for j in range(m):
                    M[j, :] = U[i*m+j, :] * V[j*p+k, :]
                try:
                    sol = np.linalg.lstsq(M.T, np.ones(m), rcond=None)[0]
                    W[k*n+i, :] = sol
                except:
                    pass
        
        # Compute error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(target_rank):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    err += (s - 1.0) ** 2
        err = np.sqrt(err / (n*m*p))
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < 1e-10:
            break
    
    # Try to round to half-integers and verify
    for scale in [1, 2, 4, 8]:
        U_r = np.round(best_U * scale) / scale
        V_r = np.round(best_V * scale) / scale
        W_r = np.round(best_W * scale) / scale
        
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(target_rank):
                        s += U_r[i*m+j, c] * V_r[j*p+k, c] * W_r[k*n+i, c]
                    if abs(s - 1.0) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            return U_r.tolist(), V_r.tolist(), W_r.tolist(), best_err
    
    return None, best_err

def _try_structured_reduction(n, m, p, max_time=80):
    """
    Try structured approaches to reduce rank below schoolbook.
    
    Key idea: The 3x3x5 problem can be decomposed. 
    For the first 3 columns of B (3x3x3), Strassen gives rank 7.
    For the last 2 columns (3x3x2), we need additional terms.
    
    Total structured rank: 7 + (3*3*2 - savings) = 7 + 18 - savings
    
    But we can do better by sharing structure. Let's try:
    - Use Strassen-like structure for the 3x3 part
    - Add terms for the extra 2 columns
    """
    start_time = time.time()
    
    # Try various target ranks
    for target_rank in range(30, 46):
        if time.time() - start_time > max_time - 10:
            break
        
        result, err = _try_find_low_rank(n, m, p, target_rank, max_time=20, seed=target_rank*1000)
        if result is not None:
            U, V, W = result
            # Verify
            if _validate(U, V, W, n, m, p):
                return U, V, W, target_rank
    
    return None, None, None, None

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook decomposition
    U, V, W = _schoolbook(n, m, p)
    best_U, best_V, best_W = U, V, W
    best_rank = len(U[0])
    
    # Try to find lower rank
    if time.time() < deadline - 20:
        result = _try_structured_reduction(n, m, p, max_time=min(60, deadline - start_time - 20))
        if result[0] is not None:
            best_U, best_V, best_W, best_rank = result
            if best_rank < 45:
                # Found something better
                pass
    
    # Final validation
    if time.time() < deadline - 5:
        if not _validate(best_U, best_V, best_W, n, m, p):
            U, V, W = _schoolbook(n, m, p)
            return {"U": U, "V": V, "W": W}
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
