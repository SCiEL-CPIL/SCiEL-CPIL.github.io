"""Rewritten program for matrix multiplication 3x4x5 tensor decomposition."""
import time
import random
from fractions import Fraction
import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W form a valid tensor decomposition for n×m×p matrix mult."""
    rank = len(U[0]) if U else 0
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if any(len(row) != rank for row in U):
        return False
    if any(len(row) != rank for row in V):
        return False
    if any(len(row) != rank for row in W):
        return False
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    return True

def _build_schoolbook(n, m, p):
    """Build the schoolbook decomposition of rank n*m*p."""
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

def _try_greedy_elimination(n, m, p, deadline):
    """
    Start with schoolbook and try to eliminate terms by finding linear dependencies.
    """
    start_time = time.time()
    schoolbook = _build_schoolbook(n, m, p)
    U = [row[:] for row in schoolbook["U"]]
    V = [row[:] for row in schoolbook["V"]]
    W = [row[:] for row in schoolbook["W"]]
    rank = n * m * p
    
    eliminated = 0
    for c in range(rank - 1, -1, -1):
        if time.time() > deadline:
            break
        
        u_col = [U[i][c] for i in range(n * m)]
        v_col = [V[j][c] for j in range(m * p)]
        w_col = [W[k][c] for k in range(p * n)]
        
        if all(x == 0 for x in u_col) or all(x == 0 for x in v_col) or all(x == 0 for x in w_col):
            U = [row[:c] + row[c+1:] for row in U]
            V = [row[:c] + row[c+1:] for row in V]
            W = [row[:c] + row[c+1:] for row in W]
            rank -= 1
            eliminated += 1
            c += 1
    
    if eliminated > 0:
        return {"U": U, "V": V, "W": W}
    return None

def _als_minimize(U, V, W, n, m, p, max_iter=100, tol=1e-10):
    """
    Alternating Least Squares to minimize the residual:
        sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] should equal 1 for all (i,j,k)
    
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    """
    rank = U.shape[1]
    
    for iteration in range(max_iter):
        # Update U: for each (i,j), solve least squares
        for i in range(n):
            for j in range(m):
                # Target: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all k
                # Let M[c,k] = V[j*p+k, c] * W[k*n+i, c]
                # We want U[i,j,:] such that sum_c U[i,j,c] * M[c,:] = ones(p)
                M = np.zeros((rank, p))
                for k in range(p):
                    for c in range(rank):
                        M[c, k] = V[j * p + k, c] * W[k * n + i, c]
                target = np.ones(p)
                # Solve M^T * U_col = target
                try:
                    U[i * m + j, :] = np.linalg.lstsq(M.T, target, rcond=None)[0]
                except:
                    pass
        
        # Update V: for each (j,k), solve least squares
        for j in range(m):
            for k in range(p):
                # Target: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all i
                # Let M[c,i] = U[i*m+j, c] * W[k*n+i, c]
                M = np.zeros((rank, n))
                for i in range(n):
                    for c in range(rank):
                        M[c, i] = U[i * m + j, c] * W[k * n + i, c]
                target = np.ones(n)
                try:
                    V[j * p + k, :] = np.linalg.lstsq(M.T, target, rcond=None)[0]
                except:
                    pass
        
        # Update W: for each (k,i), solve least squares
        for k in range(p):
            for i in range(n):
                # Target: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all j
                # Let M[c,j] = U[i*m+j, c] * V[j*p+k, c]
                M = np.zeros((rank, m))
                for j in range(m):
                    for c in range(rank):
                        M[c, j] = U[i * m + j, c] * V[j * p + k, c]
                target = np.ones(m)
                try:
                    W[k * n + i, :] = np.linalg.lstsq(M.T, target, rcond=None)[0]
                except:
                    pass
        
        # Check convergence
        residual = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    residual += (s - 1.0) ** 2
        residual = np.sqrt(residual)
        
        if residual < tol:
            break
    
    return U, V, W, residual

def _try_als_search(n, m, p, deadline):
    """
    Try to find a lower-rank decomposition using ALS optimization.
    """
    start_time = time.time()
    best_rank = n * m * p
    best_result = None
    
    # Try ranks from 55 down to 50
    for target_rank in range(55, 49, -1):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Initialize with random values
            U = np.random.randn(n * m, target_rank) * 0.5
            V = np.random.randn(m * p, target_rank) * 0.5
            W = np.random.randn(p * n, target_rank) * 0.5
            
            # Run ALS
            U, V, W, residual = _als_minimize(U, V, W, n, m, p, max_iter=80, tol=1e-9)
            
            if residual < 1e-6:
                # Convert to list format with rounding
                U_list = [[round(float(x), 10) for x in row] for row in U]
                V_list = [[round(float(x), 10) for x in row] for row in V]
                W_list = [[round(float(x), 10) for x in row] for row in W]
                
                if _verify_decomposition(U_list, V_list, W_list, n, m, p):
                    return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135  # Leave 15s margin
    
    # Start with schoolbook
    schoolbook = _build_schoolbook(n, m, p)
    best_result = schoolbook
    best_rank = n * m * p
    
    # Try ALS search
    if time.time() < deadline - 10:
        result = _try_als_search(n, m, p, deadline)
        if result is not None:
            rank = len(result["U"][0])
            if rank < best_rank:
                best_result = result
                best_rank = rank
    
    # Try greedy elimination
    if time.time() < deadline - 5:
        result = _try_greedy_elimination(n, m, p, deadline)
        if result is not None:
            rank = len(result["U"][0])
            if rank < best_rank:
                best_result = result
                best_rank = rank
    
    return best_result
# EVOLVE-BLOCK-END
