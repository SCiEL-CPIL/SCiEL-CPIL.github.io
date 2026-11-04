import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [4, 4, 5]
START_TIME = time.time()
TIME_LIMIT = 145  # seconds, leave 5s margin for the 150s kill

def _time_left():
    return TIME_LIMIT - (time.time() - START_TIME)

def _schoolbook(n=4, m=4, p=5):
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

def _verify_decomposition(U, V, W, n=4, m=4, p=5):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    # Check shapes
    if len(U) != n * m:
        return False
    if len(V) != m * p:
        return False
    if len(W) != p * n:
        return False
    for row in U:
        if len(row) != rank:
            return False
    for row in V:
        if len(row) != rank:
            return False
    for row in W:
        if len(row) != rank:
            return False
    
    # Build the tensor from U, V, W
    # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This should equal 1 if i, j, k form a valid triple (they always do for MM tensor)
    # Actually for matrix multiplication, the tensor is:
    # T[i,j,k] = 1 if the indices are "connected" properly
    # For MM: C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T[i,j,k] = delta_{i,j,k} where the structure is:
    # U has indices (i,j), V has indices (j,k), W has indices (k,i)
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This should equal 1 for all valid (i,j,k) triples
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _try_greedy_improvement(n=4, m=4, p=5, max_rank=79):
    """
    Try to find a decomposition with rank < n*m*p by using a greedy approach.
    Start from schoolbook and try to merge/remove terms.
    """
    if _time_left() < 5:
        return None
    
    best = None
    sb = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find lower rank by random search
    for target_rank in range(max_rank, best_rank - 1, -1):
        if _time_left() < 3:
            break
        
        # Try several random initializations
        for attempt in range(50):
            if _time_left() < 3:
                break
            
            # Initialize with random values
            U = np.zeros((n*m, target_rank))
            V = np.zeros((m*p, target_rank))
            W = np.zeros((p*n, target_rank))
            
            # Use schoolbook as a starting point if target_rank >= schoolbook rank
            # Otherwise, try to fit
            if target_rank >= n*m*p:
                U[:n*m, :n*m*p] = np.array(sb['U'], dtype=float)
                V[:m*p, :n*m*p] = np.array(sb['V'], dtype=float)
                W[:p*n, :n*m*p] = np.array(sb['W'], dtype=float)
                # Zero out extra columns
                if target_rank > n*m*p:
                    U[:, n*m*p:] = 0
                    V[:, n*m*p:] = 0
                    W[:, n*m*p:] = 0
            else:
                # For lower rank, use random initialization
                U = np.random.randn(n*m, target_rank) * 0.5
                V = np.random.randn(m*p, target_rank) * 0.5
                W = np.random.randn(p*n, target_rank) * 0.5
            
            # Simple gradient-free local search: try small perturbations
            # and check if we can reduce the error
            U = U.copy()
            V = V.copy()
            W = W.copy()
            
            # Compute current error
            def compute_error(U, V, W):
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for r in range(target_rank):
                                val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                            err += (val - 1.0) ** 2
                return err
            
            current_err = compute_error(U, V, W)
            
            # Try to improve with simple coordinate descent
            for step in range(100):
                if _time_left() < 2:
                    break
                
                improved = False
                # Try perturbing random entries
                for _ in range(20):
                    if _time_left() < 2:
                        break
                    
                    mat = random.choice([U, V, W])
                    ri = random.randint(0, mat.shape[0]-1)
                    rj = random.randint(0, mat.shape[1]-1)
                    delta = random.uniform(-0.1, 0.1)
                    mat[ri, rj] += delta
                    
                    new_err = compute_error(U, V, W)
                    if new_err < current_err - 1e-12:
                        current_err = new_err
                        improved = True
                    else:
                        mat[ri, rj] -= delta
                
                if not improved:
                    break
            
            # Check if we achieved exact decomposition
            if current_err < 1e-6:
                # Verify exactly
                U_list = [[float(x) for x in row] for row in U]
                V_list = [[float(x) for x in row] for row in V]
                W_list = [[float(x) for x in row] for row in W]
                
                if _verify_decomposition(U_list, V_list, W_list, n, m, p):
                    if best is None or target_rank < best['rank']:
                        best = {'U': U_list, 'V': V_list, 'W': W_list, 'rank': target_rank}
                    return best
    
    return best

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying a 4x4 matrix by a 4x5 matrix.
    Attempts to find a decomposition with rank < 80 using search methods.
    Falls back to schoolbook decomposition of rank 80 if no improvement is found.
    """
    n, m, p = DIMENSIONS
    
    # First try to find an improved decomposition
    result = _try_greedy_improvement(n, m, p, max_rank=79)
    
    if result is not None:
        return {"U": result['U'], "V": result['V'], "W": result['W']}
    
    # Fall back to schoolbook
    return _schoolbook(n, m, p)
