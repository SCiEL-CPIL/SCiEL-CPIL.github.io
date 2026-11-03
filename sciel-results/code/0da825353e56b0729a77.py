"""Search for rank-minimal tensor decomposition of matrix multiplication 3x5 by 5x5."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 5]

def _validate(U, V, W, n=3, m=5, p=5, tol=1e-9):
    """Check whether U,V,W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    # T[i,j,k] = 1 for all i<n, j<m, k<p
    # Decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _random_init(n, m, p, rank, seed=0):
    rng = np.random.RandomState(seed)
    U = rng.uniform(-1, 1, size=(n*m, rank))
    V = rng.uniform(-1, 1, size=(m*p, rank))
    W = rng.uniform(-1, 1, size=(p*n, rank))
    return U.tolist(), V.tolist(), W.tolist()

def _evaluate_residuals(U, V, W, n=3, m=5, p=5):
    """Return the residuals T[i,j,k] - sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] as a list."""
    rank = len(U[0])
    residuals = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                residuals.append(s - 1.0)
    return residuals

def _try_rank(rank, max_restarts=50, time_limit=None):
    """Attempt to find a valid decomposition of the given rank."""
    n, m, p = 3, 5, 5
    deadline = time.time() + (time_limit - (time.time() - _start_time) if time_limit else 120)
    
    for seed in range(max_restarts):
        if time.time() > deadline:
            return None
        U, V, W = _random_init(n, m, p, rank, seed=seed)
        
        # Convert to numpy for gradient-based refinement
        U_np = np.array(U, dtype=np.float64)
        V_np = np.array(V, dtype=np.float64)
        W_np = np.array(W, dtype=np.float64)
        
        # Gradient descent to minimize sum of squared residuals
        lr = 0.01
        for it in range(200):
            if time.time() > deadline:
                break
            # Compute residuals
            res = []
            res_deriv_U = np.zeros_like(U_np)
            res_deriv_V = np.zeros_like(V_np)
            res_deriv_W = np.zeros_like(W_np)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(rank):
                            s += U_np[i*m+j, c] * V_np[j*p+k, c] * W_np[k*n+i, c]
                        r = s - 1.0
                        for c in range(rank):
                            res_deriv_U[i*m+j, c] += r * V_np[j*p+k, c] * W_np[k*n+i, c]
                            res_deriv_V[j*p+k, c] += r * U_np[i*m+j, c] * W_np[k*n+i, c]
                            res_deriv_W[k*n+i, c] += r * U_np[i*m+j, c] * V_np[j*p+k, c]
                        res.append(r)
            
            U_np -= lr * res_deriv_U
            V_np -= lr * res_deriv_V
            W_np -= lr * res_deriv_W
            
            max_res = max(abs(r) for r in res)
            if max_res < 1e-9:
                break
        
        # Convert back to list and validate
        U_list = U_np.tolist()
        V_list = V_np.tolist()
        W_list = W_np.tolist()
        
        if _validate(U_list, V_list, W_list):
            return U_list, V_list, W_list
    
    return None

_start_time = time.time()

def construct():
    global _start_time
    _start_time = time.time()
    
    n, m, p = 3, 5, 5
    schoolbook_rank = n * m * p  # 75
    
    # We know the minimum rank is at most 75 (schoolbook)
    # For 3x5x5, the tensor is T_{3,5,5}. Known lower bounds suggest rank >= max(n*m, m*p, p*n) = 25
    # Actually for matrix multiplication (n x m) * (m x p), the rank is at least max(n*m, m*p, n*p)
    # and at most n*m*p.
    # For 3x5x5: lower bound = max(15, 25, 15) = 25
    # Upper bound (schoolbook) = 75
    
    # Try to find lower rank decompositions starting from lower bounds
    # However, it's very hard to find exact rational/half-integer solutions
    # so let's first ensure we return a valid schoolbook decomposition
    # and only try to improve if we have time.
    
    # Schoolbook decomposition
    U = [[0] * schoolbook_rank for _ in range(n * m)]
    V = [[0] * schoolbook_rank for _ in range(m * p)]
    W = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    
    best_result = (U, V, W)
    best_rank = schoolbook_rank
    
    remaining_time = 120 - (time.time() - _start_time)
    if remaining_time < 5:
        return {"U": best_result[0], "V": best_result[1], "W": best_result[2]}
    
    # Try to find lower rank solutions
    # The actual minimum rank for 3x5x5 is likely higher than the trivial lower bound
    # Let's try ranks in decreasing order from some upper limit
    # Due to time constraints, we'll try a limited number of ranks
    
    # Known: for small matrix multiplication, exact rank is hard
    # Let's try rank 30, 35, 40, 45, 50, 55, 60, 65, 70 with limited time each
    
    for rank in [35, 40, 45, 50, 55, 60, 65, 70]:
        remaining_time = 120 - (time.time() - _start_time)
        if remaining_time < 10:
            break
        
        time_for_this_rank = min(10, remaining_time - 5)
        result = _try_rank(rank, max_restarts=20, time_limit=time.time() + time_for_this_rank + 5)
        
        if result is not None:
            best_result = result
            best_rank = rank
            # Try even lower
            for lower_rank in [rank-5, rank-10]:
                if lower_rank < 25:
                    continue
                remaining_time = 120 - (time.time() - _start_time)
                if remaining_time < 10:
                    break
                time_for_lower = min(10, remaining_time - 5)
                lower_result = _try_rank(lower_rank, max_restarts=20, time_limit=time.time() + time_for_lower + 5)
                if lower_result is not None:
                    best_result = lower_result
                    best_rank = lower_rank
            break
    
    return {"U": best_result[0], "V": best_result[1], "W": best_result[2]}
