DIMENSIONS = [5, 5, 5]

import time
import numpy as np
from scipy.optimize import minimize

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = U.shape[1]
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    return max_err < 1e-9

def construct():
    start_time = time.time()
    DEADLINE = 140.0
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Try to find a low-rank decomposition using randomized search
    best_rank = n * m * p  # Schoolbook
    best_U = [[0.0] * best_rank for _ in range(nm)]
    best_V = [[0.0] * best_rank for _ in range(mp)]
    best_W = [[0.0] * best_rank for _ in range(pn)]
    
    # Initialize schoolbook
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i*m+j][column] = 1.0
                best_V[j*p+k][column] = 1.0
                best_W[k*n+i][column] = 1.0
                column += 1
    
    # Try to find a lower-rank decomposition
    for target_rank in range(99, 125):
        if time.time() - start_time > DEADLINE:
            break
        
        # Try a few random initializations
        for attempt in range(5):
            if time.time() - start_time > DEADLINE:
                break
            
            np.random.seed(attempt * 1000 + target_rank)
            
            # Initialize with random integer values
            U = np.random.randint(-2, 3, size=(nm, target_rank), dtype=float)
            V = np.random.randint(-2, 3, size=(mp, target_rank), dtype=float)
            W = np.random.randint(-2, 3, size=(pn, target_rank), dtype=float)
            
            # Use alternating minimization to drive residual to zero
            for iteration in range(50):
                if time.time() - start_time > DEADLINE:
                    break
                
                # Update W: for each c, W[:,c] is determined by U and V
                # The system is linear in W
                # For each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                # This is a linear system in W
                
                # Build the linear system
                A = np.zeros((n*m*p, pn * target_rank))
                b = np.ones(n*m*p)
                
                idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            for c in range(target_rank):
                                A[idx, (k*n+i) * target_rank + c] = U[i*m+j, c] * V[j*p+k, c]
                            idx += 1
                
                # Solve the linear system
                try:
                    W_flat, residuals, rank_sys, sv = np.linalg.lstsq(A, b, rcond=None)
                    W_new = W_flat.reshape(pn, target_rank)
                    
                    # Check if the solution is exact
                    max_err = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = 0.0
                                for c in range(target_rank):
                                    val += U[i*m+j, c] * V[j*p+k, c] * W_new[k*n+i, c]
                                err = abs(val - 1.0)
                                if err > max_err:
                                    max_err = err
                    
                    if max_err < 1e-9:
                        # Found a valid decomposition
                        best_rank = target_rank
                        best_U = U.tolist()
                        best_V = V.tolist()
                        best_W = W_new.tolist()
                        break
                except np.linalg.LinAlgError:
                    continue
            
            if best_rank == target_rank:
                break
        
        if best_rank == target_rank:
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
