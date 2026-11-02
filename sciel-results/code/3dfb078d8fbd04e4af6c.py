import time
import sys
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [3, 3, 3]

def _verify(U, V, W):
    """Verify that U,V,W with given indices correctly decompose matrix multiplication.
    Returns True if decomposition is exact."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _construct_schoolbook():
    n, m, p = DIMENSIONS
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

def _search_low_rank(max_rank=26, deadline=None):
    """Search for a low-rank decomposition using numerical optimization."""
    n, m, p = 3, 3, 3
    num_rows_U = n * m  # 9
    num_rows_V = m * p  # 9
    num_rows_W = p * n  # 9
    
    best_result = None
    best_rank = 27
    
    for rank in range(10, min(max_rank + 1, 27)):
        if deadline and time.time() > deadline:
            break
        
        # Variables: U (9 x rank), V (9 x rank), W (9 x rank)
        # Total: 27 * rank variables
        # Constraints: 27 equations
        
        def residual(x):
            U = x[:num_rows_U * rank].reshape(num_rows_U, rank)
            V = x[num_rows_U * rank : num_rows_U * rank + num_rows_V * rank].reshape(num_rows_V, rank)
            W = x[num_rows_U * rank + num_rows_V * rank :].reshape(num_rows_W, rank)
            
            res = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for c in range(rank):
                            s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        res.append(s - 1)
            return res
        
        # Try multiple random initializations
        np.random.seed(42)
        for attempt in range(30):
            if deadline and time.time() > deadline:
                break
                
            x0 = np.random.randn(27 * rank) * 0.5
            
            try:
                result = least_squares(residual, x0, method='lm', max_nfev=2000)
            except:
                continue
            
            if np.max(np.abs(result.fun)) < 1e-8:
                U = result.x[:num_rows_U * rank].reshape(num_rows_U, rank)
                V = result.x[num_rows_U * rank : num_rows_U * rank + num_rows_V * rank].reshape(num_rows_V, rank)
                W = result.x[num_rows_U * rank + num_rows_V * rank :].reshape(num_rows_W, rank)
                
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                
                if _verify(U_list, V_list, W_list):
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = rank
                    break
        
        if best_result is not None:
            break
    
    return best_result

def construct():
    overall_deadline = time.time() + 140
    
    # Try to find a low-rank decomposition
    result = _search_low_rank(max_rank=26, deadline=overall_deadline - 10)
    if result is not None:
        return result
    
    # Fallback to schoolbook
    return _construct_schoolbook()
