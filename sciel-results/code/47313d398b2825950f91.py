import numpy as np
import time

DIMENSIONS = [3, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    num_constraints = n * m * p
    
    start_time = time.time()
    deadline = start_time + 110
    
    # Try ranks from 46 to 53
    for rank in range(46, 54):
        if time.time() > deadline:
            break
            
        # More restarts for lower ranks
        num_restarts = 50 if rank < 50 else 20
        
        for attempt in range(num_restarts):
            if time.time() > deadline:
                break
                
            # Initialize U, V randomly with small values to avoid overflow
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            
            A_eq = np.zeros((num_constraints, rank))
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_val = U[i * m + j]
                        v_val = V[j * p + k]
                        A_eq[idx] = u_val * v_val
                        idx += 1
            
            M_big = np.zeros((num_constraints, p * n * rank))
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        w_row_start = (k * n + i) * rank
                        M_big[idx, w_row_start : w_row_start + rank] = A_eq[idx]
                        idx += 1
            
            b = np.ones(num_constraints)
            
            try:
                W_flat, residuals, rank_sol, s = np.linalg.lstsq(M_big, b, rcond=None)
            except np.linalg.LinAlgError:
                continue
                
            if len(residuals) > 0 and residuals[0] > 1e-9:
                continue
                
            W = W_flat.reshape((p * n, rank))
            
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # Fallback to schoolbook
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
