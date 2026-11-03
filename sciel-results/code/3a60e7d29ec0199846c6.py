import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    target_rank = 101
    
    # Time limit guard
    start_time = time.time()
    time_limit = 110.0 # Leave 10s margin
    
    best_result = None
    
    # Try multiple random restarts
    max_restarts = 50
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
            
        rank = target_rank
        
        # Initialize U and V with random integers
        # Using small integers to help with rounding later
        U = np.random.randint(-2, 3, size=(n * m, rank))
        V = np.random.randint(-2, 3, size=(m * p, rank))
        
        # W will be computed
        W = np.zeros((p * n, rank))
        
        # Iterative refinement
        converged = False
        for iter in range(20):
            if time.time() - start_time > time_limit:
                break
                
            # 1. Solve for W given U, V
            # For each k, i:
            # Variables: W[k*n+i, c]
            # Equations: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all j
            for k in range(p):
                for i in range(n):
                    # Build matrix A of shape (m, rank)
                    A = np.zeros((m, rank))
                    for j in range(m):
                        for c in range(rank):
                            A[j, c] = U[i * m + j, c] * V[j * p + k, c]
                    
                    # Solve A x = 1
                    try:
                        x, residuals, rank_val, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                    except np.linalg.LinAlgError:
                        x = np.zeros(rank)
                    
                    # Round to nearest 0.5
                    x_rounded = np.round(x * 2) / 2.0
                    
                    # Check if rounded solution is exact
                    # If not, we might keep the float for next step or keep rounded?
                    # Keeping rounded ensures integer/half-integer constraint.
                    # If we keep float, we might drift.
                    # Let's keep rounded to enforce the domain.
                    W[k * n + i, :] = x_rounded
            
            # 2. Solve for V given U, W
            for j in range(m):
                for k in range(p):
                    # Variables: V[j*p+k, c]
                    # Equations: sum_c U[i*m+j, c] * W[k*n+i, c] * V[j*p+k, c] = 1 for all i
                    B = np.zeros((n, rank))
                    for i in range(n):
                        for c in range(rank):
                            B[i, c] = U[i * m + j, c] * W[k * n + i, c]
                    
                    try:
                        y, residuals, rank_val, sv = np.linalg.lstsq(B, np.ones(n), rcond=None)
                    except np.linalg.LinAlgError:
                        y = np.zeros(rank)
                    
                    y_rounded = np.round(y * 2) / 2.0
                    V[j * p + k, :] = y_rounded
            
            # 3. Solve for U given V, W
            for i in range(n):
                for j in range(m):
                    # Variables: U[i*m+j, c]
                    # Equations: sum_c V[j*p+k, c] * W[k*n+i, c] * U[i*m+j, c] = 1 for all k
                    C = np.zeros((p, rank))
                    for k in range(p):
                        for c in range(rank):
                            C[k, c] = V[j * p + k, c] * W[k * n + i, c]
                    
                    try:
                        z, residuals, rank_val, sv = np.linalg.lstsq(C, np.ones(p), rcond=None)
                    except np.linalg.LinAlgError:
                        z = np.zeros(rank)
                    
                    z_rounded = np.round(z * 2) / 2.0
                    U[i * m + j, :] = z_rounded
            
            # Verify
            # Check if sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for c in range(rank):
                            val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                        if abs(val - 1.0) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                converged = True
                break
        
        if converged:
            # Convert to lists
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}

    # Fallback: Return schoolbook if no better found (should not happen if 101 is reachable)
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
