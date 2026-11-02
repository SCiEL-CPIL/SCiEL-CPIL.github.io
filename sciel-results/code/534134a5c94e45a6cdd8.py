"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product
from fractions import Fraction

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 60
    
    # For each rank from lower bound to schoolbook, try to find decomposition
    # Lower bound for matrix multiplication tensor rank is at least max(n*m, m*p, p*n)
    # For 3x4x5: max(12, 20, 15) = 20
    
    def try_decomposition(n, m, p, r, max_attempts=5000):
        """Try to find a rank-r decomposition by fixing U,V and solving for W."""
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            
            # Random integer initialization for U and V
            # Use small integers to keep numbers manageable
            scale = 2  # Try entries in {-1, 0, 1} or {-2,-1,0,1,2}
            U = np.random.randint(-scale, scale + 1, size=(n * m, r))
            V = np.random.randint(-scale, scale + 1, size=(m * p, r))
            
            # For each (i, k) pair, solve for W[k*n+i, :]
            # Equation: for all j, sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
            # This is: A_ik @ w = 1 where A_ik[j, c] = U[i*m+j, c] * V[j*p+k, c]
            
            W = np.zeros((p * n, r))
            valid = True
            
            for i in range(n):
                for k in range(p):
                    # Build A_ik: m x r matrix
                    A_ik = np.zeros((m, r))
                    for j in range(m):
                        A_ik[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    
                    # Solve A_ik @ w = ones(m)
                    try:
                        # Check if system is consistent
                        # Use least squares and check residual
                        w, residuals, rank_A, s = np.linalg.lstsq(A_ik, np.ones(m), rcond=1e-10)
                        
                        if residuals.size > 0:
                            if residuals[0] > 1e-8:
                                valid = False
                                break
                        
                        # Verify the solution exactly
                        if np.max(np.abs(A_ik @ w - np.ones(m))) > 1e-8:
                            valid = False
                            break
                        
                        W[k * n + i, :] = w
                    except:
                        valid = False
                        break
                
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Full verification
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                        if abs(val - 1.0) > 1e-8:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                # Convert to integer/half-integer entries
                # Find common denominator
                all_vals = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
                
                # Try to find a scaling factor to make everything integer or half-integer
                # First, check if they're already close to integers
                def to_half_integer(x):
                    # Round to nearest half-integer
                    return round(x * 2) / 2.0
                
                U_clean = np.array([[to_half_integer(x) for x in row] for row in U])
                V_clean = np.array([[to_half_integer(x) for x in row] for row in V])
                W_clean = np.array([[to_half_integer(x) for x in row] for row in W])
                
                # Verify cleaned version
                valid_clean = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U_clean[i * m + j, :] * V_clean[j * p + k, :] * W_clean[k * n + i, :])
                            if abs(val - 1.0) > 1e-6:
                                valid_clean = False
                                break
                        if not valid_clean:
                            break
                    if not valid_clean:
                        break
                
                if valid_clean:
                    # Return as lists
                    U_list = U_clean.tolist()
                    V_list = V_clean.tolist()
                    W_list = W_clean.tolist()
                    return U_list, V_list, W_list, r
        
        return None
    
    # Try ranks from 20 upward
    for r in range(20, schoolbook_rank):
        if time.time() > deadline:
            break
        
        result = try_decomposition(n, m, p, r)
        if result is not None:
            U, V, W, rank = result
            return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook decomposition
    rank = schoolbook_rank
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
# EVOLVE-BLOCK-END
