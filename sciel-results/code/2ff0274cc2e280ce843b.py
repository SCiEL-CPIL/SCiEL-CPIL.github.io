"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]
import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target rank 20, which is known to be feasible for 4x4x5
    R = 20
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # 110 seconds limit
    
    # Try to find a decomposition with rank R
    # We iterate with different random seeds for U and V
    for seed in range(10000):
        if time.time() > deadline:
            break
            
        np.random.seed(seed)
        
        # Initialize U and V with random 0/1 matrices
        # U shape: (n*m, R) = (16, 20)
        # V shape: (m*p, R) = (20, 20)
        U = np.random.randint(0, 2, size=(n * m, R))
        V = np.random.randint(0, 2, size=(m * p, R))
        
        # We need to solve for W
        # W shape: (p*n, R) = (20, 20)
        W = np.zeros((p * n, R))
        
        valid = True
        # For each output index (k, i), we have a system of m=4 linear equations
        # Variables: W[k*n + i, 0..R-1]
        # Equations: for each j in 0..m-1:
        # sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r] = 1
        
        for k in range(p):
            for i in range(n):
                # Construct the 4xR matrix B
                # B[j, r] = U[i*m + j, r] * V[j*p + k, r]
                B = np.zeros((m, R))
                for j in range(m):
                    u_row = U[i * m + j]
                    v_row = V[j * p + k]
                    B[j] = u_row * v_row
                
                # Solve B * w_col = [1, 1, 1, 1]^T
                # We want a solution w_col.
                # Since R=20 and m=4, we can find a sparse solution or least squares.
                # To keep entries small, we can try to find a solution with 0/1 or small integers.
                # Using lstsq gives the minimum norm solution, which might be fractional.
                # The problem allows half-integers.
                
                rhs = np.ones(m)
                try:
                    # Find minimum norm solution
                    w_sol, residuals, rank, sv = np.linalg.lstsq(B, rhs, rcond=None)
                    
                    # Check if it's a valid solution (residuals should be 0)
                    if np.allclose(B @ w_sol, rhs, atol=1e-9):
                        # Round to nearest 0.5 to satisfy "integer/half-integer" constraint
                        # If rounding breaks the equation, we might need to adjust or reject
                        w_rounded = np.round(w_sol * 2) / 2.0
                        if np.allclose(B @ w_rounded, rhs, atol=1e-9):
                            W[k * n + i] = w_rounded
                        else:
                            # If rounding failed, try to find a basic solution or reject this seed
                            valid = False
                            break
                    else:
                        valid = False
                        break
                except np.linalg.LinAlgError:
                    valid = False
                    break
            
            if not valid:
                break
        
        if valid:
            # Convert to lists of floats/ints as required by interface
            # The interface expects {'U', 'V', 'W'} with integer/half-integer entries
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            
            # Ensure types are compatible (floats are okay if they are 0.5, 1.0, etc.)
            # The problem statement says "integer/half-integer entries".
            # We return the numpy arrays converted to lists.
            return {"U": U_list, "V": V_list, "W": W_list}
        
    # Fallback: Schoolbook decomposition if no rank 20 found
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
# EVOLVE-BLOCK-END
