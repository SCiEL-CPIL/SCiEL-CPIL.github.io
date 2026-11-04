"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Schoolbook decomposition
    rank_schoolbook = n * m * p
    U_schoolbook = np.zeros((n * m, rank_schoolbook))
    V_schoolbook = np.zeros((m * p, rank_schoolbook))
    W_schoolbook = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_schoolbook[i * m + j, col] = 1
                V_schoolbook[j * p + k, col] = 1
                W_schoolbook[k * n + i, col] = 1
                col += 1
    
    best_U = U_schoolbook.tolist()
    best_V = V_schoolbook.tolist()
    best_W = W_schoolbook.tolist()
    best_rank = rank_schoolbook
    
    # Try to find lower rank decompositions
    # Strategy: for each candidate rank r < best_rank, try random U, V and solve for W
    # Check if W has integer/half-integer entries
    
    def is_half_integer(x, tol=1e-9):
        return np.allclose(2*x, np.round(2*x), atol=tol)
    
    def check_decomposition(U, V, W, n, m, p):
        """Check if the decomposition is valid."""
        for i in range(n):
            for k in range(p):
                for j1 in range(m):
                    for j2 in range(m):
                        val = np.sum(W[k*n+i, :] * U[i*m+j1, :] * V[j2*p+k, :])
                        expected = 1.0 if j1 == j2 else 0.0
                        if not np.isclose(val, expected, atol=1e-9):
                            return False
        return True
    
    # Try ranks from best_rank-1 down to some minimum
    # This is expensive, so we'll limit the search
    min_rank_to_try = max(1, best_rank - 20)  # Try at most 20 ranks below schoolbook
    
    for r in range(best_rank - 1, min_rank_to_try - 1, -1):
        if time.time() > deadline:
            break
        
        # Try a few random initializations for this rank
        for attempt in range(10):
            if time.time() > deadline:
                break
            
            # Random U, V with entries in {-1, -0.5, 0, 0.5, 1}
            U = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(n*m, r))
            V = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(m*p, r))
            
            # Build the linear system for W
            # For each (i,k), we have m^2 equations:
            # sum_{r'} W[k*n+i, r'] * U[i*m+j1, r'] * V[j2*p+k, r'] = delta_{j1,j2}
            # This is a linear system in W[k*n+i, :]
            
            # For each (i,k), we have a system:
            # A_{ik} * W_{ik} = b_{ik}
            # where A_{ik} is m^2 x r, and b_{ik} is m^2 x 1
            
            W_solution = np.zeros((p*n, r))
            valid = True
            
            for i in range(n):
                for k in range(p):
                    # Build A_{ik}: rows indexed by (j1,j2), columns by r'
                    A_ik = np.zeros((m*m, r))
                    b_ik = np.zeros(m*m)
                    idx = 0
                    for j1 in range(m):
                        for j2 in range(m):
                            A_ik[idx, :] = U[i*m+j1, :] * V[j2*p+k, :]
                            b_ik[idx] = 1.0 if j1 == j2 else 0.0
                            idx += 1
                    
                    # Solve A_ik * W_ik = b_ik
                    # Use least squares
                    try:
                        W_ik, residuals, rank_ik, sv = np.linalg.lstsq(A_ik, b_ik, rcond=None)
                        W_solution[k*n+i, :] = W_ik
                    except:
                        valid = False
                        break
                
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Check if W_solution has integer/half-integer entries
            if not is_half_integer(W_solution):
                continue
            
            # Verify the decomposition
            if check_decomposition(U, V, W_solution, n, m, p):
                best_U = U.tolist()
                best_V = V.tolist()
                best_W = W_solution.tolist()
                best_rank = r
                break  # Found a valid decomposition for this rank, move to lower rank
        
        if time.time() > deadline:
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
