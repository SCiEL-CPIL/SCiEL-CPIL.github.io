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
    
    def try_decomposition(U, V, n, m, p, r):
        """Given U and V, solve for W and check validity."""
        W_solution = np.zeros((p*n, r))
        valid = True
        
        for i in range(n):
            for k in range(p):
                A_ik = np.zeros((m*m, r))
                b_ik = np.zeros(m*m)
                idx = 0
                for j1 in range(m):
                    for j2 in range(m):
                        A_ik[idx, :] = U[i*m+j1, :] * V[j2*p+k, :]
                        b_ik[idx] = 1.0 if j1 == j2 else 0.0
                        idx += 1
                
                try:
                    W_ik, residuals, rank_ik, sv = np.linalg.lstsq(A_ik, b_ik, rcond=None)
                    W_solution[k*n+i, :] = W_ik
                except:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            return None
        
        if not is_half_integer(W_solution):
            return None
        
        if check_decomposition(U, V, W_solution, n, m, p):
            return W_solution
        return None
    
    # Strategy 1: Try structured decompositions
    # For matrix multiplication, we can try decompositions where U and V have
    # specific patterns that exploit the structure of the problem
    
    # Try rank = n*m*p - 1, n*m*p - 2, etc. with structured U, V
    # Key insight: use U and V with entries in {0, 1} or {-1, 0, 1}
    # and try to find W that works
    
    # First, try a more systematic approach:
    # For each rank r from best_rank-1 down, try multiple random U, V
    # but with a better distribution
    
    min_rank_to_try = max(1, best_rank - 30)
    
    for r in range(best_rank - 1, min_rank_to_try - 1, -1):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        num_attempts = 50 if r > best_rank - 10 else 20
        
        for attempt in range(num_attempts):
            if time.time() > deadline:
                break
            
            # Use a mix of sparse and dense random matrices
            if attempt % 3 == 0:
                # Sparse: mostly zeros with some ones
                U = np.zeros((n*m, r))
                V = np.zeros((m*p, r))
                for idx in range(n*m):
                    U[idx, np.random.randint(0, r)] = 1.0
                for idx in range(m*p):
                    V[idx, np.random.randint(0, r)] = 1.0
            elif attempt % 3 == 1:
                # Dense with {-1, 0, 1}
                U = np.random.choice([-1, 0, 1], size=(n*m, r))
                V = np.random.choice([-1, 0, 1], size=(m*p, r))
            else:
                # Dense with {-1, -0.5, 0, 0.5, 1}
                U = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(n*m, r))
                V = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(m*p, r))
            
            W_sol = try_decomposition(U, V, n, m, p, r)
            if W_sol is not None:
                best_U = U.tolist()
                best_V = V.tolist()
                best_W = W_sol.tolist()
                best_rank = r
                break
    
    # Strategy 2: Try to find a decomposition using the structure of matrix multiplication
    # The tensor for multiplying (n x m) by (m x p) has rank at most n*m*p
    # But we can do better by exploiting the algebraic structure
    
    # Try a specific construction: use U and V that are block-diagonal or have
    # Kronecker product structure
    
    # For now, if we haven't improved, return the schoolbook result
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
