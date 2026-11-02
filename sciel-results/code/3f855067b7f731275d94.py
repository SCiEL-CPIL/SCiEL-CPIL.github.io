"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3 by 4 matrix
    by a 4 by 5 matrix. The output factor follows the official trace convention W[k*n+i,r].
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 60
    
    # Build schoolbook decomposition as baseline
    U = np.zeros((n * m, rank_schoolbook), dtype=float)
    V = np.zeros((m * p, rank_schoolbook), dtype=float)
    W = np.zeros((p * n, rank_schoolbook), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank_schoolbook
    
    # Target tensor: T[i,j,k] = 1 for all i,j,k
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Try to find lower-rank decompositions by optimization
    # Strategy: minimize sum of |entries| subject to tensor equation
    # Then round to half-integers and verify
    
    def verify_decomposition(U_mat, V_mat, W_mat, tol=1e-10):
        """Verify that the decomposition is exact."""
        n, m, p = DIMENSIONS
        rank = U_mat.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U_mat[i * m + j, r] * V_mat[j * p + k, r] * W_mat[k * n + i, r]
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def round_to_half_int(matrix):
        """Round matrix entries to nearest half-integer."""
        return np.round(matrix * 2.0) / 2.0
    
    # Try random initializations for lower ranks
    np.random.seed(42)
    
    for target_rank in range(50, rank_schoolbook):
        if time.time() > deadline - 10:
            break
        
        # Try multiple random initializations
        for attempt in range(5):
            if time.time() > deadline - 10:
                break
            
            # Initialize with random values, then optimize
            U_try = np.random.randn(n * m, target_rank) * 0.5
            V_try = np.random.randn(m * p, target_rank) * 0.5
            W_try = np.random.randn(p * n, target_rank) * 0.5
            
            # Simple gradient-free optimization: try to reduce residual
            # by adjusting entries
            best_residual = np.inf
            best_U_opt, best_V_opt, best_W_opt = None, None, None
            
            for step in range(200):
                if time.time() > deadline - 10:
                    break
                
                # Compute residual
                residual = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for r in range(target_rank):
                                val += U_try[i * m + j, r] * V_try[j * p + k, r] * W_try[k * n + i, r]
                            residual += (val - 1.0) ** 2
                
                if residual < best_residual:
                    best_residual = residual
                    best_U_opt = U_try.copy()
                    best_V_opt = V_try.copy()
                    best_W_opt = W_try.copy()
                
                if best_residual < 1e-15:
                    break
                
                # Perturb a random entry
                mat_choice = np.random.randint(3)
                if mat_choice == 0:
                    row = np.random.randint(n * m)
                    col_idx = np.random.randint(target_rank)
                    U_try[row, col_idx] += np.random.randn() * 0.1
                elif mat_choice == 1:
                    row = np.random.randint(m * p)
                    col_idx = np.random.randint(target_rank)
                    V_try[row, col_idx] += np.random.randn() * 0.1
                else:
                    row = np.random.randint(p * n)
                    col_idx = np.random.randint(target_rank)
                    W_try[row, col_idx] += np.random.randn() * 0.1
            
            # Try to round to half-integers and verify
            if best_U_opt is not None:
                U_rounded = round_to_half_int(best_U_opt)
                V_rounded = round_to_half_int(best_V_opt)
                W_rounded = round_to_half_int(best_W_opt)
                
                if verify_decomposition(U_rounded, V_rounded, W_rounded):
                    best_rank = target_rank
                    best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
                    break
        
        if best_rank < rank_schoolbook:
            break
    
    # If no better decomposition found, return schoolbook
    if best_rank == rank_schoolbook:
        U_out = [[float(x) for x in U[i]] for i in range(n * m)]
        V_out = [[float(x) for x in V[i]] for i in range(m * p)]
        W_out = [[float(x) for x in W[i]] for i in range(p * n)]
    else:
        U_out = [[float(x) for x in best_U[i]] for i in range(n * m)]
        V_out = [[float(x) for x in best_V[i]] for i in range(m * p)]
        W_out = [[float(x) for x in best_W[i]] for i in range(p * n)]
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
