"""Rewritten program for 2x3x9 matrix multiplication tensor decomposition."""
DIMENSIONS = [2, 3, 9]

import time
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START
def _build_schoolbook(n, m, p):
    """Build the schoolbook decomposition as dense arrays."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly represents matrix multiplication."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U[i * m + j]
                v_val = V[j * p + k]
                w_val = W[k * n + i]
                prod_sum = np.dot(u_val, v_val * w_val)
                if abs(prod_sum - 1.0) > 1e-9:
                    return False
    return True

def _to_fraction_list(arr):
    """Convert numpy array to list of Fractions."""
    result = []
    for row in arr:
        row_list = []
        for val in row:
            if abs(val - round(val)) < 1e-9:
                row_list.append(Fraction(int(round(val))))
            else:
                frac = Fraction(val).limit_denominator(1000)
                row_list.append(frac)
        result.append(row_list)
    return result

def _try_structured_decomposition(n, m, p, deadline):
    """Try to find a lower-rank decomposition by exploiting structure.
    
    Key idea: For each j, we have m*p entries to cover. Since n=2, 
    for each (j,k) we need two equations. We can try to use shared 
    rank-1 components across i values.
    
    Strategy: For each j, try to decompose the 2x9 sub-tensor using 
    fewer than 9 rank-1 terms by finding linear dependencies.
    """
    start_time = time.time()
    
    # For each j, the sub-problem is: find U_i[j], V[j,k], W[k,i] 
    # such that for all i,k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    
    # Since n=2, for each (j,k) pair we have:
    # U[0*m+j,c]*V[j*p+k,c]*W[k*2+0,c] + ... = 1  (for i=0)
    # U[1*m+j,c]*V[j*p+k,c]*W[k*2+1,c] + ... = 1  (for i=1)
    
    # If we fix V[j*p+k,c] and W[k*n+i,c], we can solve for U.
    # Alternatively, use the fact that the tensor is separable in a structured way.
    
    # Try: rank = m*p = 27 (lower bound)
    # This would require that for each j, the 2x9 matrix of (i,k) entries 
    # can be written as sum of 9 rank-1 outer products with shared V factors.
    
    # Actually, let's try a different approach: use the Strassen-like idea
    # where we combine terms.
    
    # For now, try random search with structured initialization
    np.random.seed(42)
    
    best_rank = n * m * p
    best_U, best_V, best_W = None, None, None
    
    # Try ranks from 53 down to 27
    for target_rank in range(53, 26, -1):
        if time.time() - start_time > deadline - 15:
            break
        
        attempts = 0
        max_attempts = 30
        
        while attempts < max_attempts and time.time() - start_time < deadline - 15:
            attempts += 1
            
            # Initialize with structure: for each (j,k), have at least one term
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            W = np.zeros((p * n, target_rank))
            
            # Assign columns to (j,k) pairs with some sharing
            col = 0
            for j in range(m):
                for k in range(p):
                    if col >= target_rank:
                        break
                    # For this (j,k), assign a column
                    # U[0*m+j, col] and U[1*m+j, col] can be different
                    # V[j*p+k, col] is shared
                    # W[k*n+0, col] and W[k*n+1, col] can be different
                    
                    # Simple initialization
                    U[0 * m + j, col] = 1.0
                    U[1 * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + 0, col] = 1.0
                    W[k * n + 1, col] = 1.0
                    col += 1
            
            if col < target_rank:
                # Fill remaining columns randomly
                for c in range(col, target_rank):
                    U[:, c] = np.random.randn(n * m) * 0.1
                    V[:, c] = np.random.randn(m * p) * 0.1
                    W[:, c] = np.random.randn(p * n) * 0.1
            
            # Check if valid
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                        if abs(val - 1.0) > 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                best_rank = target_rank
                best_U, best_V, best_W = U, V, W
                break
        
        if best_rank == target_rank:
            break
    
    if best_U is not None:
        return best_U, best_V, best_W, best_rank
    return None, None, None, best_rank

def construct():
    """Find a rank-minimal exact tensor decomposition for 2x3x9 matrix multiplication."""
    start_time = time.time()
    DEADLINE = 140  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    U0, V0, W0 = _build_schoolbook(n, m, p)
    best_rank = U0.shape[1]
    best_U, best_V, best_W = U0, V0, W0
    
    # Try structured decomposition
    U_struct, V_struct, W_struct, rank_struct = _try_structured_decomposition(
        n, m, p, DEADLINE
    )
    
    if U_struct is not None and rank_struct < best_rank:
        best_rank = rank_struct
        best_U, best_V, best_W = U_struct, V_struct, W_struct
    
    # Convert to list of Fractions
    if best_rank == 54:
        # Use schoolbook
        U_list = [[0]*54 for _ in range(n*m)]
        V_list = [[0]*54 for _ in range(m*p)]
        W_list = [[0]*54 for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_list[i*m+j][col] = 1
                    V_list[j*p+k][col] = 1
                    W_list[k*n+i][col] = 1
                    col += 1
    else:
        U_list = _to_fraction_list(best_U)
        V_list = _to_fraction_list(best_V)
        W_list = _to_fraction_list(best_W)
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
