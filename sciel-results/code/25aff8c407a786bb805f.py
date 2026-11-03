"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import numpy as np
from itertools import product
import time

def _verify(U, V, W, n, m, p, tol=1e-12):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # The matrix multiplication tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We need T[i,j,k] = 1 if i,j,k are valid indices (always true for the multiplication tensor)
    # Actually, the matrix multiplication tensor has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # because (AB)[i,k] = sum_j A[i,j]*B[j,k]
    
    # Let's verify: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] should be 1
    # Wait, that's not right. The tensor for matrix multiplication A (n x m) * B (m x p) -> C (n x p)
    # is T[i,j,k] where C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]
    # So T[i,j,k] = 1 for all valid i,j,k.
    
    # Let me re-check: The multiplication tensor has entries T[i,j,k] = 1 for all i,j,k
    # and the decomposition is: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _is_valid_entries(U, V, W, tol=1e-9):
    """Check that all entries are integers or half-integers."""
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                # Check if val is close to an integer or half-integer
                v2 = 2 * val
                if abs(v2 - round(v2)) > tol:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Try ranks from lower bound up to schoolbook
    lower_bound = max(n*m, m*p, p*n)
    schoolbook = n * m * p
    
    # Known: for 3x3 * 3x3, rank is 23 (Landsberg). For 3x3 * 3x5, let's search.
    
    best_result = None
    
    # First, return schoolbook as fallback
    rank_school = schoolbook
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    
    # Try to find lower rank decompositions
    # Strategy: random search with structured patterns
    # Use the fact that entries should be integer or half-integer
    
    np.random.seed(42)
    
    # Try various ranks
    for target_rank in range(lower_bound, schoolbook + 1):
        if time.time() > deadline - 10:
            break
        
        # Try multiple random initializations
        max_attempts = 200
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                    break
            
            # Generate random matrices with integer/half-integer entries
            # Use small values for better convergence
            scale = 2
            U_rand = np.round(np.random.randn(n*m, target_rank) * scale) * 0.5
            V_rand = np.round(np.random.randn(m*p, target_rank) * scale) * 0.5
            W_rand = np.round(np.random.randn(p*n, target_rank) * scale) * 0.5
            
            # Convert to lists
            U_list = [[U_rand[i,j] for j in range(target_rank)] for i in range(n*m)]
            V_list = [[V_rand[i,j] for j in range(target_rank)] for i in range(m*p)]
            W_list = [[W_rand[i,j] for j in range(target_rank)] for i in range(p*n)]
            
            if _verify(U_list, V_list, W_list, n, m, p):
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                return best_result
        
        # If we found a valid decomposition at this rank, return it
        # (already returned above if found)
    
    return best_result
# EVOLVE-BLOCK-END
