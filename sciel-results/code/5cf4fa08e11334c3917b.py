"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 * 3x8 matrix multiplication.
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 145  # 5s safety margin
    
    # Start with schoolbook decomposition
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    best_rank = rank
    best_U, best_V, best_W = U, V, W
    
    # Try to find lower rank decompositions using random search
    # For small matrices, we can try various rank values
    
    # Known lower bounds: For 3x3*3x8, the rank is at least max(n*m, m*p, p*n) = max(9, 24, 24) = 24
    # But we can potentially do better with clever decompositions
    
    # Try ranks from a reasonable lower bound up to schoolbook
    # Lower bound for matrix multiplication rank is max(n*m, m*p, p*n)
    lower_bound = max(n * m, m * p, p * n)
    
    # For 3x3 * 3x8: lower_bound = max(9, 24, 24) = 24
    # Schoolbook is 72
    
    # Try to find decompositions with rank < schoolbook
    # Use random initialization and check validity
    
    def validate_decomposition(U_mat, V_mat, W_mat, rank_val):
        """Check if the decomposition exactly computes matrix multiplication."""
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        # The tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # For matrix multiplication, T[i,j,k] should be 1 if this corresponds to a valid product term
        
        # Actually, the matrix multiplication tensor is defined as:
        # M[A,B]_i,k = sum_j A[i,j] * B[j,k]
        # In tensor form: T[i,j,k] = 1 for all i,j,k (it's the identity-like tensor)
        # The decomposition gives: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
        
        U_arr = np.array(U_mat, dtype=float)
        V_arr = np.array(V_mat, dtype=float)
        W_arr = np.array(W_mat, dtype=float)
        
        # Check all (i,j,k) triples
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank_val):
                        val += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                    if abs(val - 1.0) > 1e-10:
                        return False
        return True
    
    # Try random search for lower rank
    # For each candidate rank, try random initializations
    
    for target_rank in range(lower_bound, best_rank):
        if time.time() > deadline:
            break
        
        # Try several random initializations for this rank
        max_attempts = 50
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Random integer/half-integer entries
            # Use small values to keep it manageable
            scale = 2
            U_rand = [[random.choice([0, 1, -1, 0.5, -0.5, scale, -scale]) for _ in range(target_rank)] 
                      for _ in range(n * m)]
            V_rand = [[random.choice([0, 1, -1, 0.5, -0.5, scale, -scale]) for _ in range(target_rank)] 
                      for _ in range(m * p)]
            W_rand = [[random.choice([0, 1, -1, 0.5, -0.5, scale, -scale]) for _ in range(target_rank)] 
                      for _ in range(p * n)]
            
            if validate_decomposition(U_rand, V_rand, W_rand, target_rank):
                best_rank = target_rank
                best_U, best_V, best_W = U_rand, V_rand, W_rand
                break  # Found a valid decomposition at this rank, try lower
        
        if time.time() > deadline:
            break
    
    # Also try structured approaches
    # For 3x3 * 3x8, we can use the fact that 3x3 multiplication has rank 23 (Strassen-like)
    # But 3x8 is different
    
    # Try a more systematic approach: use bilinear decomposition
    # The matrix multiplication can be viewed as a bilinear map
    # A (3x3) * B (3x8) = C (3x8)
    
    # Each output C[i,k] = sum_j A[i,j] * B[j,k]
    # This is a bilinear form in A and B
    
    # We can try to find a decomposition using the structure of the problem
    
    # For now, return the best found
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
