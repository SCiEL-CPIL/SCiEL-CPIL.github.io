"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # Safety margin
    
    # Schoolbook rank
    schoolbook_rank = n * m * p
    
    # Build schoolbook decomposition
    U = [[0] * schoolbook_rank for _ in range(n * m)]
    V = [[0] * schoolbook_rank for _ in range(m * p)]
    W = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    # Convert to numpy for faster computation
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Check if a given decomposition is valid
    def is_valid(U_mat, V_mat, W_mat):
        # T[i,j,k] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # And T[i,j,k] = 1 for all i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for r in range(U_mat.shape[1]):
                        val += U_mat[i*m+j, r] * V_mat[j*p+k, r] * W_mat[k*n+i, r]
                    if abs(val - 1) > 1e-9:
                        return False
        return True
    
    # Try to remove terms one by one
    # This is a simple heuristic
    current_rank = schoolbook_rank
    for term_to_remove in range(schoolbook_rank):
        if time.time() > deadline - 5:
            break
        
        if term_to_remove >= current_rank:
            continue
        
        # Try removing this term
        U_test = U_np[:, :current_rank].copy()
        V_test = V_np[:, :current_rank].copy()
        W_test = W_np[:, :current_rank].copy()
        
        # Remove the term
        U_test = np.delete(U_test, term_to_remove, axis=1)
        V_test = np.delete(V_test, term_to_remove, axis=1)
        W_test = np.delete(W_test, term_to_remove, axis=1)
        
        if is_valid(U_test, V_test, W_test):
            # Successfully removed a term
            U_np = U_test
            V_np = V_test
            W_np = W_test
            current_rank -= 1
    
    # Convert back to lists
    U_result = U_np.tolist()
    V_result = V_np.tolist()
    W_result = W_np.tolist()
    
    return {"U": U_result, "V": V_result, "W": W_result}
# EVOLVE-BLOCK-END
