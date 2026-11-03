"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank is n*m*p = 72
    schoolbook_rank = n * m * p
    
    def make_schoolbook():
        rank = schoolbook_rank
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
        return {"U": U, "V": V, "W": W}
    
    def verify(U, V, W, tol=1e-9):
        """Verify that the decomposition exactly computes matrix multiplication."""
        rank = len(U[0]) if U else 0
        # Build the multiplication tensor check
        # For each (i, j, k), sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should be 1
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U_arr[i * m + j]
                    v_val = V_arr[j * p + k]
                    w_val = W_arr[k * n + i]
                    total = np.dot(u_val, v_val) * w_val
                    # Actually the tensor is T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                    # But that's not right. The correct check is:
                    # For matrix multiplication C = A @ B where A is n x m, B is m x p
                    # C[i,k] = sum_j A[i,j] * B[j,k]
                    # The tensor T[i,j,k] = 1 for the multiplication tensor
                    # Decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    # Wait, that's a 3-way tensor decomposition. Let me re-read the task.
                    # The task says "exactly decompose the matrix multiplication tensor"
                    # The multiplication tensor for (n,m,p) has entries T[i,j,k] = 1
                    # and the decomposition is T = sum_c U_c ⊗ V_c ⊗ W_c
                    # where U_c[i*m+j], V_c[j*p+k], W_c[k*n+i]
                    # So T[i,j,k] = sum_c U_c[i*m+j] * V_c[j*p+k] * W_c[k*n+i]
                    # This should equal 1 for all (i,j,k)
                    pass
        
        # Proper verification
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U_arr[i * m + j]
                    v_val = V_arr[j * p + k]
                    w_val = W_arr[k * n + i]
                    total = 0.0
                    for c in range(rank):
                        total += u_val[c] * v_val[c] * w_val[c]
                    if abs(total - 1.0) > tol:
                        return False
        return True
    
    def try_strassen_2x2x2_block():
        """Try to use Strassen-like 2x2x2 blocks for the 3x3x8 case."""
        # For 3x3x8, we can potentially use block structure
        # Let's try a simple approach: see if we can reduce rank by exploiting structure
        
        # Actually, let's try to find a lower rank decomposition by searching
        # over small perturbations or structured decompositions
        
        # The minimal rank for 3x3x8 matrix multiplication is unknown but 
        # schoolbook gives 72. Let's try to find something better.
        
        # One approach: use the fact that some entries might be shared
        # For now, return the schoolbook decomposition
        return make_schoolbook()
    
    # Try to find a better decomposition
    # For 3x3x8, the rank is at most 72 (schoolbook)
    # Known lower bound is around 50-60, but exact value is unknown
    
    # Let's try a simple search for rank reduction
    # We'll try to see if we can eliminate some terms
    
    # For now, just return the schoolbook decomposition
    # This is safe and correct
    return make_schoolbook()
# EVOLVE-BLOCK-END
