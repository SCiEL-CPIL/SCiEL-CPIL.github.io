"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    # The tensor for matrix multiplication A (n x m) * B (m x p) -> C (n x p)
    # T_{i,j,k} = 1 if the term A[i,j]*B[j,k] contributes to C[i,k]
    # In our convention: C[k*n+i] = sum_{j} A[i*m+j] * B[j*p+k]
    # So T[i, j, k] = 1 for all i, j, k (it's the standard matmul tensor)
    # Shape: (n, m, p)
    
    # We want to find a rank-r decomposition:
    # T[i,j,k] = sum_{c=0}^{r-1} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Strategy: Use the known rank bounds and try to find a low-rank decomposition.
    # For 3x5x5, the trivial rank is 3*5*5 = 75.
    # We can try to find a better decomposition using known techniques.
    
    # First, let's try a simple approach: use the fact that we can decompose
    # the multiplication as a sum of rank-1 tensors.
    
    # The matmul tensor T has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We need: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Let's try to find a decomposition with rank r < n*m*p.
    
    # Approach: Use the "laser method" or just try small ranks and solve.
    # For now, let's try to find a decomposition with rank = n*m + m*p + n*p - n - p + 1 or similar.
    
    # Actually, let's just try to find the minimal rank by attempting decompositions.
    
    # Start with the trivial decomposition and try to reduce rank.
    
    # For 3x5x5, let's try rank 15, 20, 25, etc.
    
    # A simple approach: use the "sum of products" form.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # This is already rank m = 5 for each (i,k) pair, but globally it's more complex.
    
    # Let's try to construct a decomposition directly.
    
    # The tensor T[i,j,k] = 1 for all i,j,k.
    # We want T = sum_c U_c * V_c * W_c where:
    #   U_c is a vector in R^{n*m}, V_c in R^{m*p}, W_c in R^{p*n}
    #   and (U_c * V_c * W_c)[i,j,k] = U_c[i*m+j] * V_c[j*p+k] * W_c[k*n+i]
    
    # So we need: sum_c U_c[i*m+j] * V_c[j*p+k] * W_c[k*n+i] = 1 for all i,j,k
    
    # This is a system of n*m*p equations in r*(n*m + m*p + p*n) unknowns.
    
    # Let's try to find a solution with small r.
    
    # For r=1: U[i*m+j] * V[j*p+k] * W[k*n+i] = 1 for all i,j,k
    # This requires U[i*m+j] * V[j*p+k] * W[k*n+i] = 1
    # For fixed i,j: U[i*m+j] * V[j*p+k] * W[k*n+i] = 1 for all k
    # So V[j*p+k] * W[k*n+i] must be constant in k for each (j,i).
    # This is generally not possible for r=1 unless the tensor is rank 1, which it's not.
    
    # Let's just use the trivial decomposition for now and see if we can improve.
    
    # Trivial decomposition: rank = n*m*p
    rank_trivial = n * m * p
    U = [[0] * rank_trivial for _ in range(n * m)]
    V = [[0] * rank_trivial for _ in range(m * p)]
    W = [[0] * rank_trivial for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_trivial
    best_U, best_V, best_W = U, V, W
    
    # Try to find a better decomposition
    # Use a simple approach: try ranks from small to large
    
    # For each candidate rank r, we can try to solve the system.
    # The system is: for each (i,j,k), sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    # This is nonlinear. Let's use a different approach.
    
    # Use the fact that T[i,j,k] = 1 = delta_{i,i'} * delta_{j,j'} * delta_{k,k'} summed... no.
    
    # Actually, let's use a known result: the rank of the matmul tensor for (n,m,p)
    # is at most n*m*p, and for small cases we can compute it.
    
    # For 3x5x5, let's try to use a decomposition based on the following:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # We can write this as:
    # C[i,k] = sum_j A[i,j] B[j,k] * 1
    # This suggests rank m = 5 if we fix i,k, but globally we need to account for all i,k.
    
    # A better approach: use the "Kronecker" or "block" decomposition.
    
    # Let's try rank = n*m + m*p + n*p - n - m - p + 1 or similar.
    
    # Actually, for now, let's just return the trivial decomposition.
    # The score will be 75/75 = 1.0 (same as seed).
    
    # But wait, the score is seed/value where value is the rank.
    # So if we find rank 75, score = 75/75 = 1.0.
    # If we find rank 50, score = 75/50 = 1.5.
    
    # Let's try to find a lower rank.
    
    # Use a simple greedy or algebraic approach.
    
    # For 3x5x5, the tensor is T[i,j,k] = 1.
    # We can try to find a decomposition using the following trick:
    # 1 = (1) for all i,j,k
    # We can write 1 as a sum of products.
    
    # Let's try: for each j, define:
    # U_j[i*m+j] = 1 for all i, 0 otherwise
    # V_j[j*p+k] = 1 for all k, 0 otherwise  
    # W_j[k*n+i] = 1 for all k,i, 0 otherwise... no, this doesn't work.
    
    # Actually, the trivial decomposition is the best we can easily do.
    
    # Let's just return the trivial decomposition.
    return {"U": U, V: V, W: W}
# EVOLVE-BLOCK-END
