"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that the tensor decomposition is exact."""
    rank = len(U[0]) if U else 0
    # Compute the tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # For matrix multiplication, T[i,j,k] should be 1 if the (i,k) entry of (A*B)
    # depends on A[i,j]*B[j,k], which is always 1 for standard matmul.
    # Actually, the multiplication tensor has T[i,j,k] = 1 for all i,j,k.
    # Let's verify: (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor is T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    # U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U_reshaped = U_arr.reshape(n, m, rank)
    V_reshaped = V_arr.reshape(m, p, rank)
    W_reshaped = W_arr.reshape(p, n, rank)
    
    # T[i,j,k] = sum_c U_reshaped[i,j,c] * V_reshaped[j,k,c] * W_reshaped[k,i,c]
    # Note: W index is k*n+i, so W_reshaped[k, i, c]
    
    # Use einsum for efficiency
    # U_reshaped: (n, m, r), V_reshaped: (m, p, r), W_reshaped: (p, n, r)
    # We want T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    # This is a bit tricky with einsum because of the index permutations
    # Let's do it in a loop for clarity, or use appropriate einsum
    
    # Actually, let's compute it directly
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i,j,k] = np.dot(U_reshaped[i,j,:], V_reshaped[j,k,:]) * 0  # placeholder
                # Actually: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
                # This is not a simple dot product of two vectors
                # It's sum_c (U[i,j,c] * V[j,k,c]) * W[k,i,c]
                # = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
                
    # Better approach using einsum:
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    # Let's use: tmp[i,j,k,r] = U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Then T[i,j,k] = sum_r tmp[i,j,k,r]
    
    # Using einsum:
    # U: (n, m, r), V: (m, p, r), W: (p, n, r)
    # We want to sum over r: U[n,m,r] * V[m,p,r] * W[p,n,r] -> T[n,m,p]
    
    # einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
    # But W_reshaped is (p, n, r), so W[k,i,r] corresponds to W_reshaped[k, i, r]
    # So the index for W is (k, i, r) where k is first, i is second
    
    T_computed = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped, optimize=True)
    
    # Check if T_computed is all 1s
    if not np.allclose(T_computed, np.ones((n, m, p)), atol=tol):
        return False
    return True


def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s to be safe within 150s limit
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 54
    
    # Start with the schoolbook solution
    rank = rank_schoolbook
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
    
    best_solution = {"U": U, "V": W, "W": W}  # Wait, typo
    best_solution = {"U": U, "V": V, "W": W}
    best_rank = rank
    
    # Try to find a lower rank solution using random initialization and gradient-free optimization
    # or use known results for small matrix multiplication
    
    # For 3x3 * 3x6, the rank is at least 9 (since it's 3x3*3x6, we can think of it as 
    # 6 copies of 3x3*3x1, but that's not quite right)
    # Actually, the rank of matmul for n x m * m x p is at most n*m*p and at least max(n*m, m*p, p*n)
    # For 3x3 * 3x6: n*m=9, m*p=18, p*n=18, so lower bound is 18
    # Upper bound is 54 (schoolbook)
    
    # Let's try to find a solution with rank < 54
    # We can use the fact that matrix multiplication is a bilinear map
    
    # Try random projections to reduce rank
    # This is a heuristic approach
    
    # For now, let's try a few random rank reductions and see if we can verify
    # We'll use a simple approach: try to find a rank r solution by solving a system of equations
    
    # Actually, let's just return the schoolbook solution for now, as finding an optimal
    # decomposition is computationally hard
    
    # But let's try one heuristic: if we can express the 3x3*3x6 multiplication as
    # a combination of fewer rank-1 terms
    
    # The multiplication tensor T[i,j,k] = 1 for all i,j,k
    # We want to write T as a sum of rank-1 tensors
    # A rank-1 tensor is U[i,j] * V[j,k] * W[k,i]
    
    # Let's try to use the fact that for 3x3*3x6, we might be able to use rank 27 or so
    # by exploiting structure
    
    # For now, return the schoolbook solution
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
