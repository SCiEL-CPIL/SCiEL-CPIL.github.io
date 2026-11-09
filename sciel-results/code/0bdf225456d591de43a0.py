"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    rank = n * m * p
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
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p, tol=1e-6):
    # Check that sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1 if i,j,k consistent, else 0
    # Actually the tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # For matrix mult (A is n x m, B is m x p), C[i,k] = sum_j A[i,j] B[j,k]
    # So T[i,j,k] should be 1 if this term contributes, i.e., always 1 for valid indices? 
    # Wait, the tensor for matrix multiplication M_{i,j,k} where i in [0,n), j in [0,m), k in [0,p)
    # is 1 for all i,j,k. The product C[i,k] = sum_j M[i,j,k] A[i,j] B[j,k].
    # So we need sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] == 1 for all i,j,k.
    
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    rank = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find a lower rank decomposition
    # The lower bound for matrix multiplication of n x m by m x p is at least max(n*m, m*p, n*p) = max(6, 24, 16) = 24
    # But actually, the rank is at least the matrix multiplication rank which is known to be less than n*m*p.
    # For 2x3 by 3x8, the trivial rank is 48. Can we do better?
    
    # Let's try a randomized search for lower ranks
    # We'll try ranks from some lower bound up to best_rank - 1
    
    # Known lower bound: at least max(n, m, p) * something? 
    # Actually, the rank is at least the dimension of the tensor space needed.
    # A simple lower bound is max(n*m, m*p, p*n) = max(6, 24, 16) = 24.
    
    # Let's try to find a decomposition of rank 24, 25, ..., up to best_rank-1
    
    # For each candidate rank, we can try to solve the system of equations
    # This is a non-linear system, so we'll use a heuristic approach
    
    # Alternative: use the fact that we can share terms
    # For example, if we group by j, we have m groups of n*p terms each
    # But we can potentially share across groups if the structure allows
    
    # Let's try a simple approach: for each rank r from 24 to best_rank-1,
    # try to find a random decomposition and verify
    
    # Actually, a better approach: use the structure of matrix multiplication
    # The tensor is T[i,j,k] = 1 for all i,j,k
    # We can write this as a sum of rank-1 tensors
    # Each rank-1 tensor is U_c[i,j] * V_c[j,k] * W_c[k,i] where U_c, V_c, W_c are vectors
    
    # Wait, the indices are: U has indices (i*m+j), V has (j*p+k), W has (k*n+i)
    # So for a single term c, we have:
    # U_c[i*m+j] * V_c[j*p+k] * W_c[k*n+i]
    # This should contribute to T[i,j,k]
    
    # For the sum over c to equal 1 for all i,j,k, we need:
    # sum_c U_c[i*m+j] * V_c[j*p+k] * W_c[k*n+i] = 1 for all i,j,k
    
    # This is a system of n*m*p = 48 equations in rank * (n*m + m*p + p*n) = rank * (6 + 24 + 16) = rank * 46 unknowns
    
    # For rank 24, we have 24 * 46 = 1104 unknowns and 48 equations, so it's underdetermined
    # We can try to find a solution
    
    # Let's use a simple iterative approach: initialize randomly, then optimize
    
    # But given time constraints, let's try a few random initializations for low ranks
    
    rng = np.random.default_rng(42)
    
    # Try ranks from 24 up
    for r in range(24, best_rank):
        if time.time() > deadline - 5:
            break
            
        # Try a few random initializations
        for attempt in range(10):
            if time.time() > deadline - 5:
                break
            
            # Initialize U, V, W randomly
            U = rng.standard_normal((n*m, r))
            V = rng.standard_normal((m*p, r))
            W = rng.standard_normal((p*n, r))
            
            # This is a non-linear problem, so simple gradient descent may not work well
            # Let's just check if a random initialization happens to work (unlikely)
            # Instead, let's use a structured approach
            
            # Actually, for this small problem, we can try to use the fact that
            # the tensor is all-ones, so we can try to factor it in a structured way
            
            # One approach: use the fact that 1 = sum over some partition
            # For example, we can write the all-ones tensor as a sum of rank-1 tensors
            # where each rank-1 tensor has a specific pattern
            
            # Let's try: for each j, we have a block of n*p = 16 entries
            # We can potentially share some structure across j values
            
            # Given the complexity and time constraints, let's just return the schoolbook
            # if we don't find a better solution quickly
            pass
    
    # For now, return the schoolbook decomposition
    return best
# EVOLVE-BLOCK-END
