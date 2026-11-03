"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    # Matrix multiplication tensor: T[i, j, k] = 1 if A[i,j] * B[j,k] -> C[i,k]
    # The tensor is 3-way: T[i, j, k] where i in [0,n), j in [0,m), k in [0,p)
    # T[i,j,k] = 1 for all i,j,k (since every product contributes to the sum)
    # Actually, the matrix multiplication tensor has T[i,j,k] = 1 for all i,j,k
    # because C[i,k] = sum_j A[i,j] * B[j,k]
    
    # We want to decompose T = sum_{r=1}^R U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # where U is (n*m) x R, V is (m*p) x R, W is (p*n) x R
    # and U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] sums to T[i,j,k] = 1
    
    # Wait, the convention is: W[k*n+i, r]
    # So the tensor entry T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # And we need this to equal 1 for all i,j,k
    
    # This is equivalent to finding rank-R CP decomposition of the all-ones tensor
    # of size (n*m) x (m*p) x (p*n) with the specific index mapping
    
    # Let's use a greedy rank-1 peeling approach with exact arithmetic
    # Start with R = n*m*p (schoolbook) and try to reduce
    
    # Actually, let's try a different approach: use known results
    # For 3x4*4x6, the schoolbook rank is 72
    # Strassen-like block decompositions can do better
    
    # Let's implement a simple but effective approach:
    # 1. Try to find a decomposition using random initialization + ALS with exact rounding
    # 2. Use combinatorial search for small ranks
    
    # First, let's just return the schoolbook decomposition as a fallback
    rank_fallback = n * m * p
    U_fallback = [[0] * rank_fallback for _ in range(n * m)]
    V_fallback = [[0] * rank_fallback for _ in range(m * p)]
    W_fallback = [[0] * rank_fallback for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fallback[i * m + j][col] = 1
                V_fallback[j * p + k][col] = 1
                W_fallback[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_fallback, "V": V_fallback, "W": W_fallback}
    best_rank = rank_fallback
    
    # Try to improve using a structured approach
    # We'll use the fact that we can decompose into blocks
    # For 3x4 * 4x6, we can try various block partitions
    
    # Try a 2x2 Strassen on the 4x4 core, handling the extra rows/cols
    # This is complex, so let's try a numerical approach first
    
    # Use ALS (Alternating Least Squares) with rounding to half-integers
    def try_als(rank, max_iters=50, tol=1e-10):
        """Try to find a rank-CP decomposition using ALS with exact rounding."""
        if time.time() > deadline - 5:
            return None
        
        # Initialize with random values
        np.random.seed(42)
        U = np.random.randn(n * m, rank) * 0.1
        V = np.random.randn(m * p, rank) * 0.1
        W = np.random.randn(p * n, rank) * 0.1
        
        # The target tensor in flattened form
        # T[i, j, k] = 1 for all i, j, k
        # We need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        
        # This is a 3-way tensor. Let's unroll it.
        # For each (i, k), the equation is:
        # sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = m (sum over j of 1)
        # Wait, T[i,j,k] = 1 for each (i,j,k), so for fixed (i,k):
        # sum_j T[i,j,k] = m, but we need each T[i,j,k] = 1 separately
        
        # Let's set up the linear system properly
        # For each triple (i, j, k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        
        # This is nonlinear in U, V, W. ALS alternates:
        # Fix V, W: solve for U
        # Fix U, W: solve for V
        # Fix U, V: solve for W
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                return None
            
            # Update U: for each (i, j, r), U[i*m+j, r] appears in equations for all k
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for each k
            # This is linear in U[i*m+j, :]
            # For fixed (i,j): sum_r U[i*m+j, r] * (V[j*p+k, r] * W[k*n+i, r]) = 1 for each k
            # So we have a system: M[i,j] @ U[i*m+j, :]^T = 1_vector
            # where M[i,j][k, r] = V[j*p+k, r] * W[k*n+i, r]
            
            for i in range(n):
                for j in range(m):
                    # Build M: (p, rank)
                    M = np.zeros((p, rank))
                    for k in range(p):
                        for r in range(rank):
                            M[k, r] = V[j * p + k, r] * W[k * n + i, r]
                    # Solve M @ u = ones(p)
                    try:
                        u, _, _, _ = np.linalg.lstsq(M, np.ones(p), rcond=None)
                        U[i * m + j, :] = u
                    except:
                        return None
            
            # Update V: for each (j, k, r), V[j*p+k, r] appears in equations for all i
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for each i
            # For fixed (j,k): sum_r V[j*p+k, r] * (U[i*m+j, r] * W[k*n+i, r]) = 1 for each i
            # M[j,k][i, r] = U[i*m+j, r] * W[k*n+i, r]
            
            for j in range(m):
                for k in range(p):
                    M = np.zeros((n, rank))
                    for i in range(n):
                        for r in range(rank):
                            M[i, r] = U[i * m + j, r] * W[k * n + i, r]
                    try:
                        v, _, _, _ = np.linalg.lstsq(M, np.ones(n), rcond=None)
                        V[j * p + k, :] = v
                    except:
                        return None
            
            # Update W: for each (k, i, r), W[k*n+i, r] appears in equations for all j
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for each j
            # For fixed (k,i): sum_r W[k*n+i, r] * (U[i*m+j, r] * V[j*p+k, r]) = 1 for each j
            # M[k,i][j, r] = U[i*m+j, r] * V[j*p+k, r]
            
            for k in range(p):
                for i in range(n):
                    M = np.zeros((m, rank))
                    for j in range(m):
                        for r in range(rank):
                            M[j, r] = U[i * m + j, r] * V[j * p + k, r]
                    try:
                        w, _, _, _ = np.linalg.lstsq(M, np.ones(m), rcond=None)
                        W[k * n + i, :] = w
                    except:
                        return None
        
        # Check the solution
        max_err = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for r in range(rank):
                        val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    max_err = max(max_err, abs(val - 1))
        
        if max_err < 1e-8:
            # Round to nearest half-integer
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            # Verify rounded solution
            max_err_rounded = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for r in range(rank):
                            val += U_rounded[i * m + j, r] * V_rounded[j * p + k, r] * W_rounded[k * n + i, r]
                        max_err_rounded = max(max_err_rounded, abs(val - 1))
            
            if max_err_rounded < 1e-9:
                return {
                    "U": [[float(x) for x in row] for row in U_rounded],
                    "V": [[float(x) for x in row] for row in V_rounded],
                    "W": [[float(x) for x in row] for row in W_rounded],
                    "rank": rank
                }
        
        return None
    
    # Try various ranks from low to high
    # Known lower bound: max(n*m, m*p, p*n) = max(12, 24, 18) = 24
    # But actually the lower bound is higher. For matrix multiplication, 
    # the rank is at least max(n*p, m*n, m*p) but that's not tight.
    # The true lower bound for 3x4*4x6 is unknown but likely > 30
    
    # Let's try ranks from 30 to 60
    for rank in range(30, 65):
        if time.time() > deadline - 10:
            break
        
        result = try_als(rank, max_iters=30)
        if result is not None:
            if rank < best_rank:
                best_rank = rank
                best_result = {
                    "U": result["U"],
                    "V": result["V"],
                    "W": result["W"]
                }
            break  # Found a valid decomposition, stop
    
    return best_result
# EVOLVE-BLOCK-END
