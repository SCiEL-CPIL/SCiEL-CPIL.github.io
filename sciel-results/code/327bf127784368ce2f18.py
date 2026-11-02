"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
import numpy as np
from itertools import product, combinations
import sys
import time

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x2 matrix by a 2x6 matrix.
    
    The tensor identity is: for all i in [0,n), j in [0,m), k in [0,p):
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    We use a constructive approach: fix the structure of U and V, then solve for W.
    For rank r, we choose U (4 x r) and V (12 x r) with simple entries,
    then solve the linear system for W (12 x r).
    
    We search for the smallest r where a solution with half-integer entries exists.
    """
    n, m, p = DIMENSIONS
    dim_u = n * m  # 4
    dim_v = m * p  # 12
    dim_w = p * n  # 12
    
    def verify(U, V, W):
        """Verify that the decomposition is correct."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    def to_half_int(arr):
        """Convert numpy array to list of half-integers if possible, else None."""
        arr = np.array(arr)
        # Check if all entries are multiples of 0.5
        doubled = arr * 2
        if not np.allclose(doubled, np.round(doubled), atol=1e-9):
            return None
        result = np.round(doubled).astype(int) / 2.0
        # Convert to list of lists with proper types
        return [[float(x) for x in row] for row in result]
    
    def try_rank_structured(r):
        """
        Try to find a rank-r decomposition by fixing U and V with specific structure
        and solving for W linearly.
        """
        # Strategy: We'll try different patterns for U and V.
        # Key insight: For matmul(2,2,6), we can think of it as computing 2 output rows,
        # each with 6 columns. The "inner" dimension is 2.
        
        # Approach 1: Use a block structure
        # Let's try to use the fact that the problem has symmetry in the 6 columns.
        
        # We'll parameterize U and V and solve for W.
        # The system is: for each (i,j,k), sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        
        # This is linear in W if U and V are fixed.
        # For each output index (k*n+i) and each input pair (i,j), we have:
        # sum_c (U[i*m+j,c] * V[j*p+k,c]) * W[k*n+i,c] = 1
        
        # Let's define M[i,j,k] as a vector of length r where M[i,j,k,c] = U[i*m+j,c] * V[j*p+k,c]
        # Then for each (i,k), we need: sum_j M[i,j,k] * W[k*n+i, :] = 1
        # Wait, that's not quite right. Let me re-derive.
        
        # For fixed (i,k), the equation involves W[k*n+i, c] for all c.
        # The equation is: sum_j sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        # = sum_c W[k*n+i,c] * sum_j U[i*m+j,c] * V[j*p+k,c] = 1
        
        # So for each (i,k), we have a linear equation in W[k*n+i, :]:
        # sum_c (sum_j U[i*m+j,c] * V[j*p+k,c]) * W[k*n+i,c] = 1
        
        # This means: for each (i,k), the dot product of W[k*n+i,:] with vector 
        # A[i,k,:] where A[i,k,c] = sum_j U[i*m+j,c] * V[j*p+k,c] must equal 1.
        
        # So we need to find W such that for each (i,k):
        # W[k*n+i,:] dot A[i,k,:] = 1
        
        # This is 12 independent linear equations (one for each (i,k) pair), each of rank 1.
        # We can always satisfy this if A[i,k,:] is not zero, by setting W[k*n+i,:] = A[i,k,:] / ||A[i,k,:]||^2
        
        # But we also need W to have half-integer entries!
        
        # So the strategy is:
        # 1. Choose U and V (4x r and 12x r) with half-integer entries
        # 2. Compute A[i,k,:] for each (i,k)
        # 3. Set W[k*n+i,:] = A[i,k,:] / ||A[i,k,:]||^2
        # 4. Check if W has half-integer entries
        
        # This is still hard. Let's try a more direct approach.
        
        # Alternative: Use the structure that for each "layer" c, we have a rank-1 tensor.
        # The sum of r rank-1 tensors should equal the matmul tensor.
        
        # Let's try a random search with structured initialization.
        
        best_solution = None
        
        # Try several random initializations
        for trial in range(50):
            # Initialize U and V with small integer/half-integer values
            np.random.seed(trial * 1000 + r)
            
            # Use a structured approach: 
            # U has 4 rows, V has 12 rows.
            # Let's try to make U and V have entries from {-1, -0.5, 0, 0.5, 1}
            
            U_init = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(dim_u, r))
            V_init = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(dim_v, r))
            
            # Compute A[i,k,:]
            A = np.zeros((n, p, r))
            for i in range(n):
                for k in range(p):
                    for c in range(r):
                        A[i, k, c] = sum(U_init[i*m+j, c] * V_init[j*p+k, c] for j in range(m))
            
            # For each (i,k), we need W[k*n+i,:] dot A[i,k,:] = 1
            # Set W[k*n+i,:] = A[i,k,:] / sum(A[i,k,:]^2) if sum != 0
            W = np.zeros((dim_w, r))
            valid = True
            for i in range(n):
                for k in range(p):
                    idx = k * n + i
                    norm_sq = np.sum(A[i, k, :] ** 2)
                    if abs(norm_sq) < 1e-9:
                        valid = False
                        break
                    W[idx, :] = A[i, k, :] / norm_sq
            if not valid:
                continue
            
            # Check if W has half-integer entries
            W_doubled = W * 2
            if np.allclose(W_doubled, np.round(W_doubled), atol=1e-9):
                W = np.round(W_doubled).astype(int) / 2.0
                # Verify the solution
                if verify(U_init.tolist(), V_init.tolist(), W.tolist()):
                    best_solution = (U_init.tolist(), V_init.tolist(), W.tolist())
                    break
        
        if best_solution is not None:
            return best_solution
        
        return None
    
    # Try ranks from a reasonable lower bound up to the schoolbook rank
    # Theoretical lower bound: at least max(n*m, m*p, p*n) / something... 
    # Actually, the rank must be at least max(n*m, m*p, p*n) for some cases, but not always.
    # For matmul(2,2,6), let's try from 12 to 24.
    
    for r in range(12, 25):
        result = try_rank_structured(r)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook solution
    rank = n * m * p
    U = [[0] * rank for _ in range(dim_u)]
    V = [[0] * rank for _ in range(dim_v)]
    W = [[0] * rank for _ in range(dim_w)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
