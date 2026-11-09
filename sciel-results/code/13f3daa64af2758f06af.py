"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import random

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Schoolbook decomposition as fallback
    rank_school = n * m * p  # 128
    
    def schoolbook():
        rank = rank_school
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
    
    def verify(U, V, W, tol=1e-9):
        """Verify tensor decomposition."""
        n, m, p = DIMENSIONS
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def to_int_half(U, V, W):
        """Convert to integer/half-integer representation if possible."""
        # Check if values are close to integers or half-integers
        def clean_matrix(M):
            M = np.array(M, dtype=float)
            # Check if all values are close to half-integers
            doubled = M * 2
            rounded = np.round(doubled)
            if np.allclose(doubled, rounded, atol=1e-8):
                # Return as fractions (numerator, denominator)
                # But interface expects numeric values, so return the actual values
                return (rounded / 2).tolist()
            else:
                # Try integers
                rounded_int = np.round(M)
                if np.allclose(M, rounded_int, atol=1e-8):
                    return rounded_int.tolist()
                return M.tolist()
        
        return {"U": clean_matrix(U), "V": clean_matrix(V), "W": clean_matrix(W)}
    
    # Try to find better decomposition using structured approach
    # Strategy: Use random initialization and try to solve the system
    # The key insight: we can try rank r by setting up the equations and
    # using iterative refinement
    
    best_result = schoolbook()
    best_rank = rank_school
    
    # Try ranks from 127 down to 100 with time budget
    # For each rank, try a few random initializations
    
    random.seed(42)
    np.random.seed(42)
    
    # For efficiency, we'll try a structured approach:
    # 1. Start with schoolbook
    # 2. Try to reduce rank by finding linear dependencies
    
    # Alternative: Use the fact that for matrix multiplication,
    # we can group operations. Let's try a block-based approach.
    
    # For 4x4x8, think of it as multiplying 4x4 by 4x8.
    # We can decompose the 4x8 into two 4x4 blocks.
    # Then use a 4x4x4 decomposition for each block.
    
    # Known: 4x4x4 matrix multiplication has rank at most 49 (or lower)
    # If we use rank r for 4x4x4, then 4x4x8 might use rank 2*r
    # But we can potentially do better with shared structure.
    
    # Let's try: decompose 4x4x8 as two 4x4x4 problems
    # A(4x4) * B1(4x4) = C1(4x4)  -- uses rank r1
    # A(4x4) * B2(4x4) = C2(4x4)  -- uses rank r2
    # Total rank = r1 + r2, but if we share the A factors, we can do better.
    
    # Actually, the tensor structure doesn't allow simple sharing in this way
    # because the W matrix depends on k and i, and k ranges over 8.
    
    # Let me try a direct optimization approach:
    # For a given rank r, initialize U, V, W randomly and try to minimize
    # the error using a simple iterative method.
    
    def try_rank(r, num_trials=3):
        """Try to find a rank-r decomposition."""
        for trial in range(num_trials):
            if time.time() > deadline:
                return None
            
            # Random initialization
            U = np.random.randn(n * m, r) * 0.1
            V = np.random.randn(m * p, r) * 0.1
            W = np.random.randn(p * n, r) * 0.1
            
            # Iterative refinement: for each (i,j,k), the equation is
            # sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1
            # This is a system of nonlinear equations.
            # Simple approach: alternating least squares won't work directly
            # because of the product structure.
            
            # Instead, let's use a different approach:
            # Fix two matrices and solve for the third.
            # If we fix V and W, then for each (i,j,k):
            # sum_c U[i*m+j, c] * (V[j*p+k, c] * W[k*n+i, c]) = 1
            # This is linear in U! So we can solve for U given V, W.
            
            # Similarly for V given U, W and W given U, V.
            
            # Alternating linear solve:
            for iteration in range(50):
                if time.time() > deadline:
                    return None
                
                # Solve for U: for each (i,j), we have equations for all k
                # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all k
                # This is: sum_c U[i*m+j, c] * (V[j*p+k, c] * W[k*n+i, c]) = 1
                # For fixed (i,j), this is a system of p equations in r unknowns
                # (the r entries of U[i*m+j, :])
                
                for i in range(n):
                    for j in range(m):
                        # Build system: for each k, sum_c U[i*m+j,c] * A[k,c] = 1
                        # where A[k,c] = V[j*p+k, c] * W[k*n+i, c]
                        A = np.zeros((p, r))
                        for k in range(p):
                            A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                        # Solve A^T * x = 1 (least squares if r < p)
                        # Actually: sum_c x[c] * A[k,c] = 1 for all k
                        # This is: A @ x = ones(p)
                        try:
                            x, residuals, rank_A, sv = np.linalg.lstsq(A, np.ones(p), rcond=None)
                            U[i*m+j, :] = x
                        except:
                            pass
                
                # Solve for V: for each (j,k), sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i
                # For fixed (j,k): sum_c V[j*p+k,c] * (U[i*m+j,c] * W[k*n+i,c]) = 1 for all i
                for j in range(m):
                    for k in range(p):
                        A = np.zeros((n, r))
                        for i in range(n):
                            A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                        try:
                            x, residuals, rank_A, sv = np.linalg.lstsq(A, np.ones(n), rcond=None)
                            V[j*p+k, :] = x
                        except:
                            pass
                
                # Solve for W: for each (k,i), sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all j
                # For fixed (k,i): sum_c W[k*n+i,c] * (U[i*m+j,c] * V[j*p+k,c]) = 1 for all j
                for k in range(p):
                    for i in range(n):
                        A = np.zeros((m, r))
                        for j in range(m):
                            A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                        try:
                            x, residuals, rank_A, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                            W[k*n+i, :] = x
                        except:
                            pass
            
            # Check if we found a valid solution
            if verify(U, V, W):
                result = to_int_half(U, V, W)
                return result
        
        return None
    
    # Try ranks from 127 down
    # Start with a few trials at higher ranks, then more at lower ranks
    for r in range(rank_school - 1, 99, -1):
        if time.time() > deadline - 5:
            break
        
        # Use more trials for lower ranks
        num_trials = 2 if r > 110 else 3
        result = try_rank(r, num_trials)
        if result is not None:
            best_result = result
            best_rank = r
            break  # Found the best we can do for now
    
    # If we found something better, return it; otherwise schoolbook
    return best_result
# EVOLVE-BLOCK-END

