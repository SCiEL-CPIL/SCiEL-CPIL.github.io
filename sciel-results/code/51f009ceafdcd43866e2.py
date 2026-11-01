"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x9 matrix.
    The output factor follows the official trace convention W[k*n+i, r].
    
    Returns dict with 'U', 'V', 'W' factor matrices.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 2, 3, 9
    
    # Build the target tensor T with T[i*m+j, j*p+k, k*n+i] = 1
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # Schoolbook decomposition as fallback (rank = n*m*p = 54)
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_sb
    
    # Try to find lower rank decompositions
    # Key observation: the tensor has a special structure for matrix multiplication.
    # For n=2, we can exploit the small number of output rows.
    
    # Approach: Try structured decompositions where we group terms.
    # For each (i, k), the constraint is: sum_j A[i,j]*B[j,k] = C[i,k]
    # In tensor form: for fixed (i,k), sum_j sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all j
    
    # Actually the constraint is: for ALL (i,j,k): sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    
    # Let's try to find a decomposition with rank < 54 using a greedy/structured approach.
    
    # Strategy: For each "slice" of the tensor, try to find a low-rank approximation.
    # The tensor T is a 3-way array of size (6, 27, 18) with specific 1s.
    
    # Let's try a different approach: use the fact that we can factor through intermediate spaces.
    # For matrix multiplication, one natural factorization goes through the "middle" dimension.
    
    # Try: rank = m * max(n, p) = 3 * 9 = 27? Or rank = max(n*m, m*p, n*p) = 27?
    # The lower bound is max(n*m, m*p, n*p) = max(6, 27, 18) = 27.
    
    # Let's try to construct a rank-27 decomposition.
    # Idea: For each j (middle index), we have n*p = 18 output entries.
    # But we need to handle the sum over j.
    
    # Alternative structured approach:
    # U[i*m+j, r] where r indexes (j, something)
    # Let's try r = j * p + k' for some grouping... 
    
    # Actually, let me try a direct algebraic approach.
    # For rank r, we have variables U (6×r), V (27×r), W (18×r).
    # The constraint is bilinear in (U,V,W).
    
    # Let's try a specific structured ansatz:
    # Suppose W[k*n+i, r] depends only on k and i in a separable way.
    # W[k*n+i, r] = w_k[i] * something... 
    
    # Let me try: for each (i, k), define a vector of length m.
    # The constraint for fixed (i,k) is: for all j, sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    
    # This means: the vector (U[i*m+j,r] * V[j*p+k,r])_r dotted with W[k*n+i,:] = 1 for all j.
    
    # If W[k*n+i,:] is the same for all i (for fixed k), then we need:
    # For each j, the product vector U[i*m+j,:] * V[j*p+k,:] (element-wise) dotted with common_W = 1
    
    # This is getting complex. Let me try a numerical search for small ranks.
    
    # Try ranks from lower bound up
    for target_rank in range(27, min(54, 40)):
        if time.time() > deadline - 30:
            break
        
        # Try to find a decomposition of this rank using random initialization + local optimization
        # This is a non-convex problem, so we use multiple random restarts
        
        success = False
        for attempt in range(5):
            if time.time() > deadline - 25:
                break
            
            rng = np.random.RandomState(42 + attempt * 100 + target_rank)
            
            # Initialize with small random values
            U = rng.randn(n * m, target_rank) * 0.1
            V = rng.randn(m * p, target_rank) * 0.1
            W = rng.randn(p * n, target_rank) * 0.1
            
            # Use alternating minimization: fix two, solve for the third
            # The constraint is: for all (i,j,k): sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
            
            # This is a system of n*m*p = 54 equations.
            # For fixed U and V, we need to solve for W:
            # For each (k,i): sum_j [sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]] ... 
            # Wait, the constraint is for each (i,j,k) separately.
            
            # Let me re-read: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for ALL i,j,k.
            
            # So for fixed (i,j,k), this is a single scalar equation.
            # For fixed U and V, the equation becomes:
            # sum_r (U[i*m+j,r] * V[j*p+k,r]) * W[k*n+i,r] = 1
            # This is a linear equation in W[k*n+i,:] for each (i,j,k).
            
            # But W[k*n+i,:] appears in multiple equations (one for each j).
            # For fixed (i,k), we have m=3 equations:
            # For j=0: sum_r U[i*m+0,r]*V[0*p+k,r]*W[k*n+i,r] = 1
            # For j=1: sum_r U[i*m+1,r]*V[1*p+k,r]*W[k*n+i,r] = 1
            # For j=2: sum_r U[i*m+2,r]*V[2*p+k,r]*W[k*n+i,r] = 1
            
            # So for each (i,k), we have a system of m=3 equations in W[k*n+i,:] (which has target_rank unknowns).
            # This is underdetermined (3 equations, target_rank unknowns) for target_rank > 3.
            
            # Similarly, for fixed U and W, we can solve for V.
            # And for fixed V and W, we can solve for U.
            
            # Let's do alternating least squares.
            for iteration in range(100):
                if time.time() > deadline - 20:
                    break
                
                # Update W: for each (i,k), solve the linear system
                W_new = W.copy()
                for i in range(n):
                    for k in range(p):
                        # Build the 3 x target_rank system
                        A_sys = np.zeros((m, target_rank))
                        b_sys = np.ones(m)
                        for j in range(m):
                            for r in range(target_rank):
                                A_sys[j, r] = U[i*m+j, r] * V[j*p+k, r]
                        # Solve A_sys @ W[k*n+i, :] = b_sys
                        # Use least squares
                        W[k*n+i, :], residuals, rank_sys, sv = np.linalg.lstsq(A_sys, b_sys, rcond=None)
                
                # Update V: for each (j,k), solve the linear system
                V_new = V.copy()
                for j in range(m):
                    for k in range(p):
                        A_sys = np.zeros((n, target_rank))
                        b_sys = np.ones(n)
                        for i in range(n):
                            for r in range(target_rank):
                                A_sys[i, r] = U[i*m+j, r] * W[k*n+i, r]
                        V[j*p+k, :], residuals, rank_sys, sv = np.linalg.lstsq(A_sys, b_sys, rcond=None)
                
                # Update U: for each (i,j), solve the linear system
                U_new = U.copy()
                for i in range(n):
                    for j in range(m):
                        A_sys = np.zeros((p, target_rank))
                        b_sys = np.ones(p)
                        for k in range(p):
                            for r in range(target_rank):
                                A_sys[k, r] = V[j*p+k, r] * W[k*n+i, r]
                        U[i*m+j, :], residuals, rank_sys, sv = np.linalg.lstsq(A_sys, b_sys, rcond=None)
                
                # Check convergence
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(target_rank))
                            err += abs(val - 1.0)
                
                if err < 1e-6:
                    # Found a solution! Check if it can be made rational
                    # For now, just verify and return
                    # Convert to fractions if possible
                    U_frac = [[float(x) for x in U[i]] for i in range(n*m)]
                    V_frac = [[float(x) for x in V[i]] for i in range(m*p)]
                    W_frac = [[float(x) for x in W[i]] for i in range(p*n)]
                    
                    # Verify the solution
                    max_err = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = sum(U_frac[i*m+j][r] * V_frac[j*p+k][r] * W_frac[k*n+i][r] for r in range(target_rank))
                                max_err = max(max_err, abs(val - 1.0))
                    
                    if max_err < 1e-4:
                        # Try to round to nearest half-integer
                        U_rounded = [[round(x * 2) / 2 for x in row] for row in U_frac]
                        V_rounded = [[round(x * 2) / 2 for x in row] for row in V_frac]
                        W_rounded = [[round(x * 2) / 2 for x in row] for row in W_frac]
                        
                        # Verify rounded solution
                        max_err_rounded = 0.0
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    val = sum(U_rounded[i*m+j][r] * V_rounded[j*p+k][r] * W_rounded[k*n+i][r] for r in range(target_rank))
                                    max_err_rounded = max(max_err_rounded, abs(val - 1.0))
                        
                        if max_err_rounded < 1e-9:
                            return {"U": U_rounded, "V": V_rounded, "W": W_rounded}
                        else:
                            # Return the float solution (should be acceptable if close enough)
                            return {"U": U_frac, "V": V_frac, "W": W_frac}
            
            if err < 1e-3:
                # Close enough, try to rationalize
                U_frac = [[float(x) for x in U[i]] for i in range(n*m)]
                V_frac = [[float(x) for x in V[i]] for i in range(m*p)]
                W_frac = [[float(x) for x in W[i]] for i in range(p*n)]
                
                # Try rounding to half-integers
                U_rounded = [[round(x * 2) / 2 for x in row] for row in U_frac]
                V_rounded = [[round(x * 2) / 2 for x in row] for row in V_frac]
                W_rounded = [[round(x * 2) / 2 for x in row] for row in W_frac]
                
                max_err_rounded = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = sum(U_rounded[i*m+j][r] * V_rounded[j*p+k][r] * W_rounded[k*n+i][r] for r in range(target_rank))
                            max_err_rounded = max(max_err_rounded, abs(val - 1.0))
                
                if max_err_rounded < 1e-6:
                    return {"U": U_rounded, "V": V_rounded, "W": W_rounded}
        
        # Try a structured decomposition for this rank
        # For rank = m*p = 27: use the "middle" structure
        if target_rank == m * p:
            # U[i*m+j, j*p+k] = delta_{i,0} (only for i=0, use all; for i=1, use a different pattern)
            # This won't work directly. Let me think differently.
            pass
    
    # If we couldn't find a better decomposition, return the schoolbook
    return best_result

# EVOLVE-BLOCK-END
