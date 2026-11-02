import numpy as np
import time
import random

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # 110s limit to be safe
    
    # Build the matrix multiplication tensor T[i,j,k] = 1 for all i,j,k
    # We need: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This is a trilinear form.
    
    # First, try to find a decomposition with rank < 45 using a systematic approach.
    # Strategy: Use the fact that matrix multiplication can sometimes be decomposed
    # with fewer terms. We'll try various structured patterns.
    
    def schoolbook_decomposition():
        U = [[0] * rank_schoolbook for _ in range(n * m)]
        V = [[0] * rank_schoolbook for _ in range(m * p)]
        W = [[0] * rank_schoolbook for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to find a lower-rank decomposition using a structured approach
    # based on the idea that some entries can be shared.
    
    # Approach: Try rank 44 by merging two schoolbook terms that have overlapping structure
    # or by finding a linear dependency.
    
    # For now, let's try a randomized search for rank 44.
    # We'll set up a system: for each (i,j,k), we need sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    # This is nonlinear, so we'll try a different approach.
    
    # Alternative: Use the Strassen-like approach or just verify if rank 44 is possible.
    # Given the time constraint, let's try a few structured patterns.
    
    # Pattern 1: Try to eliminate one term by expressing it as a combination.
    # This is complex, so let's just return schoolbook for now but structure the code
    # to allow for future improvements.
    
    # Actually, let's try a simple randomized approach:
    # Fix U and V to have sparse random patterns, then solve for W.
    
    def try_rank(r, num_attempts=100):
        """Try to find a rank-r decomposition."""
        for attempt in range(num_attempts):
            if time.time() > deadline:
                return None
            
            # Create random sparse U and V
            U = np.zeros((n * m, r))
            V = np.zeros((m * p, r))
            
            # Make U and V have some structure: each row has a few nonzeros
            for row in range(n * m):
                nnz = random.randint(1, min(3, r))
                cols = random.sample(range(r), nnz)
                for c in cols:
                    U[row, c] = random.choice([-1, 1])
            
            for row in range(m * p):
                nnz = random.randint(1, min(3, r))
                cols = random.sample(range(r), nnz)
                for c in cols:
                    V[row, c] = random.choice([-1, 1])
            
            # Now solve for W: for each (k, n+i) index, we need
            # W[k*n+i, r] such that for all (i,j,k):
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            #
            # This is still nonlinear in W. Instead, let's think differently.
            #
            # Actually, the tensor decomposition is: T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
            # where U is indexed by (i,j), V by (j,k), W by (k,i).
            #
            # For fixed U and V, this is linear in W. Let's set up the linear system.
            # For each (i,j,k), we have: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
            #
            # But W[k*n+i,r] appears in multiple equations (for different j).
            # Specifically, W[k*n+i,r] appears in equations for all j.
            #
            # Let's collect all equations. There are n*m*p = 45 equations.
            # Unknowns: W has p*n*r = 15*r variables.
            #
            # For r=44, we have 660 unknowns and 45 equations - very underdetermined.
            # For r=1, we have 15 unknowns and 45 equations - overdetermined.
            #
            # The issue is that the equations are not linear in W because U*V is fixed
            # but the product U[i,j,r]*V[j,k,r]*W[k,i,r] is trilinear.
            #
            # Wait, no. For fixed U and V, the equation is linear in W.
            # sum_r (U[i*m+j,r] * V[j*p+k,r]) * W[k*n+i,r] = 1
            # This IS linear in W!
            
            # So for each (i,j,k), we have a linear equation in W.
            # Let's set up the full linear system.
            
            # Number of equations: n*m*p = 45
            # Number of unknowns: p*n*r = 15*r
            
            num_eqs = n * m * p
            num_unk = p * n * r
            
            A = np.zeros((num_eqs, num_unk))
            b = np.ones(num_eqs)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Equation: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                        for r_idx in range(r):
                            col = k * n + i * r + r_idx  # W is (p*n) x r, so index [k*n+i, r_idx]
                            # Flatten W: W[k*n+i, r_idx] is at position (k*n+i)*r + r_idx
                            col = (k * n + i) * r + r_idx
                            A[eq_idx, col] = U[i * m + j, r_idx] * V[j * p + k, r_idx]
                        eq_idx += 1
            
            # Solve A @ w_vec = b
            # This is underdetermined if num_unk > num_eqs, which it is for r >= 1.
            # Use least squares or find a particular solution.
            
            try:
                # Find a solution using pseudoinverse
                w_vec, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
                
                # Check if the solution is exact
                if residuals.size > 0:
                    if residuals[0] > 1e-10:
                        continue
                
                # Verify the solution
                w_vec = w_vec.reshape(p * n, r)
                
                # Check all equations
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0
                            for r_idx in range(r):
                                val += U[i * m + j, r_idx] * V[j * p + k, r_idx] * w_vec[k * n + i, r_idx]
                            if abs(val - 1) > 1e-10:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    # Convert to list of lists
                    U_list = [[U[i, r] for r in range(r)] for i in range(n * m)]
                    V_list = [[V[i, r] for r in range(r)] for i in range(m * p)]
                    W_list = [[w_vec[i, r] for r in range(r)] for i in range(p * n)]
                    return {"U": U_list, "V": V_list, "W": W_list}
            except:
                continue
        
        return None
    
    # Try ranks from 44 down to 1
    for r in range(rank_schoolbook - 1, 0, -1):
        if time.time() > deadline:
            break
        
        result = try_rank(r, num_attempts=50)
        if result is not None:
            return result
    
    # Fallback to schoolbook
    return schoolbook_decomposition()
# EVOLVE-BLOCK-END
