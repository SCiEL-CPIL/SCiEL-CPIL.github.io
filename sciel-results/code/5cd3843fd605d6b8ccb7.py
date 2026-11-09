"""Find rank-minimal exact tensor decomposition for 2x4 * 4x5 matrix multiplication."""
import time
import random
import numpy as np
from scipy.optimize import linprog

DIMENSIONS = [2, 4, 5]

# Known optimal rank for 2x2x5 is 10 (schoolbook)
# We try to find a decomposition for the full 2x4x5 problem

def construct():
    """
    Find rank-minimal exact tensor decomposition for multiplying a 2 by 4 matrix by a 4 by 5 matrix.
    
    The output factor follows the official trace convention W[k*n+i, r].
    
    Returns: {'U': (n*m, rank), 'V': (m*p, rank), 'W': (p*n, rank)}
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    
    # Schoolbook rank is n*m*p = 40
    # The block structure suggests we can do better.
    # For 2x4 * 4x5, we can think of it as two 2x2 * 2x5 problems if we split the 4 columns.
    # But actually, the structure is: A is 2x4, B is 4x5.
    # We can partition the middle dimension (size 4) into two blocks of size 2.
    # A = [A1 A2] where A1, A2 are 2x2
    # B = [B1; B2] where B1, B2 are 2x5
    # C = A*B = A1*B1 + A2*B2
    # Each term is a 2x2 * 2x5 multiplication.
    # The rank of 2x2 * 2x5 multiplication is at most 2*2*5 = 20 (schoolbook).
    # So total rank would be at most 40, which is the same as schoolbook.
    
    # Let's try a different approach: use the known result that for matrix multiplication,
    # the tensor rank of <n,m,p> is bounded. For 2x4x5, let's try to find a lower rank.
    
    # Strategy: Try rank values starting from a lower bound and use random initialization
    # with tensor decomposition (CP decomposition) to find an exact solution.
    
    # Lower bound: max(n*m, m*p, p*n) = max(8, 20, 10) = 20
    # Also, for matrix multiplication, there are better lower bounds.
    
    # Let's try ranks from 20 upward, using random CP decomposition with exact verification.
    
    best_result = None
    
    # Try a range of ranks
    min_rank = 20  # Lower bound from max of dimensions products
    max_rank = 40  # Schoolbook
    
    # We'll try to find a solution with rank as low as possible
    for rank in range(min_rank, max_rank + 1):
        if time.time() > deadline - 10:
            break
            
        # Try multiple random initializations
        max_attempts = 50 if rank <= 25 else 20
        if time.time() > deadline - 20:
            max_attempts = min(max_attempts, 5)
            
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                    break
                    
            # Random initialization for U, V, W
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            W = np.random.randn(p * n, rank)
            
            # We need to find U, V, W such that:
            # For all i, j, k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # (This is the matrix multiplication tensor)
            
            # This is a system of equations. Let's set up the linear system.
            # For each (i, j, k), we have:
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            
            # This is nonlinear in U, V, W. Let's use an alternating approach:
            # Fix V and W, solve for U (linear), then fix U and W, solve for V, etc.
            
            # Actually, let's use a different approach: use the fact that the tensor
            # multiplication tensor has a specific structure.
            
            # For matrix multiplication, the tensor is:
            # T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
            # Wait, no. The matrix multiplication tensor is defined such that:
            # C[i,k] = sum_j A[i,j] * B[j,k]
            # So the tensor T has T[i,j,k] = 1 for all valid i,j,k.
            
            # The CP decomposition requires:
            # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            
            # Since T[i,j,k] = 1 for all i,j,k, we need:
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
            
            # Let's use an iterative approach:
            # 1. Initialize W randomly
            # 2. For each (i,j), solve for U[i*m+j, :] given V and W
            # 3. For each (j,k), solve for V[j*p+k, :] given U and W
            # 4. For each (k,i), solve for W[k*n+i, :] given U and V
            # 5. Repeat until convergence
            
            # Let's try this iterative approach
            W_init = np.random.randn(p * n, rank)
            U_curr = np.random.randn(n * m, rank)
            V_curr = np.random.randn(m * p, rank)
            
            converged = False
            for iter in range(100):
                if time.time() > deadline - 5:
                    break
                    
                # Update U: for each (i,j), we need sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all k
                # This is: for each (i,j), U[i*m+j,:] * (V[j*p:k,:] * W[k*n+i,:])^T = 1 for all k
                # Let's define for each (i,j,k): the coefficient is V[j*p+k,r] * W[k*n+i,r]
                # So for fixed (i,j), we have a system:
                # sum_r U[i*m+j,r] * (V[j*p+k,r] * W[k*n+i,r]) = 1 for k=0,...,p-1
                # This is a linear system in U[i*m+j,:] with p equations and rank unknowns.
                
                # Similarly for V and W.
                
                # Let's build the full linear system.
                # Total number of equations: n*m*p = 40
                # Total number of unknowns: n*m*rank + m*p*rank + p*n*rank = (8+20+10)*rank = 38*rank
                
                # For rank=20, that's 760 unknowns and 40 equations. Very underdetermined.
                # So we should be able to find a solution easily if it exists.
                
                # Let's use a simpler approach: just try to solve the system directly.
                # But the system is nonlinear. Let's use the alternating linearization.
                
                # Update U
                U_new = np.zeros_like(U_curr)
                for i in range(n):
                    for j in range(m):
                        # Build the system for U[i*m+j, :]
                        # For each k: sum_r U[i*m+j,r] * V_curr[j*p+k,r] * W_curr[k*n+i,r] = 1
                        A_mat = np.zeros((p, rank))
                        for k in range(p):
                            A_mat[k, :] = V_curr[j*p+k, :] * W_curr[k*n+i, :]
                        # Solve A_mat @ U[i*m+j,:] = ones(p)
                        try:
                            sol, residuals, rank_sol, sv = np.linalg.lstsq(A_mat, np.ones(p), rcond=None)
                            U_new[i*m+j, :] = sol
                        except:
                            pass
                
                # Update V
                V_new = np.zeros_like(V_curr)
                for j in range(m):
                    for k in range(p):
                        A_mat = np.zeros((n, rank))
                        for i in range(n):
                            A_mat[i, :] = U_new[i*m+j, :] * W_curr[k*n+i, :]
                        try:
                            sol, residuals, rank_sol, sv = np.linalg.lstsq(A_mat, np.ones(n), rcond=None)
                            V_new[j*p+k, :] = sol
                        except:
                            pass
                
                # Update W
                W_new = np.zeros_like(W_curr)
                for k in range(p):
                    for i in range(n):
                        A_mat = np.zeros((m, rank))
                        for j in range(m):
                            A_mat[j, :] = U_new[i*m+j, :] * V_new[j*p+k, :]
                        try:
                            sol, residuals, rank_sol, sv = np.linalg.lstsq(A_mat, np.ones(m), rcond=None)
                            W_new[k*n+i, :] = sol
                        except:
                            pass
                
                # Check convergence
                diff = np.max(np.abs(U_new - U_curr)) + np.max(np.abs(V_new - V_curr)) + np.max(np.abs(W_new - W_curr))
                U_curr = U_new
                V_curr = V_new
                W_curr = W_new
                
                if diff < 1e-10:
                    converged = True
                    break
            
            # Verify the solution
            if converged or True:  # Always verify
                error = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U_curr[i*m+j, :] * V_curr[j*p+k, :] * W_curr[k*n+i, :])
                            error = max(error, abs(val - 1.0))
                
                if error < 1e-8:
                    # Found a valid solution!
                    # Convert to integer/half-integer if possible
                    U_int = np.round(U_curr * 2) / 2
                    V_int = np.round(V_curr * 2) / 2
                    W_int = np.round(W_curr * 2) / 2
                    
                    # Verify the integer/half-integer version
                    error_int = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = np.sum(U_int[i*m+j, :] * V_int[j*p+k, :] * W_int[k*n+i, :])
                                error_int = max(error_int, abs(val - 1.0))
                    
                    if error_int < 1e-6:
                        # Return the half-integer solution
                        return {
                            "U": U_int.tolist(),
                            "V": V_int.tolist(),
                            "W": W_int.tolist()
                        }
                    else:
                        # Return the floating point solution (should be acceptable if close enough)
                        if error < 1e-6:
                            return {
                                "U": U_curr.tolist(),
                                "V": V_curr.tolist(),
                                "W": W_curr.tolist()
                            }
    
    # Fallback: return the schoolbook decomposition
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
