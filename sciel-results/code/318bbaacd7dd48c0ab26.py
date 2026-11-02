"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import itertools
import time

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Find rank-minimal exact tensor decomposition for 3x3 * 3x4 matrix multiplication.
    
    Strategy: Fix U to sparse basis patterns (unit vectors or simple combinations),
    then solve linear systems for V and W. This transforms the nonlinear problem
    into linear feasibility checks.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 120  # 120 second deadline
    
    # The matrix multiplication tensor T[i,j,k] = 1 if and only if
    # (i,j) is in A, (j,k) is in B, and output goes to (i,k)
    # For 3x3 * 3x4: A is 3x3, B is 3x4, output is 3x4
    # T[i,j,k] = 1 for all valid (i,j,k) triples
    
    # Build the target tensor as a matrix: for each (i,k) output position,
    # we need sum_c U[i*m+j, c] * V[j*p+k, c] = delta_{i,k}
    # Actually, the tensor is: T[i,j,k] where i in [0,n), j in [0,m), k in [0,p)
    # And we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k]
    
    # For matrix multiplication, T[i,j,k] = 1 for all i,j,k (since every element
    # of A multiplies with every element of B that shares index j)
    # Wait, let me reconsider. The matrix multiplication tensor is:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T has T[i,j,k] = 1 for all valid i,j,k
    
    # A rank-r decomposition means:
    # T[i,j,k] = sum_{c=1}^r U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Strategy: Fix U columns to be standard basis vectors (one 1 per column)
    # Then for each column c, U[:,c] is a unit vector e_{u_c}
    # The constraint becomes: sum_c delta_{u_c, i*m+j} * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # This is still nonlinear. Let's try a different approach.
    
    # Alternative: Use the fact that we can fix U to have specific structure.
    # For rank r, we need r vectors in R^{n*m}, R^{m*p}, R^{p*n}.
    
    # Let's try small ranks first, starting from known lower bounds.
    # For 3x3 * 3x4, the schoolbook rank is 36.
    # Strassen-like methods might reduce this.
    
    # Actually, let me use a direct approach:
    # For each candidate rank r from 20 to 36:
    #   Try to find U, V, W of rank r that decompose T
    
    # Use randomized approach with linear algebra:
    # Fix U randomly (or with structure), then the equations become linear in V and W
    # for fixed U. But it's bilinear, so not quite linear.
    
    # Let me think differently. The tensor T[i,j,k] = 1 for all i,j,k.
    # We need: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1
    
    # If we fix U, then for each (i,j), we have:
    # sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all k
    
    # This is still nonlinear. Let's try a grid search over small ranks
    # with structured U matrices.
    
    # Known result: 3x3 matrix multiplication can be done with rank 23 (Strassen + others)
    # For 3x3 * 3x4, we can potentially use similar techniques.
    
    # Let me try a systematic search:
    # 1. Start with rank 20 and go up
    # 2. For each rank, try random initializations and use local search
    
    best_result = None
    best_rank = n * m * p  # Start with schoolbook
    
    # Generate schoolbook as fallback
    U_sb = np.zeros((n*m, best_rank))
    V_sb = np.zeros((m*p, best_rank))
    W_sb = np.zeros((p*n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    best_result = {'U': U_sb.tolist(), 'V': V_sb.tolist(), 'W': W_sb.tolist()}
    
    # Try to improve using random search with linear refinement
    np.random.seed(42)
    
    for r in range(20, best_rank + 1):
        if time.time() > deadline - 10:
            break
        
        # Try multiple random initializations for this rank
        max_attempts = 50 if r <= 25 else 20
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            U_r = np.random.randn(n*m, r)
            V_r = np.random.randn(m*p, r)
            W_r = np.random.randn(p*n, r)
            
            # Use iterative refinement:
            # Fix two matrices, solve for the third
            # This is like alternating least squares for tensor decomposition
            
            for iteration in range(20):
                if time.time() > deadline - 5:
                    break
                
                # Fix V and W, solve for U
                # For each (i,j), we need: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1
                # This is: sum_c U[i,j,c] * (V[j,:,c] * W[:,i,c]) = 1
                # Let B[i,j] = V[j,:,c] * W[:,i,c] (element-wise product, shape p)
                # Then: U[i,j,:] * B[i,j] = 1 (dot product)
                # This is underdetermined if r > p, but we can use least squares
                
                # Actually, let me set up the linear system properly.
                # For fixed V and W, the equation for U[i,j,:] is:
                # sum_c U[i,j,c] * M[i,j,k,c] = T[i,j,k] for all k
                # where M[i,j,k,c] = V[j,k,c] * W[k,i,c]
                
                # This is a linear system: A_U[i,j] * U[i,j,:] = b_U[i,j]
                # where A_U[i,j] is p x r with A_U[i,j,k,c] = V[j,k,c] * W[k,i,c]
                # and b_U[i,j,k] = 1
                
                # Solve using least squares
                U_new = np.zeros((n*m, r))
                for i in range(n):
                    for j in range(m):
                        A = np.zeros((p, r))
                        for k in range(p):
                            A[k, :] = V_r[j*p+k, :] * W_r[k*n+i, :]
                        b = np.ones(p)
                        # Solve A @ x = b in least squares sense
                        try:
                            U_new[i*m+j, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                        except:
                            pass
                
                # Fix U and W, solve for V
                V_new = np.zeros((m*p, r))
                for j in range(m):
                    for k in range(p):
                        A = np.zeros((n, r))
                        for i in range(n):
                            A[i, :] = U_r[i*m+j, :] * W_r[k*n+i, :]
                        b = np.ones(n)
                        try:
                            V_new[j*p+k, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                        except:
                            pass
                
                # Fix U and V, solve for W
                W_new = np.zeros((p*n, r))
                for k in range(p):
                    for i in range(n):
                        A = np.zeros((m, r))
                        for j in range(m):
                            A[j, :] = U_r[i*m+j, :] * V_r[j*p+k, :]
                        b = np.ones(m)
                        try:
                            W_new[k*n+i, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                        except:
                            pass
                
                # Check convergence
                U_r, V_r, W_r = U_new, V_new, W_new
                
                # Verify the decomposition
                error = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = sum(U_r[i*m+j, c] * V_r[j*p+k, c] * W_r[k*n+i, c] for c in range(r))
                            error += abs(val - 1.0)
                
                if error < 1e-10:
                    # Found a valid decomposition
                    best_rank = r
                    best_result = {'U': U_r.tolist(), 'V': V_r.tolist(), 'W': W_r.tolist()}
                    break
            
            if error < 1e-10:
                break
    
    return best_result
# EVOLVE-BLOCK-END
