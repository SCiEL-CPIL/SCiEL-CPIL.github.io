"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    # Try to find a low-rank decomposition
    # The tensor T[i,j,k] = 1 for multiplication (A[i,j]*B[j,k] -> C[i,k])
    # We need U (nm x r), V (mp x r), W (pn x r) such that:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all (i,j,k)
    
    # Known: schoolbook rank = n*m*p = 72
    # Try to find better
    
    # Let's try a structured approach based on Strassen-like ideas
    # For 2x2 x 2x2, Strassen uses 7 multiplications
    # We can try to build up from that
    
    # Approach: try various ranks and use random initialization + optimization
    best_result = None
    best_rank = n * m * p
    
    # First, return schoolbook as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank
    
    U_sch, V_sch, W_sch, rank_sch = schoolbook()
    best_result = (U_sch, V_sch, W_sch, rank_sch)
    
    # Try to find better using random search + local optimization
    # We'll try ranks from low to high
    
    def try_rank(r, max_iters=2000, seed=42):
        """Try to find a decomposition of rank r using iterative refinement."""
        if time.time() > deadline - 5:
            return None
        
        rng = np.random.RandomState(seed)
        
        # Initialize with random values
        U = rng.randn(n * m, r) * 0.1
        V = rng.randn(m * p, r) * 0.1
        W = rng.randn(p * n, r) * 0.1
        
        # Build the constraint matrix
        # For each (i,j,k), we need sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        # This is a system of nm*p equations
        
        # Use a simple gradient-like approach
        # Actually, let's use a different approach: fix two matrices and solve for the third
        
        # Better approach: use the fact that if we fix U and V, we can solve for W
        # For each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        # Group by (k,i): for each (k,i), we have m equations (one for each j)
        # sum_j' U[i*m+j',c] * V[j'*p+k,c] * W[k*n+i,c] = 1 for all j
        # Wait, that's not right. Let me re-derive.
        
        # Actually, for fixed (i,k), the equation is:
        # sum_j sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        # This is: sum_c (sum_j U[i*m+j,c] * V[j*p+k,c]) * W[k*n+i,c] = 1
        
        # So if we define M[i,k,c] = sum_j U[i*m+j,c] * V[j*p+k,c], then:
        # sum_c M[i,k,c] * W[k*n+i,c] = 1 for all (i,k)
        
        # This is a linear system in W!
        # Let's try alternating minimization
        
        best_err = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for iteration in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Fix U and V, solve for W
            # M[i,k,c] = sum_j U[i*m+j,c] * V[j*p+k,c]
            M = np.zeros((n, p, r))
            for i in range(n):
                for k in range(p):
                    for c in range(r):
                        M[i, k, c] = sum(U[i * m + j, c] * V[j * p + k, c] for j in range(m))
            
            # Now solve: for each (i,k), sum_c M[i,k,c] * W[k*n+i,c] = 1
            # This is a linear system. Let's set up the full system.
            # Number of equations: n*p, number of unknowns: p*n*r
            # We have more unknowns than equations, so we need to regularize or use least squares
            
            # Actually, we can solve this as a least squares problem
            # Build the design matrix
            A_mat = np.zeros((n * p, p * n * r))
            for i in range(n):
                for k in range(p):
                    row = k * n + i  # equation index... wait, let me be careful
                    # Equation for (i,k): sum_c M[i,k,c] * W[k*n+i,c] = 1
                    # W is indexed by (k,i) -> index k*n+i
                    for c in range(r):
                        A_mat[k * n + i, (k * n + i) * r + c] = M[i, k, c]
            
            # Solve least squares
            try:
                W_flat, residuals, rank_lsq, sv = np.linalg.lstsq(A_mat, np.ones(n * p), rcond=None)
                W_new = W_flat.reshape(p * n, r)
            except:
                W_new = W.copy()
            
            # Fix V and W, solve for U
            # For fixed (i,k): sum_c (sum_j U[i*m+j,c] * V[j*p+k,c]) * W[k*n+i,c] = 1
            # For fixed (i,j): this doesn't directly give us U[i*m+j]
            # Let me think differently.
            
            # Alternative: fix V and W, solve for U
            # The constraint is: for all (i,j,k), sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
            # For fixed (i,j): sum_c U[i*m+j,c] * (sum_k V[j*p+k,c] * W[k*n+i,c]) = 1
            # Define N[i,j,c] = sum_k V[j*p+k,c] * W[k*n+i,c]
            # Then: sum_c U[i*m+j,c] * N[i,j,c] = 1 for all (i,j)
            
            N = np.zeros((n, m, r))
            for i in range(n):
                for j in range(m):
                    for c in range(r):
                        N[i, j, c] = sum(V[j * p + k, c] * W[k * n + i, c] for k in range(p))
            
            A_mat_U = np.zeros((n * m, n * m * r))
            for i in range(n):
                for j in range(m):
                    row = i * m + j
                    for c in range(r):
                        A_mat_U[row, row * r + c] = N[i, j, c]
            
            try:
                U_flat, _, _, _ = np.linalg.lstsq(A_mat_U, np.ones(n * m), rcond=None)
                U_new = U_flat.reshape(n * m, r)
            except:
                U_new = U.copy()
            
            # Fix U and W, solve for V
            # For fixed (j,k): sum_c V[j*p+k,c] * (sum_i U[i*m+j,c] * W[k*n+i,c]) = 1
            # Define P[j,k,c] = sum_i U[i*m+j,c] * W[k*n+i,c]
            # Then: sum_c V[j*p+k,c] * P[j,k,c] = 1 for all (j,k)
            
            P = np.zeros((m, p, r))
            for j in range(m):
                for k in range(p):
                    for c in range(r):
                        P[j, k, c] = sum(U[i * m + j, c] * W[k * n + i, c] for i in range(n))
            
            A_mat_V = np.zeros((m * p, m * p * r))
            for j in range(m):
                for k in range(p):
                    row = j * p + k
                    for c in range(r):
                        A_mat_V[row, row * r + c] = P[j, k, c]
            
            try:
                V_flat, _, _, _ = np.linalg.lstsq(A_mat_V, np.ones(m * p), rcond=None)
                V_new = V_flat.reshape(m * p, r)
            except:
                V_new = V.copy()
            
            # Update
            U, V, W = U_new, V_new, W_new
            
            # Compute error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = sum(U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c] for c in range(r))
                        err += (val - 1) ** 2
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if err < 1e-10:
                break
        
        if best_err < 1e-6:
            return best_U, best_V, best_W, r, best_err
        return None
    
    # Try ranks from low to high
    # Start with a reasonable guess
    for r in range(40, min(72, n * m * p + 1)):
        if time.time() > deadline - 10:
            break
        
        result = try_rank(r, max_iters=500, seed=r)
        if result is not None:
            U_try, V_try, W_try, r_try, err = result
            if err < 1e-6:
                # Verify the solution
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = sum(U_try[i * m + j, c] * V_try[j * p + k, c] * W_try[k * n + i, c] for c in range(r_try))
                            if abs(val - 1) > 1e-5:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    best_result = (U_try, V_try, W_try, r_try)
                    break
    
    # Convert to lists and return
    U_final, V_final, W_final, rank_final = best_result
    
    # Round to nearest 0.5 if close
    def round_to_half(x):
        rounded = np.round(x * 2) / 2
        if np.allclose(x, rounded, atol=1e-4):
            return rounded
        return x
    
    U_final = round_to_half(U_final)
    V_final = round_to_half(V_final)
    W_final = round_to_half(W_final)
    
    # Verify after rounding
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U_final[i * m + j, c] * V_final[j * p + k, c] * W_final[k * n + i, c] for c in range(rank_final))
                if abs(val - 1) > 1e-3:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fall back to schoolbook
        U_final, V_final, W_final, rank_final = U_sch, V_sch, W_sch, rank_sch
    
    # Convert to list of lists
    U_list = [[float(U_final[i, c]) for c in range(rank_final)] for i in range(n * m)]
    V_list = [[float(V_final[i, c]) for c in range(rank_final)] for i in range(m * p)]
    W_list = [[float(W_final[i, c]) for c in range(rank_final)] for i in range(p * n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
