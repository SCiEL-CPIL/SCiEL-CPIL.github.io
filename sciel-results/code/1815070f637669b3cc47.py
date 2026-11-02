"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x4 matrix by a 4x7 matrix.
    Output factor follows the official trace convention W[k*n+i, r].
    
    Strategy: Use a hybrid approach combining:
    1. Start with the schoolbook decomposition (rank = n*m*p = 84)
    2. Try to reduce rank via structured random initialization and optimization
    3. Use exact rational arithmetic to verify solutions
    4. Return the best (lowest rank) valid solution found within time budget
    
    The tensor T has T[i,j,k] = 1 if (i,j,k) corresponds to a valid matrix multiplication term.
    We want U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over r to equal T[i,j,k].
    """
    n, m, p = DIMENSIONS
    nm = n * m  # 12
    mp = m * p  # 28
    pn = p * n  # 21
    
    # Build the target tensor T[i,j,k] = 1 for all valid (i,j,k) in matrix mult
    # T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    # So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    # Time management
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s safety margin
    
    # Best solution found so far
    best_rank = n * m * p
    best_solution = None
    
    # Initialize with schoolbook
    U_school = np.zeros((nm, best_rank))
    V_school = np.zeros((mp, best_rank))
    W_school = np.zeros((pn, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1
                V_school[j*p + k, col] = 1
                W_school[k*n + i, col] = 1
                col += 1
    
    best_solution = (U_school, V_school, W_school)
    
    # Try to find lower rank decompositions
    # Strategy: Use random low-rank factorization and solve least squares
    # Then round to exact values and verify
    
    # We'll try ranks from a lower bound up to the schoolbook rank
    # Lower bound: max(n*m, m*p, p*n) = max(12, 28, 21) = 28
    # But actually the tensor rank can be lower due to structure
    
    # Let's try various ranks
    ranks_to_try = []
    # Start from known lower bounds and go up
    for r in range(28, best_rank + 1):
        if time.time() > deadline:
            break
        ranks_to_try.append(r)
    
    # Also try some random intermediate ranks
    random.seed(42)
    for r in range(30, min(60, best_rank)):
        if r not in ranks_to_try:
            ranks_to_try.append(r)
    
    ranks_to_try.sort()
    
    for rank in ranks_to_try:
        if time.time() > deadline:
            break
        if rank >= best_rank:
            break
        
        # Try multiple random initializations for this rank
        max_attempts = 50 if rank < 40 else 20
        if time.time() > deadline - 30:
            max_attempts = 5
        
        found = False
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Random initialization with small values
            U = np.random.randn(nm, rank) * 0.5
            V = np.random.randn(mp, rank) * 0.5
            W = np.random.randn(pn, rank) * 0.5
            
            # Scale to get reasonable magnitudes
            U /= np.linalg.norm(U, axis=0, keepdims=True) + 1e-10
            V /= np.linalg.norm(V, axis=0, keepdims=True) + 1e-10
            W /= np.linalg.norm(W, axis=0, keepdims=True) + 1e-10
            
            # Build the linear system
            # For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # This is nonlinear. Use alternating least squares.
            
            # ALS: fix two factors, solve for the third
            # Let's use a different approach: parameterize and use gradient descent
            
            # Actually, let's use a structured approach:
            # The tensor T[i,j,k] = 1 for all i,j,k
            # We need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            
            # This is a rank-r decomposition of the tensor.
            # Use iterative refinement:
            
            U_curr = U.copy()
            V_curr = V.copy()
            W_curr = W.copy()
            
            # Alternating least squares
            for iteration in range(100):
                if time.time() > deadline:
                    break
                
                # Update U: for each (i*m+j, r), we have
                # sum_r' U[i*m+j, r'] * V[j*p+k, r'] * W[k*n+i, r'] = 1
                # For fixed V and W, this is linear in U
                # Build matrix A where A[(i,j,k), (i*m+j, r)] = V[j*p+k, r] * W[k*n+i, r]
                # Then A @ U.flatten() = ones(n*m*p)
                
                # This is a large system, so let's do it more efficiently
                # For each column r, we can update one factor at a time
                
                # Update U: fix V and W
                # For each (i,j): sum_r U[i*m+j, r] * (V[j*p+k, r] * W[k*n+i, r]) = 1 for each k
                # This gives us for each (i,j): a linear system in U[i*m+j, :]
                
                # Let's build it properly
                # Reshape approach:
                # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                
                # For fixed V, W: solve for U
                # For each (i,j): sum_r U[i*m+j, r] * (V[j*p+k, r] * W[k*n+i, r]) = 1 for all k
                # This is: U[i*m+j, :] @ diag(V[j*p+k,:]*W[k*n+i,:]) ... no
                
                # Let's think differently. Define for each k:
                # B_k[i*m+j, r] = V[j*p+k, r] * W[k*n+i, r]
                # Then: U[i*m+j, :] @ B_k[i*m+j, :].T = 1 for all i,j
                # Wait, that's not quite right either.
                
                # Let me use a simpler ALS formulation:
                # Reshape T into a matrix: T_mat[ (i,j), (k, r) ]... no, that doesn't work directly.
                
                # Standard ALS for CP decomposition:
                # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                # Note: U depends on (i,j), V on (j,k), W on (k,i)
                
                # For U: fix V and W
                # For each (i,j): sum_r U[i*m+j, r] * (V[j*p+k, r] * W[k*n+i, r]) = 1 for each k
                # Define M_k[i*m+j, r] = V[j*p+k, r] * W[k*n+i, r]
                # Then U[i*m+j, :] @ M_k[i*m+j, :].T = 1
                # But M_k depends on i and j through W and V respectively, so this isn't a standard matrix.
                
                # Let's just do direct least squares per (i,j) pair
                # For fixed (i,j), we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for k=0..p-1
                # This is a system of p equations in rank unknowns (U[i*m+j, 0..rank-1])
                
                # Build the system
                # A[(k), r] = V[j*p+k, r] * W[k*n+i, r]
                # b[k] = 1
                # Solve A @ u = b where u = U[i*m+j, :]
                
                for i in range(n):
                    for j in range(m):
                        # Build A matrix: p x rank
                        A = np.zeros((p, rank))
                        for k in range(p):
                            A[k, :] = V_curr[j*p + k, :] * W_curr[k*n + i, :]
                        b = np.ones(p)
                        # Solve least squares
                        try:
                            u_sol, residuals, rank_val, sv = np.linalg.lstsq(A, b, rcond=None)
                            U_curr[i*m + j, :] = u_sol
                        except:
                            pass
                
                # Update V: fix U and W
                # For each (j,k): sum_r V[j*p+k, r] * U[i*m+j, r] * W[k*n+i, r] = 1 for each i
                # A[(i), r] = U[i*m+j, r] * W[k*n+i, r]
                for j in range(m):
                    for k in range(p):
                        A = np.zeros((n, rank))
                        for i in range(n):
                            A[i, :] = U_curr[i*m + j, :] * W_curr[k*n + i, :]
                        b = np.ones(n)
                        try:
                            v_sol, residuals, rank_val, sv = np.linalg.lstsq(A, b, rcond=None)
                            V_curr[j*p + k, :] = v_sol
                        except:
                            pass
                
                # Update W: fix U and V
                # For each (k,i): sum_r W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r] = 1 for each j
                # A[(j), r] = U[i*m+j, r] * V[j*p+k, r]
                for k in range(p):
                    for i in range(n):
                        A = np.zeros((m, rank))
                        for j in range(m):
                            A[j, :] = U_curr[i*m + j, :] * V_curr[j*p + k, :]
                        b = np.ones(m)
                        try:
                            w_sol, residuals, rank_val, sv = np.linalg.lstsq(A, b, rcond=None)
                            W_curr[k*n + i, :] = w_sol
                        except:
                            pass
            
            # Check if we have a good solution
            # Compute the error
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U_curr[i*m + j, :] * V_curr[j*p + k, :] * W_curr[k*n + i, :])
                        error += (val - 1.0) ** 2
            error = np.sqrt(error / (n*m*p))
            
            if error < 1e-6:
                # Try to round to exact values (integers or half-integers)
                U_rounded = np.round(U_curr * 2) / 2
                V_rounded = np.round(V_curr * 2) / 2
                W_rounded = np.round(W_curr * 2) / 2
                
                # Check if rounded solution is exact
                err_rounded = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U_rounded[i*m + j, :] * V_rounded[j*p + k, :] * W_rounded[k*n + i, :])
                            err_rounded += (val - 1.0) ** 2
                err_rounded = np.sqrt(err_rounded / (n*m*p))
                
                if err_rounded < 1e-10:
                    # Valid solution!
                    # Convert to lists
                    U_list = [[float(x) for x in U_rounded[i]] for i in range(nm)]
                    V_list = [[float(x) for x in V_rounded[i]] for i in range(mp)]
                    W_list = [[float(x) for x in W_rounded[i]] for i in range(pn)]
                    
                    best_rank = rank
                    best_solution = (U_rounded, V_rounded, W_rounded)
                    found = True
                    break
            
            # Also try integer rounding
            U_int = np.round(U_curr)
            V_int = np.round(V_curr)
            W_int = np.round(W_curr)
            
            err_int = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U_int[i*m + j, :] * V_int[j*p + k, :] * W_int[k*n + i, :])
                        err_int += (val - 1.0) ** 2
            err_int = np.sqrt(err_int / (n*m*p))
            
            if err_int < 1e-10:
                U_list = [[float(x) for x in U_int[i]] for i in range(nm)]
                V_list = [[float(x) for x in V_int[i]] for i in range(mp)]
                W_list = [[float(x) for x in W_int[i]] for i in range(pn)]
                
                best_rank = rank
                best_solution = (U_int, V_int, W_int)
                found = True
                break
        
        if found:
            # Continue searching for even lower rank
            continue
    
    # If we found a better solution, return it
    if best_solution is not None and best_rank < n * m * p:
        U_arr, V_arr, W_arr = best_solution
        U_list = [[float(x) for x in U_arr[i]] for i in range(nm)]
        V_list = [[float(x) for x in V_arr[i]] for i in range(mp)]
        W_list = [[float(x) for x in W_arr[i]] for i in range(pn)]
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Return schoolbook solution
    U_list = [[float(x) for x in U_school[i]] for i in range(nm)]
    V_list = [[float(x) for x in V_school[i]] for i in range(mp)]
    W_list = [[float(x) for x in W_school[i]] for i in range(pn)]
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
