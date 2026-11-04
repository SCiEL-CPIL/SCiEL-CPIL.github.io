"""Matrix multiplication tensor decomposition for 4x5 * 5x6."""
import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [4, 5, 6]

def construct():
    """Find a rank-minimal exact tensor decomposition for 4x5 * 5x6 matrix multiplication.
    
    Returns:
        dict with keys 'U', 'V', 'W' containing factor matrices.
        U has shape (n*m, rank) = (20, rank)
        V has shape (m*p, rank) = (30, rank)
        W has shape (p*n, rank) = (24, rank)
        The entries are integers or half-integers (multiples of 1/2).
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Leave 10s margin
    
    # Schoolbook decomposition (rank = n*m*p = 120)
    rank_school = n * m * p
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    best_result = {"U": U, "V": V, "W": W}
    
    # Try to find a lower rank decomposition using numerical optimization
    # The tensor T has entries T[i,j,k] = 1 if the (i,k) entry of A*B equals A[i,j]*B[j,k]
    # We want to write T = sum_c U[:,c] * V[:,c] * W[:,c] (outer product sum)
    
    # Try structured decompositions with various ranks
    # Start with some promising ranks below 120
    
    # For 4x5x6, known bounds suggest rank might be reducible
    # Try to use a block-structured approach or other algebraic tricks
    
    # First, let's verify the schoolbook works
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Verify: for all i,j,k, sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    # This is the matrix multiplication tensor
    
    # Try a few different approaches to reduce rank:
    
    # Approach 1: Use the fact that we can factor through smaller intermediate spaces
    # C = A * B where A is 4x5, B is 5x6
    # We can think of this as a bilinear map
    
    # Approach 2: Try to find a decomposition with rank < 120 using optimization
    # This is a hard problem in general, but for small dimensions we can try
    
    # Let's try a random initialization and gradient descent approach
    # But first, let's see if there's a known structured decomposition
    
    # For 4x5x6, one approach:
    # Note that 4x5x6 = 4*5*6 = 120
    # Can we do better?
    
    # Try: decompose using the structure of the dimensions
    # 4 = 2*2, 5 = 5, 6 = 2*3
    # Maybe use some kind of Kronecker product structure?
    
    # Actually, let's try a direct numerical search for lower rank
    # We'll use a heuristic: try to find a rank-r decomposition by 
    # solving a system of equations
    
    # The constraint is:
    # For each (i,j,k), sum_{c=1}^{r} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # This is a system of n*m*p = 120 equations in r*(n*m + m*p + p*n) = r*74 unknowns
    # For r < 120, we need r*74 >= 120, so r >= ceil(120/74) = 2 (very loose bound)
    # But the equations are nonlinear (trilinear), so this is much harder
    
    # Let's try a few specific ranks and see if we can find solutions
    # We'll use a simple iterative method
    
    def try_decomposition(rank, max_iters=10000, tol=1e-6):
        """Try to find a rank decomposition using a simple iterative method."""
        if time.time() > deadline:
            return None
        
        # Initialize randomly
        np.random.seed(42)
        U_init = np.random.randn(n * m, rank)
        V_init = np.random.randn(m * p, rank)
        W_init = np.random.randn(p * n, rank)
        
        # Normalize
        for c in range(rank):
            if abs(U_init[:, c]).max() > 0:
                U_init[:, c] /= np.sqrt(np.sum(U_init[:, c]**2))
            if abs(V_init[:, c]).max() > 0:
                V_init[:, c] /= np.sqrt(np.sum(V_init[:, c]**2))
            if abs(W_init[:, c]).max() > 0:
                W_init[:, c] /= np.sqrt(np.sum(W_init[:, c]**2))
        
        # Use a simple coordinate descent approach
        U_curr = U_init.copy()
        V_curr = V_init.copy()
        W_curr = W_init.copy()
        
        # Precompute indices
        idx_i = np.array([i for i in range(n) for _ in range(m*p)])
        idx_j = np.array([j for i in range(n) for j in range(m) for _ in range(p)])
        idx_k = np.array([k for i in range(n) for j in range(m) for k in range(p)])
        
        # For each (i,j,k), we need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        # Fix U and V, solve for W:
        # For each (i,k), sum_j [A[i,j] * B[j,k]] = sum_c [sum_j U[i*m+j,c] * V[j*p+k,c]] * W[k*n+i,c]
        # Let M[i,k,c] = sum_j U[i*m+j,c] * V[j*p+k,c]
        # Then we need: sum_c M[i,k,c] * W[k*n+i,c] = 1 for all (i,k)
        # This is: sum_c M[i,k,c] * W[k*n+i,c] = 1
        
        best_error = np.inf
        best_solution = None
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Update U: fix V, W, solve for U
            # For each (i,j): sum_k [A[i,j]*B[j,k]*C[i,k]] = sum_c U[i*m+j,c] * [sum_k V[j*p+k,c]*W[k*n+i,c]]
            # Let N[j,k,c] = V[j*p+k,c]*W[k*n+i,c] -- but this depends on i too
            # Actually: for fixed (i,j), sum_k C[i,k] * B[j,k] = sum_c U[i*m+j,c] * sum_k V[j*p+k,c]*W[k*n+i,c]
            # Define P[i,j,c] = sum_k V[j*p+k,c]*W[k*n+i,c]
            # Then: sum_c U[i*m+j,c] * P[i,j,c] = sum_k B[j,k] * C[i,k]
            # But we don't have B and C, we just need the tensor equation
            
            # Let me re-think. The tensor equation is:
            # For each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
            
            # Fix V and W. For each i,j:
            # sum_k [sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]] = 1  ... no, that's not right
            # The equation is for each (i,j,k) separately, not summed over k
            
            # So if we fix V and W, the equations for U are:
            # For each (i,j,k): U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] summed over c = 1
            # This is not linear in U because V and W are fixed but the equation is trilinear
            
            # Actually, if we fix V and W, then for each (i,j,k):
            # sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c]) = 1
            # Let Q[i,j,k,c] = V[j*p+k,c] * W[k*n+i,c]
            # Then: sum_c U[i*m+j,c] * Q[i,j,k,c] = 1
            # This is linear in U! For each (i,j), we have p equations:
            # For k=0,...,p-1: sum_c U[i*m+j,c] * Q[i,j,k,c] = 1
            # So we have a system: M_i_j * U[i*m+j,:]^T = 1
            # where M_i_j is a p x rank matrix with entries M_i_j[k,c] = Q[i,j,k,c]
            
            # Similarly for V and W
            
            # Let's implement this coordinate descent
            
            # Update U
            U_new = U_curr.copy()
            for i in range(n):
                for j in range(m):
                    # Build system: for k=0..p-1, sum_c U_new[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                    A_mat = np.zeros((p, rank))
                    b_vec = np.ones(p)
                    for k in range(p):
                        for c in range(rank):
                            A_mat[k, c] = V_curr[j * p + k, c] * W_curr[k * n + i, c]
                    # Solve A_mat * x = b_vec for x = U_new[i*m+j,:]
                    try:
                        x, residuals, rank_sv, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                        U_new[i * m + j, :] = x
                    except:
                        pass
            
            # Update V
            V_new = V_curr.copy()
            for j in range(m):
                for k in range(p):
                    # Build system: for i=0..n-1, sum_c U[i*m+j,c] * V_new[j*p+k,c] * W[k*n+i,c] = 1
                    A_mat = np.zeros((n, rank))
                    b_vec = np.ones(n)
                    for i in range(n):
                        for c in range(rank):
                            A_mat[i, c] = U_new[i * m + j, c] * W_curr[k * n + i, c]
                    try:
                        x, residuals, rank_sv, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                        V_new[j * p + k, :] = x
                    except:
                        pass
            
            # Update W
            W_new = W_curr.copy()
            for i in range(n):
                for k in range(p):
                    # Build system: for j=0..m-1, sum_c U[i*m+j,c] * V[j*p+k,c] * W_new[k*n+i,c] = 1
                    A_mat = np.zeros((m, rank))
                    b_vec = np.ones(m)
                    for j in range(m):
                        for c in range(rank):
                            A_mat[j, c] = U_new[i * m + j, c] * V_new[j * p + k, c]
                    try:
                        x, residuals, rank_sv, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                        W_new[k * n + i, :] = x
                    except:
                        pass
            
            # Compute error
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for c in range(rank):
                            val += U_new[i * m + j, c] * V_new[j * p + k, c] * W_new[k * n + i, c]
                        error += (val - 1.0) ** 2
            error = np.sqrt(error)
            
            if error < best_error:
                best_error = error
                best_solution = (U_new.copy(), V_new.copy(), W_new.copy())
            
            if error < tol:
                break
            
            U_curr = U_new
            V_curr = V_new
            W_curr = W_new
        
        if best_error < tol:
            U_sol, V_sol, W_sol = best_solution
            # Round to nearest 1/2
            U_rounded = np.round(U_sol * 2) / 2
            V_rounded = np.round(V_sol * 2) / 2
            W_rounded = np.round(W_sol * 2) / 2
            
            # Verify the rounded solution
            verify_error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for c in range(rank):
                            val += U_rounded[i * m + j, c] * V_rounded[j * p + k, c] * W_rounded[k * n + i, c]
                        verify_error += (val - 1.0) ** 2
            verify_error = np.sqrt(verify_error)
            
            if verify_error < 1e-9:
                return {
                    "U": U_rounded.tolist(),
                    "V": V_rounded.tolist(),
                    "W": W_rounded.tolist()
                }
        
        return None
    
    # Try ranks from 120 down to some minimum
    # Start with a few promising values
    for rank in range(115, 50, -1):
        if time.time() > deadline:
            break
        
        result = try_decomposition(rank, max_iters=5000, tol=1e-6)
        if result is not None:
            best_result = result
            break
    
    # Also try some specific structured decompositions
    # For 4x5x6, try rank 100, 96, 90, etc.
    if time.time() < deadline:
        for rank in [100, 96, 90, 84, 80, 75, 72, 70, 68, 66, 64, 60, 56, 54, 52, 50, 48, 45, 42, 40, 36, 35, 32, 30, 28, 25, 24, 20, 18, 16, 15, 14, 12, 10, 9, 8, 7, 6, 5, 4, 3, 2]:
            if time.time() > deadline:
                break
            if rank >= 115:
                continue
            result = try_decomposition(rank, max_iters=2000, tol=1e-6)
            if result is not None:
                best_result = result
                break
    
    return best_result
