"""Matrix multiplication 3x3 by 3x7 - rank minimization."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 7]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x7 matrix.
    
    The multiplication tensor T has indices (i, j, k) where:
    - i in [0, n) = [0, 3) : row of first matrix
    - j in [0, m) = [0, 3) : column of first matrix / row of second matrix  
    - k in [0, p) = [0, 7) : column of second matrix
    
    T[i,j,k] = 1 if the product C[i,k] = sum_j A[i,j]*B[j,k]
    
    The decomposition is: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    
    Interface: U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    With W following the trace convention W[k*n+i, r]
    """
    start_time = time.time()
    deadline = 140.0  # Leave safety margin
    
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition as baseline
    rank_schoolbook = n * m * p  # 63
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U = [row[:] for row in U_school]
    best_V = [row[:] for row in V_school]
    best_W = [row[:] for row in W_school]
    
    # The tensor T[i,j,k] = 1 for all i,j,k (it's the full multiplication tensor)
    # We need to find rank-r decomposition where:
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # For small dimensions, we can try to find low-rank decompositions
    # Key insight: the multiplication tensor for n x m by m x p has a known lower bound
    # and various constructions exist
    
    # Let's try a systematic search for lower rank decompositions
    # The tensor is T[i,j,k] = 1 for all i,j,k in their respective ranges
    
    # One approach: use the fact that we can factor through intermediate indices
    # Another: try random initialization with optimization
    
    # For 3x3 x 3x7, let's try to find decompositions of rank < 63
    
    # Try structured decompositions first
    
    # Strategy 1: Try rank r from lower bounds up to 62
    # Lower bound: max(n*m, m*p, p*n) = max(9, 21, 21) = 21
    # But this is a very weak bound. Better bounds exist.
    
    # Let's try a more sophisticated approach:
    # The tensor can be seen as a 3x3x7 array of ones.
    # We want to write this as sum of r rank-1 tensors.
    
    # A rank-1 tensor has the form u_i * v_j * w_k for fixed c
    # So T[i,j,k] = sum_c u_c[i] * v_c[j] * w_c[k]
    
    # This is equivalent to finding matrices U (n x r), V (m x r), W (p x r)
    # such that for all i,j,k: sum_c U[i,c]*V[j,c]*W[k,c] = 1
    
    # Let's try to solve this systematically
    
    def verify_decomposition(U_mat, V_mat, W_mat):
        """Verify that the decomposition is exact."""
        n, m, p = DIMENSIONS
        # Check all entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U_mat[i, c] * V_mat[j, c] * W_mat[k, c] for c in range(U_mat.shape[1]))
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    def convert_to_interface(U_mat, V_mat, W_mat):
        """Convert from (n x r), (m x r), (p x r) to interface format."""
        n, m, p = DIMENSIONS
        r = U_mat.shape[1]
        U_list = [[0] * r for _ in range(n * m)]
        V_list = [[0] * r for _ in range(m * p)]
        W_list = [[0] * r for _ in range(p * n)]
        
        for i in range(n):
            for j in range(m):
                for c in range(r):
                    U_list[i * m + j][c] = U_mat[i, c]
        
        for j in range(m):
            for k in range(p):
                for c in range(r):
                    V_list[j * p + k][c] = V_mat[j, c]
        
        for k in range(p):
            for i in range(n):
                for c in range(r):
                    W_list[k * n + i][c] = W_mat[k, c]
        
        return U_list, V_list, W_list
    
    # Try to find low-rank decompositions using numerical optimization
    # Start with a reasonable lower bound and work up
    
    # The Strassen-like lower bound for 3x3x3 is 19 or 20
    # For 3x3x7, we can use block structures
    
    # Let's try a direct optimization approach
    from scipy.optimize import minimize
    
    def objective(params, r):
        """Objective function for optimization."""
        n, m, p = DIMENSIONS
        U_flat = params[:n*r].reshape(n, r)
        V_flat = params[n*r:n*r+m*r].reshape(m, r)
        W_flat = params[n*r+m*r:].reshape(p, r)
        
        # Compute tensor product sum
        # T[i,j,k] = sum_c U[i,c]*V[j,c]*W[k,c]
        errors = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_flat[i], V_flat[j] * W_flat[k])
                    errors += (val - 1.0) ** 2
        return errors
    
    # Try different ranks starting from a reasonable lower bound
    # For 3x3x7, let's try ranks from 20 to 62
    
    # Actually, let's be smarter. The tensor is all ones, so we can use
    # a simpler construction.
    
    # Simple construction: use the fact that 1 = sum of carefully chosen terms
    # For example, if we use r terms where each term is a product of three vectors
    
    # Let's try a specific construction:
    # If we can write 1 as sum_c a_c[i] * b_c[j] * c_c[k]
    
    # One simple approach: use the identity 1 = 1*1*1, but we need multiple terms
    # to get the right structure.
    
    # Actually, the schoolbook method uses n*m*p terms where each term is
    # a "delta" tensor. But we can do better by using linear combinations.
    
    # Let's try to find a decomposition using linear algebra
    # The key equation is: for all i,j,k: sum_c U[i,c]*V[j,c]*W[k,c] = 1
    
    # This is a system of n*m*p = 63 equations in 3*r unknowns (for U, V, W)
    # But it's nonlinear.
    
    # Let's try a different approach: fix W and solve for U, V
    # Or use iterative methods
    
    # For now, let's try to find a good decomposition by search
    
    # Try rank values and see if we can find exact decompositions
    # Use a combination of random initialization and local optimization
    
    np.random.seed(42)
    
    best_score = float('inf')
    
    for r in range(20, 63):  # Try ranks from 20 to 62
        if time.time() > deadline - 10:
            break
        
        # Try multiple random initializations
        n_init = 5 if r < 40 else 3
        for init in range(n_init):
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            U_init = np.random.randn(n, r) * 0.5 + 0.5
            V_init = np.random.randn(m, r) * 0.5 + 0.5
            W_init = np.random.randn(p, r) * 0.5 + 0.5
            
            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            # Try to optimize
            try:
                result = minimize(objective, params_init, args=(r,), method='Nelder-Mead',
                                  options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10})
                
                if result.fun < 1e-8:
                    # Found a good solution, verify it
                    U_mat = result.x[:n*r].reshape(n, r)
                    V_mat = result.x[n*r:n*r+m*r].reshape(m, r)
                    W_mat = result.x[n*r+m*r:].reshape(p, r)
                    
                    if verify_decomposition(U_mat, V_mat, W_mat):
                        # Convert to interface format
                        U_list, V_list, W_list = convert_to_interface(U_mat, V_mat, W_mat)
                        best_rank = r
                        best_U = U_list
                        best_V = V_list
                        best_W = W_list
                        break  # Found rank r, try lower ranks
            except Exception:
                pass
        
        if best_rank <= r:
            break
    
    # Return the best decomposition found
    return {"U": best_U, "V": best_V, "W": best_W}
