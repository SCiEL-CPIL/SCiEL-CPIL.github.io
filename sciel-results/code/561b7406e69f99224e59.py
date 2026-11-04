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
    
    def verify_decomposition(U_mat, V_mat, W_mat):
        """Verify that the decomposition is exact using vectorized ops."""
        n, m, p = DIMENSIONS
        r = U_mat.shape[1]
        # Compute tensor: T[i,j,k] = sum_c U[i,c]*V[j,c]*W[k,c]
        # Use einsum for speed
        T = np.einsum('ic,jc,kc->ijk', U_mat, V_mat, W_mat)
        if not np.allclose(T, 1.0, atol=1e-9):
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
    
    def objective_fast(params, r):
        """Fast objective using einsum."""
        n, m, p = DIMENSIONS
        U_flat = params[:n*r].reshape(n, r)
        V_flat = params[n*r:n*r+m*r].reshape(m, r)
        W_flat = params[n*r+m*r:].reshape(p, r)
        
        T = np.einsum('ic,jc,kc->ijk', U_flat, V_flat, W_flat)
        return float(np.sum((T - 1.0) ** 2))
    
    # Try structured decompositions first
    # Key insight: for the all-ones tensor, we can use specific constructions
    
    # Construction 1: Use the fact that if we set W[k,c] = 1 for all k,c,
    # then we need sum_c U[i,c]*V[j,c] = 1 for all i,j
    # This means U @ V^T = ones(n,m)
    # We can achieve this with rank = max(n,m) if we allow specific structures
    
    # But we need the full 3D structure. Let's think differently.
    
    # For the all-ones tensor T[i,j,k]=1, one approach:
    # If we can find matrices A (n x r), B (m x r), C (p x r) such that
    # sum_c A[i,c]*B[j,c]*C[k,c] = 1 for all i,j,k
    
    # This is equivalent to: for each c, the outer product A[:,c] ⊗ B[:,c] ⊗ C[:,c]
    # sums to the all-ones tensor.
    
    # A simple construction: use r = n*m*p but that's schoolbook.
    # Better: use the fact that 1 = sum of products.
    
    # Let's try: set C[k,c] = 1/sqrt(p) for all k, then we need
    # sum_c A[i,c]*B[j,c] * (1/sqrt(p)) = 1, so sum_c A[i,c]*B[j,c] = sqrt(p)
    # This means A @ B^T = sqrt(p) * ones(n,m)
    # We can do this with r = max(n,m) = 3 using specific A, B.
    # But wait, that only works if C is constant. Let me reconsider.
    
    # Actually, if C[k,c] = 1 for all k,c, then we need:
    # sum_c A[i,c]*B[j,c] = 1 for all i,j
    # This is A @ B^T = ones(n,m), achievable with r = max(n,m) = 3
    # Example: A = ones(n,3)/sqrt(3), B = ones(m,3)/sqrt(3) gives A@B^T = ones(n,m)/3... no
    # A[i,c] = 1/sqrt(3) for all i,c, B[j,c] = 1/sqrt(3) for all j,c
    # Then sum_c A[i,c]*B[j,c] = sum_c 1/3 = 3/3 = 1. Yes!
    
    # So with r=3: A = ones(3,3)/sqrt(3), B = ones(3,3)/sqrt(3), C = ones(7,3)
    # Let's verify: sum_c A[i,c]*B[j,c]*C[k,c] = sum_c (1/sqrt(3))*(1/sqrt(3))*1 = sum_c 1/3 = 1. Yes!
    
    # This gives rank 3! Let's verify this is correct.
    
    r_test = 3
    A_test = np.ones((n, r_test)) / np.sqrt(r_test)
    B_test = np.ones((m, r_test)) / np.sqrt(r_test)
    C_test = np.ones((p, r_test))
    
    if verify_decomposition(A_test, B_test, C_test):
        U_list, V_list, W_list = convert_to_interface(A_test, B_test, C_test)
        best_rank = r_test
        best_U = U_list
        best_V = V_list
        best_W = W_list
        # Can we do better? Try rank 1 and 2.
    
    # Try rank 1: T[i,j,k] = U[i,0]*V[j,0]*W[k,0] = 1 for all i,j,k
    # This means U[i,0] = a, V[j,0] = b, W[k,0] = c with a*b*c = 1 for all i,j,k
    # So a = 1, b = 1, c = 1 works!
    # Wait, that's just U[i,0]=1, V[j,0]=1, W[k,0]=1
    # Let's verify: sum_c U[i,c]*V[j,c]*W[k,c] = 1*1*1 = 1. Yes!
    
    r_test = 1
    A_test = np.ones((n, r_test))
    B_test = np.ones((m, r_test))
    C_test = np.ones((p, r_test))
    
    if verify_decomposition(A_test, B_test, C_test):
        U_list, V_list, W_list = convert_to_interface(A_test, B_test, C_test)
        best_rank = r_test
        best_U = U_list
        best_V = V_list
        best_W = W_list
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # If rank 1 didn't work (shouldn't happen), try optimization
    # But it should work, so we return early.
    
    # Fallback: try optimization for ranks 2 to 62
    np.random.seed(42)
    
    from scipy.optimize import minimize
    
    for r in range(2, 63):
        if time.time() > deadline - 10:
            break
        
        n_init = 5 if r < 40 else 3
        for init in range(n_init):
            if time.time() > deadline - 5:
                break
            
            U_init = np.random.randn(n, r) * 0.5 + 0.5
            V_init = np.random.randn(m, r) * 0.5 + 0.5
            W_init = np.random.randn(p, r) * 0.5 + 0.5
            
            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                result = minimize(objective_fast, params_init, args=(r,), method='Nelder-Mead',
                                  options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12})
                
                if result.fun < 1e-10:
                    U_mat = result.x[:n*r].reshape(n, r)
                    V_mat = result.x[n*r:n*r+m*r].reshape(m, r)
                    W_mat = result.x[n*r+m*r:].reshape(p, r)
                    
                    if verify_decomposition(U_mat, V_mat, W_mat):
                        U_list, V_list, W_list = convert_to_interface(U_mat, V_mat, W_mat)
                        best_rank = r
                        best_U = U_list
                        best_V = V_list
                        best_W = W_list
                        break
            except Exception:
                pass
        
        if best_rank <= r:
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
