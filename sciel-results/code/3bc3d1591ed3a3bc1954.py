"""Matrix multiplication 4x4 x 4x6 - rank-minimal decomposition."""
import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """Find rank-minimal exact tensor decomposition for 4x4 x 4x6 matrix mult."""
    n, m, p = 4, 4, 6
    start_time = time.time()
    deadline = start_time + 140
    
    # The tensor: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    # U: (n*m, rank) = (16, rank)
    # V: (m*p, rank) = (24, rank)  
    # W: (p*n, rank) = (24, rank)
    
    # Let's try to find a structured solution.
    # One approach: use the fact that for 2x2x2, Strassen gives rank 7.
    # We can try to compose or extend Strassen's construction.
    
    # For now, let's try a direct search with a good initialization.
    # Use a structured initialization based on block structure.
    
    def verify(U, V, W, tol=1e-9):
        """Verify that the decomposition is exact."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for r in range(rank):
                        val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # Try to find a solution using a structured ansatz.
    # Let's try rank = 24 first (which is m*p = p*n), and see if we can do better.
    
    # Actually, let's try a simple construction:
    # For each (i,k), we need sum_j sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    # This is like saying the matrix product (U_i * V_k^T * W_k) has all entries summing to 1 in a certain way.
    
    # Let me try a different approach: use the fact that if we set W[k*n+i, r] = 1 for all r,
    # then we need sum_r U[i*m+j,r]*V[j*p+k,r] = 1 for all i,j,k.
    # This means U_i * V_k^T = J (all-ones matrix) for each i,k.
    # But U_i is a 1 x rank vector and V_k is a 1 x rank vector, so U_i * V_k^T is a 1x1 scalar.
    # So we need U[i*m+j, r] * V[j*p+k, r] summed over r to equal 1 for all j, for each (i,k).
    # This means for each (i,k), the vector (sum_r U[i*m+j,r]*V[j*p+k,r])_j should be all ones.
    
    # This is getting complex. Let me just try random optimization with multiple restarts.
    
    def solve_with_optimization(rank, n_restarts=20):
        """Try to solve using scipy optimization."""
        from scipy.optimize import minimize
        
        um_size = n * m * rank
        vp_size = m * p * rank
        pn_size = p * n * rank
        total_size = um_size + vp_size + pn_size
        
        best_obj = float('inf')
        best_params = None
        
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            
            np.random.seed(1000 + restart)
            params = np.random.randn(total_size) * 0.5
            
            def obj(x):
                U = x[:um_size].reshape(n*m, rank)
                V = x[um_size:um_size+vp_size].reshape(m*p, rank)
                W = x[um_size+vp_size:].reshape(p*n, rank)
                
                # Compute sum of squared residuals
                res = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            prod = U[i*m+j] * V[j*p+k] * W[k*n+i]
                            res += (1.0 - np.sum(prod)) ** 2
                return res
            
            result = minimize(obj, params, method='L-BFGS-B', 
                            options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-12})
            
            if result.fun < best_obj:
                best_obj = result.fun
                best_params = result.x
        
        if best_params is not None and best_obj < 1e-8:
            U = best_params[:um_size].reshape(n*m, rank)
            V = best_params[um_size:um_size+vp_size].reshape(m*p, rank)
            W = best_params[um_size+vp_size:].reshape(p*n, rank)
            
            # Try to snap to half-integers
            U_snap = np.round(U * 2) / 2
            V_snap = np.round(V * 2) / 2
            W_snap = np.round(W * 2) / 2
            
            if verify(U_snap.tolist(), V_snap.tolist(), W_snap.tolist()):
                return (U_snap.tolist(), V_snap.tolist(), W_snap.tolist())
            
            # Try integer
            U_int = np.round(U)
            V_int = np.round(V)
            W_int = np.round(W)
            
            if verify(U_int.tolist(), V_int.tolist(), W_int.tolist()):
                return (U_int.tolist(), V_int.tolist(), W_int.tolist())
            
            # Return the float solution if it's close enough (but we need exact)
            # For now, skip if not exactly representable
            pass
        
        return None
    
    # Try ranks from low to high
    for rank in range(2, 40):
        if time.time() > deadline:
            break
        
        result = solve_with_optimization(rank, n_restarts=10)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook
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
# EVOLVE-BLOCK-END
