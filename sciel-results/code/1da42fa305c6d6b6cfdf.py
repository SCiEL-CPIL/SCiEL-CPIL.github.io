"""Rewritten program: search for low-rank exact tensor decomposition of 3x3 by 3x4 matmul."""
import numpy as np
from scipy.optimize import minimize
import time
import sys

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 * 3x4 matrix multiplication.
    
    The multiplication tensor T has indices (i, j, k) where:
      i in [0, n) = [0, 3)  -> row of A
      j in [0, m) = [0, 3)  -> column of A / row of B
      k in [0, p) = [0, 4)  -> column of B
    
    T[i, j, k] = 1 for all valid (i, j, k) since (AB)[i,k] = sum_j A[i,j]*B[j,k].
    
    We seek rank-r factor matrices U (n*m x r), V (m*p x r), W (p*n x r) such that:
      T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    Strategy: Use the known rank-23 algorithm for 3x3x3 (Laderman's) and extend to 3x4.
    Actually, let's try a direct optimization approach with integer/half-integer constraints.
    
    Given the complexity, let's start with the schoolbook rank=36 and try to reduce via
    a structured search or optimization.
    """
    n, m, p = DIMENSIONS
    nm = n * m  # 9
    mp = m * p  # 12
    pn = p * n  # 12
    
    # The multiplication tensor: T[i, j, k] = 1 for all i,j,k
    # We need U[nm, r], V[mp, r], W[pn, r] such that
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k
    
    # Start with schoolbook rank = n*m*p = 36
    rank_schoolbook = n * m * p
    U_schoolbook = np.zeros((nm, rank_schoolbook))
    V_schoolbook = np.zeros((mp, rank_schoolbook))
    W_schoolbook = np.zeros((pn, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_schoolbook[i * m + j, col] = 1.0
                V_schoolbook[j * p + k, col] = 1.0
                W_schoolbook[k * n + i, col] = 1.0
                col += 1
    
    def to_dict(U, V, W):
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    # Verify schoolbook works
    def verify(U, V, W):
        """Check if the decomposition exactly reconstructs the tensor."""
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(r):
                        val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(val - 1.0) > 1e-6:
                        return False
        return True
    
    if not verify(U_schoolbook, V_schoolbook, W_schoolbook):
        # Fallback: return schoolbook anyway
        return to_dict(U_schoolbook, V_schoolbook, W_schoolbook)
    
    # Try to find a lower rank decomposition using optimization
    # We'll attempt ranks from 24 up to 35
    
    best_result = to_dict(U_schoolbook, V_schoolbook, W_schoolbook)
    best_rank = rank_schoolbook
    
    start_time = time.time()
    time_limit = 120.0
    
    # For each candidate rank, try to find a solution via optimization
    for rank in range(24, rank_schoolbook):
        if time.time() - start_time > time_limit - 10:
            break
        
        # Random initialization
        np.random.seed(rank)  # Reproducible
        U = np.random.randn(nm, rank)
        V = np.random.randn(mp, rank)
        W = np.random.randn(pn, rank)
        
        # Objective: minimize sum of squared errors
        def objective(params):
            U_flat = params[:nm * rank]
            V_flat = params[nm * rank : nm * rank + mp * rank]
            W_flat = params[nm * rank + mp * rank : nm * rank + mp * rank + pn * rank]
            
            U_mat = U_flat.reshape(nm, rank)
            V_mat = V_flat.reshape(mp, rank)
            W_mat = W_flat.reshape(pn, rank)
            
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U_mat[i * m + j], V_mat[j * p + k]) * W_mat[k * n + i]
                        # Wait, that's not right. The tensor equation is:
                        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        # This is a trilinear form, not a bilinear one followed by linear.
                        # Let me recompute.
                        val = 0.0
                        for c in range(rank):
                            val += U_mat[i * m + j, c] * V_mat[j * p + k, c] * W_mat[k * n + i, c]
                        error += (val - 1.0) ** 2
            return error
        
        # This objective is expensive to evaluate with Python loops.
        # Let's vectorize it.
        
        def objective_vec(params):
            U_flat = params[:nm * rank]
            V_flat = params[nm * rank : nm * rank + mp * rank]
            W_flat = params[nm * rank + mp * rank : nm * rank + mp * rank + pn * rank]
            
            U_mat = U_flat.reshape(nm, rank)
            V_mat = V_flat.reshape(mp, rank)
            W_mat = W_flat.reshape(pn, rank)
            
            # Compute all tensor entries
            errors = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        val = np.sum(U_mat[i * m + j] * V_mat[j * p + k] * W_mat[k * n + i])
                        errors.append((val - 1.0) ** 2)
            return np.sum(errors)
        
        # Use L-BFGS-B for optimization
        try:
            x0 = np.concatenate([U.ravel(), V.ravel(), W.ravel()])
            result = minimize(objective_vec, x0, method='L-BFGS-B', 
                             options={'maxiter': 100, 'ftol': 1e-12})
            
            if result.fun < 1e-8:
                # Found a solution
                U_sol = result.x[:nm * rank].reshape(nm, rank)
                V_sol = result.x[nm * rank : nm * rank + mp * rank].reshape(mp, rank)
                W_sol = result.x[nm * rank + mp * rank : nm * rank + mp * rank + pn * rank].reshape(pn, rank)
                
                # Try to round to half-integers
                U_round = np.round(U_sol * 2) / 2
                V_round = np.round(V_sol * 2) / 2
                W_round = np.round(W_sol * 2) / 2
                
                if verify(U_round, V_round, W_round):
                    best_result = to_dict(U_round, V_round, W_round)
                    best_rank = rank
                    break
                else:
                    # Use the unrounded solution
                    best_result = to_dict(U_sol, V_sol, W_sol)
                    best_rank = rank
                    break
        except Exception:
            continue
    
    return best_result
# EVOLVE-BLOCK-END
