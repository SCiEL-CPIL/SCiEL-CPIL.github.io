"""Matrix multiplication 3,3,8 - rank-minimal decomposition."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def construct():
    """
    Returns a rank-minimal tensor decomposition for 3x3 * 3x8 matrix multiplication.
    Uses optimization to find the lowest rank decomposition.
    """
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    def error_function(params, rank):
        """Compute the squared error of the decomposition."""
        # Unpack parameters
        U_flat = params[:nm * rank]
        V_flat = params[nm * rank : nm * rank + mp * rank]
        W_flat = params[nm * rank + mp * rank : nm * rank + mp * rank + pn * rank]
        
        U = U_flat.reshape(nm, rank)
        V = V_flat.reshape(mp, rank)
        W = W_flat.reshape(pn, rank)
        
        # Compute error: for each (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] should be 1
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    err += (val - 1.0) ** 2
        return err
    
    def try_rank(rank, num_restarts=5):
        """Try to find a valid decomposition of given rank."""
        total_vars = (nm + mp + pn) * rank
        
        for restart in range(num_restarts):
            if time.time() > deadline - 10:
                break
            
            # Random initialization
            np.random.seed(42 + restart)
            x0 = np.random.randn(total_vars) * 0.5
            
            # Use Nelder-Mead for nonlinear optimization
            result = minimize(
                error_function,
                x0,
                args=(rank,),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
            
            if result.fun < 1e-10:
                # Found a valid solution, extract it
                U_flat = result.x[:nm * rank]
                V_flat = result.x[nm * rank : nm * rank + mp * rank]
                W_flat = result.x[nm * rank + mp * rank : nm * rank + mp * rank + pn * rank]
                
                U = U_flat.reshape(nm, rank).tolist()
                V = V_flat.reshape(mp, rank).tolist()
                W = W_flat.reshape(pn, rank).tolist()
                
                # Verify
                U_arr = np.array(U)
                V_arr = np.array(V)
                W_arr = np.array(W)
                
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for r in range(rank):
                                val += U_arr[i * m + j, r] * V_arr[j * p + k, r] * W_arr[k * n + i, r]
                            if abs(val - 1.0) > 1e-9:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    return U, V, W
        
        return None
    
    # Try ranks from low to high
    for rank in range(2, 73):
        if time.time() > deadline - 5:
            break
        
        result = try_rank(rank)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
