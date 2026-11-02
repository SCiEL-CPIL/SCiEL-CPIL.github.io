"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]
import numpy as np
import time
from scipy.optimize import least_squares

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    # Try to find a rank-28 decomposition
    result = _find_decomposition(n, m, p, 28, deadline)
    if result is not None:
        return result
    
    # Fall back to schoolbook
    return _schoolbook(n, m, p)

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _find_decomposition(n, m, p, rank, deadline):
    """Try to find a rank-`rank` decomposition."""
    # The constraint: for all i in [0,n), j in [0,p), k in [0,m):
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Wait, I need to check the indexing again.
    # U: (n*m, rank), indexed by [i*m + j] where i in [0,n), j in [0,m)
    # V: (m*p, rank), indexed by [j*p + k] where j in [0,m), k in [0,p)
    # W: (p*n, rank), indexed by [k*n + i] where k in [0,p), i in [0,n)
    
    # The tensor T_{i,j,k} for matrix multiplication (n×m)·(m×p)→(n×p)
    # has T_{i,j,k} = 1 if the (i,j) entry uses A_{i,k} and B_{k,j}.
    # So for all i in [0,n), j in [0,p), k in [0,m): T_{i,j,k} = 1.
    
    # The decomposition: T_{i,j,k} = sum_r U[i*m+j, r] * V[k*p+j, r] * W[j*n+i, r]
    # Wait, I need to be careful. Let me re-read the task.
    
    # "U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]"
    # So U is indexed by (i,j) where i in [0,n), j in [0,m)
    # V is indexed by (j,k) where j in [0,m), k in [0,p)
    # W is indexed by (k,i) where k in [0,p), i in [0,n)
    
    # And the product should give the matrix multiplication tensor.
    # The (i,j) entry of the product is sum_k A_{i,k} B_{k,j}.
    # So the tensor is T_{i,k,j} = 1 (using i for row of A, k for column of A / row of B, j for column of B).
    
    # In the decomposition, we have U[i*m+j, r] where j is the column of A (which is k in the product),
    # V[j*p+k, r] where j is the column of A (k) and k is the column of B (j in the product),
    # W[k*n+i, r] where k is the column of B (j) and i is the row of A.
    
    # So the indices are: i (row of A, 0..n-1), j (col of A / row of B, 0..m-1), k (col of B, 0..p-1)
    # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    # And we need: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 for all i,j,k.
    
    # So the constraint is: for all i in [0,n), j in [0,m), k in [0,p):
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This is 32 equations for n=2, m=4, p=4.
    
    def residual(params):
        # params: flattened U, V, W
        U_flat = params[:n*m*rank]
        V_flat = params[n*m*rank : n*m*rank + m*p*rank]
        W_flat = params[n*m*rank + m*p*rank :]
        
        U = U_flat.reshape(n*m, rank)
        V = V_flat.reshape(m*p, rank)
        W = W_flat.reshape(p*n, rank)
        
        res = np.zeros(n*m*p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    res[idx] = val - 1.0
                    idx += 1
        return res
    
    # Initialize with schoolbook-like structure but with rank < 32
    # This is hard. Let's try random initialization.
    np.random.seed(0)
    params = np.random.randn(n*m*rank + m*p*rank + p*n*rank) * 0.1
    
    best_params = params
    best_res = np.inf
    
    for attempt in range(10):
        if time.time() > deadline:
            break
        params = np.random.randn(n*m*rank + m*p*rank + p*n*rank) * 0.1
        try:
            result = least_squares(residual, params, method='lm', max_nfev=10000)
            if result.cost < best_res:
                best_res = result.cost
                best_params = result.x
            if result.cost < 1e-6:
                break
        except:
            pass
    
    if best_res < 1e-6:
        U_flat = best_params[:n*m*rank]
        V_flat = best_params[n*m*rank : n*m*rank + m*p*rank]
        W_flat = best_params[n*m*rank + m*p*rank :]
        
        U = U_flat.reshape(n*m, rank)
        V = V_flat.reshape(m*p, rank)
        W = W_flat.reshape(p*n, rank)
        
        # Convert to lists of floats
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        
        return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

# EVOLVE-BLOCK-END
