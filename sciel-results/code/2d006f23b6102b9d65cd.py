"""Program for 3x3x3 matrix multiplication tensor decomposition."""
import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3x3 matrix multiplication.
    The output factor follows the official trace convention W[k*n+i, r].
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    """
    n, m, p = DIMENSIONS
    # The known minimal rank for 3x3x3 matrix multiplication is 23 (Strassen-like).
    # However, we will attempt to find a valid decomposition with the lowest rank possible
    # using numerical optimization, then snap to half-integers if close enough.
    
    # The matrix multiplication tensor T is defined such that:
    # (A * B)[i, k] = sum_j A[i, j] * B[j, k]
    # In tensor decomposition form: T_{i, j, k} = sum_r U[i, r] * V[j, r] * W[k, r]
    # But the convention here is W[k*n+i, r], so the tensor is T_{i, j, k} where i is row of A, j is col of A / row of B, k is col of B.
    # The factor matrices are:
    # U: (n*m, rank) -> U[i*m+j, r]
    # V: (m*p, rank) -> V[j*p+k, r]
    # W: (p*n, rank) -> W[k*n+i, r]
    
    # The tensor T is a 3-tensor of shape (n, m, p) or flattened.
    # T[i, j, k] = 1 if it corresponds to the product A[i,j]*B[j,k] contributing to C[i,k].
    # Actually, the standard tensor for matmul is T_{i, j, k} where the entry is 1 for the term A[i,j]B[j,k].
    # The decomposition is: T_{i, j, k} = sum_r U[i, r] V[j, r] W[k, r] ?
    # Let's check the indices in the prompt:
    # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    # This suggests the tensor is indexed by (i, j, k) and the term is U[i, j, r] * V[j, k, r] * W[k, i, r]? No.
    # The prompt says: U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank).
    # This implies U is indexed by (i, j), V by (j, k), W by (k, i).
    # So the tensor T_{i, j, k} is decomposed as sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]?
    # Wait, standard tensor decomposition for matmul is:
    # C_{i, k} = sum_j A_{i, j} B_{j, k}
    # This is a 3-tensor T_{i, j, k} where T_{i, j, k} = 1 if we are computing the product.
    # The decomposition is T_{i, j, k} = sum_r U_{i, j, r} V_{j, k, r} W_{k, i, r} is not standard.
    # Usually it's T_{i, j, k} = sum_r U_{i, r} V_{j, r} W_{k, r} where U, V, W are (n, rank), (m, rank), (p, rank).
    # But here the shapes are (n*m, rank), (m*p, rank), (p*n, rank).
    # This corresponds to a decomposition of the tensor T_{i, j, k} where:
    # U is indexed by (i, j), V by (j, k), W by (k, i).
    # So T_{i, j, k} = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r].
    # Let's verify with the schoolbook:
    # For each (i, j, k), U[i, j, r] = 1, V[j, k, r] = 1, W[k, i, r] = 1.
    # This matches the prompt's schoolbook implementation.
    
    # So we need to find U, V, W of shapes (9, R), (9, R), (9, R) such that:
    # For all i, j, k: sum_r U[i, j, r] * V[j, k, r] * W[k, i, r] = T_{i, j, k}
    # where T_{i, j, k} = 1 (since every term A[i,j]B[j,k] contributes to C[i,k] with coefficient 1).
    
    # We will use a numerical optimization approach to find a low-rank decomposition.
    # Start with rank 20 and increase if necessary.
    
    # Define the target tensor T
    T = np.ones((n, m, p))
    
    # Function to compute the error for a given U, V, W
    def error_func(params, R):
        U = params[:R*n*m].reshape((n*m, R))
        V = params[R*n*m:2*R*n*m].reshape((m*p, R))
        W = params[2*R*n*m:3*R*n*m].reshape((p*n, R))
        
        # Compute the reconstructed tensor
        # T_recon[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
        # We can vectorize this using einsum
        # U is (i, j, r), V is (j, k, r), W is (k, i, r)
        # T_recon = np.einsum('ijr,jkr,kir->ijk', U, V, W)
        # But U is flattened (i*m+j, r), so we need to reshape to (i, j, r)
        U_reshaped = U.reshape((n, m, R))
        V_reshaped = V.reshape((m, p, R))
        W_reshaped = W.reshape((p, n, R))
        
        T_recon = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
        
        # Compute the squared error
        err = np.sum((T_recon - T)**2)
        return err
    
    # Try different ranks, starting from 20
    best_result = None
    best_rank = None
    start_time = time.time()
    deadline = start_time + 120  # 120 seconds
    
    for R in range(20, 28):
        if time.time() > deadline - 10:
            break
        
        # Initialize with random values
        np.random.seed(42)
        params0 = np.random.randn(3*R*n*m) * 0.1
        
        # Optimize
        result = minimize(error_func, params0, args=(R,), method='L-BFGS-B', 
                          options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8})
        
        if result.success and result.fun < 1e-6:
            U = result.x[:R*n*m].reshape((n*m, R))
            V = result.x[R*n*m:2*R*n*m].reshape((m*p, R))
            W = result.x[2*R*n*m:3*R*n*m].reshape((p*n, R))
            
            # Snap to half-integers if close
            U = np.round(U * 2) / 2
            V = np.round(V * 2) / 2
            W = np.round(W * 2) / 2
            
            # Verify the snapped version
            U_reshaped = U.reshape((n, m, R))
            V_reshaped = V.reshape((m, p, R))
            W_reshaped = W.reshape((p, n, R))
            T_recon = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
            
            if np.allclose(T_recon, T, atol=1e-6):
                best_result = (U, V, W)
                best_rank = R
                break
    
    # If no low-rank solution found, fall back to schoolbook
    if best_result is None:
        R = n * m * p
        U = np.zeros((n*m, R))
        V = np.zeros((m*p, R))
        W = np.zeros((p*n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
    else:
        U, V, W = best_result
        R = best_rank
    
    # Convert to lists of lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
