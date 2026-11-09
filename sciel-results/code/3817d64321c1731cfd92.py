"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor as a 3D array T[i,j,k] = 1."""
    T = np.zeros((n, m, p), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1
    return T

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly matches matrix multiplication."""
    rank = U.shape[1]
    T = _build_tensor(n, m, p)
    
    # Compute the tensor from the decomposition
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    T_decomp = np.zeros((n, m, p), dtype=np.int64)
    for c in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_decomp[i, j, k] += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    return np.array_equal(T, T_decomp)

def _try_rank_gradient(R, n, m, p, deadline, max_restarts=5, max_steps=2000):
    """
    Try to find a rank-R decomposition using gradient descent.
    Returns (U, V, W) as integer arrays if found, else None.
    """
    nU = n * m
    nV = m * p
    nW = p * n
    
    # Target tensor in flattened form for einsum
    # We want sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # This is equivalent to a tensor T of shape (n, m, p) where T[i,j,k]=1
    
    T_target = np.ones((n, m, p), dtype=np.float64)
    
    for restart in range(max_restarts):
        if time.time() > deadline - 5:
            break
            
        # Initialize with random values
        # Scale is important; start small
        scale = 0.5
        U = np.random.randn(nU, R) * scale
        V = np.random.randn(nV, R) * scale
        W = np.random.randn(nW, R) * scale
        
        lr = 0.01
        prev_err = np.inf
        
        for step in range(max_steps):
            if time.time() > deadline - 2:
                break
                
            # Compute reconstruction
            # U: (nU, R), V: (nV, R), W: (nW, R)
            # We need to reshape U to (n, m, R), V to (m, p, R), W to (p, n, R)
            U_reshaped = U.reshape(n, m, R)
            V_reshaped = V.reshape(m, p, R)
            W_reshaped = W.reshape(p, n, R)
            
            # T_recon[i, j, k] = sum_r U_reshaped[i, j, r] * V_reshaped[j, k, r] * W_reshaped[k, i, r]
            # Using einsum: 'ijr,jkr,kir->ijk'
            T_recon = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
            
            err_tensor = T_recon - T_target
            err = np.sum(err_tensor**2)
            
            if err < 1e-10:
                # Found a solution, now try to snap to integers/half-integers
                # Check if values are close to half-integers
                U_check = np.round(U * 2) / 2.0
                V_check = np.round(V * 2) / 2.0
                W_check = np.round(W * 2) / 2.0
                
                # Verify the snapped version
                U_r = U_check.reshape(n, m, R)
                V_r = V_check.reshape(m, p, R)
                W_r = W_check.reshape(p, n, R)
                T_snapped = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
                
                if np.allclose(T_snapped, T_target, atol=1e-9):
                    # Convert to int64 by multiplying by 2 to handle halves, then back?
                    # The problem allows integer/half-integer.
                    # Let's just return the float arrays for now, the caller can handle conversion
                    # But the interface expects integer/half-integer entries.
                    # If we return floats that are exactly x.0 or x.5, it should be fine.
                    return U_check, V_check, W_check
                
                # If snapping failed, try to continue optimizing with tighter tolerance or different init
                # For now, if err is very small but snapping fails, we might have a non-rational solution.
                # We need a rational solution.
                # Let's try to refine with higher precision or different init.
                pass
            
            # Gradients
            # dU[i, j, r] = sum_k err[i, j, k] * V[j, k, r] * W[k, i, r]
            # dV[j, k, r] = sum_i err[i, j, k] * U[i, j, r] * W[k, i, r]
            # dW[k, i, r] = sum_j err[i, j, k] * U[i, j, r] * V[j, k, r]
            
            dU = np.einsum('ijk,jkr,kir->ijr', err_tensor, V_reshaped, W_reshaped)
            dV = np.einsum('ijk,ijr,kir->jkr', err_tensor, U_reshaped, W_reshaped)
            dW = np.einsum('ijk,ijr,jkr->kir', err_tensor, U_reshaped, V_reshaped)
            
            dU = dU.reshape(nU, R)
            dV = dV.reshape(nV, R)
            dW = dW.reshape(nW, R)
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            # Adaptive learning rate
            if err > prev_err * 1.1:
                lr *= 0.5
            else:
                if err < prev_err * 0.9:
                    lr *= 1.05
                prev_err = err
        
        # Final check after loop
        U_reshaped = U.reshape(n, m, R)
        V_reshaped = V.reshape(m, p, R)
        W_reshaped = W.reshape(p, n, R)
        T_recon = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
        if np.allclose(T_recon, T_target, atol=1e-8):
            U_check = np.round(U * 2) / 2.0
            V_check = np.round(V * 2) / 2.0
            W_check = np.round(W * 2) / 2.0
            U_r = U_check.reshape(n, m, R)
            V_r = V_check.reshape(m, p, R)
            W_r = W_check.reshape(p, n, R)
            T_snapped = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
            if np.allclose(T_snapped, T_target, atol=1e-9):
                return U_check, V_check, W_check

    return None

def _try_rank_systematic(R, n, m, p, deadline):
    """
    Try to find a rank-R decomposition using a more systematic approach for small R.
    This is a placeholder for more complex searches if gradient descent fails.
    """
    # For now, just return None to let gradient descent handle it
    # We could implement a backtracking search here for very small R
    return None

def _try_rank(r, n, m, p, deadline):
    """Try to find a rank-r decomposition. Returns (U, V, W) or None."""
    # First try gradient descent
    result = _try_rank_gradient(r, n, m, p, deadline)
    if result is not None:
        return result
    
    # Try systematic search (currently disabled/placeholder)
    result = _try_rank_systematic(r, n, m, p, deadline)
    if result is not None:
        return result
        
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120  # 120 second deadline
    
    # Start with schoolbook
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
    
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank
    
    # Try to find better decompositions
    # For 2x3 x 3x3, try ranks from 17 down
    # We know schoolbook is 18.
    # Let's try 17, 16, ... down to a reasonable limit.
    # Gradient descent might be slow, so we limit the search.
    
    for r in range(rank - 1, 10, -1):
        if time.time() > deadline - 30:  # Safety margin
            break
        
        result = _try_rank(r, n, m, p, deadline)
        if result is not None:
            U_arr, V_arr, W_arr = result
            # Verify the decomposition
            if _verify_decomposition(U_arr, V_arr, W_arr, n, m, p):
                best_rank = r
                best_result = {
                    "U": U_arr.tolist(),
                    "V": V_arr.tolist(),
                    "W": W_arr.tolist()
                }
                break  # Found the best, stop
    
    return best_result
# EVOLVE-BLOCK-END
