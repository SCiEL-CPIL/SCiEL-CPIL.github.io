import numpy as np
import time
from itertools import combinations

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin

    # First, try laser method: delete one term from rank-17 and solve
    result_16 = _try_laser_rank16(n, m, p, deadline)
    if result_16 is not None:
        return result_16

    # Try continuous optimization for rank-16
    result_16b = _try_rank16(n, m, p, deadline)
    if result_16b is not None:
        return result_16b

    # Fallback: rank-17 decomposition
    return _rank17_decomposition(n, m, p)

def _rank17_decomposition(n, m, p):
    U17 = np.zeros((n * m, 17))
    V17 = np.zeros((m * p, 17))
    W17 = np.zeros((p * n, 17))

    def set_U17(i, j, r, val): U17[i * m + j, r] = val
    def set_V17(j, k, r, val): V17[j * p + k, r] = val
    def set_W17(k, i, r, val): W17[k * n + i, r] = val

    # Block 1: Strassen 2x2x2 (j in {0,1}, k in {0,1})
    set_U17(0, 0, 0, 1); set_U17(1, 1, 0, 1)
    set_V17(0, 0, 0, 1); set_V17(1, 1, 0, 1)
    set_W17(0, 0, 0, 1); set_W17(1, 1, 0, 1)

    set_U17(1, 0, 1, 1); set_U17(1, 1, 1, 1)
    set_V17(0, 0, 1, 1)
    set_W17(0, 1, 1, 1); set_W17(1, 1, 1, -1)

    set_U17(0, 0, 2, 1)
    set_V17(0, 1, 2, 1); set_V17(1, 1, 2, -1)
    set_W17(1, 0, 2, 1); set_W17(1, 1, 2, 1)

    set_U17(1, 1, 3, 1)
    set_V17(1, 0, 3, 1); set_V17(0, 0, 3, -1)
    set_W17(0, 0, 3, 1); set_W17(0, 1, 3, 1)

    set_U17(0, 0, 4, 1); set_U17(0, 1, 4, 1)
    set_V17(1, 1, 4, 1)
    set_W17(0, 0, 4, -1); set_W17(1, 0, 4, 1)

    set_U17(1, 0, 5, 1); set_U17(0, 0, 5, -1)
    set_V17(0, 0, 5, 1); set_V17(0, 1, 5, 1)
    set_W17(1, 1, 5, 1)

    set_U17(0, 1, 6, 1); set_U17(1, 1, 6, -1)
    set_V17(1, 0, 6, 1); set_V17(1, 1, 6, 1)
    set_W17(0, 0, 6, 1)

    # Block 2: Direct 2x2x1 (j in {0,1}, k=2)
    set_U17(0, 0, 7, 1)
    set_V17(0, 2, 7, 1)
    set_W17(2, 0, 7, 1)

    set_U17(0, 1, 8, 1)
    set_V17(1, 2, 8, 1)
    set_W17(2, 0, 8, 1)

    set_U17(1, 0, 9, 1)
    set_V17(0, 2, 9, 1)
    set_W17(2, 1, 9, 1)

    set_U17(1, 1, 10, 1)
    set_V17(1, 2, 10, 1)
    set_W17(2, 1, 10, 1)

    # Block 3: Direct 2x1x3 (j=2, k in {0,1,2})
    set_U17(0, 2, 11, 1)
    set_V17(2, 0, 11, 1)
    set_W17(0, 0, 11, 1)

    set_U17(0, 2, 12, 1)
    set_V17(2, 1, 12, 1)
    set_W17(1, 0, 12, 1)

    set_U17(0, 2, 13, 1)
    set_V17(2, 2, 13, 1)
    set_W17(2, 0, 13, 1)

    set_U17(1, 2, 14, 1)
    set_V17(2, 0, 14, 1)
    set_W17(0, 1, 14, 1)

    set_U17(1, 2, 15, 1)
    set_V17(2, 1, 15, 1)
    set_W17(1, 1, 15, 1)

    set_U17(1, 2, 16, 1)
    set_V17(2, 2, 16, 1)
    set_W17(2, 1, 16, 1)

    return {
        "U": U17.tolist(),
        "V": V17.tolist(),
        "W": W17.tolist()
    }

def _check_exact(U, V, W, n, m, p, tol=1e-10):
    """Check if the decomposition is exact (half-integer entries)."""
    # Round to half-integers
    U_s = np.round(U * 2) / 2.0
    V_s = np.round(V * 2) / 2.0
    W_s = np.round(W * 2) / 2.0

    # Check if the rounded values actually produce the tensor
    # Convert to integer grid (multiply by 2) for exact integer arithmetic check
    Ui = np.rint(U_s * 2).astype(int)
    Vi = np.rint(V_s * 2).astype(int)
    Wi = np.rint(W_s * 2).astype(int)

    U_t = Ui.reshape(n, m, -1)
    V_t = Vi.reshape(m, p, -1)
    W_t = Wi.reshape(p, n, -1)

    # The product in the integer domain will be 4 * (original product)
    # Original product should be 1. So integer product should be 4.
    # Wait, U, V, W are half-integers.
    # Let U = Ui/2, V = Vi/2, W = Wi/2.
    # Sum (Ui/2) (Vi/2) (Wi/2) = 1/8 Sum Ui Vi Wi.
    # So Sum Ui Vi Wi should be 8.
    
    prod = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)

    if np.all(prod == 8):
        return True, U_s, V_s, W_s
    return False, None, None, None

def _try_laser_rank16(n, m, p, deadline):
    """
    Laser method: Start from rank-17 solution, delete one term (column),
    then solve the linear system to see if we can recover the tensor with rank 16.
    
    For each of the 17 terms in the rank-17 solution, we try to delete it and
    solve for a rank-16 decomposition that matches the tensor.
    """
    base_17 = _rank17_decomposition(n, m, p)
    U17 = np.array(base_17["U"])
    V17 = np.array(base_17["V"])
    W17 = np.array(base_17["W"])
    rank17 = U17.shape[1]
    
    # The tensor T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # We want to find U,V,W of rank 16 such that this holds.
    
    # Strategy: For each term r to delete, we fix the support pattern
    # (which entries are nonzero in each of the 16 remaining terms)
    # and solve the linear system.
    
    # For each term to delete:
    for r_del in range(rank17):
        if time.time() > deadline:
            break
        
        # Keep all terms except r_del
        keep = [r for r in range(rank17) if r != r_del]
        
        # Extract the 16 remaining terms
        U_keep = U17[:, keep]
        V_keep = V17[:, keep]
        W_keep = W17[:, keep]
        
        # Now we have 16 terms. We need to check if there exist U,V,W
        # with this same support pattern (or a modified one) that work.
        
        # Approach: Set up a linear system.
        # For each output position (i,j,k), we need:
        # sum_{r in keep} U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 if (i,j,k) corresponds to T[i,j,k]=1
        # Actually, T[i,j,k] = 1 for all valid (i,j,k) in the multiplication tensor.
        
        # Wait, the multiplication tensor for (n,m,p) = (2,3,3) has:
        # T[i,j,k] = 1 if the output position (i,k) gets input from (i,j)*(j,k)
        # Actually for matrix multiplication C = A * B where A is n x m, B is m x p:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # So the tensor is T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
        # because each term A[i,j]*B[j,k] contributes to C[i,k].
        
        # Actually no. The tensor T has entries T[i,j,k] where:
        # i indexes rows of A (and C), j indexes columns of A (and rows of B), 
        # k indexes columns of B (and C).
        # T[i,j,k] = 1 for all valid (i,j,k) because A[i,j]*B[j,k] contributes to C[i,k].
        
        # So we need: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all i,j,k.
        
        # This is a system of n*m*p = 2*3*3 = 18 equations.
        # Unknowns: the entries of U, V, W for the 16 terms.
        # But it's nonlinear (trilinear).
        
        # Laser method approach: Fix one factor (say W) to its values from the
        # rank-17 solution, then solve linearly for U and V.
        
        # If we fix W, then for each (i,k), we need:
        # sum_r (U[i,:,r] dot V[:,k,r]) * W[k,i,r] = 1
        # This is still nonlinear in U and V.
        
        # Alternative: Use the fact that if we fix the supports, we can set up
        # a linear system by treating the products U[i,j,r]*V[j,k,r] as variables.
        
        # Let's try a different approach: random search with structured initialization.
        # Initialize U,V,W close to the rank-17 solution with one term removed,
        # then optimize.
        
        U_init = U_keep.copy()
        V_init = V_keep.copy()
        W_init = W_keep.copy()
        
        # Add small random perturbations
        np.random.seed(r_del * 1000)
        U_init += np.random.randn(*U_init.shape) * 0.01
        V_init += np.random.randn(*V_init.shape) * 0.01
        W_init += np.random.randn(*W_init.shape) * 0.01
        
        # Optimize
        U_opt, V_opt, W_opt, err = _optimize_decomposition(
            U_init, V_init, W_init, n, m, p, deadline, max_iter=500
        )
        
        if err < 1e-8:
            ok, Us, Vs, Ws = _check_exact(U_opt, V_opt, W_opt, n, m, p)
            if ok:
                return {
                    "U": Us.tolist(),
                    "V": Vs.tolist(),
                    "W": Ws.tolist()
                }
    
    return None

def _optimize_decomposition(U, V, W, n, m, p, deadline, max_iter=500):
    """
    Optimize U, V, W to minimize the error of the tensor decomposition.
    Uses gradient descent with adaptive learning rate.
    """
    rank = U.shape[1]
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        U_t = U.reshape(n, m, rank)
        V_t = V.reshape(m, p, rank)
        W_t = W.reshape(p, n, rank)
        
        prod = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
        res = prod - 1.0
        err = np.sum(res**2)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < 1e-10:
            break
        
        # Gradients
        dU = 2 * np.einsum('ijk,jkr,kir->ijr', res, V_t, W_t)
        dV = 2 * np.einsum('ijk,ijr,kir->jkr', res, U_t, W_t)
        dW = 2 * np.einsum('ijk,ijr,jkr->kir', res, U_t, V_t)
        
        # Normalize
        norm_U = np.linalg.norm(dU)
        norm_V = np.linalg.norm(dV)
        norm_W = np.linalg.norm(dW)
        
        if norm_U > 1e-12: dU /= norm_U
        if norm_V > 1e-12: dV /= norm_V
        if norm_W > 1e-12: dW /= norm_W
        
        # Adaptive learning rate
        lr = 0.05 * (1.0 + min(err, 10.0))
        
        U -= lr * dU.reshape(n * m, rank)
        V -= lr * dV.reshape(m * p, rank)
        W -= lr * dW.reshape(p * n, rank)
    
    return best_U, best_V, best_W, best_err

def _try_rank16(n, m, p, deadline):
    rank = 16
    
    # Random search with multiple initializations
    best_err = float('inf')
    best_state = None
    
    for seed in range(100):
        if time.time() > deadline:
            break
            
        np.random.seed(seed)
        
        # Initialize with small random values
        U = np.random.randn(n * m, rank) * 0.5
        V = np.random.randn(m * p, rank) * 0.5
        W = np.random.randn(p * n, rank) * 0.5
        
        # Optimize
        U_opt, V_opt, W_opt, err = _optimize_decomposition(
            U, V, W, n, m, p, deadline, max_iter=300
        )
        
        if err < best_err:
            best_err = err
            best_state = (U_opt, V_opt, W_opt)
        
        if err < 1e-8:
            ok, Us, Vs, Ws = _check_exact(U_opt, V_opt, W_opt, n, m, p)
            if ok:
                return {
                    "U": Us.tolist(),
                    "V": Vs.tolist(),
                    "W": Ws.tolist()
                }
    
    # Try to polish the best state
    if best_state is not None and best_err < 1e-3:
        U_b, V_b, W_b = best_state
        U_p, V_p, W_p, success = _newton_polish(U_b, V_b, W_b, n, m, p, max_iter=50, tol=1e-12)
        if success:
            ok, Us, Vs, Ws = _check_exact(U_p, V_p, W_p, n, m, p)
            if ok:
                return {
                    "U": Us.tolist(),
                    "V": Vs.tolist(),
                    "W": Ws.tolist()
                }
        
        ok, Us, Vs, Ws = _check_exact(U_b, V_b, W_b, n, m, p)
        if ok:
            return {
                "U": Us.tolist(),
                "V": Vs.tolist(),
                "W": Ws.tolist()
            }
                
    return None

def _newton_polish(U, V, W, n, m, p, max_iter=50, tol=1e-12):
    """
    Newton-type polish for the tensor decomposition.
    """
    U = U.copy().astype(np.float64)
    V = V.copy().astype(np.float64)
    W = W.copy().astype(np.float64)
    rank = U.shape[1]
    
    U_t = U.reshape(n, m, rank)
    V_t = V.reshape(m, p, rank)
    W_t = W.reshape(p, n, rank)
    
    for it in range(max_iter):
        prod = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
        res = prod - 1.0
        err = np.sum(res**2)
        
        if err < tol:
            return U, V, W, True
        
        # Gauss-Newton step
        dU = 2 * np.einsum('ijk,jkr,kir->ijr', res, V_t, W_t)
        dV = 2 * np.einsum('ijk,ijr,kir->jkr', res, U_t, W_t)
        dW = 2 * np.einsum('ijk,ijr,jkr->kir', res, U_t, V_t)
        
        # Adaptive step size
        alpha = 1.0
        for _ in range(20):
            U_new = U - alpha * dU.reshape(n*m, rank)
            V_new = V - alpha * dV.reshape(m*p, rank)
            W_new = W - alpha * dW.reshape(p*n, rank)
            
            U_t_new = U_new.reshape(n, m, rank)
            V_t_new = V_new.reshape(m, p, rank)
            W_t_new = W_new.reshape(p, n, rank)
            
            prod_new = np.einsum('ijr,jkr,kir->ijk', U_t_new, V_t_new, W_t_new)
            res_new = prod_new - 1.0
            err_new = np.sum(res_new**2)
            
            if err_new < err:
                break
            alpha *= 0.5
        
        U = U_new
        V = V_new
        W = W_new
        U_t = U.reshape(n, m, rank)
        V_t = V.reshape(m, p, rank)
        W_t = W.reshape(p, n, rank)
    
    prod = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
    err = np.sum((prod - 1.0)**2)
    return U, V, W, err < 1e-8
