"""Program for matrix multiplication 2x3x7 rank minimization."""
import time
import random
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 7]

def _verify(U, V, W, n=2, m=3, p=7, tol=1e-9):
    """Verify that U, V, W decompose the matrix multiplication tensor."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    rank = U_arr.shape[1]
    
    # Reshape to (n, m, rank), (m, p, rank), (p, n, rank)
    U_t = U_arr.reshape(n, m, rank)
    V_t = V_arr.reshape(m, p, rank)
    W_t = W_arr.reshape(p, n, rank)
    
    # Compute S[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # Using einsum: 'ijc,jkc,kic->ijk'
    S = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
    
    # Check if S is all ones
    if not np.allclose(S, 1.0, atol=tol):
        return False
    return True

def _random_init(n, m, p, rank, seed=None):
    if seed is not None:
        np.random.seed(seed)
    U = np.random.randn(n*m, rank)
    V = np.random.randn(m*p, rank)
    W = np.random.randn(p*n, rank)
    return U.tolist(), V.tolist(), W.tolist()

def _schoolbook(n, m, p):
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

def _tensor_objective(x, n, m, p, rank):
    """Objective function for tensor decomposition optimization."""
    U_t = x[:n*m*rank].reshape(n, m, rank)
    V_t = x[n*m*rank:n*m*rank + m*p*rank].reshape(m, p, rank)
    W_t = x[n*m*rank + m*p*rank:].reshape(p, n, rank)
    
    S = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
    residual = S - 1.0
    return np.sum(residual ** 2)

def _tensor_gradients(x, n, m, p, rank):
    """Gradient of the tensor decomposition objective."""
    U_t = x[:n*m*rank].reshape(n, m, rank)
    V_t = x[n*m*rank:n*m*rank + m*p*rank].reshape(m, p, rank)
    W_t = x[n*m*rank + m*p*rank:].reshape(p, n, rank)
    
    S = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
    R = S - 1.0
    
    grad_U_t = np.einsum('ijk,jkc,kic->ijc', R, V_t, W_t) * 2
    grad_V_t = np.einsum('ijk,ijc,kic->jkc', R, U_t, W_t) * 2
    grad_W_t = np.einsum('ijk,ijc,jkc->kic', R, U_t, V_t) * 2
    
    grad = np.concatenate([
        grad_U_t.flatten(),
        grad_V_t.flatten(),
        grad_W_t.flatten()
    ])
    return grad

def _gradient_descent(U, V, W, n, m, p, lr=0.01, steps=1000, tol=1e-12):
    """
    Perform gradient descent to refine a tensor decomposition.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    """
    U = np.array(U, dtype=float).copy()
    V = np.array(V, dtype=float).copy()
    W = np.array(W, dtype=float).copy()
    
    rank = U.shape[1]
    
    # Reshape
    U_t = U.reshape(n, m, rank)
    V_t = V.reshape(m, p, rank)
    W_t = W.reshape(p, n, rank)
    
    # Target tensor is all ones
    target = np.ones((n, m, p))
    
    for step in range(steps):
        # Compute S[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        S = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
        
        # Residual
        R = S - target  # We want S - target = 0
        
        # Gradient with respect to U[i,j,c]
        # dR/dU[i,j,c] = V[j,k,c] * W[k,i,c] summed over k
        # grad_U[i,j,c] = sum_k R[i,j,k] * V[j,k,c] * W[k,i,c]
        grad_U_t = np.einsum('ijk,jkc,kic->ijc', R, V_t, W_t)
        
        # Gradient with respect to V[j,k,c]
        # dR/dV[j,k,c] = U[i,j,c] * W[k,i,c] summed over i
        # grad_V[j,k,c] = sum_i R[i,j,k] * U[i,j,c] * W[k,i,c]
        grad_V_t = np.einsum('ijk,ijc,kic->jkc', R, U_t, W_t)
        
        # Gradient with respect to W[k,i,c]
        # dR/dW[k,i,c] = U[i,j,c] * V[j,k,c] summed over j
        # grad_W[k,i,c] = sum_j R[i,j,k] * U[i,j,c] * V[j,k,c]
        grad_W_t = np.einsum('ijk,ijc,jkc->kic', R, U_t, V_t)
        
        # Update
        U_t -= lr * grad_U_t
        V_t -= lr * grad_V_t
        W_t -= lr * grad_W_t
        
        # Check convergence
        if step % 10 == 0:
            S_check = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
            err = np.max(np.abs(S_check - 1.0))
            if err < tol:
                break
        
        # Adaptive learning rate
        if step % 100 == 0 and step > 0:
            S_check = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
            err = np.max(np.abs(S_check - 1.0))
            if err > 1.0:
                lr *= 0.5
            elif err < 0.01:
                lr *= 1.2
    
    # Reshape back
    U_new = U_t.reshape(n*m, rank)
    V_new = V_t.reshape(m*p, rank)
    W_new = W_t.reshape(p*n, rank)
    
    return U_new, V_new, W_new

def _scipy_optimize(U, V, W, n, m, p, time_limit=30):
    """Use scipy.optimize to refine the tensor decomposition."""
    rank = U.shape[1]
    
    x0 = np.concatenate([
        np.array(U).flatten(),
        np.array(V).flatten(),
        np.array(W).flatten()
    ])
    
    start_time = time.time()
    
    try:
        result = minimize(
            _tensor_objective,
            x0,
            jac=_tensor_gradients,
            args=(n, m, p, rank),
            method='L-BFGS-B',
            options={'maxiter': 1000, 'maxfun': 10000, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if result.success or result.fun < 1e-6:
            x_opt = result.x
            U_opt = x_opt[:n*m*rank].reshape(n*m, rank)
            V_opt = x_opt[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
            W_opt = x_opt[n*m*rank + m*p*rank:].reshape(p*n, rank)
            return U_opt, V_opt, W_opt
    except Exception:
        pass
    
    return None

def _try_round_and_verify(U, V, W, n, m, p):
    """Try to round to half-integers and verify."""
    U_r = [[round(x * 2) / 2 for x in row] for row in U]
    V_r = [[round(x * 2) / 2 for x in row] for row in V]
    W_r = [[round(x * 2) / 2 for x in row] for row in W]
    
    if _verify(U_r, V_r, W_r, n, m, p, tol=1e-9):
        return U_r, V_r, W_r
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin out of 150s
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 42
    
    # Start with schoolbook decomposition
    best = _schoolbook(n, m, p)
    best_rank = schoolbook_rank
    
    # Lower bound is max(n*m, m*p, p*n) = max(6, 21, 14) = 21
    lower_bound = max(n*m, m*p, p*n)
    
    # We will try ranks from lower_bound to best_rank - 1
    for target_rank in range(lower_bound, best_rank):
        if time.time() > deadline - 20:
            break
        
        # Try multiple random initializations
        max_attempts = 300
        found = False
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            seed = target_rank * 10000 + attempt
            U_t, V_t, W_t = _random_init(n, m, p, target_rank, seed=seed)
            
            # First try gradient descent
            U_gd, V_gd, W_gd = _gradient_descent(U_t, V_t, W_t, n, m, p, lr=0.05, steps=300)
            
            # Check if this solution works
            if _verify(U_gd.tolist(), V_gd.tolist(), W_gd.tolist(), n, m, p, tol=1e-6):
                # Try to round to half-integers
                rounded = _try_round_and_verify(U_gd.tolist(), V_gd.tolist(), W_gd.tolist(), n, m, p)
                if rounded:
                    U_r, V_r, W_r = rounded
                    best = {"U": U_r, "V": V_r, "W": W_r}
                    best_rank = target_rank
                    found = True
                    break
            else:
                # If GD didn't converge, try scipy optimization
                if time.time() < deadline - 15:
                    scipy_result = _scipy_optimize(U_gd, V_gd, W_gd, n, m, p, time_limit=25)
                    if scipy_result is not None:
                        U_scipy, V_scipy, W_scipy = scipy_result
                        if _verify(U_scipy.tolist(), V_scipy.tolist(), W_scipy.tolist(), n, m, p, tol=1e-6):
                            rounded = _try_round_and_verify(U_scipy.tolist(), V_scipy.tolist(), W_scipy.tolist(), n, m, p)
                            if rounded:
                                U_r, V_r, W_r = rounded
                                best = {"U": U_r, "V": V_r, "W": W_r}
                                best_rank = target_rank
                                found = True
                                break
        
        if found:
            break
    
    # Return the best found
    return best
