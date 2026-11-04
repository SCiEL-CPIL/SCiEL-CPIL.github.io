"""Program for matrix multiplication 2x3x7 rank minimization."""
import time
import numpy as np

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
    S = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
    
    if not np.allclose(S, 1.0, atol=tol):
        return False
    return True

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

def _newton_polish(U, V, W, n, m, p, max_iter=100, tol=1e-12):
    """
    Newton's method to polish a tensor decomposition.
    Solves the linear system for corrections delta_U, delta_V, delta_W.
    """
    U = np.array(U, dtype=float).copy()
    V = np.array(V, dtype=float).copy()
    W = np.array(W, dtype=float).copy()
    
    rank = U.shape[1]
    U_t = U.reshape(n, m, rank)
    V_t = V.reshape(m, p, rank)
    W_t = W.reshape(p, n, rank)
    
    target = np.ones((n, m, p))
    
    for it in range(max_iter):
        S = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
        R = S - target  # residual, want to make this zero
        
        err = np.max(np.abs(R))
        if err < tol:
            break
        
        # Build the linear system for Newton's method
        # We want: sum_c (dU[i,j,c]*V[j,k,c]*W[k,i,c] + U[i,j,c]*dV[j,k,c]*W[k,i,c] + U[i,j,c]*V[j,k,c]*dW[k,i,c]) = -R[i,j,k]
        # This is a linear system in the unknowns dU, dV, dW.
        
        # Unknowns: 
        # dU has n*m*rank variables
        # dV has m*p*rank variables
        # dW has p*n*rank variables
        # Total unknowns: (n*m + m*p + p*n) * rank
        
        # Equations: n*m*p equations
        
        n_unknowns = (n * m + m * p + p * n) * rank
        n_equations = n * m * p
        
        # Build the Jacobian matrix J where J[equation, unknown]
        # This can be large, but for small dimensions it's manageable.
        # n*m*p = 42 equations
        # Unknowns: (6 + 21 + 14) * rank = 41 * rank
        # For rank ~ 20-42, unknowns ~ 820 - 1722
        
        # Actually, let's use a different approach. 
        # The system is:
        # For each (i,j,k):
        #   sum_c dU[i,j,c] * V[j,k,c] * W[k,i,c] + U[i,j,c] * dV[j,k,c] * W[k,i,c] + U[i,j,c] * V[j,k,c] * dW[k,i,c] = -R[i,j,k]
        
        # Let's index the unknowns:
        # dU flat index: (i*m+j)*rank + c
        # dV flat index: (n*m + j*p+k)*rank + c
        # dW flat index: (n*m + m*p + k*n+i)*rank + c
        
        J = np.zeros((n_equations, n_unknowns))
        b = -R.flatten()  # shape (n*m*p,)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    eq_idx = i * m * p + j * p + k
                    
                    for c in range(rank):
                        # dU[i,j,c] contribution
                        u_idx = (i * m + j) * rank + c
                        J[eq_idx, u_idx] = V_t[j, k, c] * W_t[k, i, c]
                        
                        # dV[j,k,c] contribution
                        v_idx = (n * m + j * p + k) * rank + c
                        J[eq_idx, v_idx] = U_t[i, j, c] * W_t[k, i, c]
                        
                        # dW[k,i,c] contribution
                        w_idx = (n * m + m * p + k * n + i) * rank + c
                        J[eq_idx, w_idx] = U_t[i, j, c] * V_t[j, k, c]
        
        # Solve the least squares system J @ delta = b
        try:
            delta, residuals, rank_sv, sv = np.linalg.lstsq(J, b, rcond=None)
        except np.linalg.LinAlgError:
            break
        
        # Update
        dU = delta[:n * m * rank].reshape(n, m, rank)
        dV = delta[n * m * rank:n * m * rank + m * p * rank].reshape(m, p, rank)
        dW = delta[n * m * rank + m * p * rank:].reshape(p, n, rank)
        
        # Damping if step is too large
        step_norm = np.sqrt(np.sum(dU**2) + np.sum(dV**2) + np.sum(dW**2))
        if step_norm > 1.0:
            scale = 1.0 / step_norm
            dU *= scale
            dV *= scale
            dW *= scale
        
        U_t += dU
        V_t += dV
        W_t += dW
    
    U_new = U_t.reshape(n * m, rank)
    V_new = V_t.reshape(m * p, rank)
    W_new = W_t.reshape(p * n, rank)
    
    return U_new, V_new, W_new

def _gradient_descent(U, V, W, n, m, p, lr=0.05, steps=2000, tol=1e-10, momentum=0.9):
    """
    Perform gradient descent with momentum to refine a tensor decomposition.
    """
    U = np.array(U, dtype=float).copy()
    V = np.array(V, dtype=float).copy()
    W = np.array(W, dtype=float).copy()
    
    rank = U.shape[1]
    
    U_t = U.reshape(n, m, rank)
    V_t = V.reshape(m, p, rank)
    W_t = W.reshape(p, n, rank)
    
    target = np.ones((n, m, p))
    
    # Momentum buffers
    vU = np.zeros_like(U_t)
    vV = np.zeros_like(V_t)
    vW = np.zeros_like(W_t)
    
    for step in range(steps):
        S = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
        R = S - target
        
        # Gradients
        grad_U_t = np.einsum('ijk,jkc,kic->ijc', R, V_t, W_t)
        grad_V_t = np.einsum('ijk,ijc,kic->jkc', R, U_t, W_t)
        grad_W_t = np.einsum('ijk,ijc,jkc->kic', R, U_t, V_t)
        
        # Momentum update
        vU = momentum * vU + (1 - momentum) * grad_U_t
        vV = momentum * vV + (1 - momentum) * grad_V_t
        vW = momentum * vW + (1 - momentum) * grad_W_t
        
        U_t -= lr * vU
        V_t -= lr * vV
        W_t -= lr * vW
        
        if step % 50 == 0:
            S_check = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
            err = np.max(np.abs(S_check - 1.0))
            if err < tol:
                break
        
        # Adaptive learning rate
        if step % 200 == 0 and step > 0:
            S_check = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
            err = np.max(np.abs(S_check - 1.0))
            if err > 0.5:
                lr *= 0.5
            elif err < 0.001:
                lr *= 1.5
    
    U_new = U_t.reshape(n * m, rank)
    V_new = V_t.reshape(m * p, rank)
    W_new = W_t.reshape(p * n, rank)
    
    return U_new, V_new, W_new

def _try_round_and_verify(U, V, W, n, m, p):
    """Try to round to half-integers and verify."""
    U_r = [[round(x * 2) / 2 for x in row] for row in U]
    V_r = [[round(x * 2) / 2 for x in row] for row in V]
    W_r = [[round(x * 2) / 2 for x in row] for row in W]
    
    if _verify(U_r, V_r, W_r, n, m, p, tol=1e-9):
        return U_r, V_r, W_r
    return None

def _try_integer_round_and_verify(U, V, W, n, m, p):
    """Try to round to integers and verify."""
    U_r = [[round(x) for x in row] for row in U]
    V_r = [[round(x) for x in row] for row in V]
    W_r = [[round(x) for x in row] for row in W]
    
    if _verify(U_r, V_r, W_r, n, m, p, tol=1e-9):
        return U_r, V_r, W_r
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 130s safety margin out of 150s
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 42
    
    # Start with schoolbook decomposition
    best = _schoolbook(n, m, p)
    best_rank = schoolbook_rank
    
    # Lower bound is max(n*m, m*p, p*n) = max(6, 21, 14) = 21
    lower_bound = max(n * m, m * p, p * n)
    
    # Try ranks from lower_bound to best_rank - 1
    for target_rank in range(lower_bound, best_rank):
        if time.time() > deadline - 30:
            break
        
        found = False
        num_attempts = 300
        
        for attempt in range(num_attempts):
            if time.time() > deadline - 10:
                break
            
            # Random initialization with various scales
            np.random.seed(target_rank * 10000 + attempt)
            
            # Try different initialization strategies
            if attempt % 3 == 0:
                scale = 1.0
            elif attempt % 3 == 1:
                scale = 0.5
            else:
                scale = 2.0
            
            U_init = (np.random.randn(n * m, target_rank) * scale).tolist()
            V_init = (np.random.randn(m * p, target_rank) * scale).tolist()
            W_init = (np.random.randn(p * n, target_rank) * scale).tolist()
            
            # Gradient descent with momentum
            U_gd, V_gd, W_gd = _gradient_descent(
                U_init, V_init, W_init, n, m, p, 
                lr=0.05, steps=1500, tol=1e-10, momentum=0.9
            )
            
            # Check if close to solution
            if _verify(U_gd.tolist(), V_gd.tolist(), W_gd.tolist(), n, m, p, tol=1e-6):
                # Newton polish for high precision
                U_newton, V_newton, W_newton = _newton_polish(
                    U_gd, V_gd, W_gd, n, m, p, max_iter=50, tol=1e-12
                )
                
                if _verify(U_newton.tolist(), V_newton.tolist(), W_newton.tolist(), n, m, p, tol=1e-9):
                    # Try rounding to half-integers
                    rounded = _try_round_and_verify(
                        U_newton.tolist(), V_newton.tolist(), W_newton.tolist(), n, m, p
                    )
                    if rounded:
                        U_r, V_r, W_r = rounded
                        best = {"U": U_r, "V": V_r, "W": W_r}
                        best_rank = target_rank
                        found = True
                        break
                    
                    # Try rounding to integers
                    rounded_int = _try_integer_round_and_verify(
                        U_newton.tolist(), V_newton.tolist(), W_newton.tolist(), n, m, p
                    )
                    if rounded_int:
                        U_r, V_r, W_r = rounded_int
                        best = {"U": U_r, "V": V_r, "W": W_r}
                        best_rank = target_rank
                        found = True
                        break
                    else:
                        # If rounding fails, keep the floating point solution as candidate
                        # but continue searching for a cleaner one
                        pass
        
        if found:
            break
    
    return best
