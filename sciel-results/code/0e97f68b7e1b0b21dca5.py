import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    def solve_rank(R):
        max_restarts = 10
        # Use a slightly larger step size and more iterations
        for restart in range(max_restarts):
            if time.time() > deadline:
                return None
            
            # Random init
            U = np.random.randn(n, m, R) * 0.5
            V = np.random.randn(m, p, R) * 0.5
            W = np.random.randn(p, n, R) * 0.5
            
            # Gradient Descent with adaptive step
            step = 0.1
            for it in range(2000):
                if time.time() > deadline:
                    return None
                
                S = np.einsum('ijr,jkr,kir->ijk', U, V, W)
                E = S - 1.0
                loss = np.sum(E**2)
                
                if loss < 1e-10:
                    return U, V, W
                
                dU = 2.0 * np.einsum('ijk,jkr,kir->ijr', E, V, W)
                dV = 2.0 * np.einsum('ijk,ijr,kir->jkr', E, U, W)
                dW = 2.0 * np.einsum('ijk,ijr,jkr->kir', E, U, V)
                
                grad_norm = np.sqrt(np.sum(dU**2) + np.sum(dV**2) + np.sum(dW**2))
                if grad_norm < 1e-8:
                    break
                    
                # Adaptive step: reduce if loss increases? 
                # Simple: fixed direction, small step
                scale = 0.01 / (grad_norm + 1e-8)
                
                U_new = U - scale * dU
                V_new = V - scale * dV
                W_new = W - scale * dW
                
                # Check if loss improved
                S_new = np.einsum('ijr,jkr,kir->ijk', U_new, V_new, W_new)
                loss_new = np.sum((S_new - 1.0)**2)
                
                if loss_new < loss:
                    U, V, W = U_new, V_new, W_new
                else:
                    # Try smaller step
                    scale *= 0.5
                    if scale < 1e-6:
                        break
                    U_new = U - scale * dU
                    V_new = V - scale * dV
                    W_new = W - scale * dW
                    S_new = np.einsum('ijr,jkr,kir->ijk', U_new, V_new, W_new)
                    loss_new = np.sum((S_new - 1.0)**2)
                    if loss_new < loss:
                        U, V, W = U_new, V_new, W_new
                    else:
                        break
                        
        return None

    # Try ranks
    # We know 30 works.
    # Try 15 to 29.
    for R in range(15, 30):
        if time.time() > deadline - 5:
            break
            
        result = solve_rank(R)
        if result is not None:
            U, V, W = result
            # Round to nearest 0.5
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            # Verify
            S_check = np.einsum('ijr,jkr,kir->ijk', U_rounded, V_rounded, W_rounded)
            err = np.sum((S_check - 1.0)**2)
            
            if err < 1e-6:
                return {
                    "U": U_rounded.reshape(n*m, R),
                    "V": V_rounded.reshape(m*p, R),
                    "W": W_rounded.reshape(p*n, R)
                }
            else:
                # If rounding failed, maybe return floats? 
                # But contract says integer/half-integer.
                # If we can't satisfy the type constraint, it's invalid.
                # So we continue searching.
                pass

    # Fallback: Schoolbook
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
