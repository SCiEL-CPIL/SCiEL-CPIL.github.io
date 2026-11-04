import numpy as np
import time

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # Safety margin before 150s kill
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Schoolbook rank as fallback
    schoolbook_rank = n * m * p
    U_sch = np.zeros((n*m, schoolbook_rank))
    V_sch = np.zeros((m*p, schoolbook_rank))
    W_sch = np.zeros((p*n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sch[i*m+j, col] = 1
                V_sch[j*p+k, col] = 1
                W_sch[k*n+i, col] = 1
                col += 1
    
    best_result = {"U": U_sch.tolist(), "V": V_sch.tolist(), "W": W_sch.tolist()}
    
    # Lower bound is max(n*m, m*p, p*n) = 15
    # Theoretical lower bound for matmul is max(nm, mp, pn) = 15
    # Known: 3x3x3 has rank 23 (Strassen-like), but 3x3x5 is less studied
    # Let's try ranks from 15 up to schoolbook_rank
    
    def check_exact(U, V, W):
        """Check if U,V,W exactly decompose T"""
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T, T_approx, atol=1e-10)
    
    def try_integer_rounding(U, V, W):
        """Try to round to integers or half-integers and check exactness"""
        # Try integers
        U_int = np.round(U)
        V_int = np.round(V)
        W_int = np.round(W)
        if check_exact(U_int, V_int, W_int):
            return U_int, V_int, W_int
        
        # Try half-integers
        U_half = np.round(U * 2) / 2
        V_half = np.round(V * 2) / 2
        W_half = np.round(W * 2) / 2
        if check_exact(U_half, V_half, W_half):
            return U_half, V_half, W_half
        
        # Try rounding each coordinate independently to nearest integer/half-integer
        # with small perturbations
        for _ in range(50):
            U_try = np.round(U + np.random.randn(*U.shape) * 0.1)
            V_try = np.round(V + np.random.randn(*V.shape) * 0.1)
            W_try = np.round(W + np.random.randn(*W.shape) * 0.1)
            if check_exact(U_try, V_try, W_try):
                return U_try, V_try, W_try
        
        return None
    
    # For each target rank, use gradient descent to minimize the tensor error
    # then attempt to round to exact integer/half-integer solution
    
    for target_rank in range(15, schoolbook_rank + 1):
        if time.time() > deadline - 15:
            break
        
        max_restarts = 20 if target_rank < 25 else 10
        
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                break
            
            # Random initialization with small values
            U_init = np.random.randn(n*m, target_rank) * 0.3
            V_init = np.random.randn(m*p, target_rank) * 0.3
            W_init = np.random.randn(p*n, target_rank) * 0.3
            
            # Gradient descent with momentum
            U = U_init.copy()
            V = V_init.copy()
            W = W_init.copy()
            
            lr = 0.01
            momentum = 0.9
            vel_U = np.zeros_like(U)
            vel_V = np.zeros_like(V)
            vel_W = np.zeros_like(W)
            
            max_iters = 2000
            for it in range(max_iters):
                if time.time() > deadline - 8:
                    break
                
                # Compute T_approx
                T_approx = np.einsum('ar,br,cr->abc', U, V, W)
                diff = T - T_approx
                
                # Gradients
                grad_U = -2 * np.einsum('abc,br,cr->ar', diff, V, W)
                grad_V = -2 * np.einsum('abc,ar,cr->br', diff, U, W)
                grad_W = -2 * np.einsum('abc,ar,br->cr', diff, U, V)
                
                # Momentum update
                vel_U = momentum * vel_U + lr * grad_U
                vel_V = momentum * vel_V + lr * grad_V
                vel_W = momentum * vel_W + lr * grad_W
                
                U -= vel_U
                V -= vel_V
                W -= vel_W
                
                # Periodically check for exact solution
                if it % 200 == 0:
                    rounded = try_integer_rounding(U, V, W)
                    if rounded is not None:
                        return {"U": rounded[0].tolist(), "V": rounded[1].tolist(), "W": rounded[2].tolist()}
            
            # Final check
            rounded = try_integer_rounding(U, V, W)
            if rounded is not None:
                return {"U": rounded[0].tolist(), "V": rounded[1].tolist(), "W": rounded[2].tolist()}
        
        # If we found a valid rank, we can break because we are iterating upwards
        # But we didn't return, so continue to next rank
    
    return best_result
# EVOLVE-BLOCK-END
