import numpy as np
import time
import random

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s limit to be safe
    
    n, m, p = DIMENSIONS
    U_dim = n * m  # 10
    V_dim = m * p  # 25
    W_dim = p * n  # 10
    
    # Build the target tensor T
    T = np.zeros((U_dim, V_dim, W_dim))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0
    
    def snap_to_half_integers(arr):
        return np.round(arr * 2) / 2
        
    def is_valid_decomposition(U, V, W, T, atol=1e-8):
        prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return np.allclose(prod, T, atol=atol)
    
    def try_snap_and_validate(U, V, W, T, snap_atol=0.01):
        U_snap = snap_to_half_integers(U)
        V_snap = snap_to_half_integers(V)
        W_snap = snap_to_half_integers(W)
        
        # Check if original matrices are close to their snapped versions
        if not (np.allclose(U, U_snap, atol=snap_atol) and 
                np.allclose(V, V_snap, atol=snap_atol) and 
                np.allclose(W, W_snap, atol=snap_atol)):
            return False, None, None, None
            
        if is_valid_decomposition(U_snap, V_snap, W_snap, T):
            return True, U_snap, V_snap, W_snap
        return False, None, None, None

    def compute_loss(U, V, W):
        prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        diff = prod - T
        return np.sum(diff * diff)

    def schoolbook():
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1.0
                    V[j * p + k][column] = 1.0
                    W[k * n + i][column] = 1.0
                    column += 1
        return {"U": U, "V": V, "W": W}

    def polish_solution(U, V, W, max_iter=500):
        """High-precision polish pass using adaptive learning rate."""
        lr = 0.005
        best_loss = compute_loss(U, V, W)
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
                
            prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
            diff = prod - T
            loss = np.sum(diff * diff)
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if loss < 1e-12:
                break
            
            # Gradients
            gradU = np.einsum('abc,br,cr->ar', diff, V, W, optimize=True)
            gradV = np.einsum('abc,ar,cr->br', diff, U, W, optimize=True)
            gradW = np.einsum('abc,ar,br->cr', diff, U, V, optimize=True)
            
            # Adaptive learning rate
            U_new = U - lr * gradU
            V_new = V - lr * gradV
            W_new = W - lr * gradW
            
            new_loss = compute_loss(U_new, V_new, W_new)
            
            if new_loss < loss:
                U, V, W = U_new, V_new, W_new
                lr = min(lr * 1.1, 0.05)
            else:
                lr *= 0.8
                if lr < 1e-6:
                    break
        
        return best_U, best_V, best_W, best_loss

    def gradient_descent(U, V, W, max_iter=300):
        """Main gradient descent with momentum and adaptive LR."""
        lr = 0.02
        momentum = 0.9
        vU = np.zeros_like(U)
        vV = np.zeros_like(V)
        vW = np.zeros_like(W)
        
        best_loss = compute_loss(U, V, W)
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        no_improve = 0
        prev_loss = float('inf')
        
        for it in range(max_iter):
            if time.time() > deadline - 10:
                break
                
            # Check for valid solution periodically
            if it % 10 == 0:
                valid, U_snap, V_snap, W_snap = try_snap_and_validate(U, V, W, T)
                if valid:
                    return U_snap, V_snap, W_snap, True
            
            prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
            diff = prod - T
            loss = np.sum(diff * diff)
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if loss < 1e-10:
                break
            
            # Gradients
            gradU = np.einsum('abc,br,cr->ar', diff, V, W, optimize=True)
            gradV = np.einsum('abc,ar,cr->br', diff, U, W, optimize=True)
            gradW = np.einsum('abc,ar,br->cr', diff, U, V, optimize=True)
            
            # Momentum update
            vU = momentum * vU + lr * gradU
            vV = momentum * vV + lr * gradV
            vW = momentum * vW + lr * gradW
            
            U -= vU
            V -= vV
            W -= vW
            
            # Adaptive learning rate
            if loss < prev_loss - 1e-10:
                prev_loss = loss
                no_improve = 0
                lr = min(lr * 1.02, 0.1)
            else:
                no_improve += 1
                lr *= 0.95
                if no_improve > 30:
                    break
        
        # Final check
        valid, U_snap, V_snap, W_snap = try_snap_and_validate(U, V, W, T)
        if valid:
            return U_snap, V_snap, W_snap, True
        
        # Try polish on best solution
        U_pol, V_pol, W_pol, pol_loss = polish_solution(best_U, best_V, best_W)
        valid, U_snap, V_snap, W_snap = try_snap_and_validate(U_pol, V_pol, W_pol, T)
        if valid:
            return U_snap, V_snap, W_snap, True
        
        return best_U, best_V, best_W, False

    best_solution = None
    best_rank = n * m * p

    # Try ranks from 45 to 49
    ranks_to_try = [45, 46, 47, 48, 49]
    
    for target_rank in ranks_to_try:
        if time.time() > deadline - 25:
            break
            
        R = target_rank
        found = False
        
        # Number of restarts: more for lower ranks
        num_restarts = 12 if R <= 45 else 8 if R <= 47 else 5
        
        for restart in range(num_restarts):
            if time.time() > deadline - 15:
                break
            
            random.seed(42 + restart * 1000 + R)
            np.random.seed(42 + restart * 1000 + R)
            
            # Initialize with some structure: start from random but with small magnitude
            U = np.random.randn(U_dim, R) * 0.3
            V = np.random.randn(V_dim, R) * 0.3
            W = np.random.randn(W_dim, R) * 0.3
            
            # Run gradient descent
            U_final, V_final, W_final, success = gradient_descent(U, V, W)
            
            if success:
                best_solution = (U_final, V_final, W_final)
                best_rank = R
                found = True
                break
        
        if found:
            break
    
    if best_solution:
        U, V, W = best_solution
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        return schoolbook()

# EVOLVE-BLOCK-END
