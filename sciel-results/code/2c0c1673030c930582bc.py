import numpy as np
import time

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin for safety

    # Target tensor T[i, j, k] = 1
    # We want sum_r U[i, j, r] V[j, k, r] W[k, i, r] = 1
    
    best_rank = 17
    best_U, best_V, best_W = None, None, None
    
    # Store the current rank 17 solution as fallback
    U17 = np.zeros((n * m, 17))
    V17 = np.zeros((m * p, 17))
    W17 = np.zeros((p * n, 17))
    
    def set_U17(i, j, r, val): U17[i * m + j, r] = val
    def set_V17(j, k, r, val): V17[j * p + k, r] = val
    def set_W17(k, i, r, val): W17[k * n + i, r] = val

    # Copy rank 17 solution from the prompt
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
    
    best_U, best_V, best_W = U17, V17, W17

    def check_exact(U, V, W):
        # Snap to half integers
        U_s = np.round(U * 2) / 2.0
        V_s = np.round(V * 2) / 2.0
        W_s = np.round(W * 2) / 2.0
        
        # Convert to integers (multiply by 2)
        Ui = np.rint(U_s * 2).astype(int)
        Vi = np.rint(V_s * 2).astype(int)
        Wi = np.rint(W_s * 2).astype(int)
        
        # Reshape
        U_t = Ui.reshape(n, m, -1)
        V_t = Vi.reshape(m, p, -1)
        W_t = Wi.reshape(p, n, -1)
        
        # Compute product: sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        # Should be 8 (since 1/2 * 1/2 * 1/2 = 1/8, and we want 1, so 8 * 1 = 8)
        prod = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
        
        if np.all(prod == 8):
            return True, U_s, V_s, W_s
        return False, None, None, None

    def compute_grads(U, V, W, rank):
        U_t = U.reshape(n, m, rank)
        V_t = V.reshape(m, p, rank)
        W_t = W.reshape(p, n, rank)
        
        prod = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
        res = prod - 1.0
        
        dU = 2 * np.einsum('ijk,jkr,kir->ijr', res, V_t, W_t)
        dV = 2 * np.einsum('ijk,ijr,kir->jkr', res, U_t, W_t)
        dW = 2 * np.einsum('ijk,ijr,jkr->kir', res, U_t, V_t)
        
        return dU, dV, dW, np.sum(res**2)

    def polish(U, V, W, rank, max_steps=200):
        """High-precision polish pass with adaptive step size"""
        best_err = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        lr = 0.05
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
                
            dU, dV, dW, err = compute_grads(U, V, W, rank)
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if err < 1e-10:
                ok, Us, Vs, Ws = check_exact(U, V, W)
                if ok:
                    return Us, Vs, Ws, True
            
            # Adaptive learning rate
            if step % 10 == 0:
                lr = max(0.001, lr * 0.95)
            
            # Update with line search
            alpha = lr
            for _ in range(5):
                U_new = U - alpha * dU.reshape(n * m, rank)
                V_new = V - alpha * dV.reshape(m * p, rank)
                W_new = W - alpha * dW.reshape(p * n, rank)
                _, _, _, new_err = compute_grads(U_new, V_new, W_new, rank)
                if new_err < err:
                    U, V, W = U_new, V_new, W_new
                    err = new_err
                    break
                alpha *= 0.5
        
        # Final check
        if best_err < 1e-8:
            ok, Us, Vs, Ws = check_exact(best_U, best_V, best_W)
            if ok:
                return Us, Vs, Ws, True
        
        return None, None, None, False

    def try_rank_with_momentum(U, V, W, rank, max_steps=500):
        """Gradient descent with momentum and adaptive learning rate"""
        vU = np.zeros_like(U)
        vV = np.zeros_like(V)
        vW = np.zeros_like(W)
        beta = 0.9
        lr = 0.1
        best_err = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
                
            dU, dV, dW, err = compute_grads(U, V, W, rank)
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if err < 1e-6:
                # Try to polish
                ok, Us, Vs, Ws = check_exact(U, V, W)
                if ok:
                    return Us, Vs, Ws, True
                # Also try polish
                Us, Vs, Ws, found = polish(U, V, W, rank, max_steps=100)
                if found:
                    return Us, Vs, Ws, True
            
            # Momentum update
            vU = beta * vU + (1 - beta) * dU.reshape(n * m, rank)
            vV = beta * vV + (1 - beta) * dV.reshape(m * p, rank)
            vW = beta * vW + (1 - beta) * dW.reshape(p * n, rank)
            
            # Adaptive learning rate
            if step % 20 == 0:
                lr = max(0.01, lr * 0.9)
            
            U -= lr * vU
            V -= lr * vV
            W -= lr * vW
            
            # Periodic exact check
            if step % 50 == 0 and err < 0.1:
                ok, Us, Vs, Ws = check_exact(U, V, W)
                if ok:
                    return Us, Vs, Ws, True
        
        # Final polish on best found
        if best_err < 0.01:
            Us, Vs, Ws, found = polish(best_U, best_V, best_W, rank, max_steps=200)
            if found:
                return Us, Vs, Ws, True
        
        return None, None, None, False

    # Try to find lower rank
    for target_rank in [16, 15]:
        if time.time() > deadline:
            break
            
        # Increase number of seeds for lower ranks as they are harder
        num_seeds = 80 if target_rank == 16 else 120
        
        for seed in range(num_seeds):
            if time.time() > deadline:
                break
                
            np.random.seed(seed * 7919 + 13)  # Better seed spacing
            
            # Multiple initialization strategies
            init_strategy = seed % 4
            
            if init_strategy == 0:
                # Random initialization
                U = np.random.randn(n * m, target_rank) * 0.5
                V = np.random.randn(m * p, target_rank) * 0.5
                W = np.random.randn(p * n, target_rank) * 0.5
            elif init_strategy == 1:
                # Structured: perturb rank-17 solution by dropping one column
                U = U17[:, :target_rank].copy() + np.random.randn(n * m, target_rank) * 0.1
                V = V17[:, :target_rank].copy() + np.random.randn(m * p, target_rank) * 0.1
                W = W17[:, :target_rank].copy() + np.random.randn(p * n, target_rank) * 0.1
            elif init_strategy == 2:
                # Block-structured: treat rows independently then mix
                U = np.zeros((n * m, target_rank))
                V = np.zeros((m * p, target_rank))
                W = np.zeros((p * n, target_rank))
                # Row 0 contributions (first half of columns)
                half = target_rank // 2
                U[:m, :half] = np.random.randn(m, half) * 0.5
                V[:, :half] = np.random.randn(m * p, half) * 0.5
                W[:, :half] = np.random.randn(p * n, half) * 0.5
                # Row 1 contributions (second half of columns)
                U[m:, half:] = np.random.randn(m, target_rank - half) * 0.5
                V[:, half:] = np.random.randn(m * p, target_rank - half) * 0.5
                W[:, half:] = np.random.randn(p * n, target_rank - half) * 0.5
            else:
                # Sparse initialization
                U = np.zeros((n * m, target_rank))
                V = np.zeros((m * p, target_rank))
                W = np.zeros((p * n, target_rank))
                for r in range(target_rank):
                    # Random sparse pattern
                    idx_U = np.random.choice(n * m, size=max(1, n*m//3), replace=False)
                    idx_V = np.random.choice(m * p, size=max(1, m*p//3), replace=False)
                    idx_W = np.random.choice(p * n, size=max(1, p*n//3), replace=False)
                    U[idx_U, r] = np.random.choice([-1, 1], size=len(idx_U))
                    V[idx_V, r] = np.random.choice([-1, 1], size=len(idx_V))
                    W[idx_W, r] = np.random.choice([-1, 1], size=len(idx_W))
            
            # Try with momentum
            Us, Vs, Ws, found = try_rank_with_momentum(U, V, W, target_rank, max_steps=400)
            if found:
                return {
                    "U": Us.tolist(),
                    "V": Vs.tolist(),
                    "W": Ws.tolist()
                }
            
            # If close, try additional polish
            _, _, _, final_err = compute_grads(U, V, W, target_rank)
            if final_err < 0.05:
                Us, Vs, Ws, found = polish(U, V, W, target_rank, max_steps=300)
                if found:
                    return {
                        "U": Us.tolist(),
                        "V": Vs.tolist(),
                        "W": Ws.tolist()
                    }

    # Return best found
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
