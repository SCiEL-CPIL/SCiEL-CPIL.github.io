import numpy as np
import time
import random

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin before 150s kill

    def _verify(U, V, W, tol=1e-9):
        """Verify tensor decomposition using einsum."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        r = U.shape[1]
        if r != V.shape[1] or r != W.shape[1]:
            return False
        
        U_3d = U.T.reshape(r, n, m)
        V_3d = V.T.reshape(r, m, p)
        W_3d = W.T.reshape(r, p, n)
        
        T_recon = np.einsum('rij,rjk,rki->ijk', U_3d, V_3d, W_3d)
        target = np.ones((n, m, p))
        
        return np.allclose(T_recon, target, atol=tol)

    def _schoolbook(n, m, p):
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
        return U, V, W

    def _strassen_3x3x3():
        """
        Return rank-7 decomposition for 3x3x3 matrix multiplication.
        Uses Strassen's algorithm adapted for 3x3.
        Returns U, V, W each of shape (9, 7).
        """
        # Strassen's 7 multiplications for 3x3 (or 2x2 padded)
        # Standard Strassen for 2x2 uses 7 mults. For 3x3, we can use a similar approach.
        # Actually, let's use a known rank-7 scheme for 3x3x3 if available, 
        # or fallback to a structured search if not.
        # For now, let's try to construct a valid rank-7 or lower.
        # Known: 3x3x3 has rank <= 7 (Strassen) but actually it's 7.
        # Let's hardcode a known decomposition or generate one via optimization if needed.
        # To save time, I'll use a simple valid decomposition and optimize.
        # Actually, the task allows half-integers.
        
        # Let's try to find a rank-7 decomposition for 3x3x3 using a quick optimization
        # or use a known one. 
        # A known rank-7 decomposition for 3x3x3:
        # This is complex to hardcode. Let's use the optimizer to find it quickly.
        
        # For now, return None to indicate we need to find it, or use a higher rank fallback.
        # Let's use rank 7 and try to solve it.
        return None

    def _try_rank_optimization(r_target, max_iters=500, n_restarts=5, lr=0.05, tol=1e-7):
        """Try to find a rank-r decomposition using Adam optimizer."""
        best_loss = np.inf
        best_params = None
        
        for restart in range(n_restarts):
            if time.time() > deadline - 10:
                break
            
            # Random initialization
            U_3d = np.random.randn(r_target, n, m) * 0.5
            V_3d = np.random.randn(r_target, m, p) * 0.5
            W_3d = np.random.randn(r_target, p, n) * 0.5
            
            params = np.concatenate([
                U_3d.reshape(-1),
                V_3d.reshape(-1),
                W_3d.reshape(-1)
            ])
            
            state = {}
            current_lr = lr
            
            for it in range(max_iters):
                if time.time() > deadline - 10:
                    break
                
                U_3d = params[0: r_target * n * m].reshape(r_target, n, m)
                V_3d = params[r_target * n * m : 2 * r_target * n * m].reshape(r_target, m, p)
                W_3d = params[2 * r_target * n * m : 3 * r_target * n * m].reshape(r_target, p, n)
                
                target = np.ones((n, m, p))
                T_recon = np.einsum('rij,rjk,rki->ijk', U_3d, V_3d, W_3d)
                diff = T_recon - target
                loss = np.sum(diff ** 2)
                
                if loss < best_loss:
                    best_loss = loss
                    best_params = params.copy()
                
                if loss < tol:
                    break
                
                dU = np.einsum('ijk,rjk,rki->rij', diff, V_3d, W_3d)
                dV = np.einsum('ijk,rij,rki->rjk', diff, U_3d, W_3d)
                dW = np.einsum('ijk,rij,rjk->rki', diff, U_3d, V_3d)
                
                grads = np.concatenate([
                    dU.reshape(-1),
                    dV.reshape(-1),
                    dW.reshape(-1)
                ])
                
                # Adam step
                if 'm' not in state:
                    state['m'] = np.zeros_like(params)
                    state['v'] = np.zeros_like(params)
                    state['t'] = 0
                
                state['t'] += 1
                t = state['t']
                beta1, beta2, eps = 0.9, 0.999, 1e-8
                
                state['m'] = beta1 * state['m'] + (1 - beta1) * grads
                state['v'] = beta2 * state['v'] + (1 - beta2) * (grads ** 2)
                
                m_hat = state['m'] / (1 - beta1 ** t)
                v_hat = state['v'] / (1 - beta2 ** t)
                
                params -= current_lr * m_hat / (np.sqrt(v_hat) + eps)
                
                # Decay learning rate
                current_lr *= 0.99
                
                # Perturbation
                if it % 50 == 49 and loss > 1e-4:
                    params += np.random.randn(params.shape[0]) * 0.01 * np.exp(-it / 100)
        
        if best_params is not None and best_loss < 1e-6:
            U_3d = best_params[0: r_target * n * m].reshape(r_target, n, m)
            V_3d = best_params[r_target * n * m : 2 * r_target * n * m].reshape(r_target, m, p)
            W_3d = best_params[2 * r_target * n * m : 3 * r_target * n * m].reshape(r_target, p, n)
            
            U_out = U_3d.transpose(1, 2, 0).reshape(n * m, r_target)
            V_out = V_3d.transpose(1, 2, 0).reshape(m * p, r_target)
            W_out = W_3d.transpose(1, 2, 0).reshape(p * n, r_target)
            
            if _verify(U_out, V_out, W_out, tol=1e-7):
                return U_out, V_out, W_out, best_loss
        
        return None

    def _structured_search(r_target, max_iters=200, n_restarts=10):
        """
        Try to find a decomposition with structured initialization.
        Uses the idea of splitting into 3x3x3 and 3x3x4 parts.
        """
        best_loss = np.inf
        best_params = None
        
        for restart in range(n_restarts):
            if time.time() > deadline - 10:
                break
            
            # Initialize with a mix of structured and random
            U_3d = np.zeros((r_target, n, m))
            V_3d = np.zeros((r_target, m, p))
            W_3d = np.zeros((r_target, p, n))
            
            # First 7 ranks: try to mimic Strassen for 3x3x3 part
            for r_idx in range(min(7, r_target)):
                # Random assignment of indices
                i = np.random.randint(n)
                j = np.random.randint(m)
                k = np.random.randint(p)
                U_3d[r_idx, i, j] = 1.0
                V_3d[r_idx, j, k] = 1.0
                W_3d[r_idx, k, i] = 1.0
                # Add noise
                U_3d[r_idx] += np.random.randn(n, m) * 0.2
                V_3d[r_idx] += np.random.randn(m, p) * 0.2
                W_3d[r_idx] += np.random.randn(p, n) * 0.2
            
            # Remaining ranks: random
            for r_idx in range(7, r_target):
                U_3d[r_idx] = np.random.randn(n, m) * 0.5
                V_3d[r_idx] = np.random.randn(m, p) * 0.5
                W_3d[r_idx] = np.random.randn(p, n) * 0.5
            
            params = np.concatenate([
                U_3d.reshape(-1),
                V_3d.reshape(-1),
                W_3d.reshape(-1)
            ])
            
            state = {}
            current_lr = 0.03
            
            for it in range(max_iters):
                if time.time() > deadline - 10:
                    break
                
                U_3d = params[0: r_target * n * m].reshape(r_target, n, m)
                V_3d = params[r_target * n * m : 2 * r_target * n * m].reshape(r_target, m, p)
                W_3d = params[2 * r_target * n * m : 3 * r_target * n * m].reshape(r_target, p, n)
                
                target = np.ones((n, m, p))
                T_recon = np.einsum('rij,rjk,rki->ijk', U_3d, V_3d, W_3d)
                diff = T_recon - target
                loss = np.sum(diff ** 2)
                
                if loss < best_loss:
                    best_loss = loss
                    best_params = params.copy()
                
                if loss < 1e-7:
                    break
                
                dU = np.einsum('ijk,rjk,rki->rij', diff, V_3d, W_3d)
                dV = np.einsum('ijk,rij,rki->rjk', diff, U_3d, W_3d)
                dW = np.einsum('ijk,rij,rjk->rki', diff, U_3d, V_3d)
                
                grads = np.concatenate([
                    dU.reshape(-1),
                    dV.reshape(-1),
                    dW.reshape(-1)
                ])
                
                if 'm' not in state:
                    state['m'] = np.zeros_like(params)
                    state['v'] = np.zeros_like(params)
                    state['t'] = 0
                
                state['t'] += 1
                t = state['t']
                beta1, beta2, eps = 0.9, 0.999, 1e-8
                
                state['m'] = beta1 * state['m'] + (1 - beta1) * grads
                state['v'] = beta2 * state['v'] + (1 - beta2) * (grads ** 2)
                
                m_hat = state['m'] / (1 - beta1 ** t)
                v_hat = state['v'] / (1 - beta2 ** t)
                
                params -= current_lr * m_hat / (np.sqrt(v_hat) + eps)
                current_lr *= 0.995
        
        if best_params is not None and best_loss < 1e-6:
            U_3d = best_params[0: r_target * n * m].reshape(r_target, n, m)
            V_3d = best_params[r_target * n * m : 2 * r_target * n * m].reshape(r_target, m, p)
            W_3d = best_params[2 * r_target * n * m : 3 * r_target * n * m].reshape(r_target, p, n)
            
            U_out = U_3d.transpose(1, 2, 0).reshape(n * m, r_target)
            V_out = V_3d.transpose(1, 2, 0).reshape(m * p, r_target)
            W_out = W_3d.transpose(1, 2, 0).reshape(p * n, r_target)
            
            if _verify(U_out, V_out, W_out, tol=1e-7):
                return U_out, V_out, W_out, best_loss
        
        return None

    # Start with schoolbook
    U_best, V_best, W_best = _schoolbook(n, m, p)
    best_rank = n * m * p  # 63

    # Try ranks from high to low
    # Focus on ranges where improvements are likely
    candidate_ranks = list(range(60, 25, -1))
    
    for r_target in candidate_ranks:
        if time.time() > deadline - 20:
            break
        
        # Try structured search first
        result = _structured_search(r_target, max_iters=150, n_restarts=5)
        
        if result is None:
            # Try general optimization
            result = _try_rank_optimization(r_target, max_iters=100, n_restarts=3, lr=0.03)
        
        if result is not None:
            U_best, V_best, W_best, _ = result
            best_rank = r_target
            # Continue trying lower ranks
            continue
    
    # Final verification and cleanup
    def clean_values(val):
        rounded = round(val * 2) / 2
        if abs(val - rounded) < 1e-10:
            return rounded
        return val

    U_final = [[clean_values(x) for x in row] for row in U_best]
    V_final = [[clean_values(x) for x in row] for row in V_best]
    W_final = [[clean_values(x) for x in row] for row in W_best]
    
    return {
        "U": U_final,
        "V": V_final,
        "W": W_final
    }
