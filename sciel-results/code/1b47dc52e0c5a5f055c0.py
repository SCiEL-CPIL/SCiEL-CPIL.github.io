"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    # Schoolbook baseline (always valid)
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    def check_decomposition(U, V, W, tol=1e-8):
        """Check if U,V,W form a valid decomposition.
        For all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] should equal 1.
        """
        r = U.shape[1]
        
        U_reshaped = U.reshape(n, m, r)
        V_reshaped = V.reshape(m, p, r)
        W_reshaped = W.reshape(p, n, r)
        
        T = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
        
        return np.allclose(T, np.ones((n, m, p)), atol=tol)
    
    def try_rank_optimization(target_rank, num_trials=10, max_iter=50):
        """Try to find a decomposition of given rank using iterative optimization."""
        if time.time() > deadline:
            return None
        
        n, m, p = DIMENSIONS
        nm = n * m
        mp = m * p
        pn = p * n
        
        def adam_step(params, grads, state, beta1=0.9, beta2=0.999, eps=1e-8, t=1):
            """Adam optimizer step for a list of parameter arrays."""
            for idx, (param, grad) in enumerate(zip(params, grads)):
                if idx not in state:
                    state[idx] = {'m': np.zeros_like(param), 'v': np.zeros_like(param), 't': 0}
                s = state[idx]
                s['t'] += 1
                s['m'] = beta1 * s['m'] + (1 - beta1) * grad
                s['v'] = beta2 * s['v'] + (1 - beta2) * (grad ** 2)
                m_hat = s['m'] / (1 - beta1 ** s['t'])
                v_hat = s['v'] / (1 - beta2 ** s['t'])
                param -= 0.01 * m_hat / (np.sqrt(v_hat) + eps)
        
        for trial in range(num_trials):
            if time.time() > deadline:
                return None
            
            # Structured initialization exploiting the tensor structure
            if trial % 3 == 0:
                # Identity-like block structure
                U = np.zeros((nm, target_rank))
                V = np.zeros((mp, target_rank))
                W = np.zeros((pn, target_rank))
                
                # Distribute rank dimensions across indices
                for c in range(target_rank):
                    i = c % n
                    j = (c // n) % m
                    k = (c // (n * m)) % p
                    
                    U[i*m+j, c] = 1.0
                    V[j*p+k, c] = 1.0
                    W[k*n+i, c] = 1.0
                
                # Add perturbation
                U += np.random.randn(nm, target_rank) * 0.1
                V += np.random.randn(mp, target_rank) * 0.1
                W += np.random.randn(pn, target_rank) * 0.1
            elif trial % 3 == 1:
                # Random initialization with scale
                scale = 1.0 / np.sqrt(target_rank)
                U = np.random.randn(nm, target_rank) * scale
                V = np.random.randn(mp, target_rank) * scale
                W = np.random.randn(pn, target_rank) * scale
            else:
                # Hybrid: mix structured and random
                U_struct = np.zeros((nm, target_rank))
                V_struct = np.zeros((mp, target_rank))
                W_struct = np.zeros((pn, target_rank))
                
                for c in range(target_rank):
                    i = c % n
                    j = (c // n) % m
                    k = (c // (n * m)) % p
                    U_struct[i*m+j, c] = 1.0
                    V_struct[j*p+k, c] = 1.0
                    W_struct[k*n+i, c] = 1.0
                
                U = 0.5 * U_struct + 0.5 * np.random.randn(nm, target_rank) * 0.3
                V = 0.5 * V_struct + 0.5 * np.random.randn(mp, target_rank) * 0.3
                W = 0.5 * W_struct + 0.5 * np.random.randn(pn, target_rank) * 0.3
            
            # Adam optimizer
            adam_state = {}
            best_loss = float('inf')
            best_state = (U.copy(), V.copy(), W.copy())
            
            for it in range(max_iter):
                if time.time() > deadline:
                    return None
                
                # Compute current tensor
                U_r = U.reshape(n, m, target_rank)
                V_r = V.reshape(m, p, target_rank)
                W_r = W.reshape(p, n, target_rank)
                
                T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
                loss = np.sum((T - 1.0)**2)
                
                if loss < best_loss:
                    best_loss = loss
                    best_state = (U.copy(), V.copy(), W.copy())
                
                if loss < 1e-12:
                    return (U.copy(), V.copy(), W.copy())
                
                # Compute gradients
                diff = 2.0 * (T - 1.0)
                
                grad_U = np.einsum('ijk,jkc,kic->ijc', diff, V_r, W_r).reshape(nm, target_rank)
                grad_V = np.einsum('ijk,ijc,kic->jkc', diff, U_r, W_r).reshape(mp, target_rank)
                grad_W = np.einsum('ijk,ijc,jkc->kic', diff, U_r, V_r).reshape(pn, target_rank)
                
                # Adam update
                params = [U, V, W]
                grads = [grad_U, grad_V, grad_W]
                adam_step(params, grads, adam_state, t=it+1)
            
            # Final high-precision polish pass if close
            if best_loss < 1e-4:
                U, V, W = best_state
                
                # Newton-like polish with smaller steps
                for polish_iter in range(100):
                    if time.time() > deadline:
                        break
                    
                    U_r = U.reshape(n, m, target_rank)
                    V_r = V.reshape(m, p, target_rank)
                    W_r = W.reshape(p, n, target_rank)
                    
                    T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
                    loss = np.sum((T - 1.0)**2)
                    
                    if loss < 1e-14:
                        break
                    
                    diff = 2.0 * (T - 1.0)
                    grad_U = np.einsum('ijk,jkc,kic->ijc', diff, V_r, W_r).reshape(nm, target_rank)
                    grad_V = np.einsum('ijk,ijc,kic->jkc', diff, U_r, W_r).reshape(mp, target_rank)
                    grad_W = np.einsum('ijk,ijc,jkc->kic', diff, U_r, V_r).reshape(pn, target_rank)
                    
                    # Adaptive step size based on gradient magnitude
                    grad_norm = np.sqrt(np.sum(grad_U**2) + np.sum(grad_V**2) + np.sum(grad_W**2))
                    step_size = 0.005 / (1 + grad_norm * 0.1)
                    
                    U -= step_size * grad_U
                    V -= step_size * grad_V
                    W -= step_size * grad_W
                
                if check_decomposition(U, V, W):
                    return (U.copy(), V.copy(), W.copy())
            
            # Check best state
            if check_decomposition(best_state[0], best_state[1], best_state[2]):
                return best_state
        
        return None
    
    # Try various ranks - start higher and work down, with more trials for promising ranks
    # Focus on ranks that are more likely to be achievable
    for target_rank in [40, 38, 36, 34, 32, 30, 28, 26, 24, 22, 20]:
        if time.time() > deadline:
            break
        
        # Allocate more trials for lower ranks (harder to find)
        num_trials = 8 if target_rank <= 25 else 5
        max_iter = 200 if target_rank <= 25 else 150
        
        result = try_rank_optimization(target_rank, num_trials=num_trials, max_iter=max_iter)
        if result is not None:
            U, V, W = result
            if check_decomposition(U, V, W):
                best_rank = target_rank
                best_U, best_V, best_W = U, V, W
                # Don't break immediately - try one more lower rank if time permits
                if time.time() < deadline - 40:
                    continue
                else:
                    break
    
    # Final check
    if not check_decomposition(best_U, best_V, best_W):
        # Fallback to schoolbook
        best_rank = rank_sb
        best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(), 
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
