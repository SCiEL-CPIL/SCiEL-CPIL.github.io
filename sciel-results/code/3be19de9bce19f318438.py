import numpy as np
import time
import sys

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 10s buffer
    
    nm, mp, pn = n*m, m*p, p*n
    
    # Build Target Tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
                
    # Schoolbook Decomposition (Rank 60)
    rank_sb = nm * p # 12 * 5 = 60? No, n*m*p = 3*4*5 = 60.
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1

    def compute_loss_grad(U, V, W, T):
        """
        Compute loss and gradients for U, V, W.
        U: (nm, r), V: (mp, r), W: (pn, r)
        """
        r = U.shape[1]
        # Predicted Tensor
        # Efficiently: pred[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # We can compute this by iterating r, or using einsum.
        # einsum is fast for small tensors.
        pred = np.einsum('ar,br,cr->abc', U, V, W)
        error = pred - T
        loss = np.sum(error**2)
        
        # Gradients
        # dL/dU[a,r] = 2 * sum_{b,c} error[a,b,c] V[b,r] W[c,r]
        # dL/dV[b,r] = 2 * sum_{a,c} error[a,b,c] U[a,r] W[c,r]
        # dL/dW[c,r] = 2 * sum_{a,b} error[a,b,c] U[a,r] V[b,r]
        
        # Vectorized gradient computation
        # error shape (nm, mp, pn)
        # V shape (mp, r), W shape (pn, r)
        # dU shape (nm, r)
        # dU[a,r] = sum_b,c error[a,b,c] V[b,r] W[c,r]
        # This is like a tensor contraction.
        
        # Using einsum for gradients
        # dU = 2 * einsum('abc,br,cr->ar', error, V, W)
        # dV = 2 * einsum('abc,ar,cr->br', error, U, W)
        # dW = 2 * einsum('abc,ar,br->cr', error, U, V)
        
        dU = 2.0 * np.einsum('abc,br,cr->ar', error, V, W)
        dV = 2.0 * np.einsum('abc,ar,cr->br', error, U, W)
        dW = 2.0 * np.einsum('abc,ar,br->cr', error, U, V)
        
        return loss, dU, dV, dW

    def adam_step(params, grads, m_params, v_params, t, lr=0.01, b1=0.9, b2=0.999, eps=1e-8):
        """
        params: list of arrays
        grads: list of arrays
        m_params, v_params: list of moment arrays
        """
        updated = []
        for i in range(len(params)):
            p = params[i]
            g = grads[i]
            m_p = m_params[i]
            v_p = v_params[i]
            
            m_p[:] = b1 * m_p + (1 - b1) * g
            v_p[:] = b2 * v_p + (1 - b2) * (g * g)
            
            m_hat = m_p / (1 - b1**t)
            v_hat = v_p / (1 - b2**t)
            
            p_new = p - lr * m_hat / (np.sqrt(v_hat) + eps)
            updated.append(p_new)
        return updated

    def try_rank(r, seed, max_iter=1000):
        np.random.seed(seed)
        
        # Initialization: Schoolbook + Noise
        # We need r terms. If r < 60, we can't just use schoolbook.
        # We initialize randomly, but scaled to match the magnitude of T.
        # ||T||_F^2 = 60.
        # ||U|| ~ 1, ||V|| ~ 1, ||W|| ~ 1 => sum ~ r.
        # So scale by 1/sqrt(r) ?
        
        # Better Init: Random Orthogonal-like
        U = np.random.randn(nm, r) * 0.5
        V = np.random.randn(mp, r) * 0.5
        W = np.random.randn(pn, r) * 0.5
        
        # Adam State
        m_U = np.zeros_like(U)
        v_U = np.zeros_like(U)
        m_V = np.zeros_like(V)
        v_V = np.zeros_like(V)
        m_W = np.zeros_like(W)
        v_W = np.zeros_like(W)
        
        t = 0
        best_loss = float('inf')
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            loss, dU, dV, dW = compute_loss_grad(U, V, W, T)
            
            if loss < best_loss:
                best_loss = loss
                if loss < 1e-9:
                    return U, V, W, loss
            
            # Adaptive Learning Rate: Decay over time
            lr = 0.01 / (1 + 0.01 * it)
            
            t += 1
            U, V, W = adam_step([U, V, W], [dU, dV, dW], 
                                 [m_U, m_V, m_W], [v_U, v_V, v_W], t, lr=lr)
            
            # Early stop if stuck
            if it % 100 == 0 and loss > 1e-6:
                 # Try a small random kick
                 U += np.random.randn(*U.shape) * 0.01
                 V += np.random.randn(*V.shape) * 0.01
                 W += np.random.randn(*W.shape) * 0.01

        return None

    # Try ranks from 55 to 59
    # Rank 60 is the fallback
    for r in [59, 58, 57, 56, 55]:
        if time.time() > deadline - 20:
            break
            
        # Try a few seeds for each rank
        for seed in range(5):
            if time.time() > deadline - 5:
                break
                
            res = try_rank(r, seed, max_iter=500)
            if res is not None:
                U, V, W, loss = res
                if loss < 1e-8:
                    # Convert to list for return
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback to Schoolbook
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
# EVOLVE-BLOCK-END
