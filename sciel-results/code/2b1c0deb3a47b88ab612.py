import numpy as np

def construct():
        n, m, p = 4, 4, 5
        nm, mp, pn = n*m, m*p, p*n
        
        # Target tensor indices
        idx_a = []
        idx_b = []
        idx_c = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_a.append(i * m + j)
                    idx_b.append(j * p + k)
                    idx_c.append(k * n + i)
        
        idx_a = np.array(idx_a)
        idx_b = np.array(idx_b)
        idx_c = np.array(idx_c)
        
        T_target = np.zeros((nm, mp, pn))
        T_target[idx_a, idx_b, idx_c] = 1.0
        
        best_rank = 80
        best_factors = None
        
        # Try ranks from low to high
        for rank in [40, 50, 60, 70, 80]:
            # Initialize with complex random values
            np.random.seed(rank)
            U = (np.random.randn(nm, rank) + 1j * np.random.randn(nm, rank)) * 0.1
            V = (np.random.randn(mp, rank) + 1j * np.random.randn(mp, rank)) * 0.1
            W = (np.random.randn(pn, rank) + 1j * np.random.randn(pn, rank)) * 0.1
            
            # Adam Optimizer parameters
            lr = 0.01
            beta1 = 0.9
            beta2 = 0.999
            eps = 1e-8
            m_U = np.zeros_like(U)
            v_U = np.zeros_like(U)
            m_V = np.zeros_like(V)
            v_V = np.zeros_like(V)
            m_W = np.zeros_like(W)
            v_W = np.zeros_like(W)
            
            for step in range(2000):
                # Compute residual
                T_full = np.einsum('ar,br,cr->abc', U, V, W)
                residual = T_full - T_target
                loss = np.sum(np.abs(residual)**2).real
                
                # Gradients
                # dL/dU[a,r] = 2 * conj(U) * sum_{b,c} residual[a,b,c] * V[b,r] * W[c,r] ??
                # L = sum |T_full - T_target|^2
                # dL/dU = 2 * Re( conj(U) * grad_U )?
                # Let's use real-valued gradient descent on real and imaginary parts separately?
                # Or just use the fact that for real tensors, the optimal solution is real.
                # We can just take the real part of the gradient of the real part of the loss?
                # Simpler: Optimize real and imaginary parts as separate variables.
                
                # Let's stick to real-valued optimization but with a good start.
                # The complex init might help escape local minima if we allow complex steps,
                # but the target is real.
                # Let's just use real-valued Adam.
                
                break # Placeholder for logic change
            
            # Real-valued Adam implementation
            U = np.random.randn(nm, rank) * 0.1
            V = np.random.randn(mp, rank) * 0.1
            W = np.random.randn(pn, rank) * 0.1
            
            m_U = np.zeros_like(U)
            v_U = np.zeros_like(U)
            m_V = np.zeros_like(V)
            v_V = np.zeros_like(V)
            m_W = np.zeros_like(W)
            v_W = np.zeros_like(W)
            
            for step in range(3000):
                T_full = np.einsum('ar,br,cr->abc', U, V, W)
                residual = T_full - T_target
                loss = np.sum(residual**2)
                
                dU = 2 * np.einsum('abc,br,cr->ar', residual, V, W)
                dV = 2 * np.einsum('abc,ar,cr->br', residual, U, W)
                dW = 2 * np.einsum('abc,ar,br->cr', residual, U, V)
                
                t = step + 1
                m_U = beta1 * m_U + (1 - beta1) * dU
                v_U = beta2 * v_U + (1 - beta2) * (dU ** 2)
                m_U_hat = m_U / (1 - beta1 ** t)
                v_U_hat = v_U / (1 - beta2 ** t)
                U -= lr * m_U_hat / (np.sqrt(v_U_hat) + eps)
                
                m_V = beta1 * m_V + (1 - beta1) * dV
                v_V = beta2 * v_V + (1 - beta2) * (dV ** 2)
                m_V_hat = m_V / (1 - beta1 ** t)
                v_V_hat = v_V / (1 - beta2 ** t)
                V -= lr * m_V_hat / (np.sqrt(v_V_hat) + eps)
                
                m_W = beta1 * m_W + (1 - beta1) * dW
                v_W = beta2 * v_W + (1 - beta2) * (dW ** 2)
                m_W_hat = m_W / (1 - beta1 ** t)
                v_W_hat = v_W / (1 - beta2 ** t)
                W -= lr * m_W_hat / (np.sqrt(v_W_hat) + eps)
                
                if step % 500 == 0:
                    # Try to snap and check
                    U_snap = np.round(U * 2) / 2
                    V_snap = np.round(V * 2) / 2
                    W_snap = np.round(W * 2) / 2
                    T_snap = np.einsum('ar,br,cr->abc', U_snap, V_snap, W_snap)
                    err = np.sum((T_snap - T_target)**2)
                    if err < 1e-9:
                        best_rank = rank
                        best_factors = (U_snap, V_snap, W_snap)
                        break
            
            if best_factors is not None:
                break
        
        if best_factors is not None:
            U, V, W = best_factors
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        else:
            # Fallback to schoolbook rank 80
            rank_sb = n * m * p
            U_sb = [[0] * rank_sb for _ in range(n * m)]
            V_sb = [[0] * rank_sb for _ in range(m * p)]
            W_sb = [[0] * rank_sb for _ in range(p * n)]
            column = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_sb[i * m + j][column] = 1
                        V_sb[j * p + k][column] = 1
                        W_sb[k * n + i][column] = 1
                        column += 1
            return {"U": U_sb, "V": V_sb, "W": W_sb}
