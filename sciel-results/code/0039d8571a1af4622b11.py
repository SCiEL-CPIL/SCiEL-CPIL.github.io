    def try_optimize(r, U_init, V_init, W_init, max_iter=300, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        """Try to optimize decomposition to rank r with Adam."""
        U = U_init.copy().astype(float)
        V = V_init.copy().astype(float)
        W = W_init.copy().astype(float)
        
        mU = np.zeros_like(U)
        vU = np.zeros_like(U)
        mV = np.zeros_like(V)
        vV = np.zeros_like(V)
        mW = np.zeros_like(W)
        vW = np.zeros_like(W)
        
        t = 0
        best_loss = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            t += 1
            S = np.einsum('ar,br,cr->abc', U, V, W)
            err = S - T
            loss = np.sum(err**2)
            
            # Check if we are close to a half-integer solution
            if it % 10 == 0:
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                S_r = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
                if np.allclose(S_r, T, atol=1e-6):
                    return U_r, V_r, W_r, True

            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            # Gradients
            dU = 2 * np.einsum('abc,br,cr->ar', err, V, W)
            dV = 2 * np.einsum('abc,ar,cr->br', err, U, W)
            dW = 2 * np.einsum('abc,ar,br->cr', err, U, V)
            
            # Adam update
            mU = beta1 * mU + (1 - beta1) * dU
            vU = beta2 * vU + (1 - beta2) * (dU ** 2)
            mU_hat = mU / (1 - beta1 ** t)
            vU_hat = vU / (1 - beta2 ** t)
            
            mV = beta1 * mV + (1 - beta1) * dV
            vV = beta2 * vV + (1 - beta2) * (dV ** 2)
            mV_hat = mV / (1 - beta1 ** t)
            vV_hat = vV / (1 - beta2 ** t)
            
            mW = beta1 * mW + (1 - beta1) * dW
            vW = beta2 * vW + (1 - beta2) * (dW ** 2)
            mW_hat = mW / (1 - beta1 ** t)
            vW_hat = vW / (1 - beta2 ** t)
            
            U -= lr * mU_hat / (np.sqrt(vU_hat) + eps)
            V -= lr * mV_hat / (np.sqrt(vV_hat) + eps)
            W -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
            
            if loss < 1e-10:
                break
        
        # Final check with rounding
        U_r = np.round(best_U * 2) / 2
        V_r = np.round(best_V * 2) / 2
        W_r = np.round(best_W * 2) / 2
        if verify(U_r, V_r, W_r):
            return U_r, V_r, W_r, True
            
        return None, None, None, False
