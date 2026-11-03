import numpy as np

def construct():
    n, m, p = 4, 4, 5
    nm, mp, pn = n*m, m*p, p*n  # 16, 20, 20
    
    # Create the sparse tensor indices
    idx_a = []
    idx_b = []
    idx_c = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                idx_a.append(a)
                idx_b.append(b)
                idx_c.append(c)
    
    idx_a = np.array(idx_a)
    idx_b = np.array(idx_b)
    idx_c = np.array(idx_c)
    
    T_target = np.zeros((nm, mp, pn))
    T_target[idx_a, idx_b, idx_c] = 1.0
    
    def optimize(rank, seed, max_steps=20000):
        np.random.seed(seed)
        # Initialize with random complex values as per directive
        U = (np.random.randn(nm, rank) + 1j * np.random.randn(nm, rank)) * 0.1
        V = (np.random.randn(mp, rank) + 1j * np.random.randn(mp, rank)) * 0.1
        W = (np.random.randn(pn, rank) + 1j * np.random.randn(pn, rank)) * 0.1
        
        # RMSProp parameters
        lr = 0.01
        decay = 0.9
        eps = 1e-8
        mU = np.zeros_like(U)
        mV = np.zeros_like(V)
        mW = np.zeros_like(W)
        
        for step in range(max_steps):
            # Compute full tensor
            T_full = np.einsum('ar,br,cr->abc', U, V, W)
            residual = T_full - T_target
            loss = np.sum(np.abs(residual)**2)
            
            if loss < 1e-12:
                break
            
            # Gradients (Wirtinger derivatives for real-valued loss)
            # dL/dconj(U) = 2 * sum_bc residual[a,b,c] * V[b,r] * W[c,r]
            # Note: residual is complex. T_target is real.
            dU = 2 * np.einsum('abc,br,cr->ar', residual, V, W)
            dV = 2 * np.einsum('abc,ar,cr->br', residual, U, W)
            dW = 2 * np.einsum('abc,ar,br->cr', residual, U, V)
            
            # RMSProp Update
            # We update U using dU.
            # mU is moving average of |dU|^2? Or dU * conj(dU)?
            # Standard RMSProp for complex: use |grad|^2.
            mU = decay * mU + (1 - decay) * (np.abs(dU) ** 2)
            mV = decay * mV + (1 - decay) * (np.abs(dV) ** 2)
            mW = decay * mW + (1 - decay) * (np.abs(dW) ** 2)
            
            U -= lr * dU / (np.sqrt(mU) + eps)
            V -= lr * dV / (np.sqrt(mV) + eps)
            W -= lr * dW / (np.sqrt(mW) + eps)
            
        # Check if imaginary parts are small
        if np.max(np.abs(np.imag(U))) > 1e-4 or np.max(np.abs(np.imag(V))) > 1e-4 or np.max(np.abs(np.imag(W))) > 1e-4:
            # If not real, try to force real by taking real part? 
            # Or just fail this seed.
            # If the solution is real, complex init should converge to real.
            # If it doesn't, it's likely a local min or wrong rank.
            pass
            
        U_real = np.real(U)
        V_real = np.real(V)
        W_real = np.real(W)
        
        # Snap to half-integers
        U_half = np.round(U_real * 2) / 2
        V_half = np.round(V_real * 2) / 2
        W_half = np.round(W_real * 2) / 2
        
        # Verify
        T_half = np.einsum('ar,br,cr->abc', U_half, V_half, W_half)
        error = np.max(np.abs(T_half - T_target))
        
        if error < 1e-9:
            return {"U": U_half.tolist(), "V": V_half.tolist(), "W": W_half.tolist()}
        else:
            return None

    # Try ranks from low to high
    for rank in range(40, 81):
        for seed in range(10):
            result = optimize(rank, seed)
            if result is not None:
                return result
                
    # Fallback to schoolbook if optimization fails
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
