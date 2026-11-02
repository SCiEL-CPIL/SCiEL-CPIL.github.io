import numpy as np

def construct():
    n, m, p = 4, 4, 5
    nm, mp, pn = n*m, m*p, p*n  # 16, 20, 20
    
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
    
    # Try rank 60 with a few seeds
    for seed in range(3):
        np.random.seed(seed)
        rank = 60
        U = np.random.randn(nm, rank) * 0.1
        V = np.random.randn(mp, rank) * 0.1
        W = np.random.randn(pn, rank) * 0.1
        
        lr = 0.01
        for step in range(10000):
            lr_current = lr / (1 + step / 500)
            
            T_full = np.einsum('ar,br,cr->abc', U, V, W)
            residual = T_full - T_target
            loss = np.sum(residual**2)
            
            dU = 2 * np.einsum('abc,br,cr->ar', residual, V, W)
            dV = 2 * np.einsum('abc,ar,cr->br', residual, U, W)
            dW = 2 * np.einsum('abc,ar,br->cr', residual, U, V)
            
            U -= lr_current * dU
            V -= lr_current * dV
            W -= lr_current * dW
            
            # Optional: early stopping if loss is very small
            if loss < 1e-12:
                break
        
        U_half = np.round(U * 2) / 2
        V_half = np.round(V * 2) / 2
        W_half = np.round(W * 2) / 2
        
        T_half = np.einsum('ar,br,cr->abc', U_half, V_half, W_half)
        error = np.sum((T_half - T_target)**2)
        
        if error < 1e-10:
            return {"U": U_half.tolist(), "V": V_half.tolist(), "W": W_half.tolist()}
            
    # Fallback to schoolbook
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
