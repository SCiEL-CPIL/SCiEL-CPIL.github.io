def _find_rank_10_224():
    """
    Find a rank-10 decomposition for M_{2,2,4}.
    Returns (U, V, W) or None.
    """
    import numpy as np
    from scipy.optimize import minimize
    
    n, m, p = 2, 2, 4
    nm, mp, pn = 4, 8, 8
    rank = 10
    
    # Target tensor
    T_target = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i*m+j, j*p+k, k*n+i] = 1.0
    
    def objective(params):
        U = params[:nm*rank].reshape(nm, rank)
        V = params[nm*rank:nm*rank+mp*rank].reshape(mp, rank)
        W = params[nm*rank+mp*rank:].reshape(pn, rank)
        T = np.einsum('ar,br,cr->abc', U, V, W)
        return np.sum((T - T_target)**2)
    
    # Initialize with random values
    random.seed(42)
    best_loss = float('inf')
    best_params = None
    
    for _ in range(5):
        x0 = np.random.rand(nm*rank + mp*rank + pn*rank) * 2 - 1
        res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 1000})
        if res.fun < best_loss:
            best_loss = res.fun
            best_params = res.x
    
    if best_loss < 1e-6:
        U = best_params[:nm*rank].reshape(nm, rank)
        V = best_params[nm*rank:nm*rank+mp*rank].reshape(mp, rank)
        W = best_params[nm*rank+mp*rank:].reshape(pn, rank)
        # Round to nearest half-integer or integer
        U = np.round(U * 2) / 2
        V = np.round(V * 2) / 2
        W = np.round(W * 2) / 2
        
        # Verify
        T = np.einsum('ar,br,cr->abc', U, V, W)
        if np.allclose(T, T_target, atol=1e-5):
            return U.tolist(), V.tolist(), W.tolist()
            
    return None
