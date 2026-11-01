import numpy as np
from scipy.optimize import minimize

def _try_rank13(deadline=140.0):
    start_time = time.time()
    n, m, p = 2, 2, 4
    r = 13
    
    # Build target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T[a_idx, b_idx, c_idx] = 1.0
    
    # Initialize from rank-14 solution, drop one column
    sol14 = _build_rank14()
    U14 = np.array(sol14['U'], dtype=float)
    V14 = np.array(sol14['V'], dtype=float)
    W14 = np.array(sol14['W'], dtype=float)
    
    # Drop the last column to get rank-13 init
    U = U14[:, :r].copy()
    V = V14[:, :r].copy()
    W = W14[:, :r].copy()
    
    # Add small random perturbation
    np.random.seed(42)
    U += 0.1 * np.random.randn(*U.shape)
    V += 0.1 * np.random.randn(*V.shape)
    W += 0.1 * np.random.randn(*W.shape)
    
    def objective(params):
        if time.time() - start_time > deadline - 5:
            return 1e10
        U_flat = params[:4*r].reshape(4, r)
        V_flat = params[4*r:4*r+8*r].reshape(8, r)
        W_flat = params[4*r+8*r:].reshape(8, r)
        
        # Compute tensor product
        D = np.zeros((4, 8, 8))
        for ri in range(r):
            D += np.einsum('a,b,c->abc', U_flat[:, ri], V_flat[:, ri], W_flat[:, ri])
        
        return np.sum((D - T)**2)
    
    def gradient(params):
        if time.time() - start_time > deadline - 5:
            return np.zeros_like(params)
        U_flat = params[:4*r].reshape(4, r)
        V_flat = params[4*r:4*r+8*r].reshape(8, r)
        W_flat = params[4*r+8*r:].reshape(8, r)
        
        D = np.zeros((4, 8, 8))
        for ri in range(r):
            D += np.einsum('a,b,c->abc', U_flat[:, ri], V_flat[:, ri], W_flat[:, ri])
        residual = D - T
        
        grad_U = np.zeros_like(U_flat)
        grad_V = np.zeros_like(V_flat)
        grad_W = np.zeros_like(W_flat)
        
        for ri in range(r):
            grad_U[:, ri] = np.einsum('abc,bc->a', residual, V_flat[:, ri] * W_flat[:, ri])
            grad_V[:, ri] = np.einsum('abc,ac->b', residual, U_flat[:, ri] * W_flat[:, ri])
            grad_W[:, ri] = np.einsum('abc,ab->c', residual, U_flat[:, ri] * V_flat[:, ri])
        
        return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
    
    x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    # Try L-BFGS-B
    result = minimize(objective, x0, jac=gradient, method='L-BFGS-B', 
                      options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 50000})
    
    if result.fun < 1e-10:
        U_final = result.x[:4*r].reshape(4, r)
        V_final = result.x[4*r:4*r+8*r].reshape(8, r)
        W_final = result.x[4*r+8*r:].reshape(8, r)
        
        # Round to nearest half-integer
        U_round = np.round(U_final * 2) / 2
        V_round = np.round(V_final * 2) / 2
        W_round = np.round(W_final * 2) / 2
        
        # Verify
        D = np.zeros((4, 8, 8))
        for ri in range(r):
            D += np.einsum('a,b,c->abc', U_round[:, ri], V_round[:, ri], W_round[:, ri])
        
        if np.allclose(D, T, atol=1e-10):
            return {"U": U_round.tolist(), "V": V_round.tolist(), "W": W_round.tolist()}
    
    return None
