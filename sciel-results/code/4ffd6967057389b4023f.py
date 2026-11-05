import scipy.optimize as so

def try_rank(r, max_iters=1000, seed=42):
    np.random.seed(seed)
    # Initialize
    U = np.random.randn(nm, r) * 0.1
    V = np.random.randn(mp, r) * 0.1
    W = np.random.randn(pn, r) * 0.1
    
    def objective(x):
        U_flat = x[:nm*r].reshape(nm, r)
        V_flat = x[nm*r:nm*r+mp*r].reshape(mp, r)
        W_flat = x[nm*r+mp*r:].reshape(pn, r)
        
        err = 0.0
        for (a, b, c, val) in tensor_entries:
            s = np.dot(U_flat[a, :], V_flat[b, :] * W_flat[c, :])
            err += (s - val)**2
        return err

    def gradient(x):
        U_flat = x[:nm*r].reshape(nm, r)
        V_flat = x[nm*r:nm*r+mp*r].reshape(mp, r)
        W_flat = x[nm*r+mp*r:].reshape(pn, r)
        
        gU = np.zeros_like(U_flat)
        gV = np.zeros_like(V_flat)
        gW = np.zeros_like(W_flat)
        
        for (a, b, c, val) in tensor_entries:
            s = np.dot(U_flat[a, :], V_flat[b, :] * W_flat[c, :])
            err = s - val
            # d/dU[a,k] = V[b,k] * W[c,k]
            gU[a, :] += err * V_flat[b, :] * W_flat[c, :]
            # d/dV[b,k] = U[a,k] * W[c,k]
            gV[b, :] += err * U_flat[a, :] * W_flat[c, :]
            # d/dW[c,k] = U[a,k] * V[b,k]
            gW[c, :] += err * U_flat[a, :] * V_flat[b, :]
            
        return np.concatenate([gU.flatten(), gV.flatten(), gW.flatten()])

    x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    # Use L-BFGS-B
    res = so.minimize(objective, x0, jac=gradient, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12})
    
    x_opt = res.x
    U_opt = x_opt[:nm*r].reshape(nm, r)
    V_opt = x_opt[nm*r:nm*r+mp*r].reshape(mp, r)
    W_opt = x_opt[nm*r+mp*r:].reshape(pn, r)
    
    final_err = objective(x_opt)
    return U_opt, V_opt, W_opt, final_err
