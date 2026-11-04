import time
import numpy as np
import scipy.optimize as opt

DIMENSIONS = [3, 4, 6]

def _trivial_decomposition(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-6):
    Ut = U.reshape(n, m, -1)
    Vt = V.reshape(m, p, -1)
    Wt = W.reshape(p, n, -1)
    T = np.einsum('ijc,jkc,kic->ijk', Ut, Vt, Wt)
    return np.max(np.abs(T - 1.0)) < tol

def _loss_and_grad(x, n, m, p, r):
    U = x[:n*m*r].reshape(n, m, r)
    V = x[n*m*r : n*m*r + m*p*r].reshape(m, p, r)
    W = x[n*m*r + m*p*r :].reshape(p, n, r)

    T = np.einsum('ijc,jkc,kic->ijk', U, V, W)
    err = T - 1.0
    loss = np.sum(err ** 2)

    dU = np.einsum('ijk,jkc,kic->ijc', err, V, W) * 2
    dV = np.einsum('ijk,ijc,kic->jkc', err, U, W) * 2
    dW = np.einsum('ijk,ijc,jkc->kic', err, U, V) * 2

    grad = np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
    return loss, grad

def _newton_polish(U, V, W, n, m, p, r, max_iter=50, tol=1e-12):
    """Apply Newton's method to polish a near-solution."""
    U = U.copy().astype(float)
    V = V.copy().astype(float)
    W = W.copy().astype(float)
    
    for it in range(max_iter):
        Ut = U.reshape(n, m, r)
        Vt = V.reshape(m, p, r)
        Wt = W.reshape(p, n, r)
        
        T = np.einsum('ijc,jkc,kic->ijk', Ut, Vt, Wt)
        err = T - 1.0
        loss = np.sum(err ** 2)
        
        if loss < tol:
            break
        
        # Compute Hessian approximation via Gauss-Newton
        # dU = 2 * err contracted with V, W
        dU = np.einsum('ijk,jkc,kic->ijc', err, Vt, Wt) * 2
        dV = np.einsum('ijk,ijc,kic->jkc', err, Ut, Wt) * 2
        dW = np.einsum('ijk,ijc,jkc->kic', err, Ut, Vt) * 2
        
        # Simple damped Newton step
        step_size = 0.1
        U -= step_size * dU.reshape(n*m, r)
        V -= step_size * dV.reshape(m*p, r)
        W -= step_size * dW.reshape(p*n, r)
        
        # Check if loss increased, if so reduce step
        T_new = np.einsum('ijc,jkc,kic->ijk', U.reshape(n,m,r), V.reshape(m,p,r), W.reshape(p,n,r))
        loss_new = np.sum((T_new - 1.0)**2)
        if loss_new > loss:
            U += step_size * dU.reshape(n*m, r)
            V += step_size * dV.reshape(m*p, r)
            W += step_size * dW.reshape(p*n, r)
            step_size *= 0.5
            if step_size < 1e-6:
                break
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s
    n, m, p = DIMENSIONS
    trivial_rank = n * m * p  # 72
    
    U_trivial, V_trivial, W_trivial = _trivial_decomposition(n, m, p)
    best = {'U': U_trivial, 'V': V_trivial, 'W': W_trivial}
    best_rank = trivial_rank
    
    def try_rank(r, max_restarts=3, max_time_per_restart=15.0):
        """Try to find a rank-r decomposition using multiple restarts."""
        if time.time() > deadline - 20:
            return None
        
        best_result = None
        best_loss = np.inf
        
        for restart in range(max_restarts):
            if time.time() > deadline - 15:
                break
            
            # Different initialization strategies
            if restart == 0:
                # Random initialization
                rng = np.random.RandomState(42 + restart)
                scale = 1.0 / np.sqrt(max(1, r / (n * m * p)))
                U0 = rng.randn(n * m, r) * scale
                V0 = rng.randn(m * p, r) * scale
                W0 = rng.randn(p * n, r) * scale
            elif restart == 1:
                # Structured initialization based on trivial decomposition
                # Use first r columns of trivial if r < trivial_rank
                if r <= trivial_rank:
                    U0 = U_trivial[:, :r].copy()
                    V0 = V_trivial[:, :r].copy()
                    W0 = W_trivial[:, :r].copy()
                    # Add small random perturbation
                    rng = np.random.RandomState(100 + restart)
                    U0 += rng.randn(n*m, r) * 0.01
                    V0 += rng.randn(m*p, r) * 0.01
                    W0 += rng.randn(p*n, r) * 0.01
                else:
                    rng = np.random.RandomState(42 + restart)
                    scale = 1.0 / np.sqrt(max(1, r / (n * m * p)))
                    U0 = rng.randn(n * m, r) * scale
                    V0 = rng.randn(m * p, r) * scale
                    W0 = rng.randn(p * n, r) * scale
            else:
                # Another random seed
                rng = np.random.RandomState(200 + restart)
                scale = 1.0 / np.sqrt(max(1, r / (n * m * p)))
                U0 = rng.randn(n * m, r) * scale
                V0 = rng.randn(m * p, r) * scale
                W0 = rng.randn(p * n, r) * scale
            
            x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            try:
                result = opt.minimize(
                    lambda x: _loss_and_grad(x, n, m, p, r),
                    x0,
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 5000}
                )
                
                if time.time() > deadline - 10:
                    continue
                    
                U_opt = result.x[:n*m*r].reshape(n*m, r)
                V_opt = result.x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
                W_opt = result.x[n*m*r + m*p*r :].reshape(p*n, r)
                
                # Apply Newton polish
                U_pol, V_pol, W_pol = _newton_polish(U_opt, V_opt, W_opt, n, m, p, r, max_iter=100, tol=1e-14)
                
                # Verify
                if _verify(U_pol, V_pol, W_pol, n, m, p, tol=1e-8):
                    return U_pol, V_pol, W_pol
                
                # Track best loss even if not converged
                Ut = U_pol.reshape(n, m, r)
                Vt = V_pol.reshape(m, p, r)
                Wt = W_pol.reshape(p, n, r)
                T = np.einsum('ijc,jkc,kic->ijk', Ut, Vt, Wt)
                loss = np.sum((T - 1.0)**2)
                
                if loss < best_loss:
                    best_loss = loss
                    best_result = (U_pol, V_pol, W_pol)
                    
            except Exception:
                pass
        
        return best_result
    
    # Try ranks from trivial_rank-1 down to a lower bound
    # Theoretical lower bound for matrix multiplication is related to Strassen
    # For 3x4x6, try ranks 65 down to 71 first, then lower if time permits
    
    ranks_to_try = list(range(trivial_rank - 1, 64, -1))
    
    for r in ranks_to_try:
        if time.time() > deadline - 25:
            break
            
        res = try_rank(r, max_restarts=2, max_time_per_restart=20.0)
        if res is not None:
            U, V, W = res
            best = {'U': U, 'V': V, 'W': W}
            best_rank = r
            break
    
    if best_rank < trivial_rank:
        return {'U': best['U'].tolist(), 'V': best['V'].tolist(), 'W': best['W'].tolist()}
    else:
        return {'U': U_trivial.tolist(), 'V': V_trivial.tolist(), 'W': W_trivial.tolist()}
