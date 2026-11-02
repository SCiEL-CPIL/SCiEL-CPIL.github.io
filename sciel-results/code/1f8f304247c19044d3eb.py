import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 2, 9]
RNG_SEED = 42

def _build_true_tensor():
    n, m, p = DIMENSIONS
    T = np.zeros((n * m, m * p, p * n))
    for a in range(n):
        for b in range(m):
            for c in range(p):
                i = a * m + b
                j = b * p + c
                k = c * n + a
                T[i, j, k] = 1.0
    return T

def _build_rank32():
    """Strassen-based rank-32 construction."""
    n, m, p = DIMENSIONS
    rank = 32
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    def add_strassen(k0, k1):
        nonlocal col
        U[0, col] = 1; U[3, col] = 1
        V[0*p + k0, col] = 1; V[1*p + k1, col] = 1
        W[k0*n + 0, col] = 1; W[k1*n + 1, col] = 1
        col += 1
        U[2, col] = 1; U[3, col] = 1
        V[0*p + k0, col] = 1
        W[k0*n + 1, col] = 1; W[k1*n + 1, col] = -1
        col += 1
        U[0, col] = 1
        V[0*p + k1, col] = 1; V[1*p + k1, col] = -1
        W[k1*n + 0, col] = 1; W[k1*n + 1, col] = 1
        col += 1
        U[3, col] = 1
        V[1*p + k0, col] = 1; V[0*p + k0, col] = -1
        W[k0*n + 0, col] = 1; W[k0*n + 1, col] = 1
        col += 1
        U[0, col] = 1; U[1, col] = 1
        V[1*p + k1, col] = 1
        W[k0*n + 0, col] = -1; W[k1*n + 0, col] = 1
        col += 1
        U[2, col] = 1; U[0, col] = -1
        V[0*p + k0, col] = 1; V[0*p + k1, col] = 1
        W[k1*n + 1, col] = 1
        col += 1
        U[1, col] = 1; U[3, col] = -1
        V[1*p + k0, col] = 1; V[1*p + k1, col] = 1
        W[k0*n + 0, col] = 1
        col += 1

    def add_trivial(k):
        nonlocal col
        U[0, col] = 1; V[0*p + k, col] = 1; W[k*n + 0, col] = 1
        col += 1
        U[1, col] = 1; V[1*p + k, col] = 1; W[k*n + 0, col] = 1
        col += 1
        U[2, col] = 1; V[0*p + k, col] = 1; W[k*n + 1, col] = 1
        col += 1
        U[3, col] = 1; V[1*p + k, col] = 1; W[k*n + 1, col] = 1
        col += 1

    add_strassen(0, 1)
    add_strassen(2, 3)
    add_strassen(4, 5)
    add_strassen(6, 7)
    add_trivial(8)
    
    return U, V, W

def _als_polish(U, V, W, T_true, nm, mp, pn, rank, max_iters=2000, tol=1e-15):
    """ALS polishing with tighter convergence."""
    U = U.copy()
    V = V.copy()
    W = W.copy()
    for it in range(max_iters):
        # Update U
        for i in range(nm):
            B = np.zeros((mp * pn, rank))
            for r in range(rank):
                B[:, r] = (V[:, r] * W[:, r]).flatten()
            t = T_true[i, :, :].flatten()
            u, _, _, _ = np.linalg.lstsq(B, t, rcond=None)
            U[i, :] = u
        
        # Update V
        for j in range(mp):
            B = np.zeros((nm * pn, rank))
            for r in range(rank):
                B[:, r] = (U[:, r] * W[:, r]).flatten()
            t = T_true[:, j, :].flatten()
            v, _, _, _ = np.linalg.lstsq(B, t, rcond=None)
            V[j, :] = v
        
        # Update W
        for k in range(pn):
            B = np.zeros((nm * mp, rank))
            for r in range(rank):
                B[:, r] = (U[:, r] * V[:, r]).flatten()
            t = T_true[:, :, k].flatten()
            w, _, _, _ = np.linalg.lstsq(B, t, rcond=None)
            W[k, :] = w
        
        M = np.einsum('ir,jr,kr->ijk', U, V, W)
        loss = np.sum((M - T_true)**2)
        if loss < tol:
            break
    
    M = np.einsum('ir,jr,kr->ijk', U, V, W)
    max_res = np.max(np.abs(M - T_true))
    return U, V, W, max_res

def _gradient_descent_step(U, V, W, T_true, nm, mp, pn, rank, lr=0.01, steps=500):
    """Simple gradient descent with momentum for polishing."""
    U = U.copy().astype(float)
    V = V.copy().astype(float)
    W = W.copy().astype(float)
    vU = np.zeros_like(U)
    vV = np.zeros_like(V)
    vW = np.zeros_like(W)
    beta = 0.9
    
    for step in range(steps):
        M = np.einsum('ir,jr,kr->ijk', U, V, W)
        R = M - T_true
        
        gU = np.einsum('ijk,jr,kr->ir', R, V, W)
        gV = np.einsum('ijk,ir,kr->jr', R, U, W)
        gW = np.einsum('ijk,ir,jr->kr', R, U, V)
        
        vU = beta * vU + (1 - beta) * gU
        vV = beta * vV + (1 - beta) * gV
        vW = beta * vW + (1 - beta) * gW
        
        U -= lr * vU
        V -= lr * vV
        W -= lr * vW
        
        if step % 100 == 0:
            loss = np.sum(R**2)
            if loss < 1e-14:
                break
    
    M = np.einsum('ir,jr,kr->ijk', U, V, W)
    max_res = np.max(np.abs(M - T_true))
    return U, V, W, max_res

def _try_rank31_aggressive(U32, V32, W32, T_true, nm, mp, pn, time_budget):
    """Try multiple strategies to find rank-31 decomposition."""
    start = time.time()
    rank = 31
    rng = np.random.default_rng(RNG_SEED)
    
    best_result = None
    best_loss = float('inf')
    
    def check_and_update(U_sol, V_sol, W_sol, label=""):
        nonlocal best_result, best_loss
        M = np.einsum('ir,jr,kr->ijk', U_sol, V_sol, W_sol)
        residual = np.max(np.abs(M - T_true))
        if residual < best_loss:
            best_loss = residual
            best_result = (U_sol.copy(), V_sol.copy(), W_sol.copy())
            if residual < 1e-10:
                # Try to polish further
                U_p, V_p, W_p, mr = _als_polish(U_sol, V_sol, W_sol, T_true, nm, mp, pn, rank, max_iters=500, tol=1e-16)
                if mr < 1e-10:
                    best_result = (U_p, V_p, W_p)
                    best_loss = mr
                return True
        return False
    
    # Strategy 1: Drop each column from rank-32, optimize with L-BFGS-B
    for drop in range(32):
        if time.time() - start > time_budget * 0.4:
            break
        mask = [i for i in range(32) if i != drop]
        U0 = U32[:, mask].copy()
        V0 = V32[:, mask].copy()
        W0 = W32[:, mask].copy()
        
        for attempt in range(5):
            if time.time() - start > time_budget * 0.4:
                break
            noise_scale = 0.005 * (attempt + 1)
            U_init = U0 + rng.normal(0, noise_scale, size=U0.shape)
            V_init = V0 + rng.normal(0, noise_scale, size=V0.shape)
            W_init = W0 + rng.normal(0, noise_scale, size=W0.shape)
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            def objective(params):
                U_f = params[:nm * rank]
                V_f = params[nm * rank : nm * rank + mp * rank]
                W_f = params[nm * rank + mp * rank : nm * rank + mp * rank + pn * rank]
                U_m = U_f.reshape(nm, rank)
                V_m = V_f.reshape(mp, rank)
                W_m = W_f.reshape(pn, rank)
                M = np.einsum('ir,jr,kr->ijk', U_m, V_m, W_m)
                res = M - T_true
                return np.sum(res * res)
            
            def gradient(params):
                U_f = params[:nm * rank]
                V_f = params[nm * rank : nm * rank + mp * rank]
                W_f = params[nm * rank + mp * rank : nm * rank + mp * rank + pn * rank]
                U_m = U_f.reshape(nm, rank)
                V_m = V_f.reshape(mp, rank)
                W_m = W_f.reshape(pn, rank)
                M = np.einsum('ir,jr,kr->ijk', U_m, V_m, W_m)
                R = M - T_true
                gU = np.einsum('ijk,jr,kr->ir', R, V_m, W_m)
                gV = np.einsum('ijk,ir,kr->jr', R, U_m, W_m)
                gW = np.einsum('ijk,ir,jr->kr', R, U_m, V_m)
                return np.concatenate([gU.flatten(), gV.flatten(), gW.flatten()])
            
            try:
                result = minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                                  options={'maxiter': 200, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 30})
                if result.fun < 1e-8:
                    U_f = result.x[:nm * rank]
                    V_f = result.x[nm * rank : nm * rank + mp * rank]
                    W_f = result.x[nm * rank + mp * rank : nm * rank + mp * rank + pn * rank]
                    U_sol = U_f.reshape(nm, rank)
                    V_sol = V_f.reshape(mp, rank)
                    W_sol = W_f.reshape(pn, rank)
                    if check_and_update(U_sol, V_sol, W_sol, f"drop{drop}_att{attempt}"):
                        return best_result
            except Exception:
                pass
    
    # Strategy 2: Random initialization with multiple restarts
    for attempt in range(40):
        if time.time() - start > time_budget * 0.8:
            break
        
        scale = 0.3 + 0.1 * (attempt % 5)
        U_init = rng.normal(0, scale, size=(nm, rank))
        V_init = rng.normal(0, scale, size=(mp, rank))
        W_init = rng.normal(0, scale, size=(pn, rank))
        
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        def objective(params):
            U_f = params[:nm * rank]
            V_f = params[nm * rank : nm * rank + mp * rank]
            W_f = params[nm * rank + mp * rank : nm * rank + mp * rank + pn * rank]
            U_m = U_f.reshape(nm, rank)
            V_m = V_f.reshape(mp, rank)
            W_m = W_f.reshape(pn, rank)
            M = np.einsum('ir,jr,kr->ijk', U_m, V_m, W_m)
            res = M - T_true
            return np.sum(res * res)
        
        def gradient(params):
            U_f = params[:nm * rank]
            V_f = params[nm * rank : nm * rank + mp * rank]
            W_f = params[nm * rank + mp * rank : nm * rank + mp * rank + pn * rank]
            U_m = U_f.reshape(nm, rank)
            V_m = V_f.reshape(mp, rank)
            W_m = W_f.reshape(pn, rank)
            M = np.einsum('ir,jr,kr->ijk', U_m, V_m, W_m)
            R = M - T_true
            gU = np.einsum('ijk,jr,kr->ir', R, V_m, W_m)
            gV = np.einsum('ijk,ir,kr->jr', R, U_m, W_m)
            gW = np.einsum('ijk,ir,jr->kr', R, U_m, V_m)
            return np.concatenate([gU.flatten(), gV.flatten(), gW.flatten()])
        
        try:
            result = minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                              options={'maxiter': 150, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 30})
            if result.fun < 1e-8:
                U_f = result.x[:nm * rank]
                V_f = result.x[nm * rank : nm * rank + mp * rank]
                W_f = result.x[nm * rank + mp * rank : nm * rank + mp * rank + pn * rank]
                U_sol = U_f.reshape(nm, rank)
                V_sol = V_f.reshape(mp, rank)
                W_sol = W_f.reshape(pn, rank)
                if check_and_update(U_sol, V_sol, W_sol, f"rand{attempt}"):
                    return best_result
        except Exception:
            pass
    
    # Strategy 3: If we have a near-solution, polish it heavily
    if best_result is not None and best_loss < 1e-4:
        U_p, V_p, W_p, mr = _als_polish(best_result[0], best_result[1], best_result[2], 
                                         T_true, nm, mp, pn, rank, max_iters=3000, tol=1e-16)
        if mr < 1e-10:
            return (U_p, V_p, W_p)
        
        # Try gradient descent polish
        U_g, V_g, W_g, mr2 = _gradient_descent_step(best_result[0], best_result[1], best_result[2],
                                                     T_true, nm, mp, pn, rank, lr=0.005, steps=1000)
        if mr2 < 1e-10:
            return (U_g, V_g, W_g)
    
    return best_result

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    U32, V32, W32 = _build_rank32()
    T_true = _build_true_tensor()
    
    def to_list(arr):
        return arr.tolist()
    
    # Try rank 31 with aggressive optimization
    time_remaining = deadline - time.time() - 5.0
    if time_remaining > 15.0:
        result = _try_rank31_aggressive(U32, V32, W32, T_true, nm, mp, pn, time_remaining)
        if result is not None:
            U_sol, V_sol, W_sol = result
            M = np.einsum('ir,jr,kr->ijk', U_sol, V_sol, W_sol)
            residual = np.max(np.abs(M - T_true))
            if residual < 1e-6:
                return {"U": to_list(U_sol), "V": to_list(V_sol), "W": to_list(W_sol)}
    
    # Fall back to rank 32
    return {"U": to_list(U32), "V": to_list(V32), "W": to_list(W32)}
