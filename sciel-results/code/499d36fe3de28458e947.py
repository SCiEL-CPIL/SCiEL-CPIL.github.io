import numpy as np
import time
from itertools import combinations, permutations

DIMENSIONS = [2, 3, 4]
RANK_LIMIT = 23

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[a,b,c] with the official trace convention."""
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i   # W[k*n + i, r]
                T[a, b, c] = 1.0
    return T

def _reconstruction_error(params, n, m, p, rank, T_flat):
    """Squared Frobenius norm of the reconstruction error for a given parameter vector."""
    nm, mp, pn = n * m, m * p, p * n
    U = params[:nm * rank].reshape(nm, rank)
    V = params[nm * rank:nm * rank + mp * rank].reshape(mp, rank)
    W = params[nm * rank + mp * rank:].reshape(pn, rank)
    recon = np.einsum('ac,bc,dc->abd', U, V, W)
    err = recon - T_flat.reshape(nm, mp, pn)
    return float(np.sum(err * err))

def _grad(params, n, m, p, rank, T_flat):
    """Gradient of the squared Frobenius reconstruction error (analytic)."""
    nm, mp, pn = n * m, m * p, p * n
    U = params[:nm * rank].reshape(nm, rank)
    V = params[nm * rank:nm * rank + mp * rank].reshape(mp, rank)
    W = params[nm * rank + mp * rank:].reshape(pn, rank)
    T = T_flat.reshape(nm, mp, pn)
    recon = np.einsum('ac,bc,dc->abd', U, V, W)
    E = recon - T
    # dL/dU[a,c] = 2 * sum_{b,d} E[a,b,d] V[b,c] W[d,c]
    gU = 2.0 * np.einsum('abd,bc,dc->ac', E, V, W)
    gV = 2.0 * np.einsum('abd,ac,dc->bc', E, U, W)
    gW = 2.0 * np.einsum('abd,ac,bc->dc', E, U, V)
    g = np.concatenate([gU.ravel(), gV.ravel(), gW.ravel()])
    return g

def _snap_to_half_integers(U, V, W, T_flat, n, m, p):
    """Snap each matrix to the nearest half-integer and verify exactness. Return (ok, Us, Vs, Ws)."""
    T = T_flat.reshape(n * m, m * p, p * n)
    nm, mp, pn = n * m, m * p, p * n
    
    # First try direct rounding
    Us = np.round(U * 2.0) / 2.0
    Vs = np.round(V * 2.0) / 2.0
    Ws = np.round(W * 2.0) / 2.0
    recon = np.einsum('ac,bc,dc->abd', Us, Vs, Ws)
    if np.max(np.abs(recon - T)) < 1e-8:
        return True, Us, Vs, Ws
    
    # Try local optimization to find nearest half-integer solution
    # Use a penalty-based approach
    best_err = np.inf
    best_solution = None
    
    # Try different rounding strategies
    for strategy in range(10):
        rng = np.random.default_rng(strategy)
        
        # Randomly perturb before rounding
        noise_scale = 0.1 + 0.05 * strategy
        U_pert = U + rng.standard_normal(U.shape) * noise_scale
        V_pert = V + rng.standard_normal(V.shape) * noise_scale
        W_pert = W + rng.standard_normal(W.shape) * noise_scale
        
        Us2 = np.round(U_pert * 2.0) / 2.0
        Vs2 = np.round(V_pert * 2.0) / 2.0
        Ws2 = np.round(W_pert * 2.0) / 2.0
        recon2 = np.einsum('ac,bc,dc->abd', Us2, Vs2, Ws2)
        err2 = np.max(np.abs(recon2 - T))
        
        if err2 < best_err:
            best_err = err2
            best_solution = (Us2, Vs2, Ws2)
        
        if err2 < 1e-8:
            return True, Us2, Vs2, Ws2
    
    # If we found a good candidate but not exact, try to polish it
    if best_solution is not None and best_err < 0.1:
        Us2, Vs2, Ws2 = best_solution
        # Try small integer adjustments
        for du in [-1, 0, 1]:
            for dv in [-1, 0, 1]:
                for dw in [-1, 0, 1]:
                    Us3 = Us2 + du * 0.5
                    Vs3 = Vs2 + dv * 0.5
                    Ws3 = Ws2 + dw * 0.5
                    # Snap back to half-integers
                    Us3 = np.round(Us3 * 2.0) / 2.0
                    Vs3 = np.round(Vs3 * 2.0) / 2.0
                    Ws3 = np.round(Ws3 * 2.0) / 2.0
                    recon3 = np.einsum('ac,bc,dc->abd', Us3, Vs3, Ws3)
                    err3 = np.max(np.abs(recon3 - T))
                    if err3 < 1e-8:
                        return True, Us3, Vs3, Ws3
    
    return False, None, None, None

def _try_rank(n, m, p, rank, T_flat, time_budget):
    """Attempt to find an exact rank-rank decomposition within time_budget seconds."""
    start = time.time()
    nm, mp, pn = n * m, m * p, p * n
    n_params = (nm + mp + pn) * rank

    # Strategy 1: Try to use known structure - start from schoolbook and perturb
    # Strategy 2: Multiple random restarts with better initialization
    seeds = [1, 7, 42, 123, 999, 2024, 555, 888, 111, 333]
    attempt = 0
    
    # First, try a structured initialization based on the tensor structure
    # For matrix multiplication, we can use a basis that respects the block structure
    if time.time() - start < time_budget * 0.3:
        # Create a structured initialization
        U_struct = np.zeros((nm, rank))
        V_struct = np.zeros((mp, rank))
        W_struct = np.zeros((pn, rank))
        
        # Use a Kronecker-like structure
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = (i * m + j) * p + (j * p + k) % p + (k * n + i) % n
                    if idx < rank:
                        U_struct[i * m + j, idx] = 1.0
                        V_struct[j * p + k, idx] = 1.0
                        W_struct[k * n + i, idx] = 1.0
        
        params_struct = np.concatenate([U_struct.ravel(), V_struct.ravel(), W_struct.ravel()])
        bounds = [(-100.0, 100.0)] * n_params
        try:
            from scipy.optimize import minimize
            res = minimize(
                _reconstruction_error,
                params_struct,
                jac=_grad,
                args=(n, m, p, rank, T_flat),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-10, 'maxls': 50},
            )
        except Exception:
            res = None
        if res is not None:
            popt = res.x
            err = _reconstruction_error(popt, n, m, p, rank, T_flat)
            if err < 1e-6:
                U = popt[:nm * rank].reshape(nm, rank)
                V = popt[nm * rank:nm * rank + mp * rank].reshape(mp, rank)
                W = popt[nm * rank + mp * rank:].reshape(pn, rank)
                ok, Us, Vs, Ws = _snap_to_half_integers(U, V, W, T_flat, n, m, p)
                if ok:
                    return Us, Vs, Ws

    # Random restarts with varied scales
    while time.time() - start < time_budget * 0.85:
        seed = seeds[attempt % len(seeds)] + attempt
        rng = np.random.default_rng(seed)
        
        # Vary the initialization scale
        scale = 0.1 + 0.2 * (attempt % 5)
        params = rng.standard_normal(n_params) * scale
        
        # L-BFGS-B with analytic gradient
        bounds = [(-100.0, 100.0)] * n_params
        try:
            from scipy.optimize import minimize
            res = minimize(
                _reconstruction_error,
                params,
                jac=_grad,
                args=(n, m, p, rank, T_flat),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 400, 'ftol': 1e-15, 'gtol': 1e-10, 'maxls': 50},
            )
        except Exception:
            res = None
        if res is not None:
            popt = res.x
            err = _reconstruction_error(popt, n, m, p, rank, T_flat)
            if err < 1e-6:
                U = popt[:nm * rank].reshape(nm, rank)
                V = popt[nm * rank:nm * rank + mp * rank].reshape(mp, rank)
                W = popt[nm * rank + mp * rank:].reshape(pn, rank)
                ok, Us, Vs, Ws = _snap_to_half_integers(U, V, W, T_flat, n, m, p)
                if ok:
                    return Us, Vs, Ws
        attempt += 1
    return None

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _to_lists(U, V, W):
    return {
        "U": [[float(U[i][j]) for j in range(U.shape[1])] for i in range(U.shape[0])],
        "V": [[float(V[i][j]) for j in range(V.shape[1])] for i in range(V.shape[0])],
        "W": [[float(W[i][j]) for j in range(W.shape[1])] for i in range(W.shape[0])],
    }

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 135  # safety margin before 150s kill
    T_flat = _build_tensor(n, m, p).ravel()

    # Try ranks in order, allocating more time to lower ranks
    rank_budgets = {19: 40.0, 20: 35.0, 21: 25.0, 22: 15.0, 23: 10.0}
    
    for rank in (19, 20, 21, 22, 23):
        elapsed = time.time() - start
        if elapsed > deadline - 5.0:
            break
        budget = min(rank_budgets.get(rank, 10.0), deadline - elapsed - 5.0)
        if budget < 3.0:
            continue
        res = _try_rank(n, m, p, rank, T_flat, budget)
        if res is not None:
            Us, Vs, Ws = res
            return _to_lists(Us, Vs, Ws)

    # Fallback: guaranteed valid schoolbook decomposition (rank 24)
    return _schoolbook(n, m, p)
