import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    # Build index mappings
    # U indexed by (i, j) -> row i*m + j
    # V indexed by (j, k) -> row j*p + k
    # W indexed by (k, i) -> row k*n + i
    
    # Target: for all (i,j,k), sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    
    # Schoolbook fallback
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1
                V_sb[j*p + k, col] = 1
                W_sb[k*n + i, col] = 1
                col += 1
    
    best_result = None
    best_rank = rank_sb
    
    # Precompute index triples
    triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    
    def compute_residuals(U, V, W, rank):
        """Compute residual vector efficiently."""
        res = np.empty(n*m*p)
        idx = 0
        for (i, j, k) in triples:
            res[idx] = 1.0 - np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
            idx += 1
        return res
    
    def check_half_integer(U, V, W, tol=1e-9):
        """Check if snapping to half-integers still satisfies equations."""
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        max_err = 0.0
        for (i, j, k) in triples:
            val = np.dot(U_snap[i*m+j, :], V_snap[j*p+k, :] * W_snap[k*n+i, :])
            err = abs(1.0 - val)
            if err > max_err:
                max_err = err
        if max_err < tol:
            return U_snap, V_snap, W_snap
        return None
    
    def try_rank(rank, max_attempts=20):
        """Try to find a rank-r decomposition."""
        total_vars = (n*m + m*p + p*n) * rank
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            
            np.random.seed(attempt * 7919 + rank * 104729)
            
            # Structured initialization: 
            # Strategy: initialize W with small random values, U and V with moderate values
            U0 = np.random.randn(n*m, rank) * 0.5
            V0 = np.random.randn(m*p, rank) * 0.5
            W0 = np.random.randn(p*n, rank) * 0.3
            
            # For lower ranks, try a more structured init
            if rank <= 14:
                # Try to initialize so that the "average" contribution is ~1
                # Scale: each term U*V*W should contribute ~1/rank on average
                scale = (1.0 / rank) ** (1.0/3.0)
                U0 = np.random.randn(n*m, rank) * scale * 2
                V0 = np.random.randn(m*p, rank) * scale * 2
                W0 = np.random.randn(p*n, rank) * scale * 2
            
            params0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            def residuals(params):
                U = params[:n*m*rank].reshape(n*m, rank)
                V = params[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
                W = params[n*m*rank + m*p*rank:].reshape(p*n, rank)
                return compute_residuals(U, V, W, rank)
            
            try:
                result = least_squares(
                    residuals, params0,
                    method='lm', max_nfev=3000,
                    xtol=1e-12, ftol=1e-12, gtol=1e-12
                )
                
                if result.cost < 1e-10:
                    U_sol = result.x[:n*m*rank].reshape(n*m, rank)
                    V_sol = result.x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
                    W_sol = result.x[n*m*rank + m*p*rank:].reshape(p*n, rank)
                    
                    # Polish pass: refine with tighter tolerance
                    params_polish = result.x.copy()
                    try:
                        result2 = least_squares(
                            residuals, params_polish,
                            method='lm', max_nfev=5000,
                            xtol=1e-14, ftol=1e-14, gtol=1e-14
                        )
                        if result2.cost < result.cost:
                            U_sol = result2.x[:n*m*rank].reshape(n*m, rank)
                            V_sol = result2.x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
                            W_sol = result2.x[n*m*rank + m*p*rank:].reshape(p*n, rank)
                    except Exception:
                        pass
                    
                    # Try to snap to half-integers
                    snapped = check_half_integer(U_sol, V_sol, W_sol)
                    if snapped is not None:
                        return snapped
                    
                    # If snapping fails, try integer snapping as fallback
                    U_int = np.round(U_sol)
                    V_int = np.round(V_sol)
                    W_int = np.round(W_sol)
                    max_err = 0.0
                    for (i, j, k) in triples:
                        val = np.dot(U_int[i*m+j, :], V_int[j*p+k, :] * W_int[k*n+i, :])
                        err = abs(1.0 - val)
                        if err > max_err:
                            max_err = err
                    if max_err < 1e-9:
                        return (U_int.astype(float), V_int.astype(float), W_int.astype(float))
                
            except Exception:
                pass
        
        return None
    
    # Try ranks from lower bound upward
    # Lower bound: max(n*m, m*p, p*n) = 12
    lower_bound = max(n*m, m*p, p*n)
    
    for rank in range(lower_bound, rank_sb):
        if time.time() > deadline:
            break
        
        result = try_rank(rank)
        if result is not None:
            U, V, W = result
            best_rank = rank
            best_result = (U, V, W)
            break  # Found minimal rank, stop
    
    if best_result is not None:
        U, V, W = best_result
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fallback to schoolbook
        U_list = [[float(x) for x in row] for row in U_sb]
        V_list = [[float(x) for x in row] for row in V_sb]
        W_list = [[float(x) for x in row] for row in W_sb]
        return {"U": U_list, "V": V_list, "W": W_list}
