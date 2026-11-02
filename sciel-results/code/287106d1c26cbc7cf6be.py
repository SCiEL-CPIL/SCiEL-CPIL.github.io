import numpy as np
import time
import sys
from scipy.optimize import least_squares

DIMENSIONS = [3, 3, 4]

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _check_and_round(U, V, W, tol=1e-4):
    """Check if U, V, W are close to half-integers and return rounded versions if so."""
    Uf = U.flatten()
    Vf = V.flatten()
    Wf = W.flatten()
    
    dist_U = np.abs(Uf - np.round(2 * Uf) / 2.0)
    dist_V = np.abs(Vf - np.round(2 * Vf) / 2.0)
    dist_W = np.abs(Wf - np.round(2 * Wf) / 2.0)
    
    if np.max(dist_U) < tol and np.max(dist_V) < tol and np.max(dist_W) < tol:
        U_r = np.round(2 * Uf) / 2.0
        V_r = np.round(2 * Vf) / 2.0
        W_r = np.round(2 * Wf) / 2.0
        
        n, m, p = DIMENSIONS
        U_3d = U_r.reshape(n*m, -1)
        V_3d = V_r.reshape(m*p, -1)
        W_3d = W_r.reshape(p*n, -1)
        
        if _verify_decomposition(U_3d, V_3d, W_3d, n, m, p, tol=1e-8):
            return True, U_r.reshape(U.shape), V_r.reshape(V.shape), W_r.reshape(W.shape)
        
    return False, None, None, None

def _structured_init(n, m, p, rank, seed=0):
    """Create a structured initial guess by grouping schoolbook terms."""
    rng = np.random.RandomState(seed)
    
    size_U = n * m * rank
    size_V = m * p * rank
    size_W = p * n * rank
    total_size = size_U + size_V + size_W
    
    x0 = np.zeros(total_size)
    
    # For each rank term, assign it to cover a subset of (i,j,k) triples
    # Strategy: distribute the n*m*p terms across rank groups
    all_triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    
    # Shuffle and distribute
    rng.shuffle(all_triples)
    for r in range(rank):
        # Each rank term gets a subset of triples
        start = r * (len(all_triples) // rank)
        end = (r + 1) * (len(all_triples) // rank) if r < rank - 1 else len(all_triples)
        
        for (i, j, k) in all_triples[start:end]:
            # Set the corresponding entries to 1/sqrt(count) to start balanced
            x0[i*m + j + r * n * m] = 1.0
            x0[(n*m) + (j*p + k) + r * m * p] = 1.0
            x0[(n*m + m*p) + (k*n + i) + r * p * n] = 1.0
    
    # Add small noise to break symmetry
    x0 += rng.normal(0, 0.05, total_size)
    
    return x0

def _try_find_decomposition(n, m, p, rank, max_restarts=5, deadline=None):
    """Try to find a rank-rank decomposition using multiple restarts."""
    size_U = n * m * rank
    size_V = m * p * rank
    size_W = p * n * rank
    total_size = size_U + size_V + size_W
    
    def residual(x):
        Uf = x[:size_U].reshape(n*m, rank)
        Vf = x[size_U:size_U+size_V].reshape(m*p, rank)
        Wf = x[size_U+size_V:].reshape(p*n, rank)
        
        res = np.zeros(n*m*p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_row = Uf[i*m+j, :]
                    v_row = Vf[j*p+k, :]
                    w_row = Wf[k*n+i, :]
                    val = np.dot(u_row, v_row * w_row)
                    res[idx] = val - 1.0
                    idx += 1
        return res
    
    best_result = None
    
    for restart in range(max_restarts):
        if deadline is not None and time.time() > deadline:
            break
        
        # Use structured initialization with different seeds
        x0 = _structured_init(n, m, p, rank, seed=restart * 1000 + 42)
        
        try:
            res = least_squares(residual, x0, method='lm', xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=2000)
            
            if res.cost < 1e-10:
                x_sol = res.x
                U_sol = x_sol[:size_U].reshape(n*m, rank)
                V_sol = x_sol[size_U:size_U+size_V].reshape(m*p, rank)
                W_sol = x_sol[size_U+size_V:].reshape(p*n, rank)
                
                # First check if it's already close to half-integers
                valid, U_r, V_r, W_r = _check_and_round(U_sol, V_sol, W_sol)
                if valid:
                    return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                
                # Try a polish step with tighter tolerances
                try:
                    res2 = least_squares(residual, x_sol, method='lm', xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=500)
                    if res2.cost < 1e-12:
                        U_pol = res2.x[:size_U].reshape(n*m, rank)
                        V_pol = res2.x[size_U:size_U+size_V].reshape(m*p, rank)
                        W_pol = res2.x[size_U+size_V:].reshape(p*n, rank)
                        
                        valid2, U_r2, V_r2, W_r2 = _check_and_round(U_pol, V_pol, W_pol, tol=1e-3)
                        if valid2:
                            return {"U": U_r2.tolist(), "V": V_r2.tolist(), "W": W_r2.tolist()}
                except:
                    pass
                    
        except Exception as e:
            pass
        
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    # Try ranks from low to high, with more restarts for lower ranks
    # The rank of 3x3x4 is at most 16 (7 for 3x3x3 + 9 for 3x3x1)
    # Start searching from 12 to ensure we catch 16 if possible
    for rank in range(12, 37):
        if time.time() > deadline:
            break
        
        # More restarts for smaller ranks (harder to find)
        if rank <= 16:
            max_restarts = 15
        elif rank <= 20:
            max_restarts = 10
        elif rank <= 24:
            max_restarts = 5
        else:
            max_restarts = 3
        
        result = _try_find_decomposition(n, m, p, rank, max_restarts=max_restarts, deadline=deadline)
        if result:
            return result
            
    # Fallback to schoolbook
    return _schoolbook(n, m, p)
