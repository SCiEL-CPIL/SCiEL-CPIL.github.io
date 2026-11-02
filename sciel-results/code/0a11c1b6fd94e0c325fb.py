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
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
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
    
    all_triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    
    rng.shuffle(all_triples)
    for r in range(rank):
        start = r * (len(all_triples) // rank)
        end = (r + 1) * (len(all_triples) // rank) if r < rank - 1 else len(all_triples)
        
        for (i, j, k) in all_triples[start:end]:
            x0[i*m + j + r * n * m] = 1.0
            x0[(n*m) + (j*p + k) + r * m * p] = 1.0
            x0[(n*m + m*p) + (k*n + i) + r * p * n] = 1.0
    
    x0 += rng.normal(0, 0.05, total_size)
    
    return x0

def _random_init(n, m, p, rank, seed=0, scale=0.5):
    """Random initialization."""
    rng = np.random.RandomState(seed)
    size_U = n * m * rank
    size_V = m * p * rank
    size_W = p * n * rank
    total_size = size_U + size_V + size_W
    x0 = rng.normal(0, scale, total_size)
    return x0

def _residual(x, n, m, p, rank):
    size_U = n * m * rank
    size_V = m * p * rank
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

def _try_find_decomposition(n, m, p, rank, max_restarts=5, deadline=None):
    """Try to find a rank-rank decomposition using multiple restarts."""
    size_U = n * m * rank
    size_V = m * p * rank
    size_W = p * n * rank
    total_size = size_U + size_V + size_W
    
    best_result = None
    
    for restart in range(max_restarts):
        if deadline is not None and time.time() > deadline:
            break
        
        # Alternate between structured and random initialization
        if restart % 2 == 0:
            x0 = _structured_init(n, m, p, rank, seed=restart * 1000 + 42)
        else:
            x0 = _random_init(n, m, p, rank, seed=restart * 1000 + 137, scale=0.5)
        
        try:
            res = least_squares(lambda x: _residual(x, n, m, p, rank), x0, method='lm', xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=2000)
            
            if res.cost < 1e-10:
                x_sol = res.x
                U_sol = x_sol[:size_U].reshape(n*m, rank)
                V_sol = x_sol[size_U:size_U+size_V].reshape(m*p, rank)
                W_sol = x_sol[size_U+size_V:].reshape(p*n, rank)
                
                valid, U_r, V_r, W_r = _check_and_round(U_sol, V_sol, W_sol)
                if valid:
                    return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                
                # Polish step
                try:
                    res2 = least_squares(lambda x: _residual(x, n, m, p, rank), x_sol, method='lm', xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=500)
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

def _ladder_init(n, m, p, rank, seed=0):
    """
    Ladder/peeling initialization: treat the 3x3x4 tensor as 3x3x3 + extension.
    Start with a known good 3x3x3 structure embedded in the first 3 columns of W,
    then add extra rank terms to cover the 4th dimension.
    """
    rng = np.random.RandomState(seed)
    
    size_U = n * m * rank
    size_V = m * p * rank
    size_W = p * n * rank
    total_size = size_U + size_V + size_W
    x0 = np.zeros(total_size)
    
    # Base: use schoolbook for 3x3x3 part (k=0,1,2)
    # This uses n*m*3 = 27 rank terms
    base_rank = n * m * 3  # 27
    if base_rank > rank:
        base_rank = rank
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(3):  # Only first 3 columns
                if col < rank:
                    x0[i*m + j + col * n * m] = 1.0
                    x0[(n*m) + (j*p + k) + col * m * p] = 1.0
                    x0[(n*m + m*p) + (k*n + i) + col * p * n] = 1.0
                    col += 1
    
    # Remaining rank terms: initialize for k=3 coverage with small random values
    for r in range(col, rank):
        # These terms should help cover the k=3 entries
        for i in range(n):
            for j in range(m):
                x0[i*m + j + r * n * m] = rng.normal(0, 0.3)
        for k in range(p):
            for j in range(m):
                x0[(n*m) + (j*p + k) + r * m * p] = rng.normal(0, 0.3)
        for k in range(p):
            for i in range(n):
                x0[(n*m + m*p) + (k*n + i) + r * p * n] = rng.normal(0, 0.3)
    
    # Add small noise to break symmetry
    x0 += rng.normal(0, 0.02, total_size)
    
    return x0

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    # Try ranks from low to high
    # The rank of 3x3x4 is at most 36 (schoolbook)
    # Known: rank(3x3x3) = 23 (Strassen-like), so rank(3x3x4) <= 23 + 9 = 32
    # But we can do better with shared structure
    # Start from 16 and go up
    for rank in range(16, 37):
        if time.time() > deadline:
            break
        
        # More restarts for smaller ranks
        if rank <= 20:
            max_restarts = 20
        elif rank <= 24:
            max_restarts = 15
        elif rank <= 28:
            max_restarts = 10
        else:
            max_restarts = 5
        
        # First try ladder initialization
        if time.time() < deadline:
            size_U = n * m * rank
            size_V = m * p * rank
            size_W = p * n * rank
            total_size = size_U + size_V + size_W
            
            x0_ladder = _ladder_init(n, m, p, rank, seed=42)
            try:
                res = least_squares(lambda x: _residual(x, n, m, p, rank), x0_ladder, method='lm', xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=3000)
                if res.cost < 1e-10:
                    x_sol = res.x
                    U_sol = x_sol[:size_U].reshape(n*m, rank)
                    V_sol = x_sol[size_U:size_U+size_V].reshape(m*p, rank)
                    W_sol = x_sol[size_U+size_V:].reshape(p*n, rank)
                    
                    valid, U_r, V_r, W_r = _check_and_round(U_sol, V_sol, W_sol)
                    if valid:
                        return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                    
                    # Polish
                    try:
                        res2 = least_squares(lambda x: _residual(x, n, m, p, rank), x_sol, method='lm', xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=500)
                        if res2.cost < 1e-12:
                            U_pol = res2.x[:size_U].reshape(n*m, rank)
                            V_pol = res2.x[size_U:size_U+size_V].reshape(m*p, rank)
                            W_pol = res2.x[size_U+size_V:].reshape(p*n, rank)
                            
                            valid2, U_r2, V_r2, W_r2 = _check_and_round(U_pol, V_pol, W_pol, tol=1e-3)
                            if valid2:
                                return {"U": U_r2.tolist(), "V": V_r2.tolist(), "W": W_r2.tolist()}
                    except:
                        pass
            except:
                pass
        
        # Then try general search with multiple restarts
        result = _try_find_decomposition(n, m, p, rank, max_restarts=max_restarts, deadline=deadline)
        if result:
            return result
            
    # Fallback to schoolbook
    return _schoolbook(n, m, p)
