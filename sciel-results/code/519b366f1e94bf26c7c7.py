import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 5]
_DEADLINE = 135.0

def _remaining():
    return _DEADLINE - time.time()

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}

def _verify(n, m, p, U, V, W):
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    return True

def _try_structured_decomposition(n, m, p):
    """
    Try to find a lower-rank decomposition by exploiting the structure of
    2x5x5 matrix multiplication. The key insight is that we can group terms
    by the row index i and try to find efficient decompositions for each block.
    """
    if time.time() > _DEADLINE - 15:
        return None
    
    # For 2x5x5, try ranks from 49 down to some minimum
    # The theoretical lower bound is max(n*m, m*p, p*n) = max(10, 25, 10) = 25
    # But realistically, we might find something in the 40s-49 range
    
    try:
        import scipy.optimize as opt
        
        # Try a structured approach: use the fact that n=2 is small
        # We can write the multiplication as:
        # C[0,k] = sum_j A[0,j] * B[j,k]
        # C[1,k] = sum_j A[1,j] * B[j,k]
        
        # This is essentially two independent 1x5 x 5x5 multiplications
        # Each can potentially be done with rank < 25
        
        # Let's try to find a decomposition with rank 49 first (just below schoolbook)
        target_rank = 49
        
        # Use a better initialization strategy
        # Initialize with schoolbook and then try to merge terms
        np.random.seed(123)
        
        # Start with schoolbook and try to optimize
        U0 = np.zeros((n*m, target_rank))
        V0 = np.zeros((m*p, target_rank))
        W0 = np.zeros((p*n, target_rank))
        
        # Initialize with first 49 schoolbook terms
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if c < target_rank:
                        U0[i*m+j, c] = 1
                        V0[j*p+k, c] = 1
                        W0[k*n+i, c] = 1
                        c += 1
        
        # Now try to optimize to find a valid decomposition
        params = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        
        def objective(params):
            U_flat, V_flat, W_flat = np.split(params, [n*m*target_rank, 2*n*m*target_rank])
            U = U_flat.reshape(n*m, target_rank)
            V = V_flat.reshape(m*p, target_rank)
            W = W_flat.reshape(p*n, target_rank)
            
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(target_rank):
                            s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        error += (s - 1.0)**2
            return error
        
        # Try Nelder-Mead with limited iterations
        res = opt.minimize(objective, params, method='Nelder-Mead', 
                           options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-8})
        
        if res.fun < 1e-6:
            U_flat, V_flat, W_flat = np.split(res.x, [n*m*target_rank, 2*n*m*target_rank])
            U = U_flat.reshape(n*m, target_rank)
            V = V_flat.reshape(m*p, target_rank)
            W = W_flat.reshape(p*n, target_rank)
            
            # Try rounding to rationals with small denominators
            for denom in [1, 2, 4, 8]:
                U_r = np.round(U * denom) / denom
                V_r = np.round(V * denom) / denom
                W_r = np.round(W * denom) / denom
                
                if _verify(n, m, p, U_r.tolist(), V_r.tolist(), W_r.tolist()):
                    return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
        
    except Exception:
        pass
    
    return None

def _try_greedy_cover(n, m, p):
    """
    Greedy set-cover style approach: try to find rank-1 terms that cover
    multiple (i,j,k) entries simultaneously.
    """
    if time.time() > _DEADLINE - 10:
        return None
    
    # For each possible rank-1 term, determine which (i,j,k) it covers
    # A rank-1 term is defined by vectors u (n*m), v (m*p), w (p*n)
    # It covers (i,j,k) if u[i*m+j] * v[j*p+k] * w[k*n+i] != 0
    
    # Try small integer-valued terms
    best_rank = n * m * p  # schoolbook
    best_decomp = None
    
    # Generate candidate terms with small integer entries
    # Focus on terms that cover multiple entries
    candidates = []
    
    # Unit terms (cover exactly one entry)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = [0] * (n*m)
                v = [0] * (m*p)
                w = [0] * (p*n)
                u[i*m+j] = 1
                v[j*p+k] = 1
                w[k*n+i] = 1
                candidates.append((u, v, w, 1))
    
    # Try some structured terms that might cover multiple entries
    # For example, terms where u has two non-zero entries
    for i in range(n):
        for j1 in range(m):
            for j2 in range(j1+1, m):
                for k in range(p):
                    u = [0] * (n*m)
                    v = [0] * (m*p)
                    w = [0] * (p*n)
                    u[i*m+j1] = 1
                    u[i*m+j2] = 1
                    v[j1*p+k] = 1
                    v[j2*p+k] = 1
                    w[k*n+i] = 1
                    # This covers (i,j1,k) and (i,j2,k)
                    candidates.append((u, v, w, 2))
    
    # Sort by coverage (descending)
    candidates.sort(key=lambda x: -x[3])
    
    # Greedy selection
    covered = set()
    selected = []
    
    for u, v, w, _ in candidates:
        if len(selected) >= best_rank - 1:  # Can't improve
            break
        
        # Determine which entries this term covers
        term_covers = set()
        for i in range(n):
            for j in range(m):
                if u[i*m+j] == 0:
                    continue
                for k in range(p):
                    if v[j*p+k] != 0 and w[k*n+i] != 0:
                        term_covers.add((i, j, k))
        
        # Only add if it covers at least one uncovered entry
        new_covers = term_covers - covered
        if new_covers:
            selected.append((u, v, w))
            covered |= term_covers
    
    if len(covered) == n * m * p and len(selected) < best_rank:
        r = len(selected)
        U = [[0] * r for _ in range(n*m)]
        V = [[0] * r for _ in range(m*p)]
        W = [[0] * r for _ in range(p*n)]
        
        for c, (u, v, w) in enumerate(selected):
            for a in range(n*m):
                U[a][c] = u[a]
            for a in range(m*p):
                V[a][c] = v[a]
            for a in range(p*n):
                W[a][c] = w[a]
        
        if _verify(n, m, p, U, V, W):
            return {"U": U, "V": V, "W": W}
    
    return None

def _construct():
    n, m, p = DIMENSIONS
    best = _schoolbook(n, m, p)
    
    # Try structured decomposition
    result = _try_structured_decomposition(n, m, p)
    if result is not None:
        if _verify(n, m, p, result['U'], result['V'], result['W']):
            return result
    
    # Try greedy cover
    result = _try_greedy_cover(n, m, p)
    if result is not None:
        if _verify(n, m, p, result['U'], result['V'], result['W']):
            return result
    
    return best

def construct():
    try:
        return _construct()
    except Exception:
        n, m, p = DIMENSIONS
        return _schoolbook(n, m, p)
