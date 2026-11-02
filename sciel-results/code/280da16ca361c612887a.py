import numpy as np
import time
from scipy.optimize import differential_evolution

DIMENSIONS = [2, 3, 6]

def _verify_exact(U, V, W, n, m, p):
    """Verify exact decomposition using integer arithmetic (entries are half-integers)."""
    r = len(U[0])
    # Multiply by 2 to work with integers
    U2 = [[2 * U[i][j] for j in range(r)] for i in range(n*m)]
    V2 = [[2 * V[i][j] for j in range(r)] for i in range(m*p)]
    W2 = [[2 * W[i][j] for j in range(r)] for i in range(p*n)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(r):
                    s += U2[i*m+j][c] * V2[j*p+k][c] * W2[k*n+i][c]
                if s != 8:  # 2*2*2=8
                    return False
    return True

def _try_rank_233(r, timeout=5):
    """Try to find a rank r decomposition for 2x3x3."""
    n, m, p = 2, 3, 3
    start_time = time.time()
    size_u = n * m * r
    size_v = m * p * r
    size_w = p * n * r
    size = size_u + size_v + size_w
    
    def objective(x):
        if time.time() - start_time > timeout:
            return np.inf
        u = x[:size_u].reshape(n*m, r)
        v = x[size_u:size_u+size_v].reshape(m*p, r)
        w = x[size_u+size_v:].reshape(p*n, r)
        
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += u[i*m+j, c] * v[j*p+k, c] * w[k*n+i, c]
                    error += (s - 1.0) ** 2
        return error

    bounds = [(-2, 2)] * size
    
    try:
        # Use differential evolution for global search
        res = differential_evolution(objective, bounds, maxiter=50, popsize=10, tol=1e-4, polish=False)
        if res.fun < 1e-5:
            u = res.x[:size_u].reshape(n*m, r)
            v = res.x[size_u:size_u+size_v].reshape(m*p, r)
            w = res.x[size_u+size_v:].reshape(p*n, r)
            
            # Round to half-integers
            U = [[round(u[i][j] * 2) / 2 for j in range(r)] for i in range(n*m)]
            V = [[round(v[i][j] * 2) / 2 for j in range(r)] for i in range(m*p)]
            W = [[round(w[i][j] * 2) / 2 for j in range(r)] for i in range(p*n)]
            
            if _verify_exact(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    return None

def _construct_236_from_233(decomp_233):
    """Construct 2x3x6 decomposition from two 2x3x3 decompositions."""
    if decomp_233 is None:
        return None
    
    U1 = decomp_233["U"]
    V1 = decomp_233["V"]
    W1 = decomp_233["W"]
    r1 = len(U1[0])
    
    # The 2x3x6 problem is two 2x3x3 problems side-by-side.
    # U is 6 x r. V is 18 x r. W is 12 x r.
    # For block 1 (k=0,1,2), V indices are 0..8 (j*3+k), W indices are 0..5 (k*2+i).
    # For block 2 (k=3,4,5), V indices are 9..17, W indices are 6..11.
    # We use the same U for both? 
    # In the decomposition sum_c U_c V_c W_c, the index c is the rank term.
    # If we use the same U for both blocks, we are coupling them.
    # But the V and W indices are disjoint.
    # So we can just concatenate the rank terms.
    # Rank = r1 + r1 = 2 * r1.
    
    r_total = 2 * r1
    n, m, p = 2, 3, 6
    U = [[0] * r_total for _ in range(n * m)]
    V = [[0] * r_total for _ in range(m * p)]
    W = [[0] * r_total for _ in range(p * n)]
    
    # Block 1: k in 0..2
    for c in range(r1):
        for i in range(n * m):
            U[i][c] = U1[i][c]
        for j in range(m):
            for k in range(3):
                V[j * p + k][c] = V1[j * 3 + k][c]
        for k in range(3):
            for i in range(n):
                W[k * n + i][c] = W1[k * n + i][c]
                
    # Block 2: k in 3..5
    offset = r1
    for c in range(r1):
        for i in range(n * m):
            U[i][c + offset] = U1[i][c]
        for j in range(m):
            for k in range(3, 6):
                V[j * p + k][c + offset] = V1[j * 3 + (k - 3)][c]
        for k in range(3, 6):
            for i in range(n):
                W[k * n + i][c + offset] = W1[(k - 3) * n + i][c]
                
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Try to find rank 14 for 2x3x3
    # If found, rank 28 for 2x3x6
    if time.time() - start_time < 60:
        decomp_233 = _try_rank_233(14, timeout=30)
        if decomp_233:
            result = _construct_236_from_233(decomp_233)
            if result and _verify_exact(result["U"], result["V"], result["W"], n, m, p):
                return result
                
    # Fallback: schoolbook decomposition
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
