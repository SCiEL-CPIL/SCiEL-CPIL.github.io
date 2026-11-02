"""Matrix multiplication 2x3 by 3x4: rank-minimal exact tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 4
    deadline = time.time() + 130
    
    # Schoolbook decomposition: rank = n*m*p = 24
    rank_school = n * m * p
    
    # Try to find a lower rank decomposition
    # The tensor T[i,j,k] = 1 for i in [0,n), j in [0,m), k in [0,p)
    # Decomposition: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    
    def check_decomp(U, V, W, tol=1e-8):
        r = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def try_find_rank(r, n_attempts=5000, seed_base=0):
        """Search for rank-r decomposition using random init + alternating minimization."""
        for attempt in range(n_attempts):
            if time.time() > deadline:
                return None
            rng = np.random.RandomState(seed_base + attempt)
            
            # Random initialization
            U = rng.randn(n*m, r)
            V = rng.randn(m*p, r)
            W = rng.randn(p*n, r)
            
            # Alternating minimization
            for it in range(100):
                if time.time() > deadline:
                    return None
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        M = np.zeros((r, p))
                        for k in range(p):
                            M[:, k] = V[j*p+k] * W[k*n+i]
                        try:
                            U[i*m+j] = np.linalg.lstsq(M.T, np.ones(p), rcond=None)[0]
                        except:
                            pass
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        M = np.zeros((r, n))
                        for i in range(n):
                            M[:, i] = U[i*m+j] * W[k*n+i]
                        try:
                            V[j*p+k] = np.linalg.lstsq(M.T, np.ones(n), rcond=None)[0]
                        except:
                            pass
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        M = np.zeros((r, m))
                        for j in range(m):
                            M[:, j] = U[i*m+j] * V[j*p+k]
                        try:
                            W[k*n+i] = np.linalg.lstsq(M.T, np.ones(m), rcond=None)[0]
                        except:
                            pass
                
                # Quick check
                if check_decomp(U.tolist(), V.tolist(), W.tolist()):
                    return (U, V, W)
        
        return None
    
    # Try ranks from 23 down to 12
    for r in range(23, 11, -1):
        if time.time() > deadline:
            break
        result = try_find_rank(r, n_attempts=2000, seed_base=r*1000)
        if result is not None:
            U, V, W = result
            return {
                "U": [[float(x) for x in row] for row in U],
                "V": [[float(x) for x in row] for row in V],
                "W": [[float(x) for x in row] for row in W]
            }
    
    # Fall back to schoolbook
    rank = rank_school
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1.0
                V[j*p+k][col] = 1.0
                W[k*n+i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
