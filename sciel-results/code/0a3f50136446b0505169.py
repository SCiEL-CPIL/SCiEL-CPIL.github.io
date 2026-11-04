"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    
    def verify(U, V, W):
        """Verify exact decomposition. Returns True if valid."""
        n, m, p = DIMENSIONS
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        r = U.shape[1]
        if V.shape[1] != r or W.shape[1] != r:
            return False
        # For each (i,j,k), check sum_t U[i*m+j,t]*V[j*p+k,t]*W[k*n+i,t] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for t in range(r):
                        total += U[i * m + j, t] * V[j * p + k, t] * W[k * n + i, t]
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True
    
    def schoolbook():
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

    def try_rank(r, seed=0):
        """Try to find a decomposition of rank r using random U, W and solving for V."""
        np.random.seed(seed)
        # Generate random U and W with integer entries
        # We use small integers to increase chance of integer V
        U = np.random.randint(-1, 2, size=(n * m, r))
        W = np.random.randint(-1, 2, size=(p * n, r))
        
        # Build linear system A * V_flat = b
        A = np.zeros((n * m * p, m * p * r))
        b = np.ones(n * m * p)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for t in range(r):
                        coeff = U[i * m + j, t] * W[k * n + i, t]
                        if coeff != 0:
                            var_idx = (j * p + k) * r + t
                            A[eq_idx, var_idx] += coeff
                    eq_idx += 1
        
        # Check if system is solvable (rank condition)
        # If any row is zero, it's unsolvable for b=1
        if np.any(np.all(A == 0, axis=1)):
            return None
            
        try:
            V_flat, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
            V = V_flat.reshape(m * p, r)
            
            # Check if V is close to integers
            if np.allclose(V, np.round(V), atol=1e-6):
                V = np.round(V)
                if verify(U.tolist(), V.tolist(), W.tolist()):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        except:
            pass
        return None

    # Try ranks from 112 downwards
    # We start with 112 (schoolbook is guaranteed, but random might find a different one)
    # Then try lower ranks.
    for r in [112, 110, 108, 106, 104, 102, 100, 98, 96, 94, 92, 90]:
        for seed in range(50):
            res = try_rank(r, seed)
            if res:
                return res
    
    # Fallback to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
