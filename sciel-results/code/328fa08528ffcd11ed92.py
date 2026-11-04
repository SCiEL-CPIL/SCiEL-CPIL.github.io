"""Tensor decomposition for 2x3 x 3x7 matrix multiplication using iterative optimization."""
import numpy as np
import time

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 7
    nm, mp, pn = n * m, m * p, p * n  # 6, 21, 14
    start_time = time.time()
    time_limit = 100  # seconds, leaving buffer for 120s total

    def verify(U, V, W):
        """Check if the decomposition is correct."""
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for jp in range(m):
                        val = 0.0
                        for c in range(len(U[0])):
                            val += U[i*m+j][c] * V[jp*p+k][c] * W[k*n+i][c]
                        expected = 1.0 if j == jp else 0.0
                        if abs(val - expected) > 1e-6:
                            return False
        return True

    def try_rank(r):
        """Try to find rank-r decomposition using alternating least squares."""
        np.random.seed(0)
        U = np.random.randn(nm, r)
        V = np.random.randn(mp, r)
        W = np.random.randn(pn, r)
        
        for it in range(300):
            if time.time() - start_time > time_limit:
                return None
            
            # Update V rows
            for k in range(p):
                for jp in range(m):
                    rows = np.array([U[i*m+j, :] * W[k*n+i, :] for i in range(n) for j in range(m)])
                    b = np.array([1.0 if j == jp else 0.0 for i in range(n) for j in range(m)])
                    try:
                        V[jp*p+k, :] = np.linalg.lstsq(rows, b, rcond=None)[0]
                    except:
                        pass
            
            # Update U rows
            for i in range(n):
                for j in range(m):
                    rows = np.array([V[jp*p+k, :] * W[k*n+i, :] for k in range(p) for jp in range(m)])
                    b = np.array([1.0 if j == jp else 0.0 for k in range(p) for jp in range(m)])
                    try:
                        U[i*m+j, :] = np.linalg.lstsq(rows, b, rcond=None)[0]
                    except:
                        pass
            
            # Update W rows
            for k in range(p):
                for i in range(n):
                    rows = np.array([U[i*m+j, :] * V[jp*p+k, :] for j in range(m) for jp in range(m)])
                    b = np.array([1.0 if j == jp else 0.0 for j in range(m) for jp in range(m)])
                    try:
                        W[k*n+i, :] = np.linalg.lstsq(rows, b, rcond=None)[0]
                    except:
                        pass
            
            # Quick error check every 10 iterations
            if it % 10 == 0:
                err = 0.0
                for i in range(n):
                    for k in range(p):
                        for j in range(m):
                            for jp in range(m):
                                val = np.dot(U[i*m+j, :], V[jp*p+k, :] * W[k*n+i, :])
                                expected = 1.0 if j == jp else 0.0
                                err += (val - expected) ** 2
                if err < 1e-10:
                    return (U.tolist(), V.tolist(), W.tolist())
        
        return None

    # Try ranks from lower bound upward
    lower_bound = max(nm, mp, pn)  # 21
    for r in range(lower_bound, 43):
        if time.time() - start_time > time_limit:
            break
        result = try_rank(r)
        if result is not None:
            U_list, V_list, W_list = result
            # Round to nearest half-integer for cleaner output
            def round_half(x):
                return round(x * 2) / 2
            U_list = [[round_half(v) for v in row] for row in U_list]
            V_list = [[round_half(v) for v in row] for row in V_list]
            W_list = [[round_half(v) for v in row] for row in W_list]
            return {"U": U_list, "V": V_list, "W": W_list}

    # Fallback: schoolbook rank 42
    rank = n * m * p
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
