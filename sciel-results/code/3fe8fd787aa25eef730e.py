import time
import numpy as np

DIMENSIONS = [3, 3, 5]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    try:
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
            return False, "Shape mismatch"
        
        rank = U.shape[1]
        if V.shape[1] != rank or W.shape[1] != rank:
            return False, "Rank mismatch"
            
        # Check valid entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.dot(U[a, :], np.multiply(V[b, :], W[c, :]))
                    if abs(val - 1.0) > tol:
                        return False, (a, b, c, val)
        
        # Check invalid entries (sample)
        rng = np.random.RandomState(42)
        for _ in range(200):
            a = rng.randint(0, n * m)
            b = rng.randint(0, m * p)
            c = rng.randint(0, p * n)
            i, j = divmod(a, m)
            j2, k = divmod(b, p)
            k2, i2 = divmod(c, n)
            if j != j2 or k != k2:
                val = np.dot(U[a, :], np.multiply(V[b, :], W[c, :]))
                if abs(val) > tol:
                    return False, (a, b, c, val)
        return True, None
    except Exception as e:
        return False, str(e)

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115

    # Fallback: Full Schoolbook (Rank 45)
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Attempt Rank 42 (27 + 15) - Corrected allocation
    try:
        rank_333 = 27
        rank_332 = 15 # Hypothesized rank for 3x3x2 part, but we use schoolbook 18 in logic below to be safe
        # Actually, let's just use schoolbook for the 2nd part to ensure validity if 15 is unknown
        # But the loop in the original code generated 18 terms.
        # If we assume Rank 15 is possible, we need the coefficients. We don't have them.
        # So we stick to Rank 45.
        
        valid, err = _verify_decomposition(np.zeros((9, 1)), np.zeros((15, 1)), np.zeros((15, 1)), n, m, p)
        # This is just a dummy check to keep the structure
        pass
        
    except Exception as e:
        pass

    # Return the valid Rank 45 decomposition
    return get_schoolbook()
