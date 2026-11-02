import numpy as np
from fractions import Fraction
import time

def construct():
    n, m, p = 2, 3, 7
    deadline = time.time() + 140.0

    # Build valid indices
    valid_indices = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                valid_indices.add((a, b, c))

    def verify_float(U, V, W, rank):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        T_recon = np.zeros((n*m, m*p, p*n), dtype=float)
        for r in range(rank):
            T_recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        
        for (a, b, c) in valid_indices:
            if abs(T_recon[a, b, c] - 1.0) > 1e-6:
                return False
        
        mask = np.zeros((n*m, m*p, p*n), dtype=bool)
        for (a, b, c) in valid_indices:
            mask[a, b, c] = True
        if np.any(np.abs(T_recon[~mask]) > 1e-6):
            return False
        return True

    # Strategy 1: Schoolbook decomposition (rank 42)
    def schoolbook_decomposition():
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m+j
                    b = j*p+k
                    c = k*n+i
                    U[a][col] = 1
                    V[b][col] = 1
                    W[c][col] = 1
                    col += 1
        return U, V, W, rank

    U_best, V_best, W_best, rank_best = schoolbook_decomposition()

    # Verify the schoolbook decomposition
    if not verify_float(U_best, V_best, W_best, rank_best):
        # This should never happen
        pass

    return {
        "U": U_best,
        "V": V_best,
        "W": W_best
    }
