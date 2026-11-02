"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """Return the best exact tensor decomposition found within a time budget."""
    start = time.time()
    deadline = start + 140.0  # safety margin below 150s kill
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 84
    
    def _schoolbook():
        """Return the classical rank-nmp decomposition."""
        R = n * m * p
        U = [[0] * R for _ in range(n * m)]
        V = [[0] * R for _ in range(m * p)]
        W = [[0] * R for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def _verify(U, V, W):
        """Verify that the decomposition exactly reconstructs matrix multiplication."""
        R = len(U[0])
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        
        rng = np.random.RandomState(42)
        for trial in range(5):
            A = rng.randn(n, m)
            B = rng.randn(m, p)
            C_true = A @ B
            C_approx = np.zeros((n, p))
            for r in range(R):
                u_r = Ua[:, r]  # shape (n*m,)
                v_r = Va[:, r]  # shape (m*p,)
                w_r = Wa[:, r]  # shape (p*n,)
                u_scalar = np.dot(u_r, A.flatten())
                v_scalar = np.dot(v_r, B.flatten())
                C_approx += u_scalar * v_scalar * w_r.reshape(p, n).T
            if not np.allclose(C_approx, C_true, atol=1e-10):
                return False
        return True
    
    # Start with schoolbook
    result = _schoolbook()
    if _verify(result["U"], result["V"], result["W"]):
        return result
    
    # Fallback: return schoolbook anyway (should always be valid)
    return result
# EVOLVE-BLOCK-END
