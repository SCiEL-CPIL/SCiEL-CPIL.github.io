"""Matrix multiplication 2,2,12 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for multiplying a 2x2 matrix by a 2x12 matrix.
    Uses Strassen-based block construction.
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    num_pairs = p // 2  # 6
    deadline = time.time() + 140  # 10s safety margin

    def build_strassen_blocks():
        """Build the 6 independent Strassen blocks, return U, V, W as (4,42), (24,42), (24,42)."""
        rank = num_pairs * 7
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)

        for t in range(num_pairs):
            base = t * 7
            k0 = 2 * t
            k1 = 2 * t + 1

            for s in range(7):
                c = base + s
                if s == 0:
                    U[0, c] += 1; U[3, c] += 1
                    V[k0, c] += 1; V[p + k1, c] += 1
                    W[k0*n, c] += 1; W[k1*n+1, c] += 1
                elif s == 1:
                    U[2, c] += 1; U[3, c] += 1
                    V[k0, c] += 1
                    W[k0*n+1, c] += 1; W[k1*n+1, c] += -1
                elif s == 2:
                    U[0, c] += 1
                    V[k1, c] += 1; V[p+k1, c] += -1
                    W[k1*n, c] += 1; W[k1*n+1, c] += 1
                elif s == 3:
                    U[3, c] += 1
                    V[p+k0, c] += 1; V[k0, c] += -1
                    W[k0*n, c] += 1; W[k0*n+1, c] += 1
                elif s == 4:
                    U[0, c] += 1; U[1, c] += 1
                    V[p+k1, c] += 1
                    W[k0*n, c] += -1; W[k1*n, c] += 1
                elif s == 5:
                    U[2, c] += 1; U[0, c] += -1
                    V[k0, c] += 1; V[k1, c] += 1
                    W[k1*n+1, c] += 1
                elif s == 6:
                    U[1, c] += 1; U[3, c] += -1
                    V[p+k0, c] += 1; V[p+k1, c] += 1
                    W[k0*n, c] += 1
        return U, V, W

    def verify(U, V, W, tol=1e-9):
        """Check if decomposition is exact: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(U.shape[1]):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1.0) > tol:
                        return False
        return True

    # Build the base Strassen decomposition
    U0, V0, W0 = build_strassen_blocks()
    
    # Verify base case
    assert verify(U0, V0, W0), "Base Strassen decomposition failed verification"
    
    best_rank = U0.shape[1]
    best_U, best_V, best_W = U0, V0, W0
    
    # Convert to lists
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
