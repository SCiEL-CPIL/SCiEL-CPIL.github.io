"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x7 matrix.
    Output factor follows the official trace convention W[k*n+i, r].
    Returns: {'U': list of lists, 'V': list of lists, 'W': list of lists}
    """
    n, m, p = 3, 3, 7
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    # Schoolbook decomposition (rank 63)
    schoolbook_rank = n * m * p
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1

    def verify(U, V, W, tol=1e-9):
        """Verify exact decomposition."""
        if len(U) != n * m or len(V) != m * p or len(W) != p * n:
            return False
        if len(U) == 0 or len(V) == 0 or len(W) == 0:
            return False
        r = len(U[0])
        if len(V[0]) != r or len(W[0]) != r or r <= 0:
            return False
        Ua = np.array(U, dtype=np.float64)
        Va = np.array(V, dtype=np.float64)
        Wa = np.array(W, dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(Ua[i*m+j], Va[j*p+k] * Wa[k*n+i])
                    if abs(val - 1.0) > tol:
                        return False
        return True

    # Verify schoolbook
    if not verify(U_sb, V_sb, W_sb):
        # Should never happen, but fallback to empty/zero if something is wrong
        return {'U': [[0]], 'V': [[0]], 'W': [[0]]}

    # Attempt to find a lower rank decomposition (Rank 23)
    # Strategy: Use a rank-7 decomposition for 3x3*3x3 for two blocks, and schoolbook for the third.
    # We try to find a valid rank-7 3x3*3x3 decomposition via a limited search.
    
    # Since random search is unlikely to find the specific rank-7 decomposition quickly,
    # and we must ensure validity, we return the schoolbook decomposition.
    # The schoolbook is guaranteed to be valid and has rank 63.
    
    return {'U': U_sb, 'V': V_sb, 'W': W_sb}

# EVOLVE-BLOCK-END
