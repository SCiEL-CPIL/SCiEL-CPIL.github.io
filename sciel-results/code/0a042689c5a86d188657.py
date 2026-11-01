"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def _verify_numpy(U, V, W, n, m, p, tol=1e-9):
    """Verify decomposition using numpy (fast, approximate)."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = np.dot(U_arr[i * m + j], V_arr[j * p + k] * W_arr[k * n + i])
                if abs(total - 1.0) > tol:
                    return False
    return True

def _verify_exact(U, V, W, n, m, p):
    """Verify with exact rational arithmetic."""
    U_f = [[Fraction(x) for x in row] for row in U]
    V_f = [[Fraction(x) for x in row] for row in V]
    W_f = [[Fraction(x) for x in row] for row in W]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for c in range(len(U_f[i * m + j])):
                    total += U_f[i * m + j][c] * V_f[j * p + k][c] * W_f[k * n + i][c]
                if total != 1:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return schoolbook decomposition of rank n*m*p."""
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

def construct():
    start_time = time.time()
    deadline = 140  # Safety margin before 150s limit

    n, m, p = DIMENSIONS
    # Start with schoolbook as fallback
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p  # 60

    # Try to find a lower-rank decomposition via structured search.
    # Strategy: Use the Kronecker/Strassen-Winograd idea.
    # For 2x3x10, we can view it as (2x3) x (3x10).
    # The 3x10 part can be split into 3x(5+5) or 3x(3+3+3+1).
    # However, a simpler approach: try rank 50 by using 5 terms per k
    # with a valid scheme. Since 2x3x1 has rank 6, we cannot do 5 per k.
    # So we stick with schoolbook unless we find a cross-k sharing.

    # Attempt: group columns in pairs. For k and k+1, compute:
    #   T0 = A[0,0]*(B[0,k]+B[0,k+1])
    #   T1 = A[0,1]*(B[1,k]+B[1,k+1])
    #   T2 = A[0,2]*(B[2,k]+B[2,k+1])
    #   T3 = A[1,0]*(B[0,k]+B[0,k+1])
    #   T4 = A[1,1]*(B[1,k]+B[1,k+1])
    #   T5 = A[1,2]*(B[2,k]+B[2,k+1])
    #   T6 = A[0,0]*(B[0,k]-B[0,k+1])
    #   T7 = A[0,1]*(B[1,k]-B[1,k+1])
    #   T8 = A[0,2]*(B[2,k]-B[2,k+1])
    #   T9 = A[1,0]*(B[0,k]-B[0,k+1])
    #   T10 = A[1,1]*(B[1,k]-B[1,k+1])
    #   T11 = A[1,2]*(B[2,k]-B[2,k+1])
    # That's 12 terms for 2 columns = 6 per column, same as schoolbook.

    # No gain. Return schoolbook.
    return best_result
# EVOLVE-BLOCK-END
