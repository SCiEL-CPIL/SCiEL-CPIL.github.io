"""Program for 2x4 by 4x8 matrix multiplication tensor decomposition."""
import time

DIMENSIONS = [2, 4, 8]
_START_TIME = time.time()
_DEADLINE = 120.0  # seconds, with safety margin from 150s limit

def _check_deadline():
    """Return True if we should stop and return best result."""
    return (time.time() - _START_TIME) > (_DEADLINE - 5.0)

def construct():
    """
    Construct a rank-56 decomposition for 2x4 by 4x8 matrix multiplication.
    Uses 8 independent 2x2 Strassen decompositions.
    """
    n, m, p = DIMENSIONS  # A: (n, m) = (2,4), B: (m, p) = (4,8), C: (n, p) = (2,8)
    
    if _check_deadline():
        # Return a trivial but valid rank-64 decomposition as fallback
        rank_fb = n * m * p
        U_fb = [[0.0] * rank_fb for _ in range(n * m)]
        V_fb = [[0.0] * rank_fb for _ in range(m * p)]
        W_fb = [[0.0] * rank_fb for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U_fb[i * m + j][r] = 1.0
                    V_fb[j * p + k][r] = 1.0
                    W_fb[k * n + i][r] = 1.0
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    rank = 56
    U = [[0.0] * rank for _ in range(n * m)]  # shape (8, 56)
    V = [[0.0] * rank for _ in range(m * p)]  # shape (32, 56)
    W = [[0.0] * rank for _ in range(p * n)]  # shape (16, 56)
    
    # Strassen's 7 products for 2x2 matrices:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    for d in range(4):  # B column pair: 0->(0,1), 1->(2,3), 2->(4,5), 3->(6,7)
        for h in range(2):  # h=0: A left half, B top rows; h=1: A right half, B bottom rows
            base = (d * 2 + h) * 7
            if h == 0:
                def a_idx(r, c):
                    return r * m + c
                def b_idx(r, c):
                    return r * p + (2 * d + c)
                def c_idx(r, c):
                    return (2 * d + c) * n + r
            else:
                def a_idx(r, c):
                    return r * m + (2 + c)
                def b_idx(r, c):
                    return (2 + r) * p + (2 * d + c)
                def c_idx(r, c):
                    return (2 * d + c) * n + r
            
            col = base + 0
            U[a_idx(0, 0)][col] += 1.0
            U[a_idx(1, 1)][col] += 1.0
            V[b_idx(0, 0)][col] += 1.0
            V[b_idx(1, 1)][col] += 1.0
            W[c_idx(0, 0)][col] += 1.0
            W[c_idx(1, 1)][col] += 1.0
            
            col = base + 1
            U[a_idx(1, 0)][col] += 1.0
            U[a_idx(1, 1)][col] += 1.0
            V[b_idx(0, 0)][col] += 1.0
            W[c_idx(1, 0)][col] += 1.0
            W[c_idx(1, 1)][col] -= 1.0
            
            col = base + 2
            U[a_idx(0, 0)][col] += 1.0
            V[b_idx(0, 1)][col] += 1.0
            V[b_idx(1, 1)][col] -= 1.0
            W[c_idx(0, 1)][col] += 1.0
            W[c_idx(1, 1)][col] += 1.0
            
            col = base + 3
            U[a_idx(1, 1)][col] += 1.0
            V[b_idx(1, 0)][col] += 1.0
            V[b_idx(0, 0)][col] -= 1.0
            W[c_idx(0, 0)][col] += 1.0
            W[c_idx(1, 0)][col] += 1.0
            
            col = base + 4
            U[a_idx(0, 0)][col] += 1.0
            U[a_idx(0, 1)][col] += 1.0
            V[b_idx(1, 1)][col] += 1.0
            W[c_idx(0, 0)][col] -= 1.0
            W[c_idx(0, 1)][col] += 1.0
            
            col = base + 5
            U[a_idx(1, 0)][col] += 1.0
            U[a_idx(0, 0)][col] -= 1.0
            V[b_idx(0, 0)][col] += 1.0
            V[b_idx(0, 1)][col] += 1.0
            W[c_idx(1, 1)][col] += 1.0
            
            col = base + 6
            U[a_idx(0, 1)][col] += 1.0
            U[a_idx(1, 1)][col] -= 1.0
            V[b_idx(1, 0)][col] += 1.0
            V[b_idx(1, 1)][col] += 1.0
            W[c_idx(0, 0)][col] += 1.0

    return {"U": U, "V": V, "W": W}
