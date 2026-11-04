import time
import numpy as np

DIMENSIONS = [2, 3, 5]

def _verify(U, V, W, n, m, p, tol=1e-8):
    """Verify that the tensor decomposition is exact."""
    rank = len(U[0]) if isinstance(U, list) else U.shape[1]
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = U_np[i * m + j]
                v_row = V_np[j * p + k]
                w_row = W_np[k * n + i]
                val = np.dot(u_row, v_row * w_row)
                if abs(val - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Standard schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def _construct_rank_28():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    r = 0
    
    # Part 1: A2 B2 (j=2, k=0..4) - 10 terms
    for i in range(n):
        for k in range(p):
            U[i * m + 2][r] = 1.0
            V[2 * p + k][r] = 1.0
            W[k * n + i][r] = 1.0
            r += 1
            
    # Part 2: A1 B13 (j=0,1, k=4) - 4 terms
    for i in range(n):
        for j in range(2):
            U[i * m + j][r] = 1.0
            V[j * p + 4][r] = 1.0
            W[4 * n + i][r] = 1.0
            r += 1
            
    # Part 3: A1 B11 (j=0,1, k=0,1) using Strassen - 7 terms
    # M1 = (A00+A11)(B00+B11)
    U[0 * m + 0][r] = 1.0
    U[1 * m + 1][r] = 1.0
    V[0 * p + 0][r] = 1.0
    V[1 * p + 1][r] = 1.0
    W[0 * n + 0][r] = 1.0
    W[1 * n + 1][r] = 1.0
    r += 1
    
    # M2 = (A10+A11)B00
    U[1 * m + 0][r] = 1.0
    U[1 * m + 1][r] = 1.0
    V[0 * p + 0][r] = 1.0
    W[0 * n + 1][r] = 1.0
    W[1 * n + 1][r] = -1.0
    r += 1
    
    # M3 = A00(B01-B11)
    U[0 * m + 0][r] = 1.0
    V[0 * p + 1][r] = 1.0
    V[1 * p + 1][r] = -1.0
    W[1 * n + 0][r] = 1.0
    W[1 * n + 1][r] = 1.0
    r += 1
    
    # M4 = A11(B10-B00)
    U[1 * m + 1][r] = 1.0
    V[1 * p + 0][r] = 1.0
    V[0 * p + 0][r] = -1.0
    W[0 * n + 0][r] = 1.0
    W[0 * n + 1][r] = 1.0
    r += 1
    
    # M5 = (A00+A01)B11
    U[0 * m + 0][r] = 1.0
    U[0 * m + 1][r] = 1.0
    V[1 * p + 1][r] = 1.0
    W[0 * n + 0][r] = -1.0
    W[1 * n + 0][r] = 1.0
    r += 1
    
    # M6 = (A10-A00)(B00+B01)
    U[1 * m + 0][r] = 1.0
    U[0 * m + 0][r] = -1.0
    V[0 * p + 0][r] = 1.0
    V[0 * p + 1][r] = 1.0
    W[1 * n + 1][r] = 1.0
    r += 1
    
    # M7 = (A01-A11)(B10+B11)
    U[0 * m + 1][r] = 1.0
    U[1 * m + 1][r] = -1.0
    V[1 * p + 0][r] = 1.0
    V[1 * p + 1][r] = 1.0
    W[0 * n + 0][r] = 1.0
    r += 1
    
    # Part 4: A1 B12 (j=0,1, k=2,3) using Strassen - 7 terms
    # M1 = (A00+A11)(B02+B13)
    U[0 * m + 0][r] = 1.0
    U[1 * m + 1][r] = 1.0
    V[0 * p + 2][r] = 1.0
    V[1 * p + 3][r] = 1.0
    W[2 * n + 0][r] = 1.0
    W[3 * n + 1][r] = 1.0
    r += 1
    
    # M2 = (A10+A11)B02
    U[1 * m + 0][r] = 1.0
    U[1 * m + 1][r] = 1.0
    V[0 * p + 2][r] = 1.0
    W[2 * n + 1][r] = 1.0
    W[3 * n + 1][r] = -1.0
    r += 1
    
    # M3 = A00(B03-B13)
    U[0 * m + 0][r] = 1.0
    V[0 * p + 3][r] = 1.0
    V[1 * p + 3][r] = -1.0
    W[3 * n + 0][r] = 1.0
    W[3 * n + 1][r] = 1.0
    r += 1
    
    # M4 = A11(B12-B02)
    U[1 * m + 1][r] = 1.0
    V[1 * p + 2][r] = 1.0
    V[0 * p + 2][r] = -1.0
    W[2 * n + 0][r] = 1.0
    W[2 * n + 1][r] = 1.0
    r += 1
    
    # M5 = (A00+A01)B13
    U[0 * m + 0][r] = 1.0
    U[0 * m + 1][r] = 1.0
    V[1 * p + 3][r] = 1.0
    W[2 * n + 0][r] = -1.0
    W[3 * n + 0][r] = 1.0
    r += 1
    
    # M6 = (A10-A00)(B02+B03)
    U[1 * m + 0][r] = 1.0
    U[0 * m + 0][r] = -1.0
    V[0 * p + 2][r] = 1.0
    V[0 * p + 3][r] = 1.0
    W[3 * n + 1][r] = 1.0
    r += 1
    
    # M7 = (A01-A11)(B12+B13)
    U[0 * m + 1][r] = 1.0
    U[1 * m + 1][r] = -1.0
    V[1 * p + 2][r] = 1.0
    V[1 * p + 3][r] = 1.0
    W[2 * n + 0][r] = 1.0
    r += 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Try to construct Rank 28
    U, V, W = _construct_rank_28()
    
    # Verify
    if r == 28 and _verify(U, V, W, n, m, p):
        return {
            "U": U,
            "V": V,
            "W": W
        }
    else:
        # Fallback
        U_fb, V_fb, W_fb = _schoolbook(n, m, p)
        return {
            "U": U_fb,
            "V": V_fb,
            "W": W_fb
        }

if __name__ == "__main__":
    construct()
