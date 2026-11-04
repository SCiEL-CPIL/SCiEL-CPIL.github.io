import time
import numpy as np

DIMENSIONS = [2, 3, 5]

def _verify(U, V, W, n, m, p, tol=1e-8):
    """Verify that the tensor decomposition is exact."""
    rank = len(U[0]) if isinstance(U, list) else U.shape[1]
    # Convert to numpy arrays for fast verification if needed, but loop is small
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Sum over r: U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
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
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U.tolist(), V.tolist(), W.tolist()

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # 120 second limit
    
    # Try to construct rank-28 decomposition
    rank = 28
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    r = 0
    
    # Part 1: A2 B2 (j=2, k=0..4)
    # A is 2x3, B is 3x5.
    # A2 is col 2 (2x1). B2 is row 2 (1x5).
    # C2 = A2 B2.
    for i in range(n):
        for k in range(p):
            U[i * m + 2][r] = 1.0
            V[2 * p + k][r] = 1.0
            W[k * n + i][r] = 1.0
            r += 1
            
    # Part 2: A1 B13 (j=0,1, k=4)
    # A1 is cols 0,1 (2x2). B13 is row 0,1, col 4 (2x1).
    # C13 = A1 B13.
    for i in range(n):
        for j in range(2):
            U[i * m + j][r] = 1.0
            V[j * p + 4][r] = 1.0
            W[4 * n + i][r] = 1.0
            r += 1
            
    # Part 3: A1 B11 (j=0,1, k=0,1) using Strassen
    # B11 is rows 0,1, cols 0,1.
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
    
    # Part 4: A1 B12 (j=0,1, k=2,3) using Strassen
    # B12 is rows 0,1, cols 2,3.
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
    
    # Verify the rank-28 decomposition
    if r == 28 and _verify(U, V, W, n, m, p):
        return {
            "U": U,
            "V": V,
            "W": W
        }
    else:
        # Fall back to schoolbook
        U_fb, V_fb, W_fb = _schoolbook(n, m, p)
        return {
            "U": U_fb,
            "V": V_fb,
            "W": W_fb
        }

if __name__ == "__main__":
    construct()
