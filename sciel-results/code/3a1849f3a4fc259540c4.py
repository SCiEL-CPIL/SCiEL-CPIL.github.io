"""
Program to find rank-minimal exact tensor decomposition for 2x3 by 3x10 matrix multiplication.

Strategy:
  Use a block decomposition combined with Strassen's algorithm for the 2x2 blocks.
  Split A (2x3) into A_L (2x2) and A_R (2x1).
  Split B (3x10) into B_L (2x10) and B_R (1x10).
  C = A_L B_L + A_R B_R.
  
  The term A_R B_R involves 2x1 * 1x10, which has rank 20.
  The term A_L B_L involves 2x2 * 2x10. Split B_L into 5 blocks of 2x2.
  Each 2x2 * 2x2 multiplication can be done with rank 7 (Strassen).
  Total rank = 20 + 5 * 7 = 55.
"""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]
DEADLINE = 140.0

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS # n=2, m=3, p=10
    
    rank = 55
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Part 1: A_R B_R
    # A_R is columns 2 of A (index 2 in 0-based). Shape 2x1.
    # B_R is row 2 of B (index 2 in 0-based). Shape 1x10.
    # C_R[i, k] = A[i, 2] * B[2, k]
    # Rank 1 terms for each i, k
    r = 0
    for i in range(n):
        for k in range(p):
            U[i * m + 2, r] = 1.0
            V[2 * p + k, r] = 1.0
            W[k * n + i, r] = 1.0
            r += 1
            
    # Part 2: A_L B_L
    # A_L is columns 0, 1 of A. Shape 2x2.
    # B_L is rows 0, 1 of B. Shape 2x10.
    # Split B_L into 5 blocks of 2x2.
    # Block j corresponds to columns 2j, 2j+1 of B_L.
    
    # Strassen coefficients for U (A_L part)
    # X_00 = A[0,0] -> U[0]
    # X_01 = A[0,1] -> U[1]
    # X_10 = A[1,0] -> U[3]
    # X_11 = A[1,1] -> U[4]
    
    # Strassen coefficients for V (B_L,j part)
    # Y_00 = B[0, 2j] -> V[0*p + 2j]
    # Y_01 = B[0, 2j+1] -> V[0*p + 2j+1]
    # Y_10 = B[1, 2j] -> V[1*p + 2j]
    # Y_11 = B[1, 2j+1] -> V[1*p + 2j+1]
    
    # Strassen products M1..M7
    # M1: (X00+X11)(Y00+Y11)
    # M2: (X10+X11)Y00
    # M3: X00(Y10-Y11)
    # M4: X11(Y01-Y00)
    # M5: (X00+X10)Y11
    # M6: (X11-X00)(Y00+Y01)
    # M7: (X01-X11)(Y10+Y11)
    
    # W coefficients for Z (result 2x2)
    # Z00 = M1 + M4 - M5 - M7 -> C[0, 2j] -> W[2j*n + 0]
    # Z01 = M3 + M5           -> C[0, 2j+1] -> W[(2j+1)*n + 0]
    # Z10 = M2 + M4           -> C[1, 2j] -> W[2j*n + 1]
    # Z11 = M1 - M2 + M3 + M6 -> C[1, 2j+1] -> W[(2j+1)*n + 1]
    
    for j in range(5):
        col0 = 2 * j
        col1 = 2 * j + 1
        
        # M1
        U[0, r] = 1.0; U[4, r] = 1.0
        V[col0, r] = 1.0; V[p + col1, r] = 1.0
        W[col0 * n + 0, r] = 1.0
        W[col1 * n + 1, r] = 1.0
        r += 1
        
        # M2
        U[3, r] = 1.0; U[4, r] = 1.0
        V[col0, r] = 1.0
        W[col0 * n + 1, r] = 1.0
        r += 1
        
        # M3
        U[0, r] = 1.0
        V[p + col0, r] = 1.0; V[p + col1, r] = -1.0
        W[col1 * n + 0, r] = 1.0
        W[col1 * n + 1, r] = 1.0
        r += 1
        
        # M4
        U[4, r] = 1.0
        V[col1, r] = 1.0; V[col0, r] = -1.0
        W[col0 * n + 0, r] = 1.0
        W[col0 * n + 1, r] = 1.0
        r += 1
        
        # M5
        U[0, r] = 1.0; U[3, r] = 1.0
        V[p + col1, r] = 1.0
        W[col0 * n + 0, r] = -1.0
        W[col1 * n + 0, r] = 1.0
        r += 1
        
        # M6
        U[4, r] = 1.0; U[0, r] = -1.0
        V[col0, r] = 1.0; V[col1, r] = 1.0
        W[col1 * n + 1, r] = 1.0
        r += 1
        
        # M7
        U[1, r] = 1.0; U[4, r] = -1.0
        V[p + col0, r] = 1.0; V[p + col1, r] = 1.0
        W[col0 * n + 0, r] = -1.0
        r += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
    
    # Verify correctness
    n, m, p = DIMENSIONS
    U = np.array(result['U'])
    V = np.array(result['V'])
    W = np.array(result['W'])
    rank = U.shape[1]
    
    np.random.seed(42)
    A = np.random.randint(-5, 6, size=(n, m)).astype(float)
    B = np.random.randint(-5, 6, size=(m, p)).astype(float)
    
    C_decomp = np.zeros((n, p))
    for r in range(rank):
        u_r = U[:, r]
        v_r = V[:, r]
        w_r = W[:, r]
        
        A_weighted = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                A_weighted[i, j] = u_r[i * m + j]
        
        B_weighted = np.zeros((m, p))
        for j in range(m):
            for k in range(p):
                B_weighted[j, k] = v_r[j * p + k]
        
        prod_r = np.sum(A_weighted * A) * np.sum(B_weighted * B)
        
        for i in range(n):
            for k in range(p):
                C_decomp[i, k] += prod_r * w_r[k * n + i]
    
    C_direct = A @ B
    error = np.max(np.abs(C_decomp - C_direct))
    print(f"Max error: {error}")
    if error > 1e-10:
        print("WARNING: Decomposition is incorrect!")
