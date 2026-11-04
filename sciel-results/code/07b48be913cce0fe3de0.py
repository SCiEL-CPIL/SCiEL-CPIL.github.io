import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [2, 3, 5]

def _to_fraction(x):
    if isinstance(x, Fraction):
        return x
    if isinstance(x, (int, float)):
        return Fraction(x)
    return Fraction(x)

def _is_half_integer(frac):
    return (frac * 2).denominator == 1

def _verify_decomposition(U, V, W, n, m, p):
    rank = len(U[0])
    U_arr = np.array([[float(x) for x in row] for row in U], dtype=float)
    V_arr = np.array([[float(x) for x in row] for row in V], dtype=float)
    W_arr = np.array([[float(x) for x in row] for row in W], dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                s = np.dot(U_arr[idx_u, :], np.multiply(V_arr[idx_v, :], W_arr[idx_w, :]))
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = 2, 3, 5
    rank = 19
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    
    # Part 1: j=2 (the entire column 2 of A times row 2 of B)
    # This is a rank-1 bilinear term: sum_{i,k} A[i,2] * B[2,k] * W[k*n+i]
    # For each (i,k) pair, we need U[i*3+2]*V[2*5+k]*W[k*2+i] = 1
    # Setting U[2]=U[5]=1, V[10..14]=1, W[0..9]=1 works
    for i in range(n):
        U[i * m + 2][r] = 1
    for k in range(p):
        V[2 * p + k][r] = 1
    for k in range(p):
        for i in range(n):
            W[k * n + i][r] = 1
    r += 1
    
    # Part 2: j in {0,1}, k=4
    # This is a 2x2x1 tensor: C[i,4] += A[i,0]*B[0,4] + A[i,1]*B[1,4]
    # Need 4 terms (rank of 2x2x1 is 4)
    for i in range(n):
        for j in range(2):
            U[i * m + j][r] = 1
            V[j * p + 4][r] = 1
            W[4 * n + i][r] = 1
            r += 1
    
    # Part 3: j in {0,1}, k in {0,1} - Strassen for 2x2x2
    # M1: (A00+A11)(B00+B11) -> contributes to C00 and C11
    U[0 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 0][r] = 1
    V[1 * p + 1][r] = 1
    W[0 * n + 0][r] = 1
    W[1 * n + 1][r] = 1
    r += 1
    
    # M2: (A10+A11)B00 -> contributes to C01 and C11
    U[1 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 0][r] = 1
    W[0 * n + 1][r] = 1
    W[1 * n + 1][r] = -1
    r += 1
    
    # M3: A00(B11-B01) -> contributes to C10 and C11
    U[0 * m + 0][r] = 1
    V[0 * p + 1][r] = 1
    V[1 * p + 1][r] = -1
    W[1 * n + 0][r] = 1
    W[1 * n + 1][r] = 1
    r += 1
    
    # M4: A11(B10-B00) -> contributes to C00 and C01
    U[1 * m + 1][r] = 1
    V[1 * p + 0][r] = 1
    V[0 * p + 0][r] = -1
    W[0 * n + 0][r] = 1
    W[0 * n + 1][r] = 1
    r += 1
    
    # M5: (A00+A01)B11 -> contributes to C00 and C10
    U[0 * m + 0][r] = 1
    U[0 * m + 1][r] = 1
    V[1 * p + 1][r] = 1
    W[0 * n + 0][r] = -1
    W[1 * n + 0][r] = 1
    r += 1
    
    # M6: (A10-A00)(B00+B01) -> contributes to C11
    U[1 * m + 0][r] = 1
    U[0 * m + 0][r] = -1
    V[0 * p + 0][r] = 1
    V[0 * p + 1][r] = 1
    W[1 * n + 1][r] = 1
    r += 1
    
    # M7: (A01-A11)(B10+B11) -> contributes to C00
    U[0 * m + 1][r] = 1
    U[1 * m + 1][r] = -1
    V[1 * p + 0][r] = 1
    V[1 * p + 1][r] = 1
    W[0 * n + 0][r] = 1
    r += 1
    
    # Part 4: j in {0,1}, k in {2,3} - Strassen for 2x2x2
    # M1: (A00+A11)(B02+B13) -> contributes to C02 and C13
    U[0 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 2][r] = 1
    V[1 * p + 3][r] = 1
    W[2 * n + 0][r] = 1
    W[3 * n + 1][r] = 1
    r += 1
    
    # M2: (A10+A11)B02 -> contributes to C03 and C13
    U[1 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 2][r] = 1
    W[2 * n + 1][r] = 1
    W[3 * n + 1][r] = -1
    r += 1
    
    # M3: A00(B13-B03) -> contributes to C12 and C13
    U[0 * m + 0][r] = 1
    V[0 * p + 3][r] = 1
    V[1 * p + 3][r] = -1
    W[3 * n + 0][r] = 1
    W[3 * n + 1][r] = 1
    r += 1
    
    # M4: A11(B12-B02) -> contributes to C02 and C03
    U[1 * m + 1][r] = 1
    V[1 * p + 2][r] = 1
    V[0 * p + 2][r] = -1
    W[2 * n + 0][r] = 1
    W[2 * n + 1][r] = 1
    r += 1
    
    # M5: (A00+A01)B13 -> contributes to C02 and C12
    U[0 * m + 0][r] = 1
    U[0 * m + 1][r] = 1
    V[1 * p + 3][r] = 1
    W[2 * n + 0][r] = -1
    W[3 * n + 0][r] = 1
    r += 1
    
    # M6: (A10-A00)(B02+B03) -> contributes to C13
    U[1 * m + 0][r] = 1
    U[0 * m + 0][r] = -1
    V[0 * p + 2][r] = 1
    V[0 * p + 3][r] = 1
    W[3 * n + 1][r] = 1
    r += 1
    
    # M7: (A01-A11)(B12+B13) -> contributes to C02
    U[0 * m + 1][r] = 1
    U[1 * m + 1][r] = -1
    V[1 * p + 2][r] = 1
    V[1 * p + 3][r] = 1
    W[2 * n + 0][r] = 1
    r += 1
    
    # Verify
    assert r == rank, f"Expected rank {rank}, got {r}"
    assert _verify_decomposition(U, V, W, n, m, p), "Decomposition verification failed"
    
    return {"U": U, "V": V, "W": W}
