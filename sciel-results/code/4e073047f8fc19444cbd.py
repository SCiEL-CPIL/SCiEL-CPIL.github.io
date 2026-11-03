import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    if U.shape != (n * m, rank):
        return False
    if V.shape != (m * p, rank):
        return False
    if W.shape != (p * n, rank):
        return False
    
    # Check all valid (i, j, k) combinations
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = np.sum(U[i * m + j] * V[j * p + k] * W[k * n + i])
                if abs(total - 1.0) > 1e-10:
                    return False
    
    # Check a sample of invalid combinations to ensure orthogonality
    # The tensor is sparse, so most combinations should sum to 0
    # We check all combinations to be safe, as n,m,p are small
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for i2 in range(n):
                    for j2 in range(m):
                        for k2 in range(p):
                            if i2 == i and j2 == j and k2 == k:
                                continue
                            total = np.sum(U[i * m + j] * V[j * p + k] * W[k2 * n + i2])
                            if abs(total) > 1e-10:
                                return False
    return True

def _laderman_3x3x3():
    """
    Returns U, V, W for 3x3x3 matrix multiplication using Laderman's algorithm (rank 23).
    Indices: U[i*3+j], V[j*3+k], W[k*3+i]
    """
    rank = 23
    U = [[0] * rank for _ in range(9)]
    V = [[0] * rank for _ in range(9)]
    W = [[0] * rank for _ in range(9)]
    
    # Laderman's 23 multiplications coefficients
    # Source: Standard Laderman algorithm implementation
    # M1: A00 B00
    U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
    # M2: A00 B10
    U[0][1] = 1; V[3][1] = 1; W[3][1] = 1
    # M3: A00 B20
    U[0][2] = 1; V[6][2] = 1; W[6][2] = 1
    # M4: A10 B00
    U[3][3] = 1; V[0][3] = 1; W[1][3] = 1
    # M5: A20 B00
    U[6][4] = 1; V[0][4] = 1; W[2][4] = 1
    # M6: A01 B11
    U[1][5] = 1; V[3][5] = 1; W[3][5] = 1
    # M7: A02 B22
    U[2][6] = 1; V[6][6] = 1; W[6][6] = 1
    
    # M8: (A00+A10)(B01+B11)
    U[0][7] = 1; U[3][7] = 1
    V[1][7] = 1; V[4][7] = 1
    W[0][7] = 1; W[1][7] = 1 # C00 and C11? No, C01 and C11?
    # Let's check C01 = A00B01 + A01B11 + A02B21
    # M8 contributes A00B01 + A00B11 + A10B01 + A10B11
    # This is not directly C01.
    # The W coefficients must be determined by solving the linear system.
    # However, for Laderman, the W coefficients are fixed.
    # I will use the known W coefficients for Laderman.
    
    # Since deriving W is complex, I will rely on the fact that if U and V are correct, 
    # and the algorithm is known to work, the W coefficients can be found.
    # But I don't have them memorized.
    
    # Fallback: Return schoolbook for 3x3x3 if Laderman is too risky.
    # But I want to try rank 41.
    
    # I will use a verified Laderman implementation from a known source.
    # If I can't, I will return None to signal failure.
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120
    
    # Start with schoolbook
    result = _schoolbook(n, m, p)
    best_rank = len(result['U'][0])
    
    # Try to construct rank 41 solution
    # Laderman for first 3 cols, Schoolbook for last 2 cols
    
    laderman = _laderman_3x3x3()
    if laderman is not None and _verify(laderman['U'], laderman['V'], laderman['W'], 3, 3, 3):
        # Combine with schoolbook for remaining 2 columns
        U = [[0] * 41 for _ in range(9)]
        V = [[0] * 41 for _ in range(15)]
        W = [[0] * 41 for _ in range(15)]
        
        # Copy Laderman
        for i in range(9):
            for c in range(23):
                U[i][c] = laderman['U'][i][c]
                if i < 9: # V is 9x23 for 3x3x3
                    V[i][c] = laderman['V'][i][c]
                if i < 9: # W is 9x23 for 3x3x3
                    W[i][c] = laderman['W'][i][c]
        
        # Schoolbook for k=3,4
        col = 23
        for k in [3, 4]:
            for i in range(3):
                for j in range(3):
                    U[i * 3 + j][col] = 1
                    V[j * 5 + k][col] = 1
                    W[k * 3 + i][col] = 1
                    col += 1
        
        if _verify(U, V, W, 3, 3, 5):
            return {"U": U, "V": V, "W": W}
    
    return result
# EVOLVE-BLOCK-END
