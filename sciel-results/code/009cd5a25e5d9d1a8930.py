"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces the tensor using vectorized operations."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Calculate the full tensor T_{a,b,c} = sum_r U[a,r] V[b,r] W[c,r]
    # Shapes: U (nm, R), V (mp, R), W (pn, R)
    # Result shape: (nm, mp, pn)
    T = np.einsum('ar,br,cr->abc', U, V, W)
    
    # Check valid entries: T[a, b, c] should be 1.0
    # a = i*m + j, b = j*p + k, c = k*n + i
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                if abs(T[a, b, c] - 1.0) > 1e-9:
                    return False
                    
    # Check spurious entries: T[a, b, c] should be 0.0
    valid = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                valid.add((a, b, c))
                
    # Vectorized check for spurious entries
    T_masked = T.copy()
    for (a, b, c) in valid:
        T_masked[a, b, c] = 0.0
    if np.any(np.abs(T_masked) > 1e-9):
        return False
        
    return True

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
    return U, V, W

def _strassen_block(n, m, p):
    """
    For multiplying (n x m) by (m x p) where n=m=2, p=16.
    The 2x16 matrix can be seen as 8 independent 2x2 blocks.
    Each 2x2x2 multiplication has rank 7 (Strassen).
    Total rank = 8 * 7 = 56.
    """
    n, m, p = 2, 2, 16
    num_blocks = p // 2  # 8 blocks
    rank = num_blocks * 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for t in range(num_blocks):
        # V indices: j*p + k
        # j=0,k=2t -> 2t; j=0,k=2t+1 -> 2t+1
        # j=1,k=2t -> 16+2t; j=1,k=2t+1 -> 16+2t+1
        # W indices: k*n + i
        # k=2t,i=0 -> 4t; k=2t,i=1 -> 4t+1
        # k=2t+1,i=0 -> 4t+2; k=2t+1,i=1 -> 4t+3
        # U indices: A00=0, A01=1, A10=2, A11=3
        
        intermediates = [
            # M1 = (A00 + A11) * (B00 + B11)
            {'U': [(0, 1), (3, 1)], 'V': [(2*t, 1), (16+2*t+1, 1)], 'W': [(4*t, 1), (4*t+3, 1)]},
            # M2 = (A10 + A11) * B00
            {'U': [(2, 1), (3, 1)], 'V': [(2*t, 1)], 'W': [(4*t+1, 1), (4*t+3, -1)]},
            # M3 = A00 * (B01 - B11)
            {'U': [(0, 1)], 'V': [(2*t+1, 1), (16+2*t+1, -1)], 'W': [(4*t+2, 1), (4*t+3, 1)]},
            # M4 = A11 * (B10 - B00)
            {'U': [(3, 1)], 'V': [(16+2*t, 1), (2*t, -1)], 'W': [(4*t, 1), (4*t+1, 1)]},
            # M5 = (A00 + A01) * B11
            {'U': [(0, 1), (1, 1)], 'V': [(16+2*t+1, 1)], 'W': [(4*t, -1), (4*t+2, 1)]},
            # M6 = (A10 - A00) * (B00 + B01)
            {'U': [(2, 1), (0, -1)], 'V': [(2*t, 1), (2*t+1, 1)], 'W': [(4*t+3, 1)]},
            # M7 = (A01 - A11) * (B10 + B11)
            {'U': [(1, 1), (3, -1)], 'V': [(16+2*t, 1), (16+2*t+1, 1)], 'W': [(4*t, 1)]}
        ]
        
        for inter in intermediates:
            for idx, coeff in inter['U']:
                U[idx][col] += coeff
            for idx, coeff in inter['V']:
                V[idx][col] += coeff
            for idx, coeff in inter['W']:
                W[idx][col] += coeff
            col += 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Try Strassen-based construction (rank 56)
    U, V, W = _strassen_block(n, m, p)
    
    # Verify with a timeout guard
    if time.time() < deadline:
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    # If verification failed or timed out (unlikely), fall back to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
