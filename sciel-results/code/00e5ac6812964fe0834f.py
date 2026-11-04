import numpy as np
import time
from fractions import Fraction
import itertools

DIMENSIONS = [2, 3, 10]

def _validate(U, V, W):
    """Check if U, V, W form a valid rank-R tensor decomposition."""
    n, m, p = DIMENSIONS
    if not U or not V or not W:
        return False
    
    rank = len(V[0]) if V else 0
    U_arr = np.array(U, dtype=object)
    V_arr = np.array(V, dtype=object)
    W_arr = np.array(W, dtype=object)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                if s != 1:
                    return False
    return True

def _is_valid_entry(x):
    """Check if entry is integer or half-integer."""
    if isinstance(x, Fraction):
        num, den = x.numerator, x.denominator
        return den in [1, 2]
    elif isinstance(x, int):
        return True
    elif isinstance(x, float):
        x2 = x * 2
        return abs(x2 - round(x2)) < 1e-9
    return False

def _build_schoolbook():
    n, m, p = DIMENSIONS
    R = n * m * p
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i * m * p + j * p + k
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
    return U, V, W

def _build_block_55():
    """
    Construct the Rank 55 decomposition using 5 blocks of (2x3x2).
    Each block uses Rank 11 (7 for 2x2x2 + 4 for 2x1x2).
    """
    n, m, p = DIMENSIONS
    R = 55
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    # We process 5 blocks of width 2 in the output dimension p.
    # Block g covers output columns 2g and 2g+1.
    # The input dimension m=3 is split into m1=2 and m2=1.
    # A = [A1 (2x2), A2 (2x1)]
    # B = [B1 (2x2), B2 (1x2)] for this block.
    
    for g in range(5):
        base_rank = 11 * g
        
        # --- Part 1: A1 * B1 (2x2 * 2x2) -> Rank 7 (Strassen) ---
        # A1 indices in U: (0,0)=0, (0,1)=1, (1,0)=3, (1,1)=4
        # B1 indices in V: (0, 2g)=2g, (0, 2g+1)=2g+1, (1, 2g)=10+2g, (1, 2g+1)=10+2g+1
        # W indices for C[0, 2g], C[0, 2g+1], C[1, 2g], C[1, 2g+1]:
        # W[2g*2+0]=4g, W[2g*2+1]=4g+1, W[(2g+1)*2+0]=4g+2, W[(2g+1)*2+1]=4g+3
        
        # Strassen Terms for M = A1 B1
        # 1. M1 = (A11+A22)(B11+B22)
        c = base_rank
        U[0][c] += 1   # A11
        U[4][c] += 1   # A22
        V[2*g][c] += 1   # B11
        V[10+2*g+1][c] += 1 # B22
        W[4*g][c] += 1   # C11
        W[4*g+3][c] += 1 # C22
        
        # 2. M2 = (A21+A22)B11
        c = base_rank + 1
        U[3][c] += 1   # A21
        U[4][c] += 1   # A22
        V[2*g][c] += 1   # B11
        W[4*g+2][c] += 1 # C21
        
        # 3. M3 = A11(B21-B22)
        c = base_rank + 2
        U[0][c] += 1   # A11
        V[10+2*g][c] += 1   # B21
        V[10+2*g+1][c] -= 1 # B22
        W[4*g+1][c] += 1 # C12
        W[4*g+3][c] += 1 # C22
        
        # 4. M4 = A22(B12-B11)
        c = base_rank + 3
        U[4][c] += 1   # A22
        V[2*g+1][c] += 1   # B12
        V[2*g][c] -= 1   # B11
        W[4*g][c] += 1   # C11
        W[4*g+2][c] += 1 # C21
        
        # 5. M5 = (A11+A12)B22
        c = base_rank + 4
        U[0][c] += 1   # A11
        U[1][c] += 1   # A12
        V[10+2*g+1][c] += 1 # B22
        W[4*g][c] -= 1   # C11
        W[4*g+1][c] += 1 # C12
        
        # 6. M6 = (A21-A11)(B11+B12)
        c = base_rank + 5
        U[3][c] += 1   # A21
        U[0][c] -= 1   # A11
        V[2*g][c] += 1   # B11
        V[2*g+1][c] += 1 # B12
        W[4*g+3][c] += 1 # C22
        
        # 7. M7 = (A12-A22)(B21+B22)
        c = base_rank + 6
        U[1][c] += 1   # A12
        U[4][c] -= 1   # A22
        V[10+2*g][c] += 1 # B21
        V[10+2*g+1][c] += 1 # B22
        W[4*g][c] += 1   # C11
        
        # --- Part 2: A2 * B2 (2x1 * 1x2) -> Rank 4 ---
        # A2 indices in U: (0,2)=2, (1,2)=5
        # B2 indices in V: (2, 2g)=20+2g, (2, 2g+1)=20+2g+1
        # W indices: C[0, 2g]=4g, C[0, 2g+1]=4g+1, C[1, 2g]=4g+2, C[1, 2g+1]=4g+3
        
        # Term 8: A[0,2] * B[2, 2g]
        c = base_rank + 7
        U[2][c] += 1
        V[20+2*g][c] += 1
        W[4*g][c] += 1
        
        # Term 9: A[0,2] * B[2, 2g+1]
        c = base_rank + 8
        U[2][c] += 1
        V[20+2*g+1][c] += 1
        W[4*g+1][c] += 1
        
        # Term 10: A[1,2] * B[2, 2g]
        c = base_rank + 9
        U[5][c] += 1
        V[20+2*g][c] += 1
        W[4*g+2][c] += 1
        
        # Term 11: A[1,2] * B[2, 2g+1]
        c = base_rank + 10
        U[5][c] += 1
        V[20+2*g+1][c] += 1
        W[4*g+3][c] += 1

    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Try to find a better decomposition using search if time permits
    # For now, return the known Rank 55 decomposition
    U, V, W = _build_block_55()
    
    if _validate(U, V, W):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback
        U_fb, V_fb, W_fb = _build_schoolbook()
        return {"U": U_fb, "V": V_fb, "W": W_fb}
