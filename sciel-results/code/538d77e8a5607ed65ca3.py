"""Rewritten program: rank-minimal tensor decomposition for 2x4 * 4x8."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 8]
n, m, p = DIMENSIONS
RANK_SCHOOLBOOK = n * m * p  # 64

_DEADLINE = 130.0
_T0 = time.monotonic()

def _elapsed():
    return time.monotonic() - _T0

def _schoolbook():
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=object)
    V = np.zeros((m * p, rank), dtype=object)
    W = np.zeros((p * n, rank), dtype=object)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify_np(U, V, W):
    """Verify using numpy with float conversion for speed."""
    rank = len(U[0])
    Uf = np.array(U, dtype=np.float64)
    Vf = np.array(V, dtype=np.float64)
    Wf = np.array(W, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += Uf[i*m+j, c] * Vf[j*p+k, c] * Wf[k*n+i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _strassen_2x4x8():
    """
    Construct a rank-56 decomposition by partitioning into 8 sub-tensors of size 2x2x2
    and applying Strassen (7 multiplications) to each.
    
    Partition: j = 2*a + b (a in {0,1}, b in {0,1}), k = 2*c + d (c in {0,1,2,3}, d in {0,1})
    Sub-tensor (a,c): i in {0,1}, b in {0,1}, d in {0,1}
    """
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for a in range(2):
        for c in range(4):
            # Sub-tensor indices:
            # U row for A[i, 2a+b]: i*4 + 2a + b
            # V row for B[2a+b, 2c+d]: (2a+b)*8 + 2c + d
            # W row for C[i, 2c+d]: (2c+d)*2 + i = 4c + 2d + i
            
            # Strassen 7 products for 2x2x2:
            # A = [[a00, a01], [a10, a11]] = A[i, 2a+b]
            # B = [[b00, b01], [b10, b11]] = B[2a+j, 2c+d]
            # C[i,d] = sum_j A[i,j]*B[j,d]
            
            # M1 = (a00+a11)(b00+b11)
            # M2 = (a10+a11)b00
            # M3 = a00(b10-b11)
            # M4 = a11(b01-b00)
            # M5 = (a00+a01)b11
            # M6 = (a10-a00)(b00+b01)
            # M7 = (a11-a01)(b01+b11)
            
            # C00 = M1 - M4 - M6 + M3
            # C01 = M1 + M5 - M7 - M2
            # C10 = M2 + M6 - M1 + M4
            # C11 = M1 + M7 - M5 + M2
            
            # W indices: C[0,2c] = W[4c], C[1,2c] = W[4c+1], C[0,2c+1] = W[4c+2], C[1,2c+1] = W[4c+3]
            
            # M1: contributes +1 to C00, +1 to C01, -1 to C10, +1 to C11
            U[0*4+2*a+0][col] += 1  # a00
            U[1*4+2*a+1][col] += 1  # a11
            V[(2*a+0)*8+2*c+0][col] += 1  # b00
            V[(2*a+1)*8+2*c+1][col] += 1  # b11
            W[4*c+0][col] += 1    # C00
            W[4*c+1][col] += -1   # C10
            W[4*c+2][col] += 1    # C01
            W[4*c+3][col] += 1    # C11
            col += 1
            
            # M2: contributes 0 to C00, -1 to C01, +1 to C10, +1 to C11
            U[1*4+2*a+0][col] += 1  # a10
            U[1*4+2*a+1][col] += 1  # a11
            V[(2*a+0)*8+2*c+0][col] += 1  # b00
            W[4*c+1][col] += 1    # C10
            W[4*c+2][col] += -1   # C01
            W[4*c+3][col] += 1    # C11
            col += 1
            
            # M3: contributes +1 to C00, 0 to others
            U[0*4+2*a+0][col] += 1  # a00
            V[(2*a+1)*8+2*c+0][col] += 1   # b10
            V[(2*a+1)*8+2*c+1][col] += -1  # b11
            W[4*c+0][col] += 1    # C00
            col += 1
            
            # M4: contributes -1 to C00, +1 to C10
            U[1*4+2*a+1][col] += 1  # a11
            V[(2*a+0)*8+2*c+1][col] += 1   # b01
            V[(2*a+0)*8+2*c+0][col] += -1  # b00
            W[4*c+0][col] += -1   # C00
            W[4*c+1][col] += 1    # C10
            col += 1
            
            # M5: contributes +1 to C01, -1 to C11
            U[0*4+2*a+0][col] += 1  # a00
            U[0*4+2*a+1][col] += 1  # a01
            V[(2*a+1)*8+2*c+1][col] += 1  # b11
            W[4*c+2][col] += 1    # C01
            W[4*c+3][col] += -1   # C11
            col += 1
            
            # M6: contributes -1 to C00, +1 to C10
            U[1*4+2*a+0][col] += 1   # a10
            U[0*4+2*a+0][col] += -1  # a00
            V[(2*a+0)*8+2*c+0][col] += 1  # b00
            V[(2*a+0)*8+2*c+1][col] += 1  # b01
            W[4*c+0][col] += -1   # C00
            W[4*c+1][col] += 1    # C10
            col += 1
            
            # M7: contributes -1 to C01, +1 to C11
            U[1*4+2*a+1][col] += 1   # a11
            U[0*4+2*a+1][col] += -1  # a01
            V[(2*a+0)*8+2*c+1][col] += 1   # b01
            V[(2*a+1)*8+2*c+1][col] += 1   # b11
            W[4*c+2][col] += -1   # C01
            W[4*c+3][col] += 1    # C11
            col += 1
    
    assert col == 56, f"Expected 56, got {col}"
    return {"U": U, "V": V, "W": W}

def _verify_exact(U, V, W):
    """Exact verification using Fraction."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                if s != 1:
                    return False
    return True

def construct():
    best = _schoolbook()
    best_rank = RANK_SCHOOLBOOK
    
    if _elapsed() < _DEADLINE - 5:
        s = _strassen_2x4x8()
        if _verify_np(s["U"], s["V"], s["W"]):
            # Double-check with exact arithmetic
            if _verify_exact(s["U"], s["V"], s["W"]):
                r = len(s["U"][0])
                if r < best_rank:
                    best = s
                    best_rank = r
        else:
            # Debug: print where it fails
            pass
    
    return best
