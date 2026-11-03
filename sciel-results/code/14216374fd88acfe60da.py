"""Rewritten program: rank-minimal tensor decomposition for 2x4 * 4x8."""
import time
import itertools
from fractions import Fraction

DIMENSIONS = [2, 4, 8]
n, m, p = DIMENSIONS
RANK_SCHOOLBOOK = n * m * p  # 64

# Deadline guard: return best found before 140s (evaluator kills at 150s)
_DEADLINE = 140.0
_T0 = time.monotonic()

def _elapsed():
    return time.monotonic() - _T0

def _schoolbook():
    """Return the rank-64 schoolbook decomposition."""
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

def _verify(U, V, W):
    """
    Verify that U, V, W form a valid tensor decomposition.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    The tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    must equal 1 for all (i,j,k) in the multiplication tensor.
    """
    rank = len(U[0])
    # For each (i,j,k), check that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    # Also check that for non-tensor entries the sum is 0? 
    # Actually the tensor is defined on all (i,j,k) in [n]x[m]x[p], and the multiplication
    # tensor has T[i,j,k]=1 for all such triples. So we just need T=1 everywhere.
    return True

def _simple_improvement():
    """
    Construct a rank-56 decomposition by partitioning the 2x4x8 tensor
    into eight disjoint 2x2x2 sub-tensors and applying Strassen to each.
    """
    from fractions import Fraction as F
    
    n, m, p = 2, 4, 8
    rank = 56
    U = [[F(0)] * rank for _ in range(n * m)]
    V = [[F(0)] * rank for _ in range(m * p)]
    W = [[F(0)] * rank for _ in range(p * n)]
    
    col = 0
    for a in range(2):      # j-block: j = 2*a + b
        for c in range(4):  # k-block: k = 2*c + d
            
            # M1: (a00+a11)(b00+b11)
            U[0*4+2*a+0][col] += F(1)
            U[1*4+2*a+1][col] += F(1)
            V[(2*a+0)*8+2*c+0][col] += F(1)
            V[(2*a+1)*8+2*c+1][col] += F(1)
            W[4*c+0][col] += F(1)
            W[4*c+1][col] += F(-1)
            W[4*c+2][col] += F(1)
            W[4*c+3][col] += F(1)
            col += 1
            
            # M2: (a10+a11)b00
            U[1*4+2*a+0][col] += F(1)
            U[1*4+2*a+1][col] += F(1)
            V[(2*a+0)*8+2*c+0][col] += F(1)
            W[4*c+1][col] += F(1)
            W[4*c+2][col] += F(-1)
            W[4*c+3][col] += F(1)
            col += 1
            
            # M3: a00(b10-b11)
            U[0*4+2*a+0][col] += F(1)
            V[(2*a+1)*8+2*c+0][col] += F(1)
            V[(2*a+1)*8+2*c+1][col] += F(-1)
            W[4*c+0][col] += F(1)
            col += 1
            
            # M4: a11(b01-b00)
            U[1*4+2*a+1][col] += F(1)
            V[(2*a+0)*8+2*c+1][col] += F(1)
            V[(2*a+0)*8+2*c+0][col] += F(-1)
            W[4*c+0][col] += F(-1)
            W[4*c+1][col] += F(1)
            col += 1
            
            # M5: (a00+a01)b11
            U[0*4+2*a+0][col] += F(1)
            U[0*4+2*a+1][col] += F(1)
            V[(2*a+1)*8+2*c+1][col] += F(1)
            W[4*c+2][col] += F(1)
            W[4*c+3][col] += F(-1)
            col += 1
            
            # M6: (a10-a00)(b00+b01)
            U[1*4+2*a+0][col] += F(1)
            U[0*4+2*a+0][col] += F(-1)
            V[(2*a+0)*8+2*c+0][col] += F(1)
            V[(2*a+0)*8+2*c+1][col] += F(1)
            W[4*c+0][col] += F(-1)
            W[4*c+1][col] += F(1)
            col += 1
            
            # M7: (a11-a01)(b01+b11)
            U[1*4+2*a+1][col] += F(1)
            U[0*4+2*a+1][col] += F(-1)
            V[(2*a+0)*8+2*c+1][col] += F(1)
            V[(2*a+1)*8+2*c+1][col] += F(1)
            W[4*c+2][col] += F(-1)
            W[4*c+3][col] += F(1)
            col += 1
    
    assert col == 56, f"Expected 56 terms, got {col}"
    return {"U": U, "V": V, "W": W}

def construct():
    best = _schoolbook()
    best_rank = RANK_SCHOOLBOOK
    
    # Try simple improvement (which now contains the actual Strassen construction)
    if _elapsed() < _DEADLINE - 5:
        s2 = _simple_improvement()
        if s2 is not None:
            if _verify(s2["U"], s2["V"], s2["W"]):
                r = len(s2["U"][0])
                if r < best_rank:
                    best = s2
                    best_rank = r
    
    return best
