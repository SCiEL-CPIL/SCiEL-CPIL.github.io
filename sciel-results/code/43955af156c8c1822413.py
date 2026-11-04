"""
Matrix multiplication 5x5x5 tensor decomposition.
Uses block decomposition with Strassen on the 2x2 corner block.
"""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """
    Return the classic Strassen 2x2x2 decomposition with 7 terms.
    U, V, W are 4x7 matrices (row-major indexing).
    U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # Term 1: M1 = (a+d)(e+h)
    # a=A[0,0], d=A[1,1], e=B[0,0], h=B[1,1]
    # C[0,0] += M1, C[1,1] += M1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # Term 2: M2 = (c+d)e
    # c=A[1,0], d=A[1,1], e=B[0,0]
    # C[1,0] += M2
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    # Term 3: M3 = a(f-h)
    # a=A[0,0], f=B[0,1], h=B[1,1]
    # C[0,1] += M3
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1
    
    # Term 4: M4 = d(g-e)
    # d=A[1,1], g=B[1,0], e=B[0,0]
    # C[1,0] += M4
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[2][3] = 1
    
    # Term 5: M5 = (a+b)h
    # a=A[0,0], b=A[0,1], h=B[1,1]
    # C[0,0] -= M5, C[0,1] += M5
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # Term 6: M6 = (c-d)(f+g)
    # c=A[1,0], d=A[1,1], f=B[0,1], g=B[1,0]
    # C[1,1] += M6
    U[2][5] = 1; U[3][5] = -1
    V[1][5] = 1; V[2][5] = 1
    W[3][5] = 1
    
    # Term 7: M7 = (b-d)(e+f)
    # b=A[0,1], d=A[1,1], e=B[0,0], f=B[0,1]
    # C[0,0] += M7
    U[1][6] = 1; U[3][6] = -1
    V[0][6] = 1; V[1][6] = 1
    W[0][6] = 1
    
    return U, V, W


def _verify_exact(n, m, p, U, V, W):
    """
    Verify exact decomposition using Fraction arithmetic.
    T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal 1.
    """
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                u_row = U[i * m + j]
                v_row = V[j * p + k]
                w_row = W[k * n + i]
                for c in range(rank):
                    uc = u_row[c] if isinstance(u_row[c], Fraction) else Fraction(u_row[c])
                    vc = v_row[c] if isinstance(v_row[c], Fraction) else Fraction(v_row[c])
                    wc = w_row[c] if isinstance(w_row[c], Fraction) else Fraction(w_row[c])
                    s += uc * vc * wc
                if s != 1:
                    return False
    return True


def _construct_5x5x5():
    """
    Construct a 5x5x5 matrix multiplication decomposition.
    
    Strategy: Partition into a 2x2 block (top-left) and the rest.
    - Use Strassen (rank 7) for the 2x2x2 sub-tensor (i,j,k in {0,1})
    - Use schoolbook (rank 1) for all other 125-8 = 117 triples
    
    Total rank = 7 + 117 = 124
    """
    n, m, p = 5, 5, 5
    nm, mp, pn = n * m, m * p, p * n
    
    # Get Strassen decomposition for 2x2x2
    U_s, V_s, W_s = _strassen_2x2()
    
    # Initialize full matrices with rank 124
    rank = 124
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    
    # Place Strassen terms in first 7 columns
    for c in range(7):
        for r in range(4):
            U[r][c] = U_s[r][c]
            V[r][c] = V_s[r][c]
            W[r][c] = W_s[r][c]
    
    # Add schoolbook terms for all other triples
    # Skip i,j,k all in {0,1} (those are handled by Strassen)
    col = 7
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue  # Handled by Strassen
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}, rank


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Construct the 124-rank decomposition
    result, rank = _construct_5x5x5()
    
    # Verify it's correct
    if _verify_exact(n, m, p, result["U"], result["V"], result["W"]):
        return result
    
    # Fallback: schoolbook (rank 125) if verification fails
    U = [[0] * 125 for _ in range(25)]
    V = [[0] * 125 for _ in range(25)]
    W = [[0] * 125 for _ in range(25)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
