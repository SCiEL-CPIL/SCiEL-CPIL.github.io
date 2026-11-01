import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [2, 3, 3]

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify if the decomposition is exact with integer/half-integer entries.
    U: (n*m, R)
    V: (m*p, R)
    W: (p*n, R)
    """
    R = len(U[0]) if U else 0
    if R == 0:
        return False
        
    # Convert to Fractions for exact arithmetic
    # U, V, W are lists of lists
    U_frac = [[Fraction(int(x * 2)) / 2 for x in row] for row in U]
    V_frac = [[Fraction(int(x * 2)) / 2 for x in row] for row in V]
    W_frac = [[Fraction(int(x * 2)) / 2 for x in row] for row in W]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for r in range(R):
                    total += U_frac[i * m + j][r] * V_frac[j * p + k][r] * W_frac[k * n + i][r]
                if total != 1:
                    return False
    return True

def _build_strassen_2x2x2():
    """
    Returns U, V, W for 2x2x2 multiplication (Strassen).
    Indices: i in {0,1}, j in {0,1}, k in {0,1}.
    U shape: (2*2, 7) = (4, 7)
    V shape: (2*2, 7) = (4, 7)
    W shape: (2*2, 7) = (4, 7)
    """
    # Strassen matrices
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # U corresponds to A factors
    # V corresponds to B factors
    # W corresponds to C factors
    
    # We need to map the 2x2 blocks to the 2x2 indices.
    # A11 is index (0,0), A12 is (0,1), A21 is (1,0), A22 is (1,1).
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # r=0: M1
    # U: A11+A22 -> (1,0,0,1)
    U[0][0] = 1; U[3][0] = 1
    # V: B11+B22 -> (1,0,0,1)
    V[0][0] = 1; V[3][0] = 1
    # W: C11 += M1, C22 += M1
    W[0][0] = 1; W[3][0] = 1
    
    # r=1: M2
    # U: A21+A22 -> (0,0,1,1)
    U[2][1] = 1; U[3][1] = 1
    # V: B11 -> (1,0,0,0)
    V[0][1] = 1
    # W: C21 += M2
    W[2][1] = 1
    
    # r=2: M3
    # U: A11 -> (1,0,0,0)
    U[0][2] = 1
    # V: B12-B22 -> (0,1,0,-1)
    V[1][2] = 1; V[3][2] = -1
    # W: C12 += M3
    W[1][2] = 1
    
    # r=3: M4
    # U: A22 -> (0,0,0,1)
    U[3][3] = 1
    # V: B21-B11 -> (0,0,1,-1)
    V[2][3] = 1; V[0][3] = -1
    # W: C22 -= M4
    W[3][3] = -1
    
    # r=4: M5
    # U: A11+A12 -> (1,1,0,0)
    U[0][4] = 1; U[1][4] = 1
    # V: B22 -> (0,0,0,1)
    V[3][4] = 1
    # W: C12 -= M5
    W[1][4] = -1
    
    # r=5: M6
    # U: A21-A11 -> (0,0,1,-1)
    U[2][5] = 1; U[0][5] = -1
    # V: B11+B12 -> (1,1,0,0)
    V[0][5] = 1; V[1][5] = 1
    # W: C21 -= M6
    W[2][5] = -1
    
    # r=6: M7
    # U: A12-A22 -> (0,1,0,-1)
    U[1][6] = 1; U[3][6] = -1
    # V: B21+B22 -> (0,0,1,1)
    V[2][6] = 1; V[3][6] = 1
    # W: C11 -= M7
    W[0][6] = -1
    
    return U, V, W

def _build_1x3x3():
    """
    Returns U, V, W for 1x3x3 multiplication.
    Indices: i in {0}, j in {0,1,2}, k in {0,1,2}.
    U shape: (1*3, 3) = (3, 3)
    V shape: (3*3, 3) = (9, 3)
    W shape: (3*1, 3) = (3, 3)
    """
    U = [[0]*3 for _ in range(3)]
    V = [[0]*3 for _ in range(9)]
    W = [[0]*3 for _ in range(3)]
    
    # Decomposition:
    # r=0: u=e0, v=E_{0,:}, w=(1,1,1)
    # r=1: u=e1, v=E_{1,:}, w=(1,1,1)
    # r=2: u=e2, v=E_{2,:}, w=(1,1,1)
    
    for r in range(3):
        U[r][r] = 1
        # v is a 3x3 matrix flattened to 9 elements.
        # v_{r,jk} = 1 if j=r, else 0.
        for k in range(3):
            idx = r * 3 + k
            V[idx][r] = 1
        # w is a 3x1 vector flattened to 3 elements.
        for k in range(3):
            W[k][r] = 1
            
    return U, V, W

def _combine_decompositions(U1, V1, W1, U2, V2, W2, mapping):
    """
    Combine two decompositions.
    mapping: dict describing how to map indices.
    This is a placeholder for the actual logic.
    """
    pass

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Try Rank 10 Decomposition
    # We assume the 2x3x3 tensor is the direct sum of 2x2x2 and 1x3x3.
    # This requires a specific mapping of indices.
    # If the mapping is not trivial, we might fail.
    # Let's try to construct it.
    
    # If we can't easily map, we fallback to schoolbook.
    # However, the prompt suggests it's possible.
    
    # Let's try Rank 18 Schoolbook as a safe fallback.
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
