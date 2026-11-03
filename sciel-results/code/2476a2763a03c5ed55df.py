"""Matrix multiplication 3,3,4 rank-minimal exact tensor decomposition."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 4]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix
    by a 3x4 matrix.
    
    Strategy: Use a structured decomposition based on Strassen's algorithm for
    the 2x2x2 subproblem, combined with schoolbook terms for the remaining
    entries. This should give rank 35 (saving 1 term over schoolbook 36).
    
    For 3x3 * 3x4:
    - C[0:2, 0:2] = A[0:2,0:2]*B[0:2,0:2] + A[0:2,2]*B[2,0:2]
      Use Strassen (7 terms) for the 2x2*2x2 part, plus 4 schoolbook terms
    - C[0:2, 2:4] = A[0:2,0:2]*B[0:2,2:4] + A[0:2,2]*B[2,2:4]
      8 + 4 = 12 schoolbook terms
    - C[2, 0:2] = A[2,0:2]*B[0:2,0:2] + A[2,2]*B[2,0:2]
      4 + 2 = 6 schoolbook terms
    - C[2, 2:4] = A[2,0:2]*B[0:2,2:4] + A[2,2]*B[2,2:4]
      4 + 2 = 6 schoolbook terms
    
    Total: 11 + 12 + 6 + 6 = 35 terms
    """
    deadline = time.time() + 140  # safety margin before 150s kill
    
    n, m, p = 3, 3, 4  # A is n x m, B is m x p, C is n x p
    
    # Build the Strassen-based rank-35 decomposition
    U = []
    V = []
    W = []
    
    def add_term(U_dict, V_dict, W_dict):
        """Add a rank-1 term to the decomposition."""
        # U_dict: {(i,j): val} for A indices
        # V_dict: {(j,k): val} for B indices  
        # W_dict: {(i,k): val} for C indices
        U_row = [0] * 33  # temporary
        V_row = [0] * 33
        W_row = [0] * 33
        
        for (i, j), val in U_dict.items():
            U_row[i * m + j] = val
        for (j, k), val in V_dict.items():
            V_row[j * p + k] = val
        for (i, k), val in W_dict.items():
            W_row[k * n + i] = val
        
        U.append(U_row)
        V.append(V_row)
        W.append(W_row)
    
    # Strassen terms for 2x2x2
    # M1 = (A00+A11)*(B00+B11)
    add_term(
        {(0,0):1, (1,1):1},
        {(0,0):1, (1,1):1},
        {(0,0):1, (1,1):1}
    )
    # M2 = (A10+A11)*B00
    add_term(
        {(1,0):1, (1,1):1},
        {(0,0):1},
        {(1,0):1, (1,1):-1}
    )
    # M3 = A00*(B01-B11)
    add_term(
        {(0,0):1},
        {(0,1):1, (1,1):-1},
        {(0,1):1, (1,1):1}
    )
    # M4 = A11*(B10-B00)
    add_term(
        {(1,1):1},
        {(1,0):1, (0,0):-1},
        {(0,0):1, (1,0):1}
    )
    # M5 = (A00+A01)*B11
    add_term(
        {(0,0):1, (0,1):1},
        {(1,1):1},
        {(0,0):-1, (0,1):1}
    )
    # M6 = (A10-A00)*(B00+B01)
    add_term(
        {(1,0):1, (0,0):-1},
        {(0,0):1, (0,1):1},
        {(1,1):1}
    )
    # M7 = (A01-A11)*(B10+B11)
    add_term(
        {(0,1):1, (1,1):-1},
        {(1,0):1, (1,1):1},
        {(0,0):1}
    )
    
    # Schoolbook terms for A[0:2,2]*B[2,0:2] -> C[0:2,0:2]
    for i in range(2):
        for k in range(2):
            add_term(
                {(i, 2): 1},
                {(2, k): 1},
                {(i, k): 1}
            )
    
    # Schoolbook terms for C[0:2, 2:4]
    # A[0:2,0:2]*B[0:2,2:4]
    for i in range(2):
        for j in range(2):
            for k in range(2, 4):
                add_term(
                    {(i, j): 1},
                    {(j, k): 1},
                    {(i, k): 1}
                )
    # A[0:2,2]*B[2,2:4]
    for i in range(2):
        for k in range(2, 4):
            add_term(
                {(i, 2): 1},
                {(2, k): 1},
                {(i, k): 1}
            )
    
    # Schoolbook terms for C[2, 0:2]
    # A[2,0:2]*B[0:2,0:2]
    for j in range(2):
        for k in range(2):
            add_term(
                {(2, j): 1},
                {(j, k): 1},
                {(2, k): 1}
            )
    # A[2,2]*B[2,0:2]
    for k in range(2):
        add_term(
            {(2, 2): 1},
            {(2, k): 1},
            {(2, k): 1}
        )
    
    # Schoolbook terms for C[2, 2:4]
    # A[2,0:2]*B[0:2,2:4]
    for j in range(2):
        for k in range(2, 4):
            add_term(
                {(2, j): 1},
                {(j, k): 1},
                {(2, k): 1}
            )
    # A[2,2]*B[2,2:4]
    for k in range(2, 4):
        add_term(
            {(2, 2): 1},
            {(2, k): 1},
            {(2, k): 1}
        )
    
    rank = len(U)
    
    # Verify the decomposition
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                computed = 0.0
                for c in range(rank):
                    computed += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                if abs(computed - 1) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if valid:
        return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook
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
