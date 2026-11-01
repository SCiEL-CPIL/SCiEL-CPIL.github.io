import numpy as np
import random

DIMENSIONS = [2, 2, 15]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    # Try to find lower rank over F2
    # Lower bound is 30
    for r in range(30, 61):
        # Try a few random initializations
        for attempt in range(100):
            # Random U, V
            U = np.zeros((n * m, r), dtype=int)
            V = np.zeros((m * p, r), dtype=int)
            # Fill with random 0/1
            U[:] = np.random.randint(0, 2, size=(n * m, r))
            V[:] = np.random.randint(0, 2, size=(m * p, r))
            
            # Check if W exists
            W = np.zeros((p * n, r), dtype=int)
            valid = True
            for k in range(p):
                for i in range(n):
                    w_idx = k * n + i
                    # Equations:
                    # sum_c W[w_idx, c] * U[i*m+0, c] * V[0*p+k, c] = 1
                    # sum_c W[w_idx, c] * U[i*m+1, c] * V[1*p+k, c] = 1
                    
                    # Coefficients for W[w_idx, :]
                    # Row 1: U[i*m, c] * V[k, c]
                    # Row 2: U[i*m+1, c] * V[15+k, c]
                    row1 = (U[i * m] * V[k]) % 2
                    row2 = (U[i * m + 1] * V[15 + k]) % 2
                    
                    # We need W[w_idx] such that:
                    # W[w_idx] . row1 = 1
                    # W[w_idx] . row2 = 1
                    
                    # This is a linear system A x = b
                    # A is 2 x r, x is r x 1, b is 2 x 1
                    A = np.vstack([row1, row2])
                    b = np.array([1, 1])
                    
                    # Solve over F2
                    # Use Gaussian elimination
                    # Since r is small (30-60), this is fast
                    sol = solve_linear_f2(A, b)
                    if sol is None:
                        valid = False
                        break
                    W[w_idx] = sol
                if not valid:
                    break
            
            if valid:
                # Convert to lists of lists
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                
    return {
        "U": U_school,
        "V": V_school,
        "W": W_school
    }

def solve_linear_f2(A, b):
    # A is 2 x r, b is 2 x 1
    # Find x such that A x = b over F2
    # Augmented matrix
    M = np.hstack((A, b.reshape(-1, 1)))
    rows, cols = A.shape
    # Gaussian elimination
    for i in range(rows):
        # Find pivot
        pivot = -1
        for j in range(i, cols):
            if M[i, j] == 1:
                pivot = j
                break
        if pivot == -1:
            # Check if b[i] is 0
            if M[i, cols] == 1:
                return None
            else:
                # Free variable, set to 0
                continue
        # Swap
        if pivot != i:
            M[[i, pivot]] = M[[pivot, i]]
        # Eliminate
        for k in range(rows):
            if k != i and M[k, i] == 1:
                M[k] = (M[k] + M[i]) % 2
                
    # Extract solution
    x = np.zeros(cols, dtype=int)
    for i in range(rows):
        # Find pivot in row i
        pivot = -1
        for j in range(cols):
            if M[i, j] == 1:
                pivot = j
                break
        if pivot != -1:
            x[pivot] = M[i, cols]
            
    # Verify
    if (A @ x) % 2 != b % 2:
        return None
    return x
