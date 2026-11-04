"""Rewritten program for 2x5x6 matrix multiplication rank minimization."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=5, p=6
    # A is n x m (2x5), B is m x p (5x6), C is n x p (2x6)
    # W index: k*n + i
    # U index: i*m + j
    # V index: j*p + k
    
    # We decompose B into B_left (5x3) and B_right (5x3)
    # C = A * B = [A * B_left | A * B_right]
    # Each sub-problem is 2x5x3
    
    # For a 2x5x3 problem:
    # A is 2x5, B is 5x3
    # Split A into A1 (2x3) and A2 (2x2)
    # C = A1 * B1 + A2 * B2 where B1 = B[0:3,:], B2 = B[3:5,:]
    # A1 * B1 is 2x3 * 3x3 = 2x3, rank 9
    # A2 * B2 is 2x2 * 2x3 = 2x3, rank 6
    # Total for one 2x5x3: rank 15
    
    # For 2x5x6: split into two 2x5x3 problems
    # Total rank: 2 * 15 = 30
    
    # Let's verify this decomposition:
    # Global indices:
    # A: i in {0,1}, j in {0,1,2,3,4}
    # B: j in {0,1,2,3,4}, k in {0,1,2,3,4,5}
    # C: i in {0,1}, k in {0,1,2,3,4,5}
    
    # Split k into k_left in {0,1,2} and k_right in {3,4,5}
    # For k_left (k = 0,1,2):
    #   C[i, k] = sum_j A[i,j] * B[j, k]
    #   Split j into j1 in {0,1,2} and j2 in {3,4}
    #   C[i, k] = sum_{j1} A[i,j1] * B[j1,k] + sum_{j2} A[i,j2] * B[j2,k]
    #   First term: A1[i,j1] * B1[j1,k] where A1 is 2x3, B1 is 3x3
    #   Second term: A2[i,j2] * B2[j2,k] where A2 is 2x2, B2 is 2x3
    #   Rank of first: 3*3 = 9 (each product A[i,j1]*B[j1,k] is one term)
    #   Rank of second: 2*3 = 6
    #   Total for left half: 15
    
    # For k_right (k = 3,4,5): same structure
    #   C[i, k] = sum_{j1} A[i,j1] * B[j1,k] + sum_{j2} A[i,j2] * B[j2,k]
    #   First term: A1[i,j1] * B1'[j1,k] where B1' = B[0:3, 3:6] is 3x3
    #   Second term: A2[i,j2] * B2'[j2,k] where B2' = B[3:5, 3:6] is 2x3
    #   Rank: 9 + 6 = 15
    
    # Total rank: 15 + 15 = 30
    
    rank = 30
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # For each half (left: k in {0,1,2}, right: k in {3,4,5})
    for half in range(2):
        if half == 0:
            k_range = [0, 1, 2]
        else:
            k_range = [3, 4, 5]
        
        # Part 1: A1 (2x3) * B1 (3x3) - 9 terms
        # A1[i, j1] for i in {0,1}, j1 in {0,1,2}
        # B1[j1, k] for j1 in {0,1,2}, k in k_range
        for i in range(n):
            for j1 in range(3):
                for k in k_range:
                    a_idx = i * m + j1
                    b_idx = j1 * p + k
                    c_idx = k * n + i
                    U[a_idx, col] = 1.0
                    V[b_idx, col] = 1.0
                    W[c_idx, col] = 1.0
                    col += 1
        
        # Part 2: A2 (2x2) * B2 (2x3) - 6 terms
        # A2[i, j2] for i in {0,1}, j2 in {3,4}
        # B2[j2, k] for j2 in {3,4}, k in k_range
        for i in range(n):
            for j2 in [3, 4]:
                for k in k_range:
                    a_idx = i * m + j2
                    b_idx = j2 * p + k
                    c_idx = k * n + i
                    U[a_idx, col] = 1.0
                    V[b_idx, col] = 1.0
                    W[c_idx, col] = 1.0
                    col += 1
    
    # Verify the decomposition
    def verify(U, V, W, n, m, p, tol=1e-6):
        # Check all (i,j,k) combinations
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    val = np.sum(U[a_idx, :] * V[b_idx, :] * W[c_idx, :])
                    if abs(val - 1.0) > tol:
                        return False
        # Check some zero entries
        for _ in range(200):
            a = np.random.randint(0, n * m)
            b = np.random.randint(0, m * p)
            c = np.random.randint(0, p * n)
            i, j = divmod(a, m)
            j2, k = divmod(b, p)
            k2, i2 = divmod(c, n)
            if not (j == j2 and i == i2 and k == k2):
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(val) > tol:
                    return False
        return True
    
    if not verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
    
    # Convert to list of lists
    def to_list(matrix):
        return [[float(val) for val in row] for row in matrix]
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
