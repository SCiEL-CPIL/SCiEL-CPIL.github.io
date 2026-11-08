import numpy as np
import time

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # Target rank
    rank = 73
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    
    # Helper to add Strassen 2x2x2 terms
    # A indices: (r1, c1), (r1, c2), (r2, c1), (r2, c2)
    # B indices: (c1, k1), (c1, k2), (c2, k1), (c2, k2)
    # C indices: (k1, r1), (k1, r2), (k2, r1), (k2, r2)
    def add_strassen(r1, r2, c1, c2, k1, k2):
        nonlocal col
        # m1 = (A[r1,c1] + A[r2,c2]) * (B[c1,k1] + B[c2,k2])
        U[r1*m + c1, col] += 1
        U[r2*m + c2, col] += 1
        V[c1*p + k1, col] += 1
        V[c2*p + k2, col] += 1
        W[k1*n + r1, col] += 1
        W[k2*n + r2, col] += 1
        W[k1*n + r2, col] += 1
        W[k2*n + r1, col] += 1
        col += 1
        
        # m2 = (A[r2,c1] + A[r2,c2]) * B[c1,k1]
        U[r2*m + c1, col] += 1
        U[r2*m + c2, col] += 1
        V[c1*p + k1, col] += 1
        W[k1*n + r2, col] += 1
        col += 1
        
        # m3 = A[r1,c1] * (B[c1,k2] - B[c2,k2])
        U[r1*m + c1, col] += 1
        V[c1*p + k2, col] += 1
        V[c2*p + k2, col] -= 1
        W[k2*n + r1, col] += 1
        col += 1
        
        # m4 = A[r2,c2] * (B[c2,k1] - B[c1,k1])
        U[r2*m + c2, col] += 1
        V[c2*p + k1, col] += 1
        V[c1*p + k1, col] -= 1
        W[k1*n + r2, col] += 1
        col += 1
        
        # m5 = (A[r1,c1] + A[r1,c2]) * B[c2,k2]
        U[r1*m + c1, col] += 1
        U[r1*m + c2, col] += 1
        V[c2*p + k2, col] += 1
        W[k2*n + r1, col] += 1
        col += 1
        
        # m6 = (A[r2,c1] - A[r1,c1]) * (B[c1,k1] + B[c1,k2])
        U[r2*m + c1, col] += 1
        U[r1*m + c1, col] -= 1
        V[c1*p + k1, col] += 1
        V[c1*p + k2, col] += 1
        W[k1*n + r2, col] += 1
        W[k2*n + r2, col] += 1
        col += 1
        
        # m7 = (A[r1,c2] - A[r2,c2]) * (B[c2,k1] + B[c2,k2])
        U[r1*m + c2, col] += 1
        U[r2*m + c2, col] -= 1
        V[c2*p + k1, col] += 1
        V[c2*p + k2, col] += 1
        W[k1*n + r1, col] += 1
        W[k2*n + r1, col] += 1
        col += 1

    # Helper to add 2x3x2 terms
    # A: 2x3, B: 3x2
    # r1, r2: rows of A
    # c1, c2, c3: cols of A / rows of B
    # k1, k2: cols of B
    def add_2x3x2(r1, r2, c1, c2, c3, k1, k2):
        nonlocal col
        # Split into 2x2x2 (c1, c2) and 2x1x2 (c3)
        # 2x2x2 part
        add_strassen(r1, r2, c1, c2, k1, k2)
        
        # 2x1x2 part: A[:, c3] * B[c3, :]
        # C[r1, k1] += A[r1, c3] * B[c3, k1]
        U[r1*m + c3, col] += 1
        V[c3*p + k1, col] += 1
        W[k1*n + r1, col] += 1
        col += 1
        
        # C[r1, k2] += A[r1, c3] * B[c3, k2]
        U[r1*m + c3, col] += 1
        V[c3*p + k2, col] += 1
        W[k2*n + r1, col] += 1
        col += 1
        
        # C[r2, k1] += A[r2, c3] * B[c3, k1]
        U[r2*m + c3, col] += 1
        V[c3*p + k1, col] += 1
        W[k1*n + r2, col] += 1
        col += 1
        
        # C[r2, k2] += A[r2, c3] * B[c3, k2]
        U[r2*m + c3, col] += 1
        V[c3*p + k2, col] += 1
        W[k2*n + r2, col] += 1
        col += 1

    # 1. C_top (r=0,1) from A_top (2x5) and B (5x6)
    # Split c (0..4) into 0..1 and 2..4
    # Part 1: A[:, 0:2] * B[0:2, :] (2x2x6)
    # Split k (0..5) into 0..1, 2..3, 4..5
    add_strassen(0, 1, 0, 1, 0, 1)
    add_strassen(0, 1, 0, 1, 2, 3)
    add_strassen(0, 1, 0, 1, 4, 5)
    
    # Part 2: A[:, 2:5] * B[2:5, :] (2x3x6)
    # Split k (0..5) into 0..1, 2..3, 4..5
    add_2x3x2(0, 1, 2, 3, 4, 0, 1)
    add_2x3x2(0, 1, 2, 3, 4, 2, 3)
    add_2x3x2(0, 1, 2, 3, 4, 4, 5)
    
    # 2. C_bot (r=2) from A_bot (1x5) and B (5x6)
    # Rank 30
    for c in range(5):
        for k in range(6):
            U[2*m + c, col] += 1
            V[c*p + k, col] += 1
            W[k*n + 2, col] += 1
            col += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
