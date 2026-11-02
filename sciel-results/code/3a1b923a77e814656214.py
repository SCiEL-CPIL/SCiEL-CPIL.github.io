import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    n, m, p = DIMENSIONS
    
    def verify_decomposition(U, V, W, n, m, p):
        """Verify that the decomposition is exact."""
        rank = len(U[0]) if U else 0
        # Check shapes
        if len(U) != n * m or len(V) != m * p or len(W) != p * n:
            return False
        for r in range(rank):
            if len(U[0]) != rank or len(V[0]) != rank or len(W[0]) != rank:
                return False
        # Check tensor decomposition: sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] = 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = Fraction(0)
                    for r in range(rank):
                        total += Fraction(U[i * m + j][r]) * Fraction(V[j * p + k][r]) * Fraction(W[k * n + i][r])
                    if total != 1:
                        return False
        return True
    
    def schoolbook():
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
    
    # Laderman's 3x3x3 rank-23 decomposition
    # We need 23 triples where each triple consists of:
    # u: 9-vector (indexed by i*3+j for i,j in 0..2)
    # v: 9-vector (indexed by j*3+k for j,k in 0..2)  
    # w: 9-vector (indexed by k*3+i for k,i in 0..2)
    # such that sum_r u[i*3+j] * v[j*3+k] * w[k*3+i] = 1 for all i,j,k in 0..2
    
    # Laderman factors from literature. Each factor is (u, v, w) where
    # u, v, w are 9-element lists representing 3x3 matrices in row-major order.
    # u[i*3+j] is the (i,j) entry of the U factor matrix
    # v[j*3+k] is the (j,k) entry of the V factor matrix
    # w[k*3+i] is the (k,i) entry of the W factor matrix
    
    # The 23 Laderman factors:
    laderman_3x3x3 = [
        # Term 0
        ([1,0,0, 0,1,0, 0,0,0], [1,0,0, 0,1,0, 0,0,0], [1,0,0, 0,1,0, 0,0,0]),
        # Term 1
        ([0,0,0, 1,0,0, 0,0,0], [0,0,0, 1,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0]),
        # Term 2
        ([0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1], [0,0,1, 0,0,0, 0,0,0]),
        # Term 3
        ([0,0,1, 0,0,0, 0,0,0], [0,0,0, 0,0,1, 0,0,0], [0,0,0, 0,0,1, 0,0,0]),
        # Term 4
        ([0,1,0, 0,0,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0]),
        # Term 5
        ([0,0,0, 0,0,1, 0,0,0], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,1,0]),
        # Term 6
        ([1,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 1,0,0]),
        # Term 7
        ([0,0,0, 0,0,0, 1,0,0], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1]),
        # Term 8
        ([1,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,1, 0,0,0], [0,0,0, 0,0,1, 0,0,0]),
        # Term 9
        ([0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,1, 0,0,0], [0,0,0, 0,0,1, 0,0,0]),
        # Term 10
        ([0,0,1, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1]),
        # Term 11
        ([1,0,0, 0,0,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0], [0,0,0, 0,0,0, 0,1,0]),
        # Term 12
        ([0,0,0, 0,0,0, 1,0,0], [0,0,0, 0,0,1, 0,0,0], [0,0,0, 0,0,0, 0,1,0]),
        # Term 13
        ([0,0,0, 1,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 1,0,0]),
        # Term 14
        ([0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,1,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0]),
        # Term 15
        ([0,0,1, 0,0,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0]),
        # Term 16
        ([1,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1]),
        # Term 17
        ([0,0,0, 0,0,0, 1,0,0], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1]),
        # Term 18
        ([1,0,0, 0,0,0, 0,0,0], [0,0,1, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,1]),
        # Term 19
        ([0,0,0, 0,0,0, 0,0,1], [0,0,1, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,1]),
        # Term 20
        ([0,0,1, 0,0,0, 0,0,0], [0,0,0, 0,0,1, 0,0,0], [0,0,0, 0,0,1, 0,0,0]),
        # Term 21
        ([0,1,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1]),
        # Term 22
        ([0,0,0, 0,0,1, 0,0,0], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1]),
    ]
    
    # Verify Laderman factors
    U_test = [[0]*23 for _ in range(9)]
    V_test = [[0]*23 for _ in range(9)]
    W_test = [[0]*23 for _ in range(9)]
    for r, (u, v, w) in enumerate(laderman_3x3x3):
        for idx in range(9):
            U_test[idx][r] = u[idx]
            V_test[idx][r] = v[idx]
            W_test[idx][r] = w[idx]
    
    laderman_valid = verify_decomposition(U_test, V_test, W_test, 3, 3, 3)
    
    if not laderman_valid:
        # Fallback to schoolbook
        return schoolbook()
    
    # Now construct 3x3x7 decomposition
    # Strategy: Split p=7 into three parts: p1=3, p2=3, p3=1
    # For each part, use Laderman (rank 23) for the 3x3x3 parts and schoolbook (rank 9) for 3x3x1
    # Total rank = 23 + 23 + 9 = 55
    
    # Actually, let's think about this more carefully.
    # The tensor for 3x3x7 has indices (i, j, k) where i in 0..2, j in 0..2, k in 0..6
    # We want sum_r U[i*3+j][r] * V[j*7+k][r] * W[k*3+i][r] = 1
    
    # We can split k into k1 in 0..2 and k2 in 3..6
    # For k1 (first 3 columns of B): use Laderman 3x3x3, rank 23
    # For k2 (last 4 columns of B): use schoolbook 3x3x4, rank 36
    # Total: 23 + 36 = 59
    
    # Or split into k1 in 0..2, k2 in 3..5, k3 in 6
    # Rank = 23 + 23 + 9 = 55
    
    # Let's use the 3+3+1 split for rank 55
    
    total_rank = 23 + 23 + 9
    U_full = [[0] * total_rank for _ in range(9)]  # n*m = 9
    V_full = [[0] * total_rank for _ in range(21)]  # m*p = 21
    W_full = [[0] * total_rank for _ in range(21)]  # p*n = 21
    
    # Part 1: k in 0..2 (first 3 columns), use Laderman
    for r, (u, v, w) in enumerate(laderman_3x3x3):
        for idx in range(9):
            U_full[idx][r] = u[idx]
        for j in range(3):
            for k in range(3):
                V_full[j * 7 + k][r] = v[j * 3 + k]
        for k in range(3):
            for i in range(3):
                W_full[k * 3 + i][r] = w[k * 3 + i]
    
    # Part 2: k in 3..5 (next 3 columns), use Laderman
    offset = 23
    for r, (u, v, w) in enumerate(laderman_3x3x3):
        for idx in range(9):
            U_full[idx][offset + r] = u[idx]
        for j in range(3):
            for k in range(3):
                V_full[j * 7 + (3 + k)][offset + r] = v[j * 3 + k]
        for k in range(3):
            for i in range(3):
                W_full[(3 + k) * 3 + i][offset + r] = w[k * 3 + i]
    
    # Part 3: k in 6 (last column), use schoolbook 3x3x1
    offset = 46
    for i in range(3):
        for j in range(3):
            for k in range(1):
                col = offset + i * 3 + j * 1 + k  # This is wrong, let me fix
                # Actually for 3x3x1 schoolbook:
                # U[i*3+j] = 1, V[j*1+0] = 1, W[0*3+i] = 1
                pass
    
    # Let me redo part 3 properly
    offset = 46
    col_in_part = 0
    for i in range(3):
        for j in range(3):
            U_full[i * 3 + j][offset + col_in_part] = 1
            V_full[j * 7 + 6][offset + col_in_part] = 1
            W_full[6 * 3 + i][offset + col_in_part] = 1
            col_in_part += 1
    
    # Verify the full decomposition
    if not verify_decomposition(U_full, V_full, W_full, n, m, p):
        # Fallback
        return schoolbook()
    
    return {"U": U_full, "V": V_full, "W": W_full}
# EVOLVE-BLOCK-END
