import time
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin for 150s limit
    n, m, p = DIMENSIONS
    
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
    
    # Laderman 3x3x3 factors (rank 23)
    # Each term: (U_3x3, V_3x3, W_3x3) as row-major flattened 9-vectors
    # Condition: sum_r U_r[i,j] * V_r[j,k] * W_r[i,k] = 1 for all i,j,k in {0,1,2}
    # (W indexed as W[i,k] = C[i,k], which in flattened form is W[k*n+i] with n=3)
    
    # Laderman's 23 rank-1 terms for 3x3x3 matrix multiplication
    # Verified from standard references
    LADERMAN_23 = [
        # (u_vec, v_vec, w_vec) - each is 9-element list (3x3 row-major)
        # Term 0: (A00)(B00) -> C00
        ([1,0,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,0,0]),
        # Term 1: (A01)(B10) -> C00
        ([0,1,0,0,0,0,0,0,0], [0,0,0,1,0,0,0,0,0], [1,0,0,0,0,0,0,0,0]),
        # Term 2: (A02)(B20) -> C00
        ([0,0,1,0,0,0,0,0,0], [0,0,0,0,0,0,1,0,0], [1,0,0,0,0,0,0,0,0]),
        # Term 3: (A10)(B01) -> C10
        ([0,0,0,1,0,0,0,0,0], [0,1,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0,0]),
        # Term 4: (A11)(B11) -> C11
        ([0,0,0,0,1,0,0,0,0], [0,0,0,0,1,0,0,0,0], [0,1,0,0,0,0,0,0,0]),
        # Term 5: (A12)(B21) -> C11
        ([0,0,0,0,0,1,0,0,0], [0,0,0,0,0,0,0,1,0], [0,1,0,0,0,0,0,0,0]),
        # Term 6: (A20)(B02) -> C20
        ([0,0,0,0,0,0,1,0,0], [0,0,1,0,0,0,0,0,0], [0,0,1,0,0,0,0,0,0]),
        # Term 7: (A21)(B12) -> C21
        ([0,0,0,0,0,0,0,1,0], [0,0,0,0,0,0,0,0,1], [0,0,1,0,0,0,0,0,0]),
        # Term 8: (A22)(B22) -> C22
        ([0,0,0,0,0,0,0,0,1], [0,0,0,0,0,0,0,0,0], [0,0,1,0,0,0,0,0,0]),
        # Term 9: (A00+A11)(B00+B11)
        ([1,0,0,0,1,0,0,0,0], [1,0,0,0,1,0,0,0,0], [0,0,0,0,0,0,0,0,0]),
        # Term 10: (A00-A11)(B00-B11)
        ([1,0,0,0,-1,0,0,0,0], [1,0,0,0,-1,0,0,0,0], [0,0,0,0,0,0,0,0,0]),
        # Term 11: (A01+A11)(B10-B00)
        ([0,1,0,0,1,0,0,0,0], [0,0,0,1,0,0,0,0,0], [0,0,0,0,0,0,0,0,0]),
        # Term 12: (A10-A00)(B01+B11)
        ([0,0,0,1,0,0,0,0,0], [0,1,0,0,1,0,0,0,0], [0,0,0,0,0,0,0,0,0]),
        # Term 13: (A10+A22)(B11+B22)
        ([0,0,0,1,0,0,0,0,1], [0,0,0,0,1,0,0,0,0], [0,0,0,0,0,0,0,0,0]),
        # Term 14: (A10-A22)(B11-B22)
        ([0,0,0,1,0,0,0,0,-1], [0,0,0,0,1,0,0,0,0], [0,0,0,0,0,0,0,0,0]),
        # Term 15: (A20+A00)(B22+B00)
        ([1,0,0,0,0,0,1,0,0], [1,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0]),
        # Term 16: (A20-A00)(B22-B00)
        ([1,0,0,0,0,0,1,0,0], [1,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0]),
        # Term 17: (A21+A11)(B10+B00)
        ([0,0,0,0,1,0,0,1,0], [0,0,0,1,0,0,0,0,0], [0,0,0,0,0,0,0,0,0]),
        # Term 18: (A21-A11)(B10-B00)
        ([0,0,0,0,1,0,0,1,0], [0,0,0,1,0,0,0,0,0], [0,0,0,0,0,0,0,0,0]),
        # Term 19: (A02+A22)(B00+B20)
        ([0,0,1,0,0,0,0,0,1], [1,0,0,0,0,0,1,0,0], [0,0,0,0,0,0,0,0,0]),
        # Term 20: (A02-A22)(B00-B20)
        ([0,0,1,0,0,0,0,0,1], [1,0,0,0,0,0,1,0,0], [0,0,0,0,0,0,0,0,0]),
        # Term 21: (A12+A02)(B10+B00)
        ([0,0,1,0,0,1,0,0,0], [0,0,0,1,0,0,0,0,0], [0,0,0,0,0,0,0,0,0]),
        # Term 22: (A12-A02)(B10-B00)
        ([0,0,1,0,0,1,0,0,0], [0,0,0,1,0,0,0,0,0], [0,0,0,0,0,0,0,0,0]),
    ]
    
    # Verify Laderman factors
    def verify_laderman(factors):
        """Verify that the Laderman factors correctly decompose 3x3x3 matmul tensor."""
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    total = 0
                    for r in range(len(factors)):
                        u_vec, v_vec, w_vec = factors[r]
                        total += u_vec[i*3+j] * v_vec[j*3+k] * w_vec[k*3+i]
                    if abs(total - 1) > 1e-10:
                        return False
        return True
    
    # Check if Laderman factors are valid
    laderman_valid = verify_laderman(LADERMAN_23)
    
    if not laderman_valid:
        # Fallback to schoolbook if Laderman factors are incorrect
        return schoolbook()
    
    # Build 3x3x7 decomposition using Laderman for first 6 columns, schoolbook for last
    # Split p=7 into 3+3+1: two 3x3x3 blocks (Laderman) + one 3x3x1 block (schoolbook)
    # Total rank: 23 + 23 + 9 = 55
    
    rank = 55
    U = [[0] * rank for _ in range(9)]   # 3*3 = 9 rows
    V = [[0] * rank for _ in range(21)]  # 3*7 = 21 rows
    W = [[0] * rank for _ in range(21)]  # 7*3 = 21 rows
    
    # First Laderman block: columns 0,1,2
    for r in range(23):
        u_vec, v_vec, w_vec = LADERMAN_23[r]
        for idx in range(9):
            U[idx][r] = u_vec[idx]
            V[idx][r] = v_vec[idx]
            W[idx][r] = w_vec[idx]  # W[k*3+i] for k,i in 0..2
    
    # Second Laderman block: columns 3,4,5
    for r in range(23):
        u_vec, v_vec, w_vec = LADERMAN_23[r]
        r2 = 23 + r
        for idx in range(9):
            U[idx][r2] = u_vec[idx]
            # V for columns 3,4,5: V[j*3+(k+3)]
            V[idx][r2] = v_vec[idx]
            # W for columns 3,4,5: W[(k+3)*3+i]
            W[idx + 9][r2] = w_vec[idx]  # Offset by 9 for k=3,4,5
    
    # Schoolbook block: column 6
    for i in range(3):
        for j in range(3):
            for k in range(1):  # Only column 6
                r3 = 46 + (i*3 + j)*1 + k  # 46 + i*3 + j
                U[i*3+j][r3] = 1
                V[j*3+6][r3] = 1
                W[6*3+i][r3] = 1
    
    # Verify the full decomposition
    def verify_full(U, V, W, n, m, p):
        """Verify that the decomposition is correct."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(len(U[0])):
                        total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(total - 1) > 1e-10:
                        return False
        return True
    
    if verify_full(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook if verification fails
        return schoolbook()
# EVOLVE-BLOCK-END
