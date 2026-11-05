import numpy as np
import time

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Total rank is sum of parts
    # Part 1: 4x1x6 (Rank 4)
    # Part 2: 4x4x4 (Rank 49)
    # Part 3: 4x4x2 (Rank 32)
    rank = 4 + 49 + 32
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # --- Part 1: A2 (4x1) * B2 (1x6) ---
    # A2 is col 4 of A. B2 is row 4 of B.
    # U index: i*5 + 4
    # V index: 4*6 + k = 24 + k
    # W index: k*4 + i
    for r in range(4):
        for i in range(4):
            if i == r:
                U[i * 5 + 4][col + r] = 1
                W[r * 4 + i][col + r] = 1 # Wait, W index is k*4+i. Here k varies.
                # Correction: W[k*4+i, col+r] should be delta(i,r) for all k?
                # Check: sum_r U[i,4,r] V[24+k,r] W[k*4+i,r]
                # U[i,4,r] = delta(i,r)
                # V[24+k,r] = 1
                # W[k*4+i,r] = delta(i,r)
                # Sum = delta(i,r) * 1 * delta(i,r) = delta(i,r).
                # This is 1 if i=r, 0 else.
                # We need T[i, 24+k, k*4+i] = 1 for all i,k.
                # My check: if i=r, sum is 1. If i!=r, sum is 0.
                # But T is 1 for ALL i,k.
                # So this decomposition is WRONG.
                # T[i, k, k*4+i] = 1.
                # We need U[i, r] V[k, r] W[k*4+i, r] = 1 for all i,k.
                # This implies U[i,r] and W[k*4+i,r] must be non-zero for all i,k.
                # If we use rank 4, we can't cover all i,k with delta.
                # Rank 4 is NOT sufficient for 4x1x6?
                # 4x1x6 has 24 non-zero entries.
                # Rank 1: U[i,r]=1, V[k,r]=1, W[k*4+i,r]=1. Sum=1.
                # YES! Rank 1 is sufficient for 4x1x6.
                # U[i, 4*5+j? No, U[i*5+4, 0] = 1 for all i.
                # V[24+k, 0] = 1 for all k.
                # W[k*4+i, 0] = 1 for all k,i.
                # Sum = 1.
                # So Rank 1 for Part 1.
                pass
    # Reset col
    col = 0
    # Part 1: Rank 1
    for i in range(4):
        U[i * 5 + 4][col] = 1
    for k in range(6):
        V[24 + k][col] = 1
    for k in range(6):
        for i in range(4):
            W[k * 4 + i][col] = 1
    col += 1

    # --- Part 2: A1 (4x4) * B1a (4x4) ---
    # A1 cols 0-3. B1a rows 0-3, cols 0-3.
    # U index: i*5 + j (j=0..3)
    # V index: j*6 + k (j=0..3, k=0..3)
    # W index: k*4 + i
    # Rank 49 (Strassen 4x4x4)
    # We need to generate Strassen 4x4x4 factors.
    # Strassen 2x2x2 rank 7 factors:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A10)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M7
    # C10 = M2 + M5
    # C11 = M1 - M2 + M3 + M6
    
    # For 4x4x4, we apply this to 2x2 blocks.
    # There are 7 "M" terms, each is a 2x2 matrix product.
    # Each 2x2 product has rank 7.
    # Total 49.
    
    # We will hardcode the 49 rank decomposition for 4x4x4.
    # This is complex to generate on the fly correctly without bugs.
    # Alternative: Use schoolbook 64 for 4x4x4?
    # Rank 64 + 32 + 1 = 97. Still < 120.
    # Let's use schoolbook for simplicity and reliability.
    # 4x4x4 schoolbook rank 64.
    
    # Part 2: Rank 64
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U[i * 5 + j][col] = 1
                V[j * 6 + k][col] = 1
                W[k * 4 + i][col] = 1
                col += 1

    # --- Part 3: A1 (4x4) * B1b (4x2) ---
    # A1 cols 0-3. B1b rows 0-3, cols 4-5.
    # U index: i*5 + j (j=0..3)
    # V index: j*6 + k (j=0..3, k=4..5)
    # W index: k*4 + i
    # Rank 32 (Schoolbook)
    for i in range(4):
        for j in range(4):
            for k in range(4, 6):
                U[i * 5 + j][col] = 1
                V[j * 6 + k][col] = 1
                W[k * 4 + i][col] = 1
                col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
