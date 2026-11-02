import time
import numpy as np

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # We will build the decomposition manually.
    # A is n x m (4x5), B is m x p (5x5).
    # Split A into A1 (4x4) and A2 (4x1).
    # Split B into B1 (4x5) and B2 (1x5).
    # B1 = B1a (4x4) + b5 (4x1).
    # AB = A1 B1a + A1 b5 + A2 B2.
    
    # Part 1: A1 B1a (4x4 * 4x4) using Strassen with schoolbook 2x2.
    # Strassen 4x4: 7 multiplications of 2x2 matrices.
    # Each 2x2 multiplication is schoolbook (rank 8).
    # Total rank for core: 7 * 8 = 56.
    
    # Part 2: A1 b5 (4x4 * 4x1). Schoolbook.
    # Rank: 4 * 4 * 1 = 16.
    
    # Part 3: A2 B2 (4x1 * 1x5). Schoolbook.
    # Rank: 4 * 1 * 5 = 20.
    
    # Total rank: 56 + 16 + 20 = 92.
    # Wait, A2 is 4x1, B2 is 1x5. Rank is 4*1*5 = 20.
    # Previous calculation was wrong.
    
    # Let's re-evaluate.
    # A1 B1a: 56
    # A1 b5: 16
    # A2 B2: 20
    # Total: 92.
    
    # Can we do better?
    # Use Strassen for 2x2 blocks: 7 * 7 = 49.
    # Total: 49 + 16 + 20 = 85.
    
    # Let's try to optimize A2 B2.
    # A2 is 4x1, B2 is 1x5. This is an outer product.
    # Rank is 4*5 = 20.
    
    # Is there a better partition?
    # Maybe split B into 3x2 and 2x3?
    # A is 4x5. B is 5x5.
    # Split B into B1 (3x5) and B2 (2x5).
    # A1 (4x3) B1 (3x5) + A2 (4x2) B2 (2x5).
    # A1 B1: 4x3 * 3x5. Rank 4*3*5 = 60.
    # A2 B2: 4x2 * 2x5. Rank 4*2*5 = 40.
    # Total 100. No improvement.
    
    # The 4x4 core is the key.
    # Let's implement the 4x4 Strassen with schoolbook 2x2.
    
    U = []
    V = []
    W = []
    
    # We need to map the tensor indices.
    # T[i, j, k] = 1 if A[i,j] * B[j,k] contributes to C[i,k].
    # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r].
    
    # Let's define the schoolbook for 2x2.
    # A2x2 = [[a11, a12], [a21, a22]]
    # B2x2 = [[b11, b12], [b21, b22]]
    # C11 = a11*b11 + a12*b21
    # C12 = a11*b12 + a12*b22
    # C21 = a21*b11 + a22*b21
    # C22 = a21*b12 + a22*b22
    
    # Strassen for 4x4:
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Each Mi is a 2x2 matrix product.
    # We use schoolbook for each Mi.
    
    # We will generate the U, V, W matrices for the core 4x4*4x4.
    # Then add the remaining parts.
    
    # This is complex to implement directly.
    # Let's use a known rank 69 decomposition if possible.
    # Actually, the rank of 4x4x4 is 49 (Strassen) or 48 (Laderman).
    # So 4x4x4 rank is 49.
    # Then 4x4*4x1 is 16.
    # 4x1*1x5 is 20.
    # Total 49 + 16 + 20 = 85.
    
    # Let's implement the 4x4 Strassen with Strassen 2x2.
    
    # For now, let's return the schoolbook to ensure correctness, 
    # and then improve.
    
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
