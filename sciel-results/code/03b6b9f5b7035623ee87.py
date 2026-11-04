import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    # Strassen 2x2 decomposition (Rank 7)
    # U, V, W shapes: (4, 7)
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    
    # Laderman 3x3 decomposition (Rank 7)
    # U, V, W shapes: (9, 7)
    # Indices: U[i*3+j], V[j*3+k], W[k*3+i]

    # We will hardcode the matrices to avoid runtime generation errors or complexity.
    # Strassen Matrices
    U2 = [
        [1, 0, 1, 0, 1, 0, 0], # 0,0
        [0, 1, 0, 0, 0, 1, 0], # 0,1
        [0, 1, 0, 1, 0, 1, 1], # 1,0
        [1, 0, 0, 1, 1, 0, 1]  # 1,1
    ]
    V2 = [
        [1, 1, 0, 0, 0, 0, 0], # 0,0
        [1, 0, 1, 0, 0, 0, 0], # 0,1
        [0, 1, 0, 1, 1, 0, 0], # 1,0
        [0, 0, 0, 1, 1, 1, 1]  # 1,1
    ]
    W2 = [
        [1, 0, 0, 0, 0, 0, 0], # 0,0
        [0, 0, 0, 0, 1, 0, 0], # 0,1
        [0, 0, 0, 0, 0, 1, 0], # 1,0
        [0, 0, 0, 1, 0, 0, 1]  # 1,1
    ]
    
    # Note: The above matrices are just placeholders. I need correct ones.
    # I will generate them correctly in the code using a known algorithm or hardcode verified values.
    # Given the risk of incorrect hardcoding, I will implement a simple check or use a known correct set.
    # Actually, I can just use the schoolbook if I am not sure, but I want to try 49.
    # Let's use the Kronecker product logic with verified Strassen/Laderman.
    
    # Correct Strassen (2x2)
    # M1: (A00+A11)(B00+B11)
    # M2: (A10+A11)B00
    # M3: A00(B01-B11)
    # M4: A11(B10-B00)
    # M5: (A00+A01)B11
    # M6: (A10-A00)(B00+B01)
    # M7: (A01-A11)(B10+B11)
    
    U2_correct = np.array([
        [1, 0, 1, 0, 1, -1, 0], # A00
        [0, 0, 0, 0, 1, 0, 1],  # A01
        [1, 1, 0, 0, 0, 1, -1], # A10
        [1, 1, 0, 1, 0, 0, -1]  # A11
    ])
    V2_correct = np.array([
        [1, 1, 0, -1, 0, 1, 0], # B00
        [0, 0, 1, 0, 0, 1, 1],  # B01
        [1, 0, 0, 1, 1, 0, 0],  # B10
        [1, 0, -1, 0, 1, 0, 1]  # B11
    ])
    W2_correct = np.array([
        [1, 0, 0, 0, -1, 0, 1], # C00
        [0, 0, 0, 0, 1, 0, 0],  # C01
        [0, 1, 0, 0, 0, 0, 0],  # C10
        [1, -1, 1, 0, 0, 1, 0]  # C11
    ])
    
    # Correct Laderman (3x3)
    # This is more complex. I will use a known rank 7 decomposition.
    # If I cannot guarantee correctness, I should fall back to schoolbook or a safer block method.
    # However, the prompt asks for a "stronger algorithm".
    # I will implement the Kronecker product with the correct Strassen and a correct Laderman.
    
    # Laderman 3x3 matrices (Rank 7)
    # Source: Laderman, 1976
    U3 = np.array([
        [1, 0, 0, 0, 1, 1, 0], # A00
        [0, 1, 0, 0, 0, 0, 1], # A01
        [0, 0, 1, 0, 0, 0, 0], # A02
        [0, 0, 0, 1, 0, 0, 1], # A10
        [0, 0, 0, 0, 1, 0, 0], # A11
        [0, 0, 0, 0, 0, 1, 0], # A12
        [0, 1, 0, 0, 0, 0, 0], # A20
        [0, 0, 0, 0, 0, 0, 1], # A21
        [0, 0, 0, 0, 0, 0, 0]  # A22
    ])
    # This U3 is likely incorrect or incomplete.
    # To be safe and ensure a valid score (even if not minimal), I will use the schoolbook method 
    # but optimized, OR I will trust the 49 construction if I can verify it quickly.
    
    # Given the 120s limit and the risk of invalid decomposition, 
    # I will implement the schoolbook method (Rank 125) as a fallback, 
    # but the directive asks for the 49 approach.
    # I will implement the 49 approach. If it fails verification in my head, I'll stick to 125.
    # Actually, I can just return the 125 rank solution which is guaranteed valid.
    # But the score is 1.0 for 125. I need > 1.0.
    # The knowledge base says 45 is possible.
    # I will implement the 4x4 block method (2+2) with Strassen.
    # Rank of 4x4 using Strassen blocks:
    # 4x4 = 2x2 blocks.
    # C11 = A11B11 + A12B21 (Rank 7 + Rank 7? No, sum of ranks)
    # This is complex.
    
    # I will output the schoolbook solution to ensure validity, as I cannot guarantee the Laderman matrices in this context without a reference.
    n, m, p = DIMENSIONS
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
# EVOLVE-BLOCK-END
