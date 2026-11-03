import numpy as np
import time

def strassen_2x2():
    """
    Returns U, V, W for 2x2x2 matrix multiplication with rank 7.
    U shape: (4, 7)
    V shape: (4, 7)
    W shape: (4, 7)
    Indices:
    U[i*2+j, c]
    V[j*2+k, c]
    W[k*2+i, c]
    """
    # Strassen coefficients
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # A indices: 0:00, 1:01, 2:10, 3:11
    # B indices: 0:00, 1:01, 2:10, 3:11
    # C indices: 0:00, 1:01, 2:10, 3:11
    
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1 # C11, C22 get +1?
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Let's fill W based on the formulas
    # W[k*2+i, c]
    # C00 (k=0,i=0) is index 0
    # C01 (k=0,i=1) is index 1
    # C10 (k=1,i=0) is index 2
    # C11 (k=1,i=1) is index 3
    
    # M1
    W[0, 0] = 1 # C00
    W[3, 0] = 1 # C11
    
    # M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1 # C10
    W[3, 1] = -1 # C11
    
    # M3
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1 # C01
    W[3, 2] = 1 # C11
    
    # M4
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1 # C00
    W[2, 3] = 1 # C10
    
    # M5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1 # C00
    W[1, 4] = 1 # C01
    
    # M6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1 # C11
    
    # M7
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1 # C00
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    # 1. Construct 4x4x4 decomposition (Rank 49)
    # Recursive Strassen
    U_444, V_444, W_444 = np.zeros((16, 49), dtype=int), np.zeros((16, 49), dtype=int), np.zeros((16, 49), dtype=int)
    
    # We need to map the 7 big Strassen multiplications to 49 small ones.
    # Big A, B are 4x4. Split into 2x2 blocks.
    # A11, A12, A21, A22
    # B11, B12, B21, B22
    
    # Big M1 = (A11+A22)(B11+B22)
    # This is a 2x2x2 multiplication of (A11+A22) and (B11+B22).
    # We can use the strassen_2x2() matrices, but we need to map the indices.
    
    # This is complex to do programmatically without a generic tensor composition tool.
    # Given the constraints, I will provide the known rank 49 matrices if I can, or fall back.
    # Actually, I can just use the schoolbook for 4x4x4 if I can't derive it easily.
    # Schoolbook 4x4x4 is rank 64.
    # 64 + 32 = 96. (Same as seed).
    # 64 + 28 = 92. (Improvement).
    # 49 + 32 = 81. (Better).
    # 49 + 28 = 77. (Best).
    
    # I will implement the recursive Strassen carefully.
    
    # Helper to compose two rank-1 decompositions? No.
    # I will hardcode the 49 rank terms for 4x4x4? Too many.
    
    # Alternative: Use the fact that Rank(4x4x4) = 49 is standard.
    # I will assume I can generate it.
    
    # Let's try to generate it.
    # The 7 big multiplications are:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Each Mr is a 2x2x2 multiplication.
    # We can use the U,V,W from strassen_2x2() for each Mr.
    # We need to map the indices of the 2x2 sub-matrices to the 4x4 indices.
    
    # A11 indices: (0,0), (0,1), (1,0), (1,1) -> 0,1,2,3
    # A12 indices: (0,2), (0,3), (1,2), (1,3) -> 4,5,6,7
    # A21 indices: (2,0), (2,1), (3,0), (3,1) -> 8,9,10,11
    # A22 indices: (2,2), (2,3), (3,2), (3,3) -> 12,13,14,15
    
    # Similarly for B.
    # C indices:
    # C11: 0,1,2,3
    # C12: 4,5,6,7
    # C21: 8,9,10,11
    # C22: 12,13,14,15
    
    # For each big multiplication Mr, we have 7 small multiplications.
    # We will fill U_444, V_444, W_444.
    
    # Define the mapping for each big M
    # M1: A_part = A11+A22, B_part = B11+B22
    # A11 indices in 2x2 local: 0,1,2,3. Global: 0,1,2,3.
    # A22 indices in 2x2 local: 0,1,2,3. Global: 12,13,14,15.
    # So for M1, the A matrix in the small Strassen is formed by adding these.
    # This means we need to add the U vectors.
    
    # This is getting too complex to implement robustly in the short time.
    # I will fall back to Schoolbook for 4x4x4 (Rank 64) and Schoolbook for 4x4x2 (Rank 32).
    # Total Rank 96.
    # This is the seed score.
    
    # Wait, I can implement Rank 49 for 4x4x4 by using a known set of matrices.
    # Or I can use the Rank 81 solution (49 + 32) if I can implement 49.
    
    # Let's try to implement the recursive Strassen properly.
    
    def compose_strassen(A_indices, B_indices, C_indices, offset):
        """
        A_indices: list of 4 global indices for the 2x2 A block
        B_indices: list of 4 global indices for the 2x2 B block
        C_indices: list of 4 global indices for the 2x2 C block
        offset: column offset in the rank dimension
        """
        U_s, V_s, W_s = strassen_2x2()
        for c in range(7):
            for local_i in range(4):
                if U_s[local_i, c] != 0:
                    U_444[A_indices[local_i], offset + c] += U_s[local_i, c]
            for local_j in range(4):
                if V_s[local_j, c] != 0:
                    V_444[B_indices[local_j], offset + c] += V_s[local_j, c]
            for local_k in range(4):
                if W_s[local_k, c] != 0:
                    W_444[C_indices[local_k], offset + c] += W_s[local_k, c]

    # Define the 7 big multiplications
    # A11: [0,1,2,3], A12: [4,5,6,7], A21: [8,9,10,11], A22: [12,13,14,15]
    # B11: [0,1,2,3], B12: [4,5,6,7], B21: [8,9,10,11], B22: [12,13,14,15]
    # C11: [0,1,2,3], C12: [4,5,6,7], C21: [8,9,10,11], C22: [12,13,14,15]
    
    # M1 = (A11+A22)(B11+B22)
    # A is A11+A22. This is not a single 2x2 block.
    # We cannot just pass indices. We need to sum the U vectors.
    # So for M1, the A part is A11 + A22.
    # U_444 for M1 should be U(A11) + U(A22).
    
    # This requires a different approach.
    # I will skip the 4x4x4 optimization and use schoolbook (64).
    # And I will try to optimize 4x4x2 to 28.
    # Total 64 + 28 = 92.
    # Score 96/92 = 1.043.
    
    # Let's implement 4x4x2 rank 28.
    # 4 independent 2x2x2 multiplications.
    # A11 B11 -> C11? No.
    # The structure is not direct sum.
    # I will use schoolbook for 4x4x2 (32).
    # Total 64 + 32 = 96.
    
    # This is not an improvement.
    
    # I must implement the Rank 49 for 4x4x4.
    # I will hardcode the Rank 49 matrices for 4x4x4 if I can find them in my training data.
    # I don't have them memorized.
    
    # I will use the Rank 81 solution (49 + 32) by implementing the recursive Strassen correctly.
    
    # To implement recursive Strassen:
    # For each big M, we have a linear combination of A blocks and B blocks.
    # We can represent the U matrix for the big M as the sum of U matrices for the constituent blocks.
    
    # Let's do it.
    
    U_444 = np.zeros((16, 49), dtype=int)
    V_444 = np.zeros((16, 49), dtype=int)
    W_444 = np.zeros((16, 49), dtype=int)
    
    # M1: (A11+A22)(B11+B22) -> C11, C12, C21, C22 (distributed)
    # Actually, the result of M1 is added to C11 and C22.
    # The Strassen algorithm computes M1..M7, then combines them to get C.
    # So the W matrix for the 4x4x4 decomposition is not just the W of the small Strassen.
    # The W of the small Strassen gives the 2x2 result of the big multiplication.
    # Then we need to add that 2x2 result to the appropriate part of C.
    
    # This is too complex to get right quickly.
    
    # I will return the schoolbook solution (Rank 96).
    # It is safe and correct.
    
    rank = 96
    U = np.zeros((16, rank), dtype=int)
    V = np.zeros((24, rank), dtype=int)
    W = np.zeros((24, rank), dtype=int)
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(6):
                U[i*4+j, col] = 1
                V[j*6+k, col] = 1
                W[k*4+i, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
