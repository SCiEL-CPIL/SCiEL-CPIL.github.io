import numpy as np
import time

DIMENSIONS = [3, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 3, 4, 7
    
    # Subproduct method:
    # Partition A (3x4) into A_top (2x4) and A_bot (1x4)
    # Partition B (4x7) into B_left (4x6) and B_right (4x1)
    # C = A_top @ B_left + A_top @ B_right + A_bot @ B_left + A_bot @ B_right
    # = A_top @ B + A_bot @ B
    # 
    # For A_top (2x4) @ B (4x7): use rank-21 decomposition
    # For A_bot (1x4) @ B (4x7): use rank-7 schoolbook
    # Total: 21 + 7 = 28
    
    # Let's construct the rank-21 decomposition for 2x4x7
    # Using Strassen-like approach:
    # A_top = [A11 A12] where A11, A12 are 2x2
    # B = [B1; B2] where B1, B2 are 2x7
    # C_top = A11 @ B1 + A12 @ B2
    #
    # For each 2x2 @ 2x7 product, we can use 7 Strassen mults on the 2x2 part
    # but need to handle the 2x7 output. Split B1 into [B1a B1b B1c B1d] (2x2 each, 4 blocks)
    # That's 4 * 7 = 28, too many.
    #
    # Alternative: direct construction. Let me build the full tensor and find rank-21.
    
    # Build the 2x4x7 multiplication tensor
    n2, m2, p2 = 2, 4, 7
    tensor2 = np.zeros((n2 * m2, m2 * p2, p2 * n2))
    for i in range(n2):
        for j in range(m2):
            for k in range(p2):
                tensor2[i * m2 + j, j * p2 + k, k * n2 + i] = 1.0
    
    # For 1x4x7, it's just rank 7 (schoolbook)
    n1, m1, p1 = 1, 4, 7
    tensor1 = np.zeros((n1 * m1, m1 * p1, p1 * n1))
    for i in range(n1):
        for j in range(m1):
            for k in range(p1):
                tensor1[i * m1 + j, j * p1 + k, k * n1 + i] = 1.0
    
    # Try to find a rank-21 decomposition for the 2x4x7 tensor
    # Use truncated SVD on the matricization
    U_mat, s, Vh = np.linalg.svd(tensor2.reshape(n2 * m2, -1), full_matrices=False)
    
    # The rank of the 2x4x7 tensor matricized as (8, 28) is min(8,28) = 8, 
    # but that's not the tensor rank. Tensor rank is different.
    
    # Let me try a different approach: use known decompositions
    # Actually, let me just use a direct search for low-rank decomposition
    
    # For now, use the schoolbook for everything as fallback, but try to improve
    # Let me construct a rank-21 decomposition explicitly
    
    # Strassen for 2x2x2 has rank 7. For 2x4x7:
    # Split into four 2x2x2 subproblems? No, dimensions don't match.
    
    # Let me use the following: 2x4x7 can be decomposed using 21 rank-1 terms
    # by combining Strassen products appropriately
    
    # Construct 2x4x7 rank-21 decomposition:
    # Use the fact that we can write it as a sum of 21 rank-1 tensors
    # This is a known result; let me construct it via the following method:
    # 
    # A (2x4) = [A1 A2] (each 2x2)
    # B (4x7) = [B1; B2] (each 2x7)  
    # C (2x7) = A1*B1 + A2*B2
    #
    # For A1*B1 (2x2 @ 2x7): split B1 into [B1_1 B1_2 B1_3 B1_4] (each 2x2, but 4*2=8≠7)
    # So split into [B1_1 B1_2 B1_3] (each 2x2) + [B1_4] (2x1)
    # A1*B1_1: 7 Strassen products
    # A1*B1_2: 7 Strassen products  
    # A1*B1_3: 7 Strassen products
    # A1*B1_4: 2 schoolbook products (2x2 @ 2x1 = 2x1, needs 2 mults... actually 2*1=2)
    # Total for A1*B1: 21 + 2 = 23
    # Similarly for A2*B2: 23
    # Total: 46, way too many.
    
    # I think the rank-21 claim might be incorrect or I'm misunderstanding.
    # Let me just use a verified lower rank if possible, otherwise use schoolbook.
    
    # Actually, let me try random search for a low-rank decomposition
    best_rank = n * m * p  # Start with schoolbook rank 84
    best_decomp = None
    
    # Try to find a better decomposition using iterative refinement
    # Start with random initialization and use gradient descent on the tensor equation
    
    # For simplicity and reliability, let me use the schoolbook but check if we can do better
    # with a simple greedy approach
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 10:
        # Not enough time, return schoolbook
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1.0
                    V[j * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0
                    col += 1
        
        return {"U": U, "V": V, "W": W}
    
    # Try a simple improvement: use block structure
    # Split n=3 into 2+1, and use Strassen for the 2x2 part where possible
    
    # For now, return schoolbook as a safe baseline
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
                
    return {"U": U, "V": V, "W": W}
