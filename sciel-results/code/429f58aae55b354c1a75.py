"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # Use a block decomposition approach:
    # Split B (4x9) into B1 (4x4) and B2 (4x5)
    # C = A*B1 + A*B2
    # 
    # For A*B1 (4x4x4): use Strassen's algorithm (rank 28)
    # For A*B2 (4x4x5): split into A*B2a (4x4x4) + A*B2b (4x4x1)
    #   A*B2a: Strassen (rank 28)
    #   A*B2b: schoolbook (rank 4)
    # Total rank: 28 + 28 + 4 = 60
    
    # First, let's implement Strassen for 2x2x2
    # Then build up to 4x4x4 using Strassen recursively
    
    # Strassen 2x2x2 factors:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A10)*B11
    # M6 = (A01-A00)*(B00+B10)
    # M7 = (A10-A11)*(B01+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For 4x4x4, we can use Strassen recursively:
    # Split each 4x4 matrix into four 2x2 blocks
    # Use Strassen to get rank 7*7 = 49? No, that's not right.
    # Actually, for 4x4x4, we can use Strassen on 2x2 blocks:
    # The 4x4 matrix multiplication can be decomposed into 7 Strassen products
    # of 2x2 matrices, each of which has rank 7.
    # Total rank: 7*7 = 49? No, that's not correct either.
    
    # Let me think more carefully. For 4x4x4:
    # We can view it as multiplying two 4x4 matrices.
    # Using Strassen recursively: split into 2x2 blocks.
    # Each 2x2x2 multiplication has rank 7.
    # The 4x4x4 multiplication requires 7 such 2x2x2 multiplications.
    # So total rank = 7 * 7 = 49.
    
    # Wait, that's not right. The rank of the tensor for 4x4x4 matrix multiplication
    # is known to be at most 49 using Strassen recursively, but actually the best known
    # upper bound is lower. Let me just use a known decomposition.
    
    # Actually, let me just construct a valid decomposition with rank < 144.
    # The simplest approach: use schoolbook but be smart about it.
    
    # For 4x4x9, schoolbook gives rank 4*4*9 = 144.
    # We can do better by exploiting structure.
    
    # Let's try: split B into three 4x3 blocks.
    # C = A*B1 + A*B2 + A*B3 where Bi are 4x3.
    # Each A*Bi is 4x4x3, schoolbook rank 4*4*3 = 48.
    # Total: 3*48 = 144. Not helpful.
    
    # Better: use the fact that 4x4x4 has a known decomposition.
    # The minimal rank for 4x4x4 is not known exactly, but we know it's <= 49
    # using Strassen recursively.
    
    # For simplicity, let's just return a valid decomposition.
    # I'll use a direct construction with rank 132 as a starting point,
    # then try to improve.
    
    # Actually, let me just implement a proper Strassen-based decomposition.
    
    # Helper: get Strassen factors for 2x2x2
    def strassen_2x2():
        """Return U, V, W for 2x2x2 Strassen decomposition."""
        rank = 7
        U = np.zeros((4, rank))
        V = np.zeros((4, rank))
        W = np.zeros((4, rank))
        
        # M1 = (A00+A11)*(B00+B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2 = (A10+A11)*B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1
        
        # M3 = A00*(B01-B11)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1
        
        # M4 = A11*(B10-B00)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[2, 3] = 1
        
        # M5 = (A00+A10)*B11
        U[0, 4] = 1; U[2, 4] = 1
        V[3, 4] = 1
        W[1, 4] = 1; W[3, 4] = 1
        
        # M6 = (A01-A00)*(B00+B10)
        U[1, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[2, 5] = 1
        W[1, 5] = 1
        
        # M7 = (A10-A11)*(B01+B11)
        U[2, 6] = 1; U[3, 6] = -1
        V[1, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1; W[3, 6] = 1
        
        return U, V, W
    
    # Now build 4x4x4 using Strassen recursively.
    # Split each 4x4 into four 2x2 blocks.
    # Use Strassen on the 2x2 blocks.
    
    def strassen_4x4():
        """Return U, V, W for 4x4x4 Strassen decomposition."""
        # This is complex. Let me just use a known result.
        # The rank of 4x4x4 matrix multiplication is at most 49 using Strassen.
        # But constructing it explicitly is tedious.
        
        # For now, let's just use schoolbook for 4x4x4 (rank 64)
        # and combine with the rest.
        
        rank = 64
        U = np.zeros((16, rank))
        V = np.zeros((16, rank))
        W = np.zeros((16, rank))
        
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U[i*4+j, col] = 1
                    V[j*4+k, col] = 1
                    W[k*4+i, col] = 1
                    col += 1
        
        return U, V, W, rank
    
    # Given the complexity, let me just return a valid decomposition.
    # I'll use a simple but valid approach.
    
    # Split B (4x9) into B1 (4x4) and B2 (4x5)
    # C = A*B1 + A*B2
    
    # For A*B1: use schoolbook (rank 64)
    # For A*B2: split into B2a (4x4) and B2b (4x1)
    #   A*B2a: schoolbook (rank 64)
    #   A*B2b: schoolbook (rank 4)
    # Total: 64 + 64 + 4 = 132
    
    rank = 132
    U = np.zeros((16, rank))
    V = np.zeros((36, rank))
    W = np.zeros((36, rank))
    
    col = 0
    # A*B1: B1 is columns 0-3 of B
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U[i*4+j, col] = 1
                V[j*9+k, col] = 1
                W[k*4+i, col] = 1
                col += 1
    
    # A*B2a: B2a is columns 4-7 of B
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U[i*4+j, col] = 1
                V[j*9+(4+k), col] = 1
                W[k*4+i, col] = 1
                col += 1
    
    # A*B2b: B2b is column 8 of B
    for i in range(4):
        for j in range(4):
            U[i*4+j, col] = 1
            V[j*9+8, col] = 1
            W[8*4+i, col] = 1
            col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

