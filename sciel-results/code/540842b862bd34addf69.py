import time
import sys
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    # For 3x5x6, the schoolbook rank is 90.
    # We will use a hybrid approach:
    # 1. Use Strassen for the 2x2x2 blocks to reduce rank.
    # 2. Use schoolbook for the remaining 1x5x6 or similar strips.
    # 3. Or, more generally, use a known low-rank decomposition if available.
    # However, for small matrices, exact rank is often the schoolbook rank unless a specific low-rank algorithm is known.
    # For 3x5x6, no widely known low-rank algorithm beats 90 significantly in standard literature without large overhead or non-integer entries that might not be "exact" in the required sense (though Strassen is exact).
    
    # Let's try to construct a decomposition using Strassen for 2x2x2 blocks.
    # We can split 3x5x6 into:
    # - 2x2x2 (rank 7)
    # - 2x2x4 (rank ?)
    # - 1x5x6 (rank 30)
    # This is getting complex to optimize manually.
    
    # Instead, let's use the standard schoolbook decomposition but check if we can merge any terms.
    # The tensor is T = sum_{i,j,k} e_i (in A) * e_j (in B) * e_k (in C) where C_{ik} = sum_j A_{ij} B_{jk}.
    # The rank is the minimum number of terms in a bilinear decomposition.
    
    # For 3x5x6, the schoolbook rank is 90.
    # Is there a known lower rank?
    # For 2x2x2, rank is 7.
    # For 3x3x3, rank is 23 (Landsberg).
    # For 3x5x6, we can view it as 3x5x2 + 3x5x4? No.
    # We can view it as (2+1) x 5 x (2+4)?
    
    # Let's try a direct construction using Strassen for 2x2x2 blocks embedded in the larger matrix.
    # Split A into A1 (2x5) and A2 (1x5).
    # Split B into B1 (5x2) and B2 (5x4).
    # Split C into C1 (2x2), C2 (2x4), C3 (1x2), C4 (1x4).
    
    # C1 = A1 * B1 -> 2x2x2, rank 7
    # C2 = A1 * B2 -> 2x5x4
    # C3 = A2 * B1 -> 1x5x2
    # C4 = A2 * B2 -> 1x5x4
    
    # Rank(C1) = 7
    # Rank(C2) = 2*5*4 = 40 (schoolbook) or lower?
    # Rank(C3) = 1*5*2 = 10
    # Rank(C4) = 1*5*4 = 20
    # Total = 7 + 40 + 10 + 20 = 77.
    
    # Can we do better for C2 (2x5x4)?
    # 2x5x4 can be split into 2x2x2 + 2x3x2?
    # 2x2x2 rank 7.
    # 2x3x2 rank 12 (schoolbook) or lower?
    # 2x3x2 is small. Schoolbook is 12.
    # Is there a low rank for 2x3x2?
    # T = sum_{i=0..1, j=0..2, k=0..1} e_i e_j e_k.
    # This is equivalent to multiplying a 2x3 matrix by a 3x2 matrix.
    # Rank is 12.
    # So C2 rank = 7 + 12 = 19.
    # Total = 7 + 19 + 10 + 20 = 56.
    
    # Let's verify this construction.
    # We need to build the U, V, W matrices for the whole 3x5x6.
    
    # Let's define a helper to build the schoolbook decomposition for n x m x p.
    def build_schoolbook(n, m, p):
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank
    
    # Helper to build Strassen for 2x2x2.
    def build_strassen_2x2x2():
        # 7 terms
        # M1 = (A11 + A22)(B21 - B22)
        # M2 = (A11 + A12)B22
        # M3 = A11(B11 - B21)
        # M4 = A22(B12 - B11)
        # M5 = (A11 + A21)B11
        # M6 = (A12 - A22)(B21 + B22)
        # M7 = (A11 - A12)(B11 + B12)
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
        # B indices: 00, 01, 10, 11 -> 0, 1, 2, 3
        # C indices: 00, 01, 10, 11 -> 0, 1, 2, 3
        
        # M1
        U[2, 0] = 1; U[3, 0] = 1
        V[2, 0] = 1; V[3, 0] = -1
        W[2, 0] = 1; W[3, 0] = 1 # C10, C11
        
        # M2
        U[0, 1] = 1; U[1, 1] = 1
        V[3, 1] = 1
        W[1, 1] = 1; W[3, 1] = 1 # C01, C11
        
        # M3
        U[0, 2] = 1
        V[0, 2] = 1; V[2, 2] = -1
        W[0, 2] = 1; W[2, 2] = 1 # C00, C10
        
        # M4
        U[3, 3] = 1
        V[1, 3] = 1; V[0, 3] = -1
        W[1, 3] = 1; W[3, 3] = 1 # C01, C11
        
        # M5
        U[0, 4] = 1; U[2, 4] = 1
        V[0, 4] = 1
        W[0, 4] = 1; W[2, 4] = 1 # C00, C10
        
        # M6
        U[1, 5] = 1; U[3, 5] = -1
        V[2, 5] = 1; V[3, 5] = 1
        W[2, 5] = 1; W[3, 5] = 1 # C10, C11
        
        # M7
        U[0, 6] = 1; U[1, 6] = -1
        V[0, 6] = 1; V[1, 6] = 1
        W[0, 6] = 1; W[1, 6] = 1 # C00, C01
        
        return U, V, W, 7
    
    # We will manually construct the 3x5x6 decomposition using the block strategy.
    # Blocks:
    # A: 3x5. Split into A1 (2x5) and A2 (1x5).
    # B: 5x6. Split into B1 (5x2) and B2 (5x4).
    # C: 3x6. Split into C1 (2x2), C2 (2x4), C3 (1x2), C4 (1x4).
    
    # Sub-problems:
    # 1. C1 = A1 * B1 (2x2x2). Rank 7.
    # 2. C2 = A1 * B2 (2x5x4). Split B2 into B21 (5x2) and B22 (5x2).
    #    C2 = A1 * B21 (2x5x2) + A1 * B22 (2x5x2).
    #    2x5x2 schoolbook rank 20.
    #    Can we do better for 2x5x2?
    #    2x5x2 = 2x2x2 (rank 7) + 2x3x2 (rank 12). Total 19.
    #    So C2 rank = 19 + 19 = 38.
    # 3. C3 = A2 * B1 (1x5x2). Rank 10.
    # 4. C4 = A2 * B2 (1x5x4). Rank 20.
    
    # Total rank = 7 + 38 + 10 + 20 = 75.
    
    # Let's refine 2x5x2.
    # A is 2x5. B is 5x2.
    # Split B into B1 (5x2) and B2 (5x0)? No.
    # Split B into B1 (5x1) and B2 (5x1)? No, B is 5x2.
    # Split B into B1 (5x2) is the whole thing.
    # We can split the 5 rows of B into 2 and 3.
    # B = [B_top (2x2); B_bottom (3x2)]? No, B is 5x2.
    # B rows are indexed 0..4.
    # Split rows into 0..1 (2 rows) and 2..4 (3 rows).
    # A columns corresponding to these rows: A[:, 0..1] and A[:, 2..4].
    # C = A[:, 0..1] * B[0..1, :] + A[:, 2..4] * B[2..4, :]
    # First term: 2x2x2. Rank 7.
    # Second term: 2x3x2. Rank 12.
    # Total 19.
    
    # So for 2x5x2, we use rank 19.
    
    # Now we need to assemble the full U, V, W matrices.
    # This is complex to do by hand in code without a general assembler.
    # Given the time constraints and complexity, and that 75 is a significant improvement over 90,
    # we will implement this specific decomposition.
    
    # However, writing the assembler for this specific case is error-prone.
    # Let's stick to the schoolbook if we can't guarantee correctness quickly.
    # But 75 is much better.
    
    # Let's try a simpler improvement:
    # Use Strassen for 2x2x2 blocks wherever possible.
    # 3x5x6.
    # Split 3 into 2+1.
    # Split 6 into 2+2+2.
    # 2x5x2 (rank 19) * 3 = 57.
    # 1x5x2 (rank 10) * 3 = 30.
    # Total 87.
    # This is worse than 75.
    
    # Let's go with the 75 rank construction.
    # We will build the matrices for each block and then assemble them.
    
    def assemble(U_blocks, V_blocks, W_blocks, row_offsets_U, row_offsets_V, row_offsets_W, col_offset):
        pass # Too complex to do generally here.
        
    # Given the risk of bugs in a complex manual assembly, and that the previous program scored 1.0 (rank 90),
    # a safer bet is to use the schoolbook rank but with a slightly optimized construction if possible.
    # Or, use the known rank for 3x3x3 (23) and pad.
    # 3x5x6 can be seen as 3x3x3 + 3x3x3 + 3x3x3 + ...?
    # 5 = 3+2, 6 = 3+3.
    # 3x3x3 (23) + 3x3x3 (23) + 3x2x3 (18) + 3x2x3 (18) = 82.
    # This is better than 90.
    
    # Let's try to implement the 3x3x3 rank 23 decomposition.
    # This is the Landsberg decomposition.
    # It is quite complex to write from scratch.
    
    # Alternative: Use the schoolbook but check if we can merge any terms.
    # For 3x5x6, the terms are independent in the schoolbook.
    
    # Let's return the schoolbook decomposition for now, but with a clean implementation.
    # If we have time, we can try to improve.
    
    # Actually, the prompt says "Keep the function names and signatures".
    # And "score is the rank".
    
    # Let's try the 82 rank construction (3x3x3 + 3x3x3 + 3x2x3 + 3x2x3).
    # We need a 3x3x3 rank 23 decomposition.
    # I will use a known one.
    
    # Due to complexity, I will stick to the schoolbook for safety, but I will try to see if I can find a simple low-rank decomposition for 3x5x6.
    
    # Wait, the inspiration says "score 1.0000" for the current program.
    # The current program uses a DP which might find 90.
    
    # Let's try to implement the 75 rank construction carefully.
    
    # I will define the blocks explicitly.
    
    # Block 1: 2x2x2 (Strassen)
    U1, V1, W1, r1 = build_strassen_2x2x2()
    
    # Block 2: 2x5x2 (Split into 2x2x2 + 2x3x2)
    # 2x2x2 part
    U2a, V2a, W2a, r2a = build_strassen_2x2x2()
    # 2x3x2 part (Schoolbook)
    U2b, V2b, W2b, r2b = build_schoolbook(2, 3, 2)
    
    # Block 3: 2x5x2 (Another one for the second 2x2 block of B2)
    U3a, V3a, W3a, r3a = build_strassen_2x2x2()
    U3b, V3b, W3b, r3b = build_schoolbook(2, 3, 2)
    
    # Block 4: 1x5x2 (Schoolbook)
    U4, V4, W4, r4 = build_schoolbook(1, 5, 2)
    
    # Block 5: 1x5x4 (Schoolbook)
    U5, V5, W5, r5 = build_schoolbook(1, 5, 4)
    
    # Total rank
    total_rank = r1 + r2a + r2b + r3a + r3b + r4 + r5
    # 7 + 7 + 12 + 7 + 12 + 10 + 20 = 75
    
    # Now assemble U, V, W
    U = np.zeros((3 * 5, total_rank))
    V = np.zeros((5 * 6, total_rank))
    W = np.zeros((6 * 3, total_rank))
    
    col = 0
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # A rows: 0:2, cols 0:2. U rows: 0:2 * 5 + 0:2 = 0:2, 5:7? No.
    # U row index: i * m + j.
    # i=0, j=0..1 -> 0, 1
    # i=1, j=0..1 -> 5, 6
    # So U rows for A1 (2x5) cols 0:2 are 0,1,5,6.
    # But Strassen U is 4x7. It assumes A is 2x2.
    # We need to map the 2x2 A block to the 4x7 U matrix.
    # The 2x2 A block is A[0:2, 0:2].
    # Rows of U correspond to A[0,0], A[0,1], A[1,0], A[1,1].
    # In the full U, these are at indices:
    # A[0,0] -> 0*5+0 = 0
    # A[0,1] -> 0*5+1 = 1
    # A[1,0] -> 1*5+0 = 5
    # A[1,1] -> 1*5+1 = 6
    
    # We will copy the U1 values to the correct rows.
    # U1 is 4x7. Rows 0,1,2,3 correspond to A[0,0], A[0,1], A[1,0], A[1,1].
    # Full U rows: 0, 1, 5, 6.
    
    # This mapping is getting complicated.
    # Given the time, I will return the schoolbook decomposition to ensure correctness.
    
    U, V, W, rank = build_schoolbook(n, m, p)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
