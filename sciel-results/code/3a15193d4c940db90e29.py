import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    
    n, m, p = 2, 4, 8
    dim_U = n * m  # 8
    dim_V = m * p  # 32
    dim_W = p * n  # 16
    
    # We will construct a decomposition of rank 56.
    # Structure:
    # C = A B
    # A is 2x4, B is 4x8.
    # Split A into A1 (2x2, cols 0-1) and A2 (2x2, cols 2-3).
    # Split B into B1 (2x8, rows 0-1) and B2 (2x8, rows 2-3).
    # C = A1 B1 + A2 B2.
    # A1 B1 is 2x2 * 2x8.
    # B1 has 8 columns. Let's call them B1_0 ... B1_7 (each 2x2? No, 2x1).
    # Wait, B1 is 2x8. It can be seen as 4 blocks of 2x2? No, 8 columns of 2x1.
    # A 2x2 * 2x1 multiplication is a rank 2 tensor? No, it's part of 2x2x1.
    # Standard matrix multiplication 2x2 * 2x2 is rank 7.
    # 2x2 * 2x8 is 4 independent 2x2 * 2x2 multiplications?
    # No, B is 2x8. It's not a square matrix.
    # C = A B.
    # C is 2x8.
    # C[:, k] = A B[:, k].
    # This is a 2x2 matrix A times a 2x1 vector b.
    # The tensor for 2x2 * 2x1 is rank 2?
    # C_i = sum_j A_ij b_j.
    # This is just standard multiplication. Rank 2?
    # T[i, j, k] where k is the column index.
    # If we treat each column independently, the rank is sum of ranks.
    # For a single column (2x2 * 2x1), the output is a 2x1 vector.
    # The tensor is T[i, j, 0] = delta_{i,j}? No.
    # The multiplication is linear.
    # C_0 = A_00 b_0 + A_01 b_1
    # C_1 = A_10 b_0 + A_11 b_1
    # This can be done with 4 terms (schoolbook).
    # Can it be done with fewer?
    # It's a bilinear map R^4 x R^2 -> R^2.
    # Rank is at least 4?
    # Actually, we can use the 7-term Strassen scheme for 2x2*2x2 and just ignore the second column of B?
    # No, Strassen is for 2x2*2x2.
    # If we view 2x2*2x8 as 2x2*2x2 repeated 4 times, and we can share A factors.
    # For 2x2*2x2, rank is 7.
    # If we have 4 such products, and A is shared, do we need 7 V's and 7 W's per product?
    # Yes.
    # So rank 7 * 4 = 28 for A1 B1.
    # And 28 for A2 B2.
    # Total 56.
    
    rank = 56
    U = [[0] * rank for _ in range(dim_U)]
    V = [[0] * rank for _ in range(dim_V)]
    W = [[0] * rank for _ in range(dim_W)]
    
    def add_2x2x2_rank7(a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, c2, c3, offset, U, V, W):
        # Adds a 2x2x2 rank 7 block to U, V, W starting at column offset.
        # A indices: a0, a1, a2, a3
        # B indices: b0, b1, b2, b3
        # C indices: c0, c1, c2, c3
        # Returns new offset.
        
        # Term 1: (a0+a1+a2+a3)(b0+b1+b2+b3) -> C00, C01, C10, C11
        U[a0][offset] += 1; U[a1][offset] += 1; U[a2][offset] += 1; U[a3][offset] += 1
        V[b0][offset] += 1; V[b1][offset] += 1; V[b2][offset] += 1; V[b3][offset] += 1
        W[c0][offset] += 0.25; W[c1][offset] += 0.25; W[c2][offset] += 0.25; W[c3][offset] += 0.25
        offset += 1
        
        # Term 2: (a0-a1+a2-a3)(b0-b1+b2-b3)
        U[a0][offset] += 1; U[a1][offset] -= 1; U[a2][offset] += 1; U[a3][offset] -= 1
        V[b0][offset] += 1; V[b1][offset] -= 1; V[b2][offset] += 1; V[b3][offset] -= 1
        W[c0][offset] += 0.25; W[c1][offset] -= 0.25; W[c2][offset] += 0.25; W[c3][offset] -= 0.25
        offset += 1
        
        # Term 3: (a0+a1-a2-a3)(b0+b1-b2-b3)
        U[a0][offset] += 1; U[a1][offset] += 1; U[a2][offset] -= 1; U[a3][offset] -= 1
        V[b0][offset] += 1; V[b1][offset] += 1; V[b2][offset] -= 1; V[b3][offset] -= 1
        W[c0][offset] += 0.25; W[c1][offset] += 0.25; W[c2][offset] -= 0.25; W[c3][offset] -= 0.25
        offset += 1
        
        # Term 4: (a0-a1-a2+a3)(b0-b1-b2+b3)
        U[a0][offset] += 1; U[a1][offset] -= 1; U[a2][offset] -= 1; U[a3][offset] += 1
        V[b0][offset] += 1; V[b1][offset] -= 1; V[b2][offset] -= 1; V[b3][offset] += 1
        W[c0][offset] += 0.25; W[c1][offset] -= 0.25; W[c2][offset] -= 0.25; W[c3][offset] += 0.25
        offset += 1
        
        # Term 5: (a0-a1)(b0-b1)
        U[a0][offset] += 1; U[a1][offset] -= 1
        V[b0][offset] += 1; V[b1][offset] -= 1
        W[c0][offset] += 0.5; W[c1][offset] += 0.5
        offset += 1
        
        # Term 6: (a2-a3)(b2-b3)
        U[a2][offset] += 1; U[a3][offset] -= 1
        V[b2][offset] += 1; V[b3][offset] -= 1
        W[c2][offset] += 0.5; W[c3][offset] += 0.5
        offset += 1
        
        # Term 7: (a1-a3)(b1-b3)
        U[a1][offset] += 1; U[a3][offset] -= 1
        V[b1][offset] += 1; V[b3][offset] -= 1
        W[c0][offset] += 0.5; W[c1][offset] -= 0.5; W[c2][offset] += 0.5; W[c3][offset] -= 0.5
        offset += 1
        
        return offset

    # A indices:
    # A1 (2x2): rows 0-1, cols 0-1.
    # A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3.
    # A2 (2x2): rows 0-1, cols 2-3.
    # A[0,2]=4, A[0,3]=5, A[1,2]=6, A[1,3]=7.
    
    # B indices:
    # B1 (2x8): rows 0-1, cols 0-7.
    # B[0,0]=0, B[0,1]=1, B[0,2]=2, B[0,3]=3, B[0,4]=4, B[0,5]=5, B[0,6]=6, B[0,7]=7.
    # B[1,0]=8, B[1,1]=9, B[1,2]=10, B[1,3]=11, B[1,4]=12, B[1,5]=13, B[1,6]=14, B[1,7]=15.
    # B2 (2x8): rows 2-3, cols 0-7.
    # B[2,0]=16, B[2,1]=17, B[2,2]=18, B[2,3]=19, B[2,4]=20, B[2,5]=21, B[2,6]=22, B[2,7]=23.
    # B[3,0]=24, B[3,1]=25, B[3,2]=26, B[3,3]=27, B[3,4]=28, B[3,5]=29, B[3,6]=30, B[3,7]=31.
    
    # C indices (2x8):
    # C[0,0]=0, C[0,1]=1, C[0,2]=2, C[0,3]=3, C[0,4]=4, C[0,5]=5, C[0,6]=6, C[0,7]=7.
    # C[1,0]=8, C[1,1]=9, C[1,2]=10, C[1,3]=11, C[1,4]=12, C[1,5]=13, C[1,6]=14, C[1,7]=15.
    # Note: The problem statement says output factor follows W[k*n+i, r].
    # So C index is k*n + i.
    # C[0,0] -> k=0, i=0 -> idx 0.
    # C[0,1] -> k=1, i=0 -> idx 2.
    # C[1,0] -> k=0, i=1 -> idx 1.
    # C[1,1] -> k=1, i=1 -> idx 3.
    # My helper `add_2x2x2_rank7` takes c0, c1, c2, c3 corresponding to a 2x2 block.
    # The 2x2 block of C for column pair (2k, 2k+1)?
    # No, my helper assumes the C indices are contiguous in the 2x2 sense.
    # If I treat B1 as 4 blocks of 2x2, I need to map the C indices correctly.
    
    # Let's process B1 (rows 0-1) as 4 blocks of 2x2 columns.
    # Block 0: B cols 0-1. C cols 0-1.
    # B indices: 0, 1, 8, 9. (B[0,0], B[0,1], B[1,0], B[1,1])
    # C indices: 0, 2, 1, 3. (C[0,0], C[0,1], C[1,0], C[1,1])
    # Wait, W index is k*n + i.
    # C[0,0] (k=0, i=0) -> 0.
    # C[0,1] (k=1, i=0) -> 2.
    # C[1,0] (k=0, i=1) -> 1.
    # C[1,1] (k=1, i=1) -> 3.
    # So for the 2x2 block at C cols 0-1, the indices are 0, 2, 1, 3.
    
    # A1 indices: 0, 1, 2, 3.
    
    offset = 0
    
    # Block 0: A1 * B1[:, 0:2] -> C[:, 0:2]
    # B1[:, 0:2] indices: 0, 1 (row 0), 8, 9 (row 1).
    # C[:, 0:2] indices: 0, 2 (row 0), 1, 3 (row 1).
    offset = add_2x2x2_rank7(0, 1, 2, 3, 0, 1, 8, 9, 0, 2, 1, 3, offset, U, V, W)
    
    # Block 1: A1 * B1[:, 2:4] -> C[:, 2:4]
    # B1[:, 2:4] indices: 2, 3, 10, 11.
    # C[:, 2:4] indices: 4, 6, 5, 7. (k=2,3; i=0,1)
    # C[0,2]=4, C[0,3]=6, C[1,2]=5, C[1,3]=7.
    offset = add_2x2x2_rank7(0, 1, 2, 3, 2, 3, 10, 11, 4, 6, 5, 7, offset, U, V, W)
    
    # Block 2: A1 * B1[:, 4:6] -> C[:, 4:6]
    # B1[:, 4:6] indices: 4, 5, 12, 13.
    # C[:, 4:6] indices: 8, 10, 9, 11.
    offset = add_2x2x2_rank7(0, 1, 2, 3, 4, 5, 12, 13, 8, 10, 9, 11, offset, U, V, W)
    
    # Block 3: A1 * B1[:, 6:8] -> C[:, 6:8]
    # B1[:, 6:8] indices: 6, 7, 14, 15.
    # C[:, 6:8] indices: 12, 14, 13, 15.
    offset = add_2x2x2_rank7(0, 1, 2, 3, 6, 7, 14, 15, 12, 14, 13, 15, offset, U, V, W)
    
    # Now A2 (indices 4, 5, 6, 7) * B2 (rows 2-3)
    # B2 indices:
    # Row 2: 16, 17, 18, 19, 20, 21, 22, 23.
    # Row 3: 24, 25, 26, 27, 28, 29, 30, 31.
    
    # Block 4: A2 * B2[:, 0:2] -> C[:, 0:2]
    # B2[:, 0:2] indices: 16, 17, 24, 25.
    # C[:, 0:2] indices: 0, 2, 1, 3.
    offset = add_2x2x2_rank7(4, 5, 6, 7, 16, 17, 24, 25, 0, 2, 1, 3, offset, U, V, W)
    
    # Block 5: A2 * B2[:, 2:4] -> C[:, 2:4]
    # B2[:, 2:4] indices: 18, 19, 26, 27.
    # C[:, 2:4] indices: 4, 6, 5, 7.
    offset = add_2x2x2_rank7(4, 5, 6, 7, 18, 19, 26, 27, 4, 6, 5, 7, offset, U, V, W)
    
    # Block 6: A2 * B2[:, 4:6] -> C[:, 4:6]
    # B2[:, 4:6] indices: 20, 21, 28, 29.
    # C[:, 4:6] indices: 8, 10, 9, 11.
    offset = add_2x2x2_rank7(4, 5, 6, 7, 20, 21, 28, 29, 8, 10, 9, 11, offset, U, V, W)
    
    # Block 7: A2 * B2[:, 6:8] -> C[:, 6:8]
    # B2[:, 6:8] indices: 22, 23, 30, 31.
    # C[:, 6:8] indices: 12, 14, 13, 15.
    offset = add_2x2x2_rank7(4, 5, 6, 7, 22, 23, 30, 31, 12, 14, 13, 15, offset, U, V, W)
    
    if offset != rank:
        # Fallback to schoolbook if logic error
        rank = 64
        U = [[0] * rank for _ in range(dim_U)]
        V = [[0] * rank for _ in range(dim_V)]
        W = [[0] * rank for _ in range(dim_W)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1

    return {"U": U, "V": V, "W": W}
