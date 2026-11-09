"""Matrix multiplication 4x5x5: find a rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a tensor decomposition for 4x5 * 5x5 matrix multiplication.
    
    Uses block decomposition:
    - Split the 4x5 matrix into 2x2 and 2x3 blocks (rows)
    - Split the 5x5 matrix into 2x2, 2x3, 3x2, 3x3 blocks (cols/rows)
    - For 2x2x2 sub-problems, use Strassen's rank-7 decomposition
    - For other sub-problems, use schoolbook (rank n*m*p)
    
    Blocks:
    Row split of A (4x5): rows 0:2 and 2:4
    Col split of B (5x5): cols 0:2 and 2:5
    Row split of B (5x5): rows 0:2 and 2:5
    
    Sub-blocks (i, j) where i is row-block of A, j is (row-block, col-block) of B:
    - (0, (0,0)): 2x2 * 2x2 -> 2x2x2, rank 7 (Strassen)
    - (0, (0,1)): 2x2 * 2x3 -> 2x2x3, rank 12 (schoolbook)
    - (0, (1,0)): 2x3 * 3x2 -> 2x3x2, rank 12 (schoolbook)
    - (0, (1,1)): 2x3 * 3x3 -> 2x3x3, rank 18 (schoolbook)
    - (1, (0,0)): 2x2 * 2x2 -> 2x2x2, rank 7 (Strassen)
    - (1, (0,1)): 2x2 * 2x3 -> 2x2x3, rank 12 (schoolbook)
    - (1, (1,0)): 2x3 * 3x2 -> 2x3x2, rank 12 (schoolbook)
    - (1, (1,1)): 2x3 * 3x3 -> 2x3x3, rank 18 (schoolbook)
    
    Total rank = 7 + 12 + 12 + 18 + 7 + 12 + 12 + 18 = 98
    
    Wait, that's only a small improvement. Let me reconsider.
    
    Actually, let's split differently. Split 4 = 2+2 and 5 = 3+2.
    
    A is 4x5, split rows into 2+2.
    B is 5x5, split rows into 3+2 and cols into 3+2.
    
    Sub-blocks:
    - A[0:2] * B[0:3, 0:3] -> 2x3x3, rank 18
    - A[0:2] * B[0:3, 3:5] -> 2x3x2, rank 12
    - A[0:2] * B[3:5, 0:3] -> 2x2x3, rank 12
    - A[0:2] * B[3:5, 3:5] -> 2x2x2, rank 7 (Strassen)
    - A[2:4] * B[0:3, 0:3] -> 2x3x3, rank 18
    - A[2:4] * B[0:3, 3:5] -> 2x3x2, rank 12
    - A[2:4] * B[3:5, 0:3] -> 2x2x3, rank 12
    - A[2:4] * B[3:5, 3:5] -> 2x2x2, rank 7 (Strassen)
    
    Total = 18+12+12+7+18+12+12+7 = 98
    
    Hmm, still 98. The previous knowledge said Strassen on 4x2x2x2 gave 96.
    Let me try splitting 4 into 2+2 and 5 into 2+3 for both dimensions of B.
    
    Actually, let's just implement the 4x2x2x2 Strassen reduction properly.
    
    Split A (4x5) into two 2x5 blocks vertically.
    Split B (5x5) into two 2x5 and two 3x5... no, B is 5x5.
    
    Let me think again. We want to find 2x2x2 sub-problems.
    
    If we split A into 2+2 rows, and B into 2+2+1 rows and 2+2+1 cols, we get
    2x2, 2x2, 2x1, 2x2, 2x2, 2x1, 1x2, 1x2, 1x1 sub-blocks of B.
    
    The 2x2x2 sub-problems come from A_block (2x?) * B_block (?x?) where both are 2x2.
    
    Let's split B's 5 rows into 2+2+1 and 5 cols into 2+2+1.
    
    Sub-blocks of B:
    - B[0:2, 0:2]: 2x2
    - B[0:2, 2:4]: 2x2
    - B[0:2, 4:5]: 2x1
    - B[2:4, 0:2]: 2x2
    - B[2:4, 2:4]: 2x2
    - B[2:4, 4:5]: 2x1
    - B[4:5, 0:2]: 1x2
    - B[4:5, 2:4]: 1x2
    - B[4:5, 4:5]: 1x1
    
    A blocks: A[0:2] (2x5), A[2:4] (2x5)
    
    For A[0:2] * B:
    - A[0:2] * B[0:2, 0:2]: 2x5 * 2x2... wait, A is 4x5, so A[0:2] is 2x5.
      B[0:2, 0:2] is 2x2. But A[0:2] has 5 columns, B[0:2, 0:2] has 2 rows.
      This doesn't align!
    
    I see the issue. A is 4x5, B is 5x5. The inner dimension is 5.
    So A[i,j] * B[j,k] where i in [0,4), j in [0,5), k in [0,5).
    
    To do block decomposition, I need to split the inner dimension j.
    Split j into 2+2+1.
    
    A (4x5) -> A[0:2] (4x2), A[2:4] (4x2), A[4:5] (4x1)
    B (5x5) -> B[0:2] (2x5), B[2:4] (2x5), B[4:5] (1x5)
    
    But this doesn't give me 2x2x2 sub-problems directly.
    
    Let me try a different approach. I'll just implement the known Strassen reduction
    for the 2x2x2 parts where they naturally occur.
    
    Actually, the simplest safe improvement is to use the 4x2x2x2 partition:
    Split A's 4 rows into 2+2, and B's 5 cols into 2+2+1, and B's 5 rows into 2+2+1.
    
    But the inner dimension must match. So I need to split the inner dimension consistently.
    
    Let's just do: split inner dimension 5 into 2+2+1.
    A: 4x(2+2+1) = (4x2) + (4x2) + (4x1)
    B: (2+2+1)x5 = (2x5) + (2x5) + (1x5)
    
    Products:
    A[0:2] * B[0:2]: 4x2 * 2x5 = 4x5, rank 4*2*5=40 schoolbook, but we can do better
    A[0:2] * B[2:4]: 4x2 * 2x5 = 4x5, rank 40
    A[0:2] * B[4:5]: 4x1 * 1x5 = 4x5, rank 4*1*5=20
    A[2:4] * B[0:2]: 4x2 * 2x5 = 4x5, rank 40
    A[2:4] * B[2:4]: 4x2 * 2x5 = 4x5, rank 40
    A[2:4] * B[4:5]: 4x1 * 1x5 = 4x5, rank 20
    
    This is getting complicated. Let me just use the simple approach from the knowledge:
    "Replacing 4 disjoint 2x2x2 standard sub-problems with Strassen's 7-term decomposition
    reduces the tensor rank from 100 to 96."
    
    So I need to identify 4 disjoint 2x2x2 sub-problems within the 4x5x5 tensor.
    
    The 4x5x5 tensor has indices (i, j, k) where i in [0,4), j in [0,5), k in [0,5).
    A 2x2x2 sub-problem uses 2 values of i, 2 values of j, 2 values of k.
    
    I can find 4 disjoint 2x2x2 sub-problems if I partition i into {0,1} and {2,3},
    and find disjoint pairs of j and k.
    
    For i in {0,1}: I need 2x2x2 sub-problems using j pairs and k pairs from [0,5).
    I can use (j,k) = ({0,1},{0,1}), ({0,1},{2,3}), ({2,3},{0,1}), ({2,3},{2,3})
    That's 4 sub-problems for i={0,1}.
    
    Similarly for i={2,3}: another 4 sub-problems.
    
    But that's 8 sub-problems of size 2x2x2, using j in {0,1,2,3} and k in {0,1,2,3}.
    The remaining j=4 and k=4 are unused.
    
    So I have 8 2x2x2 sub-problems, each replaced by Strassen rank 7.
    Original rank for these 8: 8 * 8 = 64
    New rank: 8 * 7 = 56
    Remaining: the parts involving j=4 or k=4.
    
    The remaining tensor involves:
    - i in [0,4), j=4, k in [0,5): 4*1*5 = 20
    - i in [0,4), j in [0,4), k=4: 4*4*1 = 16
    - But (j=4, k=4) is counted twice, so subtract 4*1*1=4
    - Total remaining: 20 + 16 - 4 = 32
    
    Total rank = 56 + 32 = 88
    
    That's a bigger improvement! Let me implement this.
    
    Actually wait, I need to be more careful. The 8 2x2x2 sub-problems use:
    i in {0,1,2,3} (all 4 values), j in {0,1,2,3}, k in {0,1,2,3}
    
    So the remaining part is:
    - All entries where j=4 OR k=4 (within i in [0,4))
    - This is a union of two sets:
      Set 1: j=4, i in [0,4), k in [0,5) -> 4*1*5 = 20
      Set 2: k=4, i in [0,4), j in [0,4) -> 4*4*1 = 16 (excluding j=4 which is in Set 1)
      But Set 2 with j=4 is already in Set 1.
      
    Actually, let's just compute it as:
    Total tensor: 4*5*5 = 100
    Used by 8 2x2x2 blocks: 8*8 = 64
    Remaining: 100 - 64 = 36
    
    Hmm, 36 not 32. Let me recount.
    
    The 8 2x2x2 blocks cover:
    i: 0,1,2,3 (all)
    j: 0,1,2,3 (4 values)
    k: 0,1,2,3 (4 values)
    So they cover 4*4*4 = 64 entries.
    
    Remaining entries: those where j=4 or k=4.
    - j=4, k in [0,5), i in [0,4): 4*1*5 = 20
    - j in [0,4), k=4, i in [0,4): 4*4*1 = 16
    - Intersection: j=4, k=4, i in [0,4): 4*1*1 = 4
    - Union: 20 + 16 - 4 = 32
    
    But 64 + 32 = 96, not 100. What am I missing?
    
    Ah, the 2x2x2 blocks use specific pairs. Let me list them:
    Block 1: i={0,1}, j={0,1}, k={0,1} -> 8 entries
    Block 2: i={0,1}, j={0,1}, k={2,3} -> 8 entries
    Block 3: i={0,1}, j={2,3}, k={0,1} -> 8 entries
    Block 4: i={0,1}, j={2,3}, k={2,3} -> 8 entries
    Block 5: i={2,3}, j={0,1}, k={0,1} -> 8 entries
    Block 6: i={2,3}, j={0,1}, k={2,3} -> 8 entries
    Block 7: i={2,3}, j={2,3}, k={0,1} -> 8 entries
    Block 8: i={2,3}, j={2,3}, k={2,3} -> 8 entries
    
    Total: 8*8 = 64 entries. These cover i in {0,1,2,3}, j in {0,1,2,3}, k in {0,1,2,3}.
    
    Remaining: j=4 or k=4.
    - j=4, i in [0,4), k in [0,5): 20
    - k=4, i in [0,4), j in [0,4): 16 (j in {0,1,2,3} since j=4 is already counted)
    - Total remaining: 20 + 16 = 36
    
    Wait, I double-counted? No:
    - Entries with j=4: (i,4,k) for i in [0,4), k in [0,5). That's 4*5=20.
    - Entries with k=4 but j!=4: (i,j,4) for i in [0,4), j in [0,4). That's 4*4=16.
    - Total: 20+16=36.
    
    So 64+36=100. Correct.
    
    So remaining rank = 36 (schoolbook).
    Total rank = 8*7 + 36 = 56 + 36 = 92.
    
    That's better than 96! Let me implement this.
    """
    deadline = time.time() + 140.0
    
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    # We'll build U, V, W as lists of lists
    # U: (n*m) x rank, V: (m*p) x rank, W: (p*n) x rank
    
    # First, let's figure out the total rank
    # 8 2x2x2 Strassen blocks: 8*7 = 56
    # Remaining 36 schoolbook terms: 36
    # Total: 92
    
    total_rank = 92
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen 2x2x2 decomposition
    # For a 2x2x2 sub-problem with:
    #   A indices: (i1, j1), (i1, j2), (i2, j1), (i2, j2)
    #   B indices: (j1, k1), (j1, k2), (j2, k1), (j2, k2)
    #   C indices: (i1, k1), (i1, k2), (i2, k1), (i2, k2)
    #
    # Strassen:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    def strassen_2x2x2(i1, i2, j1, j2, k1, k2, col_start):
        """
        Add Strassen 2x2x2 decomposition starting at column col_start.
        Returns the next column index.
        """
        # M1 = (A11+A22)*(B11+B22)
        # U: (i1,j1) and (i2,j2) get +1
        # V: (j1,k1) and (j2,k2) get +1
        # W: (k1,i1) and (k2,i2) get +1
        U[i1 * m + j1][col_start] = 1
        U[i2 * m + j2][col_start] = 1
        V[j1 * p + k1][col_start] = 1
        V[j2 * p + k2][col_start] = 1
        W[k1 * n + i1][col_start] = 1
        W[k2 * n + i2][col_start] = 1
        col_start += 1
        
        # M2 = (A21+A22)*B11
        # U: (i2,j1) and (i2,j2) get +1
        # V: (j1,k1) gets +1
        # W: (k1,i2) gets +1
        U[i2 * m + j1][col_start] = 1
        U[i2 * m + j2][col_start] = 1
        V[j1 * p + k1][col_start] = 1
        W[k1 * n + i2][col_start] = 1
        col_start += 1
        
        # M3 = A11*(B12-B22)
        # U: (i1,j1) gets +1
        # V: (j1,k2) gets +1, (j2,k2) gets -1
        # W: (k2,i1) gets +1
        U[i1 * m + j1][col_start] = 1
        V[j1 * p + k2][col_start] = 1
        V[j2 * p + k2][col_start] = -1
        W[k2 * n + i1][col_start] = 1
        col_start += 1
        
        # M4 = A22*(B21-B11)
        # U: (i2,j2) gets +1
        # V: (j2,k1) gets +1, (j1,k1) gets -1
        # W: (k1,i2) gets +1
        U[i2 * m + j2][col_start] = 1
        V[j2 * p + k1][col_start] = 1
        V[j1 * p + k1][col_start] = -1
        W[k1 * n + i2][col_start] = 1
        col_start += 1
        
        # M5 = (A11+A12)*B22
        # U: (i1,j1) and (i1,j2) get +1
        # V: (j2,k2) gets +1
        # W: (k2,i1) gets +1
        U[i1 * m + j1][col_start] = 1
        U[i1 * m + j2][col_start] = 1
        V[j2 * p + k2][col_start] = 1
        W[k2 * n + i1][col_start] = 1
        col_start += 1
        
        # M6 = (A21-A11)*(B11+B12)
        # U: (i2,j1) gets +1, (i1,j1) gets -1
        # V: (j1,k1) and (j1,k2) get +1
        # W: (k1,i2) gets +1, (k2,i2) gets +1
        U[i2 * m + j1][col_start] = 1
        U[i1 * m + j1][col_start] = -1
        V[j1 * p + k1][col_start] = 1
        V[j1 * p + k2][col_start] = 1
        W[k1 * n + i2][col_start] = 1
        W[k2 * n + i2][col_start] = 1
        col_start += 1
        
        # M7 = (A12-A22)*(B21+B22)
        # U: (i1,j2) gets +1, (i2,j2) gets -1
        # V: (j2,k1) and (j2,k2) get +1
        # W: (k1,i1) gets +1, (k2,i1) gets +1
        U[i1 * m + j2][col_start] = 1
        U[i2 * m + j2][col_start] = -1
        V[j2 * p + k1][col_start] = 1
        V[j2 * p + k2][col_start] = 1
        W[k1 * n + i1][col_start] = 1
        W[k2 * n + i1][col_start] = 1
        col_start += 1
        
        return col_start
    
    # Define the 8 2x2x2 blocks
    i_blocks = [(0, 1), (2, 3)]
    j_blocks = [(0, 1), (2, 3)]
    k_blocks = [(0, 1), (2, 3)]
    
    for i1, i2 in i_blocks:
        for j1, j2 in j_blocks:
            for k1, k2 in k_blocks:
                col = strassen_2x2x2(i1, i2, j1, j2, k1, k2, col)
    
    # Now handle the remaining 36 entries
    # These are entries where j=4 or k=4
    # For each such (i, j, k), add a rank-1 term
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if j == 4 or k == 4:
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
    
    # Verify col == total_rank
    assert col == total_rank, f"Expected {total_rank} columns, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
