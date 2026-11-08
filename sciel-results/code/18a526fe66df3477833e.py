"""
Matrix multiplication 2x3 by 3x6 - rank-minimal tensor decomposition.

Strategy:
- The multiplication C = A * B where A is 2x3, B is 3x6, C is 2x6.
- We can view C as having two 2x3 blocks (columns 0-2 and 3-5).
- For each block, we multiply a 2x3 matrix by a 3x3 matrix.
- The 2x3 by 3x3 multiplication can be decomposed using Strassen-like techniques
  on the 2x2 sub-blocks plus handling of the extra column.
- Total rank: 2 * (rank of 2x3 by 3x3) 
- Using Strassen on the 2x2 part (rank 7) plus 3 additional terms for the 3rd column
  gives rank 10 per block, total rank 20.
- Alternatively, we can use a more careful decomposition.

Let me think more carefully.

A is 2x3: rows i=0,1; columns j=0,1,2
B is 3x6: rows j=0,1,2; columns k=0..5
C is 2x6: rows i=0,1; columns k=0..5

C[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k]

We can split B into B[:, 0:3] and B[:, 3:6], each 3x3.
Then C[:, 0:3] = A * B[:, 0:3] and C[:, 3:6] = A * B[:, 3:6].

For each 2x3 by 3x3 multiplication:
A is 2x3, B is 3x3, C is 2x3.
C[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k]

We can write A as [A_left | A_right] where A_left is 2x2 (j=0,1) and A_right is 2x1 (j=2).
Similarly B = [B_left | B_right] where B_left is 3x2 (k=0,1) and B_right is 3x1 (k=2).

Wait, that's not quite right for the column split. Let me reconsider.

Actually, for the 2x3 by 3x3 case:
C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]

We can use Strassen on the 2x2 part (j=0,1 and k=0,1) which gives rank 7 for the 2x2 by 2x2 part,
plus we need to handle the contributions from j=2 and k=2.

The 2x2 by 2x2 part (j in {0,1}, k in {0,1}) has rank 7 via Strassen.
The remaining terms:
- A[i,2]*B[2,k] for k=0,1,2: this is 2*3 = 6 terms, but we can share.
- A[i,j]*B[j,2] for j=0,1: this is 2*2 = 4 terms.

Actually, let me just use a direct approach with rank 18 (2*3*3) for each 2x3 by 3x3, giving total rank 36. That's the schoolbook.

To do better, I'll use the fact that the two 2x3 by 3x3 multiplications share the same A.

Let me try a different approach: use rank 18 total by exploiting structure.

Actually, the simplest improvement is to use Strassen on the 2x2 sub-blocks.

Let me just implement a clean rank-33 solution using Strassen on the 2x2 blocks as described in the knowledge.

For the 2x3 by 3x6:
- Split B into 3 blocks of 3x2: B[:, 0:2], B[:, 2:4], B[:, 4:6]
- For each block, multiply 2x3 by 3x2
- The 2x3 by 3x2 can be seen as 2x2 by 2x2 (using j=0,1 and k=0,1 of the block) plus 2x1 by 1x2 (using j=2)

Hmm, this is getting complicated. Let me just go with the rank 33 solution from the knowledge:
- Use Strassen (rank 7) on the 2x2 sub-blocks for the first 2 columns of B
- Handle the 3rd column of A (j=2) with direct terms

Actually, let me re-read the knowledge: "Reduces rank from 36 to 33 by applying Strassen's algorithm (rank 7) to the 2x2 sub-blocks"

So the approach is:
- For each of the 3 column-pairs in B (columns 0-1, 2-3, 4-5), we have a 2x3 by 3x2 multiplication
- The 2x2 part (j=0,1) uses Strassen with rank 7
- The j=2 part uses rank 2 (one term per output column)
- Total per column-pair: 7 + 2 = 9
- But wait, 3 * 9 = 27, not 33.

Let me think again. Maybe the decomposition is:
- The 2x3 by 3x6 can be viewed as three 2x3 by 3x2 multiplications
- Each 2x3 by 3x2: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] for k in {2t, 2t+1}
- The 2x2 part (j=0,1, k=0,1 within the pair) uses Strassen rank 7
- The j=2 part: A[i,2]*B[2,2t] and A[i,2]*B[2,2t+1] for i=0,1: that's 4 terms, but we can do it with 2 rank-1 tensors (one for each output column, sharing the A[i,2] factor)
- So per column-pair: 7 + 2 = 9
- Total: 3 * 9 = 27

But the knowledge says 33. Let me just implement a correct solution and see.

I'll go with a straightforward approach: use Strassen on the 2x2 blocks for the first two rows of B, and handle the third row directly.

For each column-pair t (0, 1, 2):
- B_sub is 3x2: B[0:3, 2t:2t+2]
- We want C_sub = A * B_sub, which is 2x2
- C_sub[i,k] = A[i,0]*B[0,2t+k] + A[i,1]*B[1,2t+k] + A[i,2]*B[2,2t+k]

Strassen on the 2x2 part (A[0:2, 0:2] and B[0:2, 0:2]):
M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
M2 = (A[1,0]+A[1,1])*B[0,0]
M3 = A[0,0]*(B[1,0]-B[0,0])
M4 = A[1,1]*(B[1,1]-B[0,1])
M5 = (A[0,0]+A[0,1])*B[1,1]
M6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
M7 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])

C[0,0] = M1 + M4 - M5 + M7
C[0,1] = M3 + M5
C[1,0] = M2 + M4
C[1,1] = M1 - M2 + M3 + M6

Plus the j=2 contributions:
C[i,k] += A[i,2]*B[2,2t+k]

So for each column-pair, we have 7 + 4 = 11 terms? No, the j=2 part is:
C[0,0] += A[0,2]*B[2,2t]
C[0,1] += A[0,2]*B[2,2t+1]
C[1,0] += A[1,2]*B[2,2t]
C[1,1] += A[1,2]*B[2,2t+1]

These are 4 separate rank-1 terms, unless we can share. We can't easily share because each involves a different product.

So per column-pair: 7 + 4 = 11, total 33. That matches!

Let me implement this.
"""

import time

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=6
    
    # U: (n*m, rank) = (6, rank)
    # V: (m*p, rank) = (18, rank)
    # W: (p*n, rank) = (12, rank)
    
    # For each column-pair t in 0,1,2:
    #   Strassen part: 7 terms
    #   j=2 part: 4 terms
    # Total rank: 3 * 11 = 33
    
    rank = 33
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for t in range(3):
        # Strassen part for columns 2t and 2t+1
        # M1 = (A[0,0]+A[1,1]) * (B[0,2t]+B[1,2t+1])
        # U: A[0,0] and A[1,1]
        # V: B[0,2t] and B[1,2t+1]
        # W: C[0,0], C[1,1] (with appropriate signs)
        
        # M1
        U[0][col] = 1   # A[0,0]
        U[4][col] = 1   # A[1,1] (index 1*3+1=4)
        V[0][col] = 1   # B[0,2t]
        V[2+2*t+1][col] = 1  # B[1,2t+1] (index 1*6+2t+1)
        W[2*t*2+0][col] = 1  # C[0, 2t] (index 2t*2+0)
        W[2*t+1][col] = 1    # C[0, 2t+1] (index (2t+1)*2+0)
        W[2*t*2+1][col] = 1  # C[1, 2t] (index 2t*2+1)
        W[2*t+1+1][col] = 1  # C[1, 2t+1] (index (2t+1)*2+1)
        # Wait, I need to be more careful with the W indices.
        # W index: k*n + i where k is column of C, i is row of C
        # C[0, 2t]: k=2t, i=0 -> W[2t*2+0]
        # C[0, 2t+1]: k=2t+1, i=0 -> W[(2t+1)*2+0]
        # C[1, 2t]: k=2t, i=1 -> W[2t*2+1]
        # C[1, 2t+1]: k=2t+1, i=1 -> W[(2t+1)*2+1]
        
        # M1 contributes +1 to C[0,0] and C[1,1] of the 2x2 block
        # So W[2t*2+0] += 1, W[(2t+1)*2+1] += 1
        # Let me redo this more carefully.
        
        # I'll clear and redo.
        col += 1  # placeholder, I'll fix below
    
    # Let me restart with a cleaner implementation.
    col = 0
    for t in range(3):
        k0 = 2*t
        k1 = 2*t+1
        
        # Strassen 7 terms for the 2x2 sub-problem (rows 0,1 of A; cols k0,k1 of B; rows 0,1 of B)
        
        # M1 = (A[0,0]+A[1,1]) * (B[0,k0]+B[1,k1])
        # Contributes to C[0,k0] and C[1,k1]
        U[0][col] = 1    # A[0,0]
        U[4][col] = 1    # A[1,1]
        V[0*6+k0][col] = 1      # B[0,k0]
        V[1*6+k1][col] = 1      # B[1,k1]
        W[k0*2+0][col] = 1      # C[0,k0]
        W[k1*2+1][col] = 1      # C[1,k1]
        col += 1
        
        # M2 = (A[1,0]+A[1,1]) * B[0,k0]
        # Contributes to C[1,k0] and C[1,k1] with -1
        U[3][col] = 1    # A[1,0]
        U[4][col] = 1    # A[1,1]
        V[0*6+k0][col] = 1      # B[0,k0]
        W[k0*2+1][col] = 1      # C[1,k0]
        W[k1*2+1][col] = -1     # C[1,k1]
        col += 1
        
        # M3 = A[0,0] * (B[1,k0]-B[0,k0])
        # Contributes to C[0,k1] and C[1,k1]
        U[0][col] = 1    # A[0,0]
        V[1*6+k0][col] = 1      # B[1,k0]
        V[0*6+k0][col] = -1     # B[0,k0]
        W[k1*2+0][col] = 1      # C[0,k1]
        W[k1*2+1][col] = 1      # C[1,k1]
        col += 1
        
        # M4 = A[1,1] * (B[1,k1]-B[0,k1])
        # Contributes to C[0,k0] and C[1,k0]
        U[4][col] = 1    # A[1,1]
        V[1*6+k1][col] = 1      # B[1,k1]
        V[0*6+k1][col] = -1     # B[0,k1]
        W[k0*2+0][col] = 1      # C[0,k0]
        W[k0*2+1][col] = 1      # C[1,k0]
        col += 1
        
        # M5 = (A[0,0]+A[0,1]) * B[1,k1]
        # Contributes to C[0,k1] and C[0,k0] with -1
        U[0][col] = 1    # A[0,0]
        U[1][col] = 1    # A[0,1]
        V[1*6+k1][col] = 1      # B[1,k1]
        W[k1*2+0][col] = 1      # C[0,k1]
        W[k0*2+0][col] = -1     # C[0,k0]
        col += 1
        
        # M6 = (A[1,0]-A[0,0]) * (B[0,k0]+B[0,k1])
        # Contributes to C[1,k1]
        U[3][col] = 1    # A[1,0]
        U[0][col] = -1   # A[0,0]
        V[0*6+k0][col] = 1      # B[0,k0]
        V[0*6+k1][col] = 1      # B[0,k1]
        W[k1*2+1][col] = 1      # C[1,k1]
        col += 1
        
        # M7 = (A[0,1]-A[1,1]) * (B[1,k0]+B[1,k1])
        # Contributes to C[0,k0]
        U[1][col] = 1    # A[0,1]
        U[4][col] = -1   # A[1,1]
        V[1*6+k0][col] = 1      # B[1,k0]
        V[1*6+k1][col] = 1      # B[1,k1]
        W[k0*2+0][col] = 1      # C[0,k0]
        col += 1
        
        # j=2 part: 4 terms
        # C[0,k0] += A[0,2]*B[2,k0]
        U[2][col] = 1    # A[0,2]
        V[2*6+k0][col] = 1      # B[2,k0]
        W[k0*2+0][col] = 1      # C[0,k0]
        col += 1
        
        # C[0,k1] += A[0,2]*B[2,k1]
        U[2][col] = 1    # A[0,2]
        V[2*6+k1][col] = 1      # B[2,k1]
        W[k1*2+0][col] = 1      # C[0,k1]
        col += 1
        
        # C[1,k0] += A[1,2]*B[2,k0]
        U[5][col] = 1    # A[1,2]
        V[2*6+k0][col] = 1      # B[2,k0]
        W[k0*2+1][col] = 1      # C[1,k0]
        col += 1
        
        # C[1,k1] += A[1,2]*B[2,k1]
        U[5][col] = 1    # A[1,2]
        V[2*6+k1][col] = 1      # B[2,k1]
        W[k1*2+1][col] = 1      # C[1,k1]
        col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
