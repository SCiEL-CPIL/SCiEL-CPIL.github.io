"""Matrix multiplication 6x6x6 tensor decomposition.
Uses Laderman's rank-16 decomposition for 4x4, combined with Strassen for the
remaining degrees of freedom, to achieve a lower rank than schoolbook.

Strategy: We use the known Laderman decomposition for 4x4 matrix multiplication
(rank 16). For 6x6, we can view it as a block structure. However, a more direct
approach is to use the best known decomposition for small matrices.

Actually, the known best rank for 6x6 matrix multiplication is not trivial.
The schoolbook is 216. Strassen gives about 189 (2^(log2(6)) ~ 2^5.9 = ~59? No, Strassen on 6x6 isn't direct).

Let's use a known decomposition. The Laderman decomposition is for 4x4 with rank 16.
For 6x6, we can pad to 8x8 and use Strassen? No, that's worse.

Actually, let's think about the tensor structure. The multiplication tensor for n x n is of order n^3.
For 6x6, the standard decomposition has rank 216.

A better approach: Use the fact that 6 = 2 * 3. We can use a Kronecker product of decompositions.
If we have a decomposition for 2x2 (Strassen, rank 7) and for 3x3 (schoolbook, rank 27),
the Kronecker product gives rank 7 * 27 = 189. This is better than 216.

Can we do better? What about using Laderman for 4x4 (rank 16) and something for 6?
6 is not a power of 2.

Let's use the Kronecker product of Strassen (2x2, rank 7) and schoolbook (3x3, rank 27).
This gives rank 189.

Actually, we can do even better. The best known rank for 6x6 matrix multiplication over a field
is 73 (for the 4x4 case it's 49, but for 6x6 it's higher). Wait, no. The exponent is log2(7) ~ 2.807.
For 6x6, the rank should be around 6^2.807 ~ 189. So 189 is near-optimal.

Let's implement the Kronecker product of Strassen (2x2) and schoolbook (3x3).

Strassen decomposition for 2x2:
M1 = (A00 + A11)(B00 + B11)
M2 = (A10 + A11)B00
M3 = A00(B10 - B11)
M4 = A11(B01 - B00)
M5 = (A00 + A01)B11
M6 = (A10 - A00)(B00 + B01)
M7 = (A01 - A11)(B10 + B11)

C00 = M1 + M4 - M5 + M7
C01 = M3 + M5
C10 = M2 + M4
C11 = M1 - M2 + M3 + M6

The tensor entries for Strassen (2x2):
U (input A, size 4 x 7), V (input B, size 4 x 7), W (output C, size 4 x 7)
Indices: A00=0, A01=1, A10=2, A11=3; B00=0, B01=1, B10=2, B11=3; C00=0, C01=1, C10=2, C11=3

U[0] = [1,0,0,0,1,0,0]  # A00 in M1, M3, M5
U[1] = [0,1,0,0,0,1,0]  # A01 in M5, M6, M7
U[2] = [0,0,1,0,0,0,1]  # A10 in M2, M6
U[3] = [1,0,0,1,0,0,0]  # A11 in M1, M2, M4

Wait, let me re-derive:
M1 = (A00+A11)(B00+B11): A00 coeff 1, A11 coeff 1
M2 = (A10+A11)B00: A10 coeff 1, A11 coeff 1
M3 = A00(B10-B11): A00 coeff 1
M4 = A11(B01-B00): A11 coeff 1
M5 = (A00+A01)B11: A00 coeff 1, A01 coeff 1
M6 = (A10-A00)(B00+B01): A10 coeff 1, A00 coeff -1
M7 = (A01-A11)(B10+B11): A01 coeff 1, A11 coeff -1

So U (4x7):
Row 0 (A00): [1,0,1,0,1,-1,0]
Row 1 (A01): [0,1,0,0,1,0,1]
Row 2 (A10): [0,0,1,0,0,1,0]
Row 3 (A11): [1,1,0,1,0,0,-1]

V (4x7):
Row 0 (B00): [1,1,0,-1,0,1,0]
Row 1 (B01): [0,0,0,1,0,1,0]
Row 2 (B10): [0,0,1,0,0,0,1]
Row 3 (B11): [1,0,-1,0,1,0,1]

W (4x7):
Row 0 (C00): [1,0,0,1,-1,0,1]
Row 1 (C01): [0,0,1,0,1,0,0]
Row 2 (C10): [0,1,0,1,0,0,0]
Row 3 (C11): [1,-1,1,0,0,1,0]

Now, for the 3x3 schoolbook, rank 27.
U3 (9x27), V3 (9x27), W3 (9x27)

The Kronecker product: U = U2 ⊗ U3, V = V2 ⊗ V3, W = W2 ⊗ W3
Rank = 7 * 27 = 189.

Let's implement this.
"""

import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strassen 2x2 decomposition
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 1, 0, 0, 1, 0, 1],
        [0, 0, 1, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ])
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 0, 1, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ])
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ])
    
    # Schoolbook 3x3 decomposition
    rank3 = 27
    U3 = np.zeros((9, rank3))
    V3 = np.zeros((9, rank3))
    W3 = np.zeros((9, rank3))
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U3[i * 3 + j, col] = 1
                V3[j * 3 + k, col] = 1
                W3[k * 3 + i, col] = 1
                col += 1
    
    # Kronecker product
    U = np.kron(U2, U3)
    V = np.kron(V2, V3)
    W = np.kron(W2, W3)
    
    # Convert to list of lists as required
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
