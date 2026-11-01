"""
Strategy: Constructive Embedding of Rank-28 Decomposition.
The current program uses random search which rarely finds non-trivial decompositions.
This program explicitly constructs a rank-28 decomposition by viewing the 2x4x4 multiplication
as two independent 2x2x2 multiplications (using Strassen's algorithm, rank 7 each) 
combined with a trivial 2x2x1 multiplication (rank 4) for the remaining terms, or more simply,
by permuting the matrix indices to map the 2x4x4 problem onto a structure that allows 
reusing the 2x2x2 Strassen algorithm on blocks.

Actually, a simpler constructive approach for 2x4x4:
We can view the 2x4 matrix A as two 2x2 matrices A1, A2.
We can view the 4x4 matrix B as four 2x2 matrices B1, B2, B3, B4.
C1 = A1 B1 + A2 B3 (2x2 product)
C2 = A1 B2 + A2 B4 (2x2 product)
Each 2x2 product can be done with rank 7 (Strassen).
So total rank = 7 + 7 = 14? 
Wait, the indices are:
A is 2x4. B is 4x4.
C is 2x4.
C[i, k] = sum_{j=0}^3 A[i, j] B[j, k].
Let's split j into 0,1 and 2,3.
C[i, k] = A[i, 0]B[0, k] + A[i, 1]B[1, k] + A[i, 2]B[2, k] + A[i, 3]B[3, k].
This is not simply two 2x2 matrix multiplications because the "middle" dimension j is split,
but the output k is not split in the same way for both terms.
However, we can group the columns of B.
Let B be partitioned into B_top (rows 0,1) and B_bottom (rows 2,3).
Let A be partitioned into A_left (cols 0,1) and A_right (cols 2,3).
C = A_left * B_top + A_right * B_bottom.
A_left is 2x2. B_top is 2x4.
A_right is 2x2. B_bottom is 2x4.
The product of a 2x2 and a 2x4 matrix can be viewed as two 2x2 matrix multiplications:
A_left * [B_top_left, B_top_right] = [A_left B_top_left, A_left B_top_right].
So C = [A_left B_top_left + A_right B_bottom_left, A_left B_top_right + A_right B_bottom_right].
Let C_left = A_left B_top_left + A_right B_bottom_left.
Let C_right = A_left B_top_right + A_right B_bottom_right.
Each of these is a sum of two 2x2 matrix products.
The rank of a 2x2 matrix product is 7 (Strassen).
The rank of a sum of two 2x2 matrix products (with independent inputs) is 7+7=14?
Yes, if the inputs are independent, the rank adds.
So we need 14 for C_left and 14 for C_right. Total rank 28.

We will construct the rank-28 decomposition explicitly.
Strassen's 7 multiplications for X*Y (2x2):
M1 = (X11+X22)(Y11+Y22)
M2 = (X21+X22)Y11
M3 = X11(Y21-Y22)
M4 = X22(Y12-Y11)
M5 = (X11+X12)Y22
M6 = (X21-X11)(Y11+Y12)
M7 = (X12-X22)(Y21+Y22)

Z11 = M1 + M4 - M5 + M7
Z12 = M3 + M5
Z21 = M2 + M4
Z22 = M1 - M2 + M3 + M6

We need to map the 2x4x4 problem to this structure.
Let A be 2x4, B be 4x4.
Split A into A_L (cols 0,1) and A_R (cols 2,3).
Split B into B_T (rows 0,1) and B_B (rows 2,3).
B_T is 2x4, split into B_TL (cols 0,1) and B_TR (cols 2,3).
B_B is 2x4, split into B_BL (cols 0,1) and B_BR (cols 2,3).

C_L (cols 0,1 of C) = A_L B_TL + A_R B_BL.
C_R (cols 2,3 of C) = A_L B_TR + A_R B_BR.

For C_L:
Term 1: A_L * B_TL. Both 2x2. Use Strassen.
Term 2: A_R * B_BL. Both 2x2. Use Strassen.
Total 14 multiplications for C_L.

For C_R:
Term 1: A_L * B_TR. Both 2x2. Use Strassen.
Term 2: A_R * B_BR. Both 2x2. Use Strassen.
Total 14 multiplications for C_R.

Total Rank = 28.

We need to output U, V, W matrices.
U shape: (2*4, 28) = (8, 28)
V shape: (4*4, 28) = (16, 28)
W shape: (4*2, 28) = (8, 28)

Indices:
U[i*m + j, r] -> U[i*4 + j, r]
V[j*p + k, r] -> V[j*4 + k, r]
W[k*n + i, r] -> W[k*2 + i, r]

Let's define the 28 terms.
Terms 0-6: Strassen for A_L * B_TL.
Terms 7-13: Strassen for A_R * B_BL.
Terms 14-20: Strassen for A_L * B_TR.
Terms 21-27: Strassen for A_R * B_BR.

For each Strassen instance, we define the coefficients for U, V, W.
Strassen instance for X * Y:
X is 2x2, Y is 2x2.
X indices in U: 
  X11 -> U[0, r]
  X12 -> U[1, r]
  X21 -> U[2, r]
  X22 -> U[3, r]
(Note: X is a submatrix of A. A_L is A[:, 0:2]. So A_L[0,0]=A[0,0], A_L[0,1]=A[0,1], A_L[1,0]=A[1,0], A_L[1,1]=A[1,1].
So U indices for A_L:
  A_L[0,0] -> U[0*4+0, r] = U[0, r]
  A_L[0,1] -> U[0*4+1, r] = U[1, r]
  A_L[1,0] -> U[1*4+0, r] = U[4, r]
  A_L[1,1] -> U[1*4+1, r] = U[5, r]
Wait, U index is i*m + j.
i=0, j=0 -> 0
i=0, j=1 -> 1
i=1, j=0 -> 4
i=1, j=1 -> 5

Similarly for A_R (cols 2,3):
A_R[0,0] = A[0,2] -> U[2, r]
A_R[0,1] = A[0,3] -> U[3, r]
A_R[1,0] = A[1,2] -> U[6, r]
A_R[1,1] = A[1,3] -> U[7, r]

Y indices in V:
Y is a submatrix of B.
B_TL is B[0:2, 0:2].
B_TL[0,0] = B[0,0] -> V[0*4+0, r] = V[0, r]
B_TL[0,1] = B[0,1] -> V[1, r]
B_TL[1,0] = B[1,0] -> V[4, r]
B_TL[1,1] = B[1,1] -> V[5, r]

B_BL is B[2:4, 0:2].
B_BL[0,0] = B[2,0] -> V[8, r]
B_BL[0,1] = B[2,1] -> V[9, r]
B_BL[1,0] = B[3,0] -> V[12, r]
B_BL[1,1] = B[3,1] -> V[13, r]

B_TR is B[0:2, 2:4].
B_TR[0,0] = B[0,2] -> V[2, r]
B_TR[0,1] = B[0,3] -> V[3, r]
B_TR[1,0] = B[1,2] -> V[6, r]
B_TR[1,1] = B[1,3] -> V[7, r]

B_BR is B[2:4, 2:4].
B_BR[0,0] = B[2,2] -> V[10, r]
B_BR[0,1] = B[2,3] -> V[11, r]
B_BR[1,0] = B[3,2] -> V[14, r]
B_BR[1,1] = B[3,3] -> V[15, r]

Z indices in W:
Z is a submatrix of C.
C_L is C[:, 0:2].
C_L[0,0] = C[0,0] -> W[0*2+0, r] = W[0, r]
C_L[0,1] = C[0,1] -> W[1*2+0, r] = W[2, r]
C_L[1,0] = C[1,0] -> W[0*2+1, r] = W[1, r]
C_L[1,1] = C[1,1] -> W[1*2+1, r] = W[3, r]

C_R is C[:, 2:4].
C_R[0,0] = C[0,2] -> W[2*2+0, r] = W[4, r]
C_R[0,1] = C[0,3] -> W[3*2+0, r] = W[6, r]
C_R[1,0] = C[1,2] -> W[2*2+1, r] = W[5, r]
C_R[1,1] = C[1,3] -> W[3*2+1, r] = W[7, r]

Strassen Coefficients (for X*Y=Z):
M1: X11+X22, Y11+Y22, Z11+1, Z22+1
M2: X21+X22, Y11, Z21+1
M3: X11, Y21-Y22, Z12+1, Z22+1
M4: X22, Y12-Y11, Z11+1, Z21+1
M5: X11+X12, Y22, Z11-1, Z12+1
M6: X21-X11, Y11+Y12, Z22+1
M7: X12-X22, Y21+Y22, Z11+1, Z12+1

We will build the U, V, W matrices by iterating through the 4 Strassen instances and adding the coefficients.
"""

import time
import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    start_time = time.time()
    deadline = 140
    
    n, m, p = DIMENSIONS
    R = 28
    
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    # Helper to add a Strassen instance
    def add_strassen(offset, X_indices, Y_indices, Z_indices):
        # X_indices: [X11, X12, X21, X22]
        # Y_indices: [Y11, Y12, Y21, Y22]
        # Z_indices: [Z11, Z12, Z21, Z22]
        
        # M1
        U[X_indices[0], offset + 0] += 1
        U[X_indices[3], offset + 0] += 1
        V[Y_indices[0], offset + 0] += 1
        V[Y_indices[3], offset + 0] += 1
        W[Z_indices[0], offset + 0] += 1
        W[Z_indices[3], offset + 0] += 1
        
        # M2
        U[X_indices[2], offset + 1] += 1
        U[X_indices[3], offset + 1] += 1
        V[Y_indices[0], offset + 1] += 1
        W[Z_indices[2], offset + 1] += 1
        
        # M3
        U[X_indices[0], offset + 2] += 1
        V[Y_indices[2], offset + 2] += 1
        V[Y_indices[3], offset + 2] -= 1
        W[Z_indices[1], offset + 2] += 1
        W[Z_indices[3], offset + 2] += 1
        
        # M4
        U[X_indices[3], offset + 3] += 1
        V[Y_indices[1], offset + 3] += 1
        V[Y_indices[0], offset + 3] -= 1
        W[Z_indices[0], offset + 3] += 1
        W[Z_indices[2], offset + 3] += 1
        
        # M5
        U[X_indices[0], offset + 4] += 1
        U[X_indices[1], offset + 4] += 1
        V[Y_indices[3], offset + 4] += 1
        W[Z_indices[0], offset + 4] -= 1
        W[Z_indices[1], offset + 4] += 1
        
        # M6
        U[X_indices[2], offset + 5] += 1
        U[X_indices[0], offset + 5] -= 1
        V[Y_indices[0], offset + 5] += 1
        V[Y_indices[1], offset + 5] += 1
        W[Z_indices[3], offset + 5] += 1
        
        # M7
        U[X_indices[1], offset + 6] += 1
        U[X_indices[3], offset + 6] -= 1
        V[Y_indices[2], offset + 6] += 1
        V[Y_indices[3], offset + 6] += 1
        W[Z_indices[0], offset + 6] += 1
        W[Z_indices[1], offset + 6] += 1

    # Indices for U (A)
    # A_L (cols 0,1):
    X11_L = 0 # A[0,0]
    X12_L = 1 # A[0,1]
    X21_L = 4 # A[1,0]
    X22_L = 5 # A[1,1]
    X_L = [X11_L, X12_L, X21_L, X22_L]
    
    # A_R (cols 2,3):
    X11_R = 2 # A[0,2]
    X12_R = 3 # A[0,3]
    X21_R = 6 # A[1,2]
    X22_R = 7 # A[1,3]
    X_R = [X11_R, X12_R, X21_R, X22_R]
    
    # Indices for V (B)
    # B_TL (rows 0,1; cols 0,1):
    Y11_TL = 0 # B[0,0]
    Y12_TL = 1 # B[0,1]
    Y21_TL = 4 # B[1,0]
    Y22_TL = 5 # B[1,1]
    Y_TL = [Y11_TL, Y12_TL, Y21_TL, Y22_TL]
    
    # B_BL (rows 2,3; cols 0,1):
    Y11_BL = 8 # B[2,0]
    Y12_BL = 9 # B[2,1]
    Y21_BL = 12 # B[3,0]
    Y22_BL = 13 # B[3,1]
    Y_BL = [Y11_BL, Y12_BL, Y21_BL, Y22_BL]
    
    # B_TR (rows 0,1; cols 2,3):
    Y11_TR = 2 # B[0,2]
    Y12_TR = 3 # B[0,3]
    Y21_TR = 6 # B[1,2]
    Y22_TR = 7 # B[1,3]
    Y_TR = [Y11_TR, Y12_TR, Y21_TR, Y22_TR]
    
    # B_BR (rows 2,3; cols 2,3):
    Y11_BR = 10 # B[2,2]
    Y12_BR = 11 # B[2,3]
    Y21_BR = 14 # B[3,2]
    Y22_BR = 15 # B[3,3]
    Y_BR = [Y11_BR, Y12_BR, Y21_BR, Y22_BR]
    
    # Indices for W (C)
    # C_L (cols 0,1):
    Z11_L = 0 # C[0,0] -> W[0*2+0]
    Z12_L = 2 # C[0,1] -> W[1*2+0]
    Z21_L = 1 # C[1,0] -> W[0*2+1]
    Z22_L = 3 # C[1,1] -> W[1*2+1]
    Z_L = [Z11_L, Z12_L, Z21_L, Z22_L]
    
    # C_R (cols 2,3):
    Z11_R = 4 # C[0,2] -> W[2*2+0]
    Z12_R = 6 # C[0,3] -> W[3*2+0]
    Z21_R = 5 # C[1,2] -> W[2*2+1]
    Z22_R = 7 # C[1,3] -> W[3*2+1]
    Z_R = [Z11_R, Z12_R, Z21_R, Z22_R]
    
    # Add Strassen instances
    # 1. A_L * B_TL -> contributes to C_L
    add_strassen(0, X_L, Y_TL, Z_L)
    
    # 2. A_R * B_BL -> contributes to C_L
    add_strassen(7, X_R, Y_BL, Z_L)
    
    # 3. A_L * B_TR -> contributes to C_R
    add_strassen(14, X_L, Y_TR, Z_R)
    
    # 4. A_R * B_BR -> contributes to C_R
    add_strassen(21, X_R, Y_BR, Z_R)
    
    # Convert to lists of lists
    U_out = U.tolist()
    V_out = V.tolist()
    W_out = W.tolist()
    
    # Verify
    T_recon = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for r in range(R):
                    val += U_out[i * m + j][r] * V_out[j * p + k][r] * W_out[k * n + i][r]
                T_recon[i, j, k] = val
    
    if not np.allclose(T_recon, 1.0):
        # Fallback to schoolbook if construction failed (should not happen)
        rank_school = n * m * p
        U_school = [[0] * rank_school for _ in range(n * m)]
        V_school = [[0] * rank_school for _ in range(m * p)]
        W_school = [[0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i * m + j][col] = 1
                    V_school[j * p + k][col] = 1
                    W_school[k * n + i][col] = 1
                    col += 1
        return {"U": U_school, "V": V_school, "W": W_school}
    
    return {"U": U_out, "V": V_out, "W": W_out}

if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
