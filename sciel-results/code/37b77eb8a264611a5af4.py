"""
Rank-minimal exact tensor decomposition for 3x4 * 4x4 matrix multiplication.

Strategy: Use Strassen's algorithm on the 2x2x2 sub-blocks for the top two
output rows (i=0,1) and trivial multiplication for the bottom row (i=2).

The 3x4 * 4x4 multiplication can be decomposed as:
- A is 3x4, B is 4x4, C is 3x4
- Split A into A_top (2x4) and A_bottom (1x4)
- Split B into B_left (4x2) and B_right (4x2)
- C_top = A_top * B (2x4), C_bottom = A_bottom * B (1x4)

For C_top (2x4 = 2x2 * 2x2 blocks):
  Let A_top = [[A11, A12], [A21, A22]] where each Aij is 1x2
  Let B = [[B11, B12], [B21, B22]] where each Bij is 2x2
  Then C_top = [[A11*B11 + A12*B21, A11*B12 + A12*B22],
                [A21*B11 + A22*B21, A21*B12 + A22*B22]]
  
  Each 1x2 * 2x2 = 1x2 product can use Strassen-like tricks.
  Actually, let me think more carefully.

Better approach: Use the known result that M(3,4,4) <= 44 via block Strassen.

Let me use a direct construction:
- n=3, m=4, p=4
- Trivial rank = 3*4*4 = 48
- We can save 4 by using Strassen on a 2x2x2 subproblem.

The 3x4 * 4x4 can be seen as:
Row 0: [a00 a01 a02 a03] * B = c0
Row 1: [a10 a11 a12 a13] * B = c1
Row 2: [a20 a21 a22 a23] * B = c2

For rows 0 and 1, we can group them:
[a00 a01 a02 a03]
[a10 a11 a12 a13]
This is a 2x4 matrix. Multiplying by B (4x4) gives 2x4.

We can write this as:
[A_left A_right] * [B_top; B_bottom] where A_left is 2x2, A_right is 2x2,
B_top is 2x4, B_bottom is 2x4.

Hmm, this is getting complicated. Let me use a simpler block decomposition.

Actually, the cleanest way: partition the 4 columns of A into two groups of 2,
and the 4 rows of B into two groups of 2.

A = [A1 A2] where A1, A2 are 3x2
B = [B1; B2] where B1, B2 are 2x4

C = A1*B1 + A2*B2

Each A_i is 3x2, B_i is 2x4, so A_i*B_i is 3x4.
Trivial rank for each: 3*2*4 = 24, total 48.

Using Strassen on each 3x2 * 2x4:
This is not a square problem, so standard Strassen doesn't directly apply.

Alternative: Use the fact that 2x2 * 2x2 has rank 7 (Strassen).
Partition A into 2x2 blocks and B into 2x2 blocks.

A (3x4) can be padded to 4x4, but that changes the problem.

Let me try a different partition:
A is 3x4. Write A as rows: r0, r1, r2 (each 1x4).
B is 4x4. Write B as columns: c0, c1, c2, c3 (each 4x1).

C[i,j] = r_i . c_j

For i=0,1 and j=0,1: we have a 2x2 subproblem with 4x1 vectors.
This isn't a standard tensor.

Let me just use the hybrid approach from the inspiration:
- For the top 2 output rows, use Strassen-like decomposition on 2x2x2 blocks
- For the bottom row, use trivial multiplication

The 3x4 * 4x4 tensor:
C[i,k] = sum_j A[i,j] * B[j,k] for i in 0..2, j in 0..3, k in 0..3

Partition j into {0,1} and {2,3}, and k into {0,1} and {2,3}:
C[i, 0:2] = A[i, 0:2] * B[0:2, 0:2] + A[i, 2:4] * B[2:4, 0:2]
C[i, 2:4] = A[i, 0:2] * B[0:2, 2:4] + A[i, 2:4] * B[2:4, 2:4]

For each i, the computation involves four 1x2 * 2x2 products.
Each 1x2 * 2x2 = 1x2 product has rank 4 trivially.

But we can use Strassen for the 2x2 * 2x2 part if we group rows.

Let me try: For i=0,1, treat A[0:2, :] as a 2x4 matrix.
C[0:2, 0:2] = A[0:2, 0:2] * B[0:2, 0:2] + A[0:2, 2:4] * B[2:4, 0:2]
C[0:2, 2:4] = A[0:2, 0:2] * B[0:2, 2:4] + A[0:2, 2:4] * B[2:4, 2:4]

Each term is 2x2 * 2x2 = 2x2, which can be done in rank 7 (Strassen).
So for i=0,1: 4 Strassen blocks, each rank 7 = 28
For i=2: trivial, 4 blocks of 1x2 * 2x2, each rank 4 = 16
Total: 28 + 16 = 44.

This matches the inspiration! Let me implement this.
"""

import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 4]

def _strassen_2x2():
    """
    Strassen decomposition of 2x2 * 2x2 = 2x2.
    Returns U, V, W matrices for the rank-7 decomposition.
    
    A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    C = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    
    Strassen:
    M1 = (a+d)(e+h)
    M2 = (c+d)e
    M3 = a(f-h)
    M4 = d(g-e)
    M5 = (a+b)h
    M6 = (c-a)(e+f)
    M7 = (b-d)(g+h)
    
    C11 = M1 - M4 + M5 - M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M3 + M2 - M6
    """
    # Indices: A is 2x2, B is 2x2, C is 2x2
    # U: (2*2, 7) = (4, 7), V: (2*2, 7) = (4, 7), W: (2*2, 7) = (4, 7)
    # U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    
    U = np.zeros((4, 7), dtype=object)
    V = np.zeros((4, 7), dtype=object)
    W = np.zeros((4, 7), dtype=object)
    
    # A indices: 00=0, 01=1, 10=2, 11=3
    # B indices: 00=4, 01=5, 10=6, 11=7 -> but we use local indices 0,1,2,3 for B
    # C indices: 00=0, 01=1, 10=2, 11=3
    
    # M1 = (a+d)(e+h)
    U[0, 0] = 1  # a
    U[3, 0] = 1  # d
    V[0, 0] = 1  # e
    V[3, 0] = 1  # h
    W[0, 0] = 1  # C11
    W[3, 0] = 1  # C22
    
    # M2 = (c+d)e
    U[2, 1] = 1  # c
    U[3, 1] = 1  # d
    V[0, 1] = 1  # e
    W[2, 1] = 1  # C21
    W[3, 1] = 1  # C22
    
    # M3 = a(f-h)
    U[0, 2] = 1  # a
    V[1, 2] = 1  # f
    V[3, 2] = -1 # h
    W[1, 2] = 1  # C12
    W[3, 2] = -1 # C22
    
    # M4 = d(g-e)
    U[3, 3] = 1  # d
    V[2, 3] = 1  # g
    V[0, 3] = -1 # e
    W[0, 3] = -1 # C11
    W[2, 3] = 1  # C21
    
    # M5 = (a+b)h
    U[0, 4] = 1  # a
    U[1, 4] = 1  # b
    V[3, 4] = 1  # h
    W[0, 4] = 1  # C11
    W[1, 4] = 1  # C12
    
    # M6 = (c-a)(e+f)
    U[2, 5] = 1  # c
    U[0, 5] = -1 # a
    V[0, 5] = 1  # e
    V[1, 5] = 1  # f
    W[3, 5] = -1 # C22
    
    # M7 = (b-d)(g+h)
    U[1, 6] = 1  # b
    U[3, 6] = -1 # d
    V[2, 6] = 1  # g
    V[3, 6] = 1  # h
    W[0, 6] = -1 # C11
    
    return U, V, W

def _trivial_1x2_2x2():
    """
    Trivial decomposition of 1x2 * 2x2 = 1x2.
    A = [a, b], B = [[e, f], [g, h]]
    C = [ae+bg, af+bh]
    
    Rank 4:
    Term 1: a*e -> C[0]
    Term 2: b*g -> C[0]
    Term 3: a*f -> C[1]
    Term 4: b*h -> C[1]
    
    U: (1*2, 4) = (2, 4), V: (2*2, 4) = (4, 4), W: (2*1, 4) = (2, 4)
    U[i*2+j, r] where i=0, j=0,1
    V[j*2+k, r] where j=0,1, k=0,1
    W[k*1+i, r] where k=0,1, i=0
    """
    U = np.zeros((2, 4), dtype=object)
    V = np.zeros((4, 4), dtype=object)
    W = np.zeros((2, 4), dtype=object)
    
    # Term 1: a * e -> C[0]
    U[0, 0] = 1  # a (i=0, j=0)
    V[0, 0] = 1  # e (j=0, k=0)
    W[0, 0] = 1  # C[0] (k=0, i=0)
    
    # Term 2: b * g -> C[0]
    U[1, 1] = 1  # b (i=0, j=1)
    V[2, 1] = 1  # g (j=1, k=0)
    W[0, 1] = 1  # C[0] (k=0, i=0)
    
    # Term 3: a * f -> C[1]
    U[0, 2] = 1  # a (i=0, j=0)
    V[1, 2] = 1  # f (j=0, k=1)
    W[1, 2] = 1  # C[1] (k=1, i=0)
    
    # Term 4: b * h -> C[1]
    U[1, 3] = 1  # b (i=0, j=1)
    V[3, 3] = 1  # h (j=1, k=1)
    W[1, 3] = 1  # C[1] (k=1, i=0)
    
    return U, V, W

def construct():
    """
    Construct rank-44 decomposition for 3x4 * 4x4 matrix multiplication.
    
    U: (3*4, rank) = (12, rank)
    V: (4*4, rank) = (16, rank)
    W: (4*3, rank) = (12, rank)
    
    W convention: W[k*n+i, r] where n=3
    """
    n, m, p = DIMENSIONS  # n=3, m=4, p=4
    
    # We'll build the decomposition in blocks.
    # For i=0,1 (top 2 rows of A), use Strassen on 2x2 * 2x2 sub-blocks.
    # For i=2 (bottom row), use trivial 1x2 * 2x2.
    
    # Partition:
    # A rows: i in {0,1,2}
    # A columns: j in {0,1,2,3} -> split into {0,1} and {2,3}
    # B rows: j in {0,1,2,3} -> split into {0,1} and {2,3}
    # B cols: k in {0,1,2,3} -> split into {0,1} and {2,3}
    
    # C[i, k] = sum_j A[i,j] B[j,k]
    
    # For i=0,1:
    # C[i, 0:2] = A[i, 0:2] * B[0:2, 0:2] + A[i, 2:4] * B[2:4, 0:2]
    # C[i, 2:4] = A[i, 0:2] * B[0:2, 2:4] + A[i, 2:4] * B[2:4, 2:4]
    
    # For the 2x2 * 2x2 blocks (when i=0,1), we group the two rows.
    # A[0:2, 0:2] is 2x2, B[0:2, 0:2] is 2x2, result C[0:2, 0:2] is 2x2.
    # Use Strassen (rank 7).
    
    # Similarly for other blocks.
    
    # Let's build up the full U, V, W matrices.
    
    # Total rank: 4 Strassen blocks (rank 7 each) + 4 trivial blocks (rank 4 each) = 28 + 16 = 44
    rank = 44
    
    U = np.zeros((n * m, rank), dtype=object)
    V = np.zeros((m * p, rank), dtype=object)
    W = np.zeros((p * n, rank), dtype=object)
    
    col_offset = 0
    
    # Strassen block template
    S_U, S_V, S_W = _strassen_2x2()
    # Trivial block template
    T_U, T_V, T_W = _trivial_1x2_2x2()
    
    # Define the 4 Strassen blocks for i=0,1:
    # Block (i_range, j_range, k_range)
    # i_range: 0:2, j_range: 0:2 or 2:4, k_range: 0:2 or 2:4
    
    strassen_blocks = [
        # (i_start, j_start, k_start)
        (0, 0, 0),  # A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
        (0, 0, 2),  # A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
        (0, 2, 0),  # A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
        (0, 2, 2),  # A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    ]
    
    for i_s, j_s, k_s in strassen_blocks:
        for r in range(7):
            # U: A[i, j] where i in i_s:i_s+2, j in j_s:j_s+2
            # U index: i*m + j
            for ii in range(2):
                for jj in range(2):
                    val = S_U[ii * 2 + jj, r]
                    if val != 0:
                        U[(i_s + ii) * m + (j_s + jj), col_offset + r] += val
            
            # V: B[j, k] where j in j_s:j_s+2, k in k_s:k_s+2
            # V index: j*p + k
            for jj in range(2):
                for kk in range(2):
                    val = S_V[jj * 2 + kk, r]
                    if val != 0:
                        V[(j_s + jj) * p + (k_s + kk), col_offset + r] += val
            
            # W: C[i, k] where i in i_s:i_s+2, k in k_s:k_s+2
            # W index: k*n + i
            for kk in range(2):
                for ii in range(2):
                    val = S_W[kk * 2 + ii, r]
                    if val != 0:
                        W[(k_s + kk) * n + (i_s + ii), col_offset + r] += val
        
        col_offset += 7
    
    # Trivial blocks for i=2:
    # C[2, 0:2] = A[2, 0:2] * B[0:2, 0:2] + A[2, 2:4] * B[2:4, 0:2]
    # C[2, 2:4] = A[2, 0:2] * B[0:2, 2:4] + A[2, 2:4] * B[2:4, 2:4]
    
    trivial_blocks = [
        # (j_start, k_start)
        (0, 0),  # A[2, 0:2] * B[0:2, 0:2] -> C[2, 0:2]
        (0, 2),  # A[2, 0:2] * B[0:2, 2:4] -> C[2, 2:4]
        (2, 0),  # A[2, 2:4] * B[2:4, 0:2] -> C[2, 0:2]
        (2, 2),  # A[2, 2:4] * B[2:4, 2:4] -> C[2, 2:4]
    ]
    
    for j_s, k_s in trivial_blocks:
        for r in range(4):
            # U: A[2, j] where j in j_s:j_s+2
            # U index: 2*m + j
            for jj in range(2):
                val = T_U[jj, r]
                if val != 0:
                    U[2 * m + (j_s + jj), col_offset + r] += val
            
            # V: B[j, k] where j in j_s:j_s+2, k in k_s:k_s+2
            # V index: j*p + k
            for jj in range(2):
                for kk in range(2):
                    val = T_V[jj * 2 + kk, r]
                    if val != 0:
                        V[(j_s + jj) * p + (k_s + kk), col_offset + r] += val
            
            # W: C[2, k] where k in k_s:k_s+2
            # W index: k*n + 2
            for kk in range(2):
                val = T_W[kk, r]
                if val != 0:
                    W[(k_s + kk) * n + 2, col_offset + r] += val
        
        col_offset += 4
    
    # Convert to lists
    U_list = [[int(U[i, j]) if U[i, j] == int(U[i, j]) else U[i, j] for j in range(rank)] for i in range(n * m)]
    V_list = [[int(V[i, j]) if V[i, j] == int(V[i, j]) else V[i, j] for j in range(rank)] for i in range(m * p)]
    W_list = [[int(W[i, j]) if W[i, j] == int(W[i, j]) else W[i, j] for j in range(rank)] for i in range(p * n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
