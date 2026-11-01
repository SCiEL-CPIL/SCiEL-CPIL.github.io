"""
Matrix multiplication 2x4 x 4x4 tensor decomposition.

Strategy:
1. Start with a Strassen-based block decomposition (rank 28).
2. Use iterative rank reduction via structured random perturbations
   and verification, trying to find a valid decomposition at lower rank.
3. If no reduction found within time budget, return the best known (rank 28).

The tensor for multiplying A (2x4) by B (4x4) has entries T[i, j, k, l] = 1
if the (i,l) entry of A*B depends on A[i,j]*B[j,k]. Specifically,
(A*B)[i, l] = sum_j A[i,j] * B[j, l].

So the tensor is T[i, j, l] where i in 0..1, j in 0..3, l in 0..3.
Wait, let me re-read the contract.

The contract says:
- U has shape (n*m, rank) = (8, rank)
- V has shape (m*p, rank) = (16, rank)
- W has shape (p*n, rank) = (8, rank)

The decomposition is: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]

This should equal the (i, k) entry of A*B where A is n x m and B is m x p.
(A*B)[i, k] = sum_j A[i,j] * B[j,k]

So the tensor T[i, j, k] = 1 for all i, j, k (since every A[i,j] multiplies B[j,k]).
Wait, that's the schoolbook. The tensor has T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p).

The decomposition requires:
sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i, j, k.

For schoolbook: rank = n*m*p = 32.

For Strassen on 2x4 x 4x4:
We can view A as [A1 | A2] where A1, A2 are 2x2.
B as [[B11, B12], [B21, B22]] where each is 2x2.

(A*B)[i, l] = sum_{j=0}^{3} A[i,j] * B[j,l]
= sum_{j=0}^{1} A[i,j] * B[j,l] + sum_{j=2}^{3} A[i,j] * B[j,l]

Let's index j as 0,1 for first block and 2,3 for second block.
Let j' = j % 2, block = j // 2.

A[i, block*2 + j'] * B[block*2 + j', l]

For a fixed block b (0 or 1), we have a 2x2 x 2x2 multiplication.
Using Strassen, each 2x2 x 2x2 needs 7 products.
So total rank = 2 * 7 = 14? No, that's not right.

Wait, Strassen is for 2x2 x 2x2. Here we have two independent 2x2 x 2x2 multiplications?
No, because B's columns are shared. Let me think again.

Actually, A is 2x4, B is 4x4. 
A = [A_left | A_right] where A_left is 2x2, A_right is 2x2.
B = [[B_topleft, B_topright], [B_botleft, B_botright]] each 2x2.

(A*B)[:, 0:2] = A_left * B_topleft + A_right * B_botleft
(A*B)[:, 2:4] = A_left * B_topright + A_right * B_botright

So we need:
C_left = A_left * B_topleft + A_right * B_botleft
C_right = A_left * B_topright + A_right * B_botright

Each of these is a sum of two 2x2 x 2x2 products.
Using Strassen for each 2x2 x 2x2:
A_left * B_topleft: 7 products
A_right * B_botleft: 7 products
A_left * B_topright: 7 products
A_right * B_botright: 7 products

But we can reuse some Strassen products? No, they are independent.
Total: 4 * 7 = 28 products.

So rank 28 is achievable via Strassen blocks.

Let me construct this.

For a 2x2 x 2x2 multiplication X*Y = Z:
Strassen products:
M1 = (X00 + X11) * (Y00 + Y11)
M2 = (X10 + X11) * Y00
M3 = X00 * (Y10 - Y11)
M4 = X11 * (Y01 - Y00)
M5 = (X00 + X10) * Y11
M6 = (X11 - X00) * (Y00 + Y10)
M7 = (X01 - X11) * (Y01 + Y11)

Z00 = M1 + M4 - M5 + M7
Z01 = M3 + M5
Z10 = M2 + M4
Z11 = M1 - M2 + M3 + M6

For our case:
We need to compute:
C_left[i, l] for i in 0,1 and l in 0,1:
C_left[i, l] = sum_{j=0,1} A[i,j] * B[j,l] + sum_{j=0,1} A[i, 2+j] * B[2+j, l]

Let A_left = A[:, 0:2], A_right = A[:, 2:4]
B_tl = B[0:2, 0:2], B_bl = B[2:4, 0:2]

C_left = A_left * B_tl + A_right * B_bl

Using Strassen for A_left * B_tl (7 products) and A_right * B_bl (7 products).
Similarly for C_right.

Total 28 products.

Let me code this up.
"""

import time
import random
import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # Start with schoolbook rank 32
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    # Try Strassen block decomposition for rank 28
    strassen_rank = _strassen_decomposition(n, m, p)
    if strassen_rank is not None:
        U, V, W = strassen_rank
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}

def _strassen_decomposition(n, m, p):
    """
    For n=2, m=4, p=4:
    Split A (2x4) into A_left (2x2) and A_right (2x2).
    Split B (4x4) into four 2x2 blocks.
    
    C = A*B
    C[:, 0:2] = A_left * B[0:2, 0:2] + A_right * B[2:4, 0:2]
    C[:, 2:4] = A_left * B[0:2, 2:4] + A_right * B[2:4, 2:4]
    
    Each 2x2*2x2 uses Strassen (7 products).
    Total: 4 * 7 = 28.
    """
    if n != 2 or m != 4 or p != 4:
        return None
    
    rank = 28
    U = [[0] * rank for _ in range(8)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(8)]
    
    col = 0
    
    # Helper: add Strassen decomposition for X (2x2) * Y (2x2)
    # X indices in U: x_base + i*2 + j (i,j in 0,1)
    # Y indices in V: y_base + i*2 + j
    # Z indices in W: z_base + i*2 + j
    # Returns number of columns used
    def add_strassen(x_base, y_base, z_base):
        nonlocal col
        # Strassen products:
        # M1 = (X00 + X11) * (Y00 + Y11)
        # M2 = (X10 + X11) * Y00
        # M3 = X00 * (Y10 - Y11)
        # M4 = X11 * (Y01 - Y00)
        # M5 = (X00 + X10) * Y11
        # M6 = (X11 - X00) * (Y00 + Y10)
        # M7 = (X01 - X11) * (Y01 + Y11)
        
        # M1
        U[x_base + 0][col] = 1
        U[x_base + 3][col] = 1
        V[y_base + 0][col] = 1
        V[y_base + 3][col] = 1
        W[z_base + 0][col] = 1
        W[z_base + 2][col] = 1
        W[z_base + 1][col] = -1
        W[z_base + 3][col] = 1
        col += 1
        
        # M2
        U[x_base + 2][col] = 1
        U[x_base + 3][col] = 1
        V[y_base + 0][col] = 1
        W[z_base + 2][col] = 1
        W[z_base + 3][col] = -1
        col += 1
        
        # M3
        U[x_base + 0][col] = 1
        V[y_base + 2][col] = 1
        V[y_base + 3][col] = -1
        W[z_base + 1][col] = 1
        W[z_base + 3][col] = 1
        col += 1
        
        # M4
        U[x_base + 3][col] = 1
        V[y_base + 1][col] = 1
        V[y_base + 0][col] = -1
        W[z_base + 0][col] = 1
        W[z_base + 2][col] = 1
        col += 1
        
        # M5
        U[x_base + 0][col] = 1
        U[x_base + 2][col] = 1
        V[y_base + 3][col] = 1
        W[z_base + 0][col] = -1
        W[z_base + 1][col] = 1
        col += 1
        
        # M6
        U[x_base + 3][col] = 1
        U[x_base + 0][col] = -1
        V[y_base + 0][col] = 1
        V[y_base + 2][col] = 1
        W[z_base + 3][col] = 1
        col += 1
        
        # M7
        U[x_base + 1][col] = 1
        U[x_base + 3][col] = -1
        V[y_base + 1][col] = 1
        V[y_base + 3][col] = 1
        W[z_base + 0][col] = 1
        col += 1
    
    # A_left: rows 0-1, cols 0-1 -> U indices 0,1,2,3
    # A_right: rows 0-1, cols 2-3 -> U indices 4,5,6,7
    
    # B[0:2, 0:2]: V indices 0,1,2,3
    # B[0:2, 2:4]: V indices 4,5,6,7
    # B[2:4, 0:2]: V indices 8,9,10,11
    # B[2:4, 2:4]: V indices 12,13,14,15
    
    # C[:, 0:2]: W indices 0,1,2,3 (k*n+i for k=0,1 and i=0,1)
    # C[:, 2:4]: W indices 4,5,6,7 (k*n+i for k=2,3 and i=0,1)
    
    # C_left = A_left * B_tl + A_right * B_bl
    # A_left * B_tl: x_base=0, y_base=0, z_base=0
    add_strassen(0, 0, 0)
    # A_right * B_bl: x_base=4, y_base=8, z_base=0
    add_strassen(4, 8, 0)
    
    # C_right = A_left * B_tr + A_right * B_br
    # A_left * B_tr: x_base=0, y_base=4, z_base=4
    add_strassen(0, 4, 4)
    # A_right * B_br: x_base=4, y_base=12, z_base=4
    add_strassen(4, 12, 4)
    
    return U, V, W

def _verify(U, V, W, n, m, p):
    """
    Verify that sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i, j, k.
    """
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for c in range(rank):
                    total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(total - 1) > 1e-9:
                    return False
    return True
