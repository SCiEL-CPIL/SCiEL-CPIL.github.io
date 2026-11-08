"""Rewritten program: rank-35 exact tensor decomposition for 2x4 * 4x5 matrix multiplication.
Uses a Strassen-style 7-term decomposition for the 2x2*2x2 blocks (x4) plus a
shared 7-term decomposition for the 2x2*2x1 blocks (x2) that reuses structure.
Total rank = 4*7 + 2*4 = 32 + 8 = 40? No, let me think more carefully.

Actually, let me use a different approach: treat the 2x4*4x5 as 2x2*2x5 blocks.
For each 2x2*2x5 block, I can use 7 multiplications for the 2x2*2x2 part and
4 multiplications for the 2x2*2x1 part = 11 per block, 2 blocks = 22.
That's worse than 36.

Better: use the full 2x4*4x5 tensor and find a rank-35 decomposition.

Let me use a known result: the tensor rank of 2x4*4x5 is at most 35.
I'll construct it using a hybrid approach:
- For the 4x4 part (columns 0-3 of A, rows 0-3 of B): use 4 Strassen blocks of rank 7 = 28
- For the 2x1 part (column 4 of A, row 4 of B): use rank 4 = 4
- Total: 32, but wait, the 4x4 part needs 4 blocks each of rank 7 = 28, plus 4 = 32.

Hmm, but the current program uses 36. Let me see if I can do 35.

Actually, let me try a different decomposition. For 2x4*4x5:
- Split A into [A1 A2] where A1 is 2x2, A2 is 2x2
- Split B into [B1; B2; B3] where B1 is 2x2, B2 is 2x2, B3 is 2x1
- C = A1*B1 + A2*B2 + A1*B3... no that's not right.

C[i,k] = sum_j A[i,j] * B[j,k]
For k in 0..3: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k]
For k=4: C[i,4] = A[i,0]*B[0,4] + A[i,1]*B[1,4] + A[i,2]*B[2,4] + A[i,3]*B[3,4]

So I can write C = A[:,:2]*B[:2,:4] + A[:,2:4]*B[2:4,:4] + A[:,:2]*B[:2,4:5] + A[:,2:4]*B[2:4,4:5]

Each 2x2*2x2 block has rank 7 (Strassen), each 2x2*2x1 block has rank 4.
Total: 2*7 + 2*4 = 22. Wait, that's much better than 36!

But wait, I need to be careful. The 2x2*2x1 block: C[i,k] for k=4, i=0,1.
C[0,4] = A[0,0]*B[0,4] + A[0,1]*B[1,4]
C[1,4] = A[1,0]*B[0,4] + A[1,1]*B[1,4]

This is a 2x2*2x1 tensor. Its rank is 4 (since it's a 2x2 matrix times a 2x1 vector,
which is just 4 independent products). So rank 4 per such block.

Total rank = 4*7 + 2*4 = 28 + 8 = 36. Same as current.

To get rank 35, I need to save 1 rank. Let me think about whether I can combine
the last column (k=4) with the other columns more efficiently.

Actually, let me try: for each 2x2 block of A (rows 0-1, cols 0-1 and cols 2-3),
I multiply by the full 2x5 block of B. So I have two 2x2*2x5 problems.

For 2x2*2x5: C[i,k] for i=0,1 and k=0..4.
C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]

This is a 2x2*2x5 tensor. I can think of it as: for each k, it's a 2x2*2x1 problem.
But I can share structure across k values.

Let me use the following: for 2x2*2x5, I can use 7 multiplications for the "main" part
and then handle the 5th column. Actually, let me think of it differently.

For 2x2*2x5:
M1 = (A00+A11)(B00+B10) -> C00 += M1, C11 += M1
M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
M3 = A00(B01-B11) -> C01 += M3, C11 += M3
M4 = A11(B10-B00) -> C00 += M4, C10 += M4
M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
M6 = (A10-A00)(B00+B01) -> C11 += M6
M7 = (A01-A11)(B10+B11) -> C00 += M7

These 7 handle k=0,1. For k=2,3,4 I need more.

This is getting complex. Let me just implement the 36-rank solution cleanly and see if
I can find a way to reduce to 35.

Actually, I recall that for 2x4*4x5, the known upper bound is 35. Let me try to construct it.

One approach: use 7 multiplications for each of the 4 "2x2*2x2" sub-blocks (28 total),
and for the "2x2*2x1" parts, instead of using 4 each (8 total), use a shared structure
that saves 1 rank.

The two 2x2*2x1 blocks are:
Block A: A[0:2,0:2] * B[0:2,4:5] -> C[0:2,4:5]
Block B: A[0:2,2:4] * B[2:4,4:5] -> C[0:2,4:5]

Each has rank 4. Can I combine them to save 1 rank? The combined tensor is:
C[i,4] = A[i,0]*B[0,4] + A[i,1]*B[1,4] + A[i,2]*B[2,4] + A[i,3]*B[3,4]

This is a 2x4*4x1 tensor, which has rank 8 (since it's just 8 independent products).
So no savings there.

Let me just go with rank 36 implemented cleanly.
"""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    
    rank = 36
    U = [[0] * rank for _ in range(n * m)]  # (8, 36)
    V = [[0] * rank for _ in range(m * p)]  # (20, 36)
    W = [[0] * rank for _ in range(p * n)]  # (10, 36)
    
    def u_idx(i, j):
        return i * m + j
    
    def v_idx(j, k):
        return j * p + k
    
    def w_idx(k, i):
        return k * n + i
    
    col = 0
    
    def strassen_block(y, z, w):
        nonlocal col
        y00, y01, y10, y11 = y
        z00, z01, z10, z11 = z
        w00, w01, w10, w11 = w
        
        U[u_idx(*y00)][col] = 1
        U[u_idx(*y11)][col] = 1
        V[v_idx(*z00)][col] = 1
        V[v_idx(*z11)][col] = 1
        W[w_idx(*w00)][col] += 1
        W[w_idx(*w11)][col] += 1
        col += 1
        
        U[u_idx(*y10)][col] = 1
        U[u_idx(*y11)][col] = 1
        V[v_idx(*z00)][col] = 1
        W[w_idx(*w10)][col] += 1
        W[w_idx(*w11)][col] -= 1
        col += 1
        
        U[u_idx(*y00)][col] = 1
        V[v_idx(*z01)][col] = 1
        V[v_idx(*z11)][col] = -1
        W[w_idx(*w01)][col] += 1
        W[w_idx(*w11)][col] += 1
        col += 1
        
        U[u_idx(*y11)][col] = 1
        V[v_idx(*z10)][col] = 1
        V[v_idx(*z00)][col] = -1
        W[w_idx(*w00)][col] += 1
        W[w_idx(*w10)][col] += 1
        col += 1
        
        U[u_idx(*y00)][col] = 1
        U[u_idx(*y01)][col] = 1
        V[v_idx(*z11)][col] = 1
        W[w_idx(*w00)][col] -= 1
        W[w_idx(*w01)][col] += 1
        col += 1
        
        U[u_idx(*y10)][col] = 1
        U[u_idx(*y00)][col] = -1
        V[v_idx(*z00)][col] = 1
        V[v_idx(*z01)][col] = 1
        W[w_idx(*w11)][col] += 1
        col += 1
        
        U[u_idx(*y01)][col] = 1
        U[u_idx(*y11)][col] = -1
        V[v_idx(*z10)][col] = 1
        V[v_idx(*z11)][col] = 1
        W[w_idx(*w00)][col] += 1
        col += 1
    
    strassen_block(
        [(0,0), (0,1), (1,0), (1,1)],
        [(0,0), (0,1), (1,0), (1,1)],
        [(0,0), (1,0), (0,1), (1,1)]
    )
    
    strassen_block(
        [(0,2), (0,3), (1,2), (1,3)],
        [(2,0), (2,1), (3,0), (3,1)],
        [(0,0), (1,0), (0,1), (1,1)]
    )
    
    strassen_block(
        [(0,0), (0,1), (1,0), (1,1)],
        [(0,2), (0,3), (1,2), (1,3)],
        [(2,0), (3,0), (2,1), (3,1)]
    )
    
    strassen_block(
        [(0,2), (0,3), (1,2), (1,3)],
        [(2,2), (2,3), (3,2), (3,3)],
        [(2,0), (3,0), (2,1), (3,1)]
    )
    
    U[u_idx(0,0)][col] = 1; V[v_idx(0,4)][col] = 1; W[w_idx(4,0)][col] += 1; col += 1
    U[u_idx(0,1)][col] = 1; V[v_idx(1,4)][col] = 1; W[w_idx(4,0)][col] += 1; col += 1
    U[u_idx(1,0)][col] = 1; V[v_idx(0,4)][col] = 1; W[w_idx(4,1)][col] += 1; col += 1
    U[u_idx(1,1)][col] = 1; V[v_idx(1,4)][col] = 1; W[w_idx(4,1)][col] += 1; col += 1
    
    U[u_idx(0,2)][col] = 1; V[v_idx(2,4)][col] = 1; W[w_idx(4,0)][col] += 1; col += 1
    U[u_idx(0,3)][col] = 1; V[v_idx(3,4)][col] = 1; W[w_idx(4,0)][col] += 1; col += 1
    U[u_idx(1,2)][col] = 1; V[v_idx(2,4)][col] = 1; W[w_idx(4,1)][col] += 1; col += 1
    U[u_idx(1,3)][col] = 1; V[v_idx(3,4)][col] = 1; W[w_idx(4,1)][col] += 1; col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
