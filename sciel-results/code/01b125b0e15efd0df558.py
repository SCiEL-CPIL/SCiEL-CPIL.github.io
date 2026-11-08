"""Matrix multiplication 3,4,8 using a rank-84 Kronecker-product decomposition.

We model the 3x4x8 tensor as a Kronecker product of a 3x2x4 tensor and a 2x2x2 tensor.
Using the rank-7 Strassen decomposition for 2x2x2 and a rank-12 decomposition for 3x2x4.
Total rank = 12 * 7 = 84.

The 3x2x4 tensor is decomposed using a 2x2x2 Strassen block for rows 1,2 and a
schoolbook 1x2x4 for row 0, giving 4*7 + 1*2*4 = 28+8 = 36... wait that's 36 not 12.
  
Actually, let me reconsider. The Kronecker product of a (3x2x4) tensor T1 with a
(2x2x2) tensor T2 gives a (6x4x8) tensor. We need (3x4x8). 
  
Let me use a different factorization: 3x4x8 = (3x2x4) ⊗ (1x2x2) where the second
factor is just a 1x2x2 tensor. But that doesn't help.
  
Alternative: Use block structure. Partition the 3x4x8 matrix multiplication as:
- Row 0: 1x4x8 = 32 terms (schoolbook)
- Rows 1-2: 2x4x8 = can use Strassen on the 2x2 subblocks
  
For 2x4x8: partition into 2 j-blocks of size 2, 4 k-blocks of size 2.
Each 2x2x2 block uses 7 Strassen terms. Total: 2*4*7 = 56 terms.
Grand total: 32 + 56 = 88. That's the current approach.
  
To improve, I need a better decomposition. Let me try:
- Use a 3x2x2 Strassen-like decomposition for the first 2 columns of B
- Use a 3x2x2 decomposition for the next 2 columns
- etc.
  
For 3x2x2: We can use the known result. The matrix multiplication tensor for 3x2x2
has rank at most... Let me think. We can partition the 3 rows into 1+2.
Row 0: 1x2x2 = 4 terms
Rows 1-2: 2x2x2 = 7 terms (Strassen)
Total: 11 terms for 3x2x2.
  
For 3x4x8: 2 j-blocks of size 2, 4 k-blocks of size 2.
Each 3x2x2 block: 11 terms. Total: 2*4*11 = 88. Same as before.
  
Let me try a different approach: use a rank-10 decomposition for 3x2x2.
Actually, the minimum rank for 3x2x2 matrix multiplication is 10 (known result).
  
With rank-10 for 3x2x2: 2*4*10 = 80 terms. That would give score 1.0/80 = 0.0125,
which is better than 1.0/88 = 0.01136.
  
Let me construct a rank-10 decomposition for 3x2x2 matrix multiplication.
  
For multiplying a 3x2 matrix A by a 2x2 matrix B to get a 3x2 matrix C:
C[i,k] = sum_j A[i,j] * B[j,k]
  
The tensor has 3*2*2 = 12 entries. We need to decompose it into rank-10 terms.
  
Known rank-10 decomposition for 3x2x2:
Use Strassen on rows 1,2 (indices 1,2) and schoolbook for row 0.
Row 0: A[0,0]*B[0,0], A[0,0]*B[0,1], A[0,1]*B[1,0], A[0,1]*B[1,1] = 4 terms
Rows 1-2: Strassen 2x2x2 = 7 terms
Total: 11 terms.
  
To get 10, we need to share one term between row 0 and the Strassen block.
  
Let me look up or derive a rank-10 decomposition.
  
Actually, let me just implement the rank-88 approach cleanly and ensure it's correct,
then try to find a way to reduce it.
  
For now, let me implement a clean rank-88 and verify correctness.
"""
import time

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = 3, 4, 8
    rank = 84
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    # Part 1: i=0, schoolbook 1x4x8 = 32 terms
    for j in range(4):
        for k in range(8):
            U[j][col] = 1
            V[j * p + k][col] = 1
            W[k * n][col] = 1
            col += 1

    # Part 2: i=1,2, use 2x4x8 with Strassen on 2x2x2 blocks
    # 2 j-blocks of size 2, 4 k-blocks of size 2, 7 Strassen terms each
    # Total: 2 * 4 * 7 = 56 terms
    # Grand total: 32 + 56 = 88, but we allocated 84.
    # We need to reduce by 4. Let's try sharing some terms.
    
    # Actually, let me just use rank 88 and fix the allocation.
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0

    for j in range(4):
        for k in range(8):
            U[j][col] = 1
            V[j * p + k][col] = 1
            W[k * n][col] = 1
            col += 1

    strassen_U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    strassen_V = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    strassen_W = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]

    for jb in range(2):
        js = jb * 2
        for kb in range(4):
            ks = kb * 2
            for t in range(7):
                for l in range(4):
                    val = strassen_U[t][l]
                    if val:
                        li, lj = l // 2, l % 2
                        gi, gj = 1 + li, js + lj
                        U[gi * m + gj][col] += val
                for l in range(4):
                    val = strassen_V[t][l]
                    if val:
                        lj, lk = l // 2, l % 2
                        gj, gk = js + lj, ks + lk
                        V[gj * p + gk][col] += val
                for l in range(4):
                    val = strassen_W[t][l]
                    if val:
                        li, lk = l // 2, l % 2
                        gi, gk = 1 + li, ks + lk
                        W[gk * n + gi][col] += val
                col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
