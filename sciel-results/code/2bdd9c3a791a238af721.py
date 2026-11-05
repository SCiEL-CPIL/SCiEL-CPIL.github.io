"""
Rank-124 exact decomposition for 5x5 x 5x5 matrix multiplication.

Uses a constructive tensor decomposition: the multiplication tensor T[i,j,k]
is decomposed as sum over r of u_r[i,j] * v_r[j,k] * w_r[k,i] where the
factor matrices follow the trace convention:
  U has shape (n*m, rank), indexed by (i*m+j, r)
  V has shape (m*p, rank), indexed by (j*p+k, r)  
  W has shape (p*n, rank), indexed by (k*n+i, r)

We partition the 5x5 matrices into a 3x3 top-left block and a 2x2 bottom-right block.
The cross blocks are handled by the schoolbook method (rank 12 each for 3x2 and 2x3).
The 3x3 block uses schoolbook (rank 27).
The 2x2 block uses Strassen's algorithm (rank 7).

Total rank: 27 + 12 + 12 + 7 = 58? No wait.
Actually, let's think about the partitioning more carefully.

For 5x5, we can partition as:
- A[0:3, 0:3] (3x3), A[0:3, 3:5] (3x2), A[3:5, 0:3] (2x3), A[3:5, 3:5] (2x2)
- Similarly for B and C.

The matrix multiplication C = A * B gives:
C[0:3, 0:3] = A[0:3, 0:3]*B[0:3, 0:3] + A[0:3, 3:5]*B[3:5, 0:3]
C[0:3, 3:5] = A[0:3, 0:3]*B[0:3, 3:5] + A[0:3, 3:5]*B[3:5, 3:5]
C[3:5, 0:3] = A[3:5, 0:3]*B[0:3, 0:3] + A[3:5, 3:5]*B[3:5, 0:3]
C[3:5, 3:5] = A[3:5, 0:3]*B[0:3, 3:5] + A[3:5, 3:5]*B[3:5, 3:5]

Each block multiplication requires its own decomposition. The total rank is the sum of ranks of all block multiplications since they are independent.

- A[0:3, 0:3]*B[0:3, 0:3]: 3x3 * 3x3, schoolbook rank 27
- A[0:3, 3:5]*B[3:5, 0:3]: 3x2 * 2x3, schoolbook rank 12
- A[0:3, 0:3]*B[0:3, 3:5]: 3x3 * 3x2, schoolbook rank 18
- A[0:3, 3:5]*B[3:5, 3:5]: 3x2 * 2x2, schoolbook rank 12
- A[3:5, 0:3]*B[0:3, 0:3]: 2x3 * 3x3, schoolbook rank 18
- A[3:5, 3:5]*B[3:5, 0:3]: 2x2 * 2x3, schoolbook rank 12
- A[3:5, 0:3]*B[0:3, 3:5]: 2x3 * 3x2, schoolbook rank 18
- A[3:5, 3:5]*B[3:5, 3:5]: 2x2 * 2x2, Strassen rank 7

Total: 27 + 12 + 18 + 12 + 18 + 12 + 18 + 7 = 124

This gives rank 124, which is an improvement over 125.
"""

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 124
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a schoolbook term for multiplying A[i1:i2, j1:j2] * B[j1:j2, k1:k2]
    # contributing to C[i1:i2, k1:k2]
    def add_schoolbook(i1, i2, j1, j2, k1, k2):
        nonlocal col
        for i in range(i1, i2):
            for j in range(j1, j2):
                for k in range(k1, k2):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
    
    # 3x3 * 3x3 -> C[0:3, 0:3]
    add_schoolbook(0, 3, 0, 3, 0, 3)  # rank 27
    
    # 3x2 * 2x3 -> C[0:3, 0:3]
    add_schoolbook(0, 3, 3, 5, 0, 3)  # rank 12
    
    # 3x3 * 3x2 -> C[0:3, 3:5]
    add_schoolbook(0, 3, 0, 3, 3, 5)  # rank 18
    
    # 3x2 * 2x2 -> C[0:3, 3:5]
    add_schoolbook(0, 3, 3, 5, 3, 5)  # rank 12
    
    # 2x3 * 3x3 -> C[3:5, 0:3]
    add_schoolbook(3, 5, 0, 3, 0, 3)  # rank 18
    
    # 2x2 * 2x3 -> C[3:5, 0:3]
    add_schoolbook(3, 5, 3, 5, 0, 3)  # rank 12
    
    # 2x3 * 3x2 -> C[3:5, 3:5]
    add_schoolbook(3, 5, 0, 3, 3, 5)  # rank 18
    
    # 2x2 * 2x2 -> C[3:5, 3:5] using Strassen (rank 7)
    # Local indices: i, j, k in {0, 1} map to global i+3, j+3, k+3
    # Strassen's 7 products for 2x2:
    # P1 = (A00 + A11)(B00 + B11)
    # P2 = (A10 + A11)B00
    # P3 = A00(B10 - B00)
    # P4 = A11(B01 - B11)
    # P5 = (A00 + A01)B11
    # P6 = (A10 - A00)(B00 + B01)
    # P7 = (A01 - A11)(B10 + B11)
    #
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # P1: (A00+A11)(B00+B11)
    # A00: i=3,j=3; A11: i=4,j=4
    # B00: j=3,k=3; B11: j=4,k=4
    U[3*5+3][col] = 1
    U[4*5+4][col] = 1
    V[3*5+3][col] = 1
    V[4*5+4][col] = 1
    W[3*5+3][col] = 1  # C00
    W[4*5+4][col] = 1  # C11
    col += 1
    
    # P2: (A10+A11)B00
    # A10: i=4,j=3; A11: i=4,j=4
    # B00: j=3,k=3
    U[4*5+3][col] = 1
    U[4*5+4][col] = 1
    V[3*5+3][col] = 1
    W[3*5+4][col] = 1  # C10
    W[4*5+4][col] = -1  # C11
    col += 1
    
    # P3: A00(B10-B00)
    # A00: i=3,j=3
    # B10: j=4,k=3; B00: j=3,k=3
    U[3*5+3][col] = 1
    V[4*5+3][col] = 1
    V[3*5+3][col] = -1
    W[4*5+3][col] = 1  # C01
    W[4*5+4][col] = 1  # C11
    col += 1
    
    # P4: A11(B01-B11)
    # A11: i=4,j=4
    # B01: j=3,k=4; B11: j=4,k=4
    U[4*5+4][col] = 1
    V[3*5+4][col] = 1
    V[4*5+4][col] = -1
    W[3*5+3][col] = 1  # C00
    W[3*5+4][col] = 1  # C10
    col += 1
    
    # P5: (A00+A01)B11
    # A00: i=3,j=3; A01: i=3,j=4
    # B11: j=4,k=4
    U[3*5+3][col] = 1
    U[3*5+4][col] = 1
    V[4*5+4][col] = 1
    W[4*5+3][col] = 1  # C01
    W[3*5+3][col] = -1  # C00
    col += 1
    
    # P6: (A10-A00)(B00+B01)
    # A10: i=4,j=3; A00: i=3,j=3
    # B00: j=3,k=3; B01: j=3,k=4
    U[4*5+3][col] = 1
    U[3*5+3][col] = -1
    V[3*5+3][col] = 1
    V[3*5+4][col] = 1
    W[4*5+4][col] = 1  # C11
    col += 1
    
    # P7: (A01-A11)(B10+B11)
    # A01: i=3,j=4; A11: i=4,j=4
    # B10: j=4,k=3; B11: j=4,k=4
    U[3*5+4][col] = 1
    U[4*5+4][col] = -1
    V[4*5+3][col] = 1
    V[4*5+4][col] = 1
    W[3*5+3][col] = 1  # C00
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
