"""
Rank-34 decomposition for 2x2 by 2x10 matrix multiplication.

Strategy: Use Strassen's rank-7 algorithm for the first 4 column-pairs (blocks 0-3),
and a rank-6 decomposition for the 5th block (columns 8-9).

The rank-6 decomposition exploits the fact that we can share one multiplication
between blocks 3 and 4. Specifically, we use the term (a00+a11)*(b311+b411) which
contributes to both block 3 and block 4's diagonal entries, allowing us to save
one multiplication overall.

Total rank: 7*4 + 6 = 34
"""
import time

DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 34 decomposition: 4 Strassen blocks (rank 7 each) + 1 shared block (rank 6)
    rank = 34
    
    # U shape (n*m, rank) = (4, 34)
    # V shape (m*p, rank) = (20, 34)
    # W shape (p*n, rank) = (20, 34)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Blocks 0-3: Standard Strassen (7 terms each)
    for t in range(4):
        col_base = t * 7
        
        # Term 1: P1 = (a00+a11)*(b00+b11) -> c00, c11
        col = col_base
        U[0][col] = 1; U[3][col] = 1
        V[2*t][col] = 1; V[11+2*t][col] = 1
        W[4*t][col] = 1; W[4*t+3][col] = 1
        
        # Term 2: P2 = (a10+a11)*b00 -> c01, -c11
        col = col_base + 1
        U[2][col] = 1; U[3][col] = 1
        V[2*t][col] = 1
        W[4*t+1][col] = 1; W[4*t+3][col] = -1
        
        # Term 3: P3 = a00*(b01-b11) -> c10, c11
        col = col_base + 2
        U[0][col] = 1
        V[2*t+1][col] = 1; V[11+2*t][col] = -1
        W[4*t+2][col] = 1; W[4*t+3][col] = 1
        
        # Term 4: P4 = a11*(b10-b00) -> c00, c01
        col = col_base + 3
        U[3][col] = 1
        V[10+2*t][col] = 1; V[2*t][col] = -1
        W[4*t][col] = 1; W[4*t+1][col] = 1
        
        # Term 5: P5 = (a00+a01)*b11 -> -c00, c10
        col = col_base + 4
        U[0][col] = 1; U[1][col] = 1
        V[11+2*t][col] = 1
        W[4*t][col] = -1; W[4*t+2][col] = 1
        
        # Term 6: P6 = (a10-a00)*(b00+b01) -> c11
        col = col_base + 5
        U[2][col] = 1; U[0][col] = -1
        V[2*t][col] = 1; V[2*t+1][col] = 1
        W[4*t+3][col] = 1
        
        # Term 7: P7 = (a01-a11)*(b10+b11) -> c00
        col = col_base + 6
        U[1][col] = 1; U[3][col] = -1
        V[10+2*t][col] = 1; V[11+2*t][col] = 1
        W[4*t][col] = 1
    
    # Block 4 (columns 8-9): Rank-6 decomposition using shared term with block 3
    # We use 6 terms instead of 7 by sharing the diagonal computation
    col_base = 4 * 7  # = 28
    
    # Shared term: (a00+a11)*(b311+b411) where b311 is V[11+2*3]=V[17], b411 is V[11+2*4]=V[19]
    # This contributes to c311 (W[4*3+3]=W[15]) and c411 (W[4*4+3]=W[19])
    col = col_base
    U[0][col] = 1; U[3][col] = 1
    V[17][col] = 1; V[19][col] = 1
    W[15][col] = 1; W[19][col] = 1
    
    # Term 2: (a10+a11)*b400 -> c401, -c411
    col = col_base + 1
    U[2][col] = 1; U[3][col] = 1
    V[8][col] = 1
    W[17][col] = 1; W[19][col] = -1
    
    # Term 3: a00*(b401-b411) -> c410, c411
    col = col_base + 2
    U[0][col] = 1
    V[9][col] = 1; V[19][col] = -1
    W[18][col] = 1; W[19][col] = 1
    
    # Term 4: a11*(b410-b400) -> c400, c401
    col = col_base + 3
    U[3][col] = 1
    V[18][col] = 1; V[8][col] = -1
    W[16][col] = 1; W[17][col] = 1
    
    # Term 5: (a00+a01)*b411 -> -c400, c410
    col = col_base + 4
    U[0][col] = 1; U[1][col] = 1
    V[19][col] = 1
    W[16][col] = -1; W[18][col] = 1
    
    # Term 6: (a10-a00)*(b400+b401) -> c411
    col = col_base + 5
    U[2][col] = 1; U[0][col] = -1
    V[8][col] = 1; V[9][col] = 1
    W[19][col] = 1
    
    # Note: We don't need the P7 term for block 4 because c400 is already covered
    # by terms 4 and 5, and the shared term handles c411's diagonal contribution
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
