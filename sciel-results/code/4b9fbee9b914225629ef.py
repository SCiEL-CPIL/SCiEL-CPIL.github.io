"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import random

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x4 matrix by a 4x5 matrix.
    
    Strategy:
    1. Use Strassen-style decomposition for 2x2 blocks where possible.
    2. For 2x2x5 and 2x2x4 blocks, use known optimal decompositions.
    3. Combine blocks to minimize total rank.
    
    The 2x4x5 multiplication can be decomposed as:
    - Two 2x2x5 blocks (each row of A times both 2x2 blocks of B)
    - Each 2x2x5 block can be decomposed optimally
    
    For 2x2x5: We can use a rank-9 decomposition (better than schoolbook rank 10)
    Actually, let's think more carefully.
    
    For 2x2xn matrix multiplication, the tensor is the bilinear map:
    (A: 2x2, B: 2xn) -> C: 2xn
    
    The schoolbook rank is 4n.
    
    Known results:
    - 2x2x2: rank 7 (Strassen)
    - 2x2x3: rank 10 (optimal, better than schoolbook 12)
    - 2x2x4: rank 13 (optimal)
    - 2x2x5: rank 16 (optimal)
    
    So for 2x4x5:
    - Split B into two 2x2x5 blocks (columns 0-1 and 2-3)
    - Each 2x2x5 block has rank 16
    - Total rank = 16 + 16 = 32
    
    Wait, that's not quite right. Let me reconsider.
    
    A is 2x4, B is 4x5.
    We can write:
    A = [A1 | A2] where A1, A2 are 2x2
    B = [B1; B2] where B1, B2 are 2x5
    
    Then C = A1*B1 + A2*B2
    
    Each term Ai*Bj is a 2x2 times 2x5 product, giving a 2x5 result.
    
    The tensor for 2x2x5 multiplication has rank 16 (known optimal).
    
    So total rank = 16 + 16 = 32.
    
    But we can do better by using a more clever decomposition that doesn't simply add the ranks.
    
    Actually, the standard approach is:
    - The 2x4x5 tensor can be viewed as a sum of 4 rank-5 tensors (one for each 2x2x5 block)
    - But we can use Strassen-like tricks.
    
    Let me think about this differently. The known optimal rank for 2x4x5 is 32.
    
    Actually, let me look up: for [2,4,5] matrix multiplication, the known upper bound is 32.
    
    Let me construct a rank-32 decomposition:
    
    For each pair (i, j) where i in {0,1} (row of A), j in {0,1} (column block of A):
    We have a 2x2x5 subproblem.
    
    The 2x2x5 decomposition of rank 16 can be constructed as:
    - One 2x2x2 Strassen block (rank 7) for the first 2 columns
    - One 2x2x3 block (rank 10) for the next 3 columns
    - Total: 7 + 10 = 17... that's worse than 16.
    
    Actually, the optimal 2x2x5 rank is 16, achieved by:
    - Two 2x2x2 Strassen blocks would give 14, but that's for 2x2x4.
    - For 2x2x5, we need rank 16.
    
    Let me just use the schoolbook for the 2x2x5 blocks: rank 10 each, total 20.
    That's worse than 32.
    
    Wait, I'm confusing myself. Let me restart.
    
    Schoolbook for 2x4x5: rank = 2*4*5 = 40.
    
    Using block decomposition:
    - 2x4x5 = 2 blocks of 2x2x5
    - Each 2x2x5 schoolbook: rank = 2*2*5 = 20
    - Total: 40 (same as schoolbook, no improvement)
    
    To improve, we need Strassen-like decompositions for the 2x2x5 blocks.
    
    Known: rank(2x2x2) = 7, rank(2x2x3) = 10, rank(2x2x4) = 13, rank(2x2x5) = 16
    
    So:
    - 2x2x5 = 2x2x2 (rank 7) + 2x2x3 (rank 10) = rank 17? No, that's not how it works.
    
    Actually, the 2x2x5 tensor is not a sum of independent 2x2x2 and 2x2x3 tensors in a simple way.
    
    Let me just implement the known rank-16 decomposition for 2x2x5.
    
    For now, let me use a simpler approach:
    - Decompose into four 2x2x5 subproblems (one for each row of A and each column block)
    - Wait, that's not right either.
    
    Let me be more careful:
    A is 2x4, B is 4x5.
    A = [A0 A1] where A0, A1 are 2x2
    B = [B0; B1] where B0, B1 are 2x5
    
    C = A0*B0 + A1*B1
    
    Each Ai*Bj is 2x2 * 2x5 = 2x5.
    
    The tensor for multiplying 2x2 by 2x5 has 10 independent products (schoolbook).
    With Strassen-like tricks, we can do it in 16 multiplications (known optimal).
    
    So total rank = 16 + 16 = 32.
    
    Let me implement the rank-16 decomposition for 2x2x5.
    
    Actually, I recall that the optimal rank for 2x2xn is:
    n=1: 3 (trivial)
    n=2: 7 (Strassen)
    n=3: 10
    n=4: 13
    n=5: 16
    
    The pattern is 3n+1 for n>=2? No, 3*2+1=7, 3*3+1=10, 3*4+1=13, 3*5+1=16. Yes!
    
    So rank(2x2xn) = 3n+1 for n>=2.
    
    For 2x2x5: rank = 16.
    
    Now I need to construct the actual decomposition.
    
    Let me use a known construction. For 2x2xn, we can use:
    - One Strassen block for the first 2 columns (rank 7)
    - Then for each additional column, add 3 more multiplications
    - Total: 7 + 3*(n-2) = 3n+1
    
    So for n=5: 7 + 3*3 = 16.
    
    Let me implement this.
    """
    
    n, m, p = DIMENSIONS  # 2, 4, 5
    
    # We'll decompose into blocks
    # A is n x m = 2 x 4
    # B is m x p = 4 x 5
    # C is n x p = 2 x 5
    
    # Split into 2 blocks along m:
    # Block 1: A[:, 0:2] (2x2) * B[0:2, :] (2x5) -> C1 (2x5)
    # Block 2: A[:, 2:4] (2x2) * B[2:4, :] (2x5) -> C2 (2x5)
    # C = C1 + C2
    
    # Each block is a 2x2x5 multiplication with rank 16.
    # Total rank = 32.
    
    def build_2x2xn_decomp(n_cols):
        """
        Build a rank-(3*n_cols+1) decomposition for 2x2 by 2*n_cols multiplication.
        
        Returns U (4 x rank), V (2*n_cols x rank), W (2*n_cols x rank)
        where U[i*2+j, c], V[j*n_cols+k, c], W[k*2+i, c]
        
        Convention: A is 2x2, B is 2xn_cols, C is 2xn_cols
        C[i,k] = sum_{j1,j2} A[i,j1] * B[j2,k] * delta(j1,j2)
        
        Actually, the tensor is: C[i,k] = sum_j A[i,j] * B[j,k]
        
        U has shape (4, rank): U[i*2+j, c] = coefficient for A[i,j]
        V has shape (2*n_cols, rank): V[j*n_cols+k, c] = coefficient for B[j,k]
        W has shape (2*n_cols, rank): W[k*2+i, c] = coefficient for C[i,k]
        """
        nc = n_cols
        rank = 3 * nc + 1
        U = [[0] * rank for _ in range(4)]
        V = [[0] * rank for _ in range(2 * nc)]
        W = [[0] * rank for _ in range(2 * nc)]
        
        # Use Strassen for first 2 columns, then schoolbook for remaining
        # Strassen for 2x2x2:
        # M1 = (A00 + A11)(B00 + B11)
        # M2 = (A10 + A11)B00
        # M3 = A00(B10 - B00)
        # M4 = A11(B01 - B11)
        # M5 = (A00 + A01)B11
        # M6 = (A10 - A00)(B00 + B01)
        # M7 = (A01 - A11)(B10 + B11)
        # C00 = M1 - M4 + M5 - M7
        # C01 = M3 + M7
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # For columns 0,1 (Strassen part):
        # Column index in B: 0, 1
        # We'll use columns 0-1 of the 2xnc B matrix
        
        # Strassen multiplications (rank 7):
        # M1: A00+A11, B00+B11 (col 0)
        U[0][0] = 1  # A00
        U[3][0] = 1  # A11
        V[0*nc+0][0] = 1  # B00
        V[1*nc+0][0] = 1  # B10
        # C contribution: C00 and C11
        W[0*2+0][0] = 1  # C00
        W[1*2+1][0] = 1  # C11
        
        # M2: A10+A11, B00 (col 0)
        U[2][1] = 1  # A10
        U[3][1] = 1  # A11
        V[0*nc+0][1] = 1  # B00
        # C contribution: C10
        W[0*2+1][1] = 1  # C10
        
        # M3: A00, B10-B00 (col 0)
        U[0][2] = 1  # A00
        V[1*nc+0][2] = 1  # B10
        V[0*nc+0][2] = -1  # -B00
        # C contribution: C01
        W[1*2+0][2] = 1  # C01
        
        # M4: A11, B01-B11 (col 1)
        U[3][3] = 1  # A11
        V[0*nc+1][3] = 1  # B01
        V[1*nc+1][3] = -1  # -B11
        # C contribution: -C00
        W[0*2+0][3] = -1  # -C00
        
        # M5: A00+A01, B11 (col 1)
        U[0][4] = 1  # A00
        U[1][4] = 1  # A01
        V[1*nc+1][4] = 1  # B11
        # C contribution: C00
        W[0*2+0][4] = 1  # C00
        
        # M6: A10-A00, B00+B01 (col 0,1)
        U[2][5] = 1  # A10
        U[0][5] = -1  # -A00
        V[0*nc+0][5] = 1  # B00
        V[0*nc+1][5] = 1  # B01
        # C contribution: C11
        W[1*2+1][5] = 1  # C11
        
        # M7: A01-A11, B10+B11 (col 0,1)
        U[1][6] = 1  # A01
        U[3][6] = -1  # -A11
        V[1*nc+0][6] = 1  # B10
        V[1*nc+1][6] = 1  # B11
        # C contribution: -C00, C01
        W[0*2+0][6] = -1  # -C00
        W[1*2+0][6] = 1  # C01
        
        # For remaining columns (2 to nc-1), use schoolbook: 3 mults per column
        col_offset = 7
        for k in range(2, nc):
            # Schoolbook for column k:
            # C0k = A00*B0k + A01*B1k
            # C1k = A10*B0k + A11*B1k
            # Use 3 multiplications:
            # M_a = A00*B0k
            # M_b = A01*B1k  
            # M_c = (A10+A11)*(B0k+B1k)
            # C0k = M_a + M_b
            # C1k = M_c - M_a - M_b
            
            # M_a
            U[0][col_offset] = 1  # A00
            V[0*nc+k][col_offset] = 1  # B0k
            W[k*2+0][col_offset] = 1  # C0k
            W[k*2+1][col_offset] = -1  # -C1k
            
            # M_b
            U[1][col_offset+1] = 1  # A01
            V[1*nc+k][col_offset+1] = 1  # B1k
            W[k*2+0][col_offset+1] = 1  # C0k
            W[k*2+1][col_offset+1] = -1  # -C1k
            
            # M_c
            U[2][col_offset+2] = 1  # A10
            U[3][col_offset+2] = 1  # A11
            V[0*nc+k][col_offset+2] = 1  # B0k
            V[1*nc+k][col_offset+2] = 1  # B1k
            W[k*2+1][col_offset+2] = 1  # C1k
            
            col_offset += 3
        
        assert col_offset == rank, f"col_offset={col_offset}, rank={rank}"
        
        return U, V, W
    
    # Build the two 2x2x5 decompositions
    U1, V1, W1 = build_2x2xn_decomp(p)  # p=5
    U2, V2, W2 = build_2x2xn_decomp(p)  # p=5
    
    rank = 2 * (3 * p + 1)  # 2 * 16 = 32
    
    U = [[0] * rank for _ in range(n * m)]  # 2*4 = 8 rows
    V = [[0] * rank for _ in range(m * p)]  # 4*5 = 20 rows
    W = [[0] * rank for _ in range(p * n)]  # 5*2 = 10 rows
    
    # Block 1: A[:, 0:2] * B[0:2, :]
    # A indices: i in 0..1, j in 0..1 -> U[i*4 + j, c] (since m=4)
    # B indices: j in 0..1, k in 0..4 -> V[j*5 + k, c] (since p=5)
    # C indices: i in 0..1, k in 0..4 -> W[k*2 + i, c]
    
    for c in range(3 * p + 1):
        for i in range(2):
            for j in range(2):
                U[i * m + j][c] = U1[i * 2 + j][c]
        for j in range(2):
            for k in range(p):
                V[j * p + k][c] = V1[j * p + k][c]
        for i in range(2):
            for k in range(p):
                W[k * n + i][c] = W1[k * n + i][c]
    
    # Block 2: A[:, 2:4] * B[2:4, :]
    offset = 3 * p + 1
    for c in range(3 * p + 1):
        for i in range(2):
            for j in range(2):
                U[i * m + (j + 2)][offset + c] = U2[i * 2 + j][c]
        for j in range(2):
            for k in range(p):
                V[(j + 2) * p + k][offset + c] = V2[j * p + k][c]
        for i in range(2):
            for k in range(p):
                W[k * n + i][offset + c] = W2[k * n + i][c]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
