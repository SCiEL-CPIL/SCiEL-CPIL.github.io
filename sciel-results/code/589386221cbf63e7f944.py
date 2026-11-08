"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Block-Strassen decomposition for 4x5 * 5x5.
    Split A (4x5) into 2x2 blocks of size 2x5: A = [[A11, A12],[A21, A22]]
    Split B (5x5) into 2x2 blocks of size 5x5: B = [[B11, B12],[B21, B22]]
    Wait - that doesn't work cleanly. Let me reconsider.
    
    Actually, split A (4x5) into top 2 rows and bottom 2 rows: A_top (2x5), A_bot (2x5)
    Split B (5x5) into left 5 cols and right 5 cols... no, B is 5x5 so we can't split it evenly.
    
    Let me try: Split A into 2 blocks of 2x5. Split B into 2 blocks of 5x2 and 5x3.
    C = A*B is 4x5. C_top = A_top * B (2x5), C_bot = A_bot * B (2x5).
    C_top = A_top * B1 + A_top * B2 where B1 is 5x2, B2 is 5x3.
    This gives 4 multiplications of (2x5)*(5x2) and (2x5)*(5x3).
    That's not helpful directly.
    
    Better approach: Use the known rank for 4x5*5x5.
    The minimum rank is not trivially known. Let me try a different block structure.
    
    Split A (4x5) into 4 blocks of 1x5.
    Split B (5x5) into 5 blocks of 5x1.
    That's just schoolbook.
    
    Let me try: Use 2x2 block Strassen on a 4x4 subproblem.
    Take first 4 columns of B: B4 (5x4). Then A*B4 is 4x4.
    Use Strassen on 4x4*4x4 with rank 7 (if we can decompose 4x4*4x4 with rank 7).
    But 4x4*4x4 has known minimum rank... it's not 7.
    
    Actually, let me just use a simpler improvement: exploit that some entries can be shared.
    For 4x5*5x5, the schoolbook rank is 100.
    
    Let me try a 2x2 block decomposition where we split the 4x5 matrix into two 2x5 blocks
    and the 5x5 matrix into two 5x2 and one 5x1 block... this is getting complicated.
    
    Let me use a known technique: for multiplying an m x n matrix by an n x p matrix,
    if we can find a bilinear algorithm with fewer than m*n*p multiplications.
    
    For 4x5 * 5x5: Let's split A into 2 blocks of (2x5) and B into 2 blocks of (5x5) but
    use Strassen's trick on the 2x2 block structure.
    
    A = [A1; A2] where A1, A2 are 2x5
    B = B (5x5)
    C = [A1*B; A2*B]
    
    This is just 2 independent (2x5)*(5x5) multiplications, each of rank 50. Total rank 100.
    No improvement.
    
    Let me try splitting both:
    A (4x5) -> split rows into 2+2: A_top (2x5), A_bot (2x5)
    B (5x5) -> split columns into 2+3: B_left (5x2), B_right (5x3)
    
    C = A*B is 4x5.
    C[:, 0:2] = A * B_left (4x2)
    C[:, 2:5] = A * B_right (4x3)
    
    C[:, 0:2] = [A_top*B_left; A_bot*B_left]
    C[:, 2:5] = [A_top*B_right; A_bot*B_right]
    
    Each sub-multiplication:
    A_top*B_left: (2x5)*(5x2) = 2x2, rank 20
    A_bot*B_left: (2x5)*(5x2) = 2x2, rank 20
    A_top*B_right: (2x5)*(5x3) = 2x3, rank 30
    A_bot*B_right: (2x5)*(5x3) = 2x3, rank 30
    Total: 100. No improvement.
    
    I need to find actual shared structure. Let me use the fact that we can combine terms.
    
    For a (2x5)*(5x2) multiplication, can we do better than rank 20?
    The minimum rank for 2x2*2x2 is 7 (Strassen). For 2x5*5x2, it's more complex.
    
    Let me try a different approach: use the known result that the rank of matrix multiplication
    tensor [m,n,p] satisfies certain bounds. For [4,5,5], the rank is at most...
    
    Actually, let me just implement a careful 2x2 block Strassen on the 4x4 part.
    Take B' = B[:, 0:4] (5x4). Compute C' = A * B' (4x4) using a rank-7 decomposition for 4x4*4x4?
    But 4x4*4x4 doesn't have rank 7.
    
    Let me just keep the schoolbook for now but try to find any small improvement.
    """
    n, m, p = DIMENSIONS  # 4, 5, 5
    rank = n * m * p  # 100
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
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
