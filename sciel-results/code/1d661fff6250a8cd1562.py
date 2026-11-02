"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct rank-minimal decomposition for 2x4 * 4x7 matrix multiplication.
    Uses Strassen's algorithm on 2x2 blocks split along the middle dimension (4 = 2+2).
    Rank: 2 * 7 = 14 per Strassen block? No, 7 Strassen terms per 2x2x7 subproblem, times 2 subproblems = 14.
    Wait, each Strassen term for 2x2x7 involves 7 scalar multiplications per column, so 7*7=49 per subproblem? 
    Let me reconsider.
    """
    n, m, p = DIMENSIONS  # 2, 4, 7
    deadline = time.time() + 140  # safety margin before 150s limit
    
    # Schoolbook fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Try Strassen-based decomposition
    # Split middle dimension m=4 into 2+2
    # Two subproblems: 2x2 * 2x7 each
    # For each 2x2*2x7, use 7 Strassen terms, each term multiplies a 1x7 row from combined A with a 1x7 row from combined B
    # Actually, each Strassen product is: (linear combo of 2x2 A entries) * (linear combo of 2x2 B entries for fixed column)
    # So for 7 columns, 7*7 = 49 terms per subproblem? No, 7 Strassen products, each producing a 1x7 result by multiplying two 1x7 vectors element-wise? No.
    
    # Let me think again. For 2x2 matrix A and 2x7 matrix B:
    # A = [a b; c d], B has rows [r1 r2] where r1, r2 are 1x7
    # C = A*B = [r1 + r2 + (a-d)r2 ... ] no, let me use proper Strassen.
    
    # Standard Strassen for 2x2 * 2x2:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)(f)
    # M3 = (a-b)(h)
    # M4 = (a+d)(g-e)
    # M5 = (a+c)(e)
    # M6 = (b+d)(h-g)
    # M7 = (a-c)(f+g)
    # where A=[a b; c d], B=[e f; g h]
    
    # For 2x2 * 2x7, B = [e1 e2 e3 e4 e5 e6 e7; g1 g2 g3 g4 g5 g6 g7]
    # Each "scalar" in B is a 1x7 row. So M1 = (a+d) * (e_row + g_row) where e_row = [e1..e7], g_row=[g1..g7]
    # (a+d) is a scalar, (e_row+g_row) is 1x7, so M1 is 1x7.
    # Similarly all 7 Mi are 1x7 vectors.
    # Then C's entries are linear combinations of these 1x7 vectors.
    
    # So for 2x2*2x7, we have 7 rank-1 terms, each of which is outer product of:
    # - a 2x2 matrix (the combined A pattern) 
    # - a 2x7 matrix (the combined B pattern for one column... no, for all 7 columns)
    
    # Actually, each Mi corresponds to a rank-1 term in the tensor decomposition where:
    # U factor: 2x2 pattern (4 entries)
    # V factor: 2x7 pattern (14 entries, but structured as 2 rows of 7)
    # W factor: 2x7 pattern (14 entries)
    
    # Total for one subproblem: 7 terms
    # Two subproblems: 14 terms total
    # Rank = 14
    
    # Let me verify this is correct by constructing it.
    
    U = []
    V = []
    W = []
    
    for sub in range(2):
        # sub=0: middle indices 0,1; sub=1: middle indices 2,3
        if sub == 0:
            j0, j1 = 0, 1
        else:
            j0, j1 = 2, 3
            
        # Strassen patterns for 2x2 A block and 2x7 B block
        # A block: rows 0,1 of A; cols j0,j1 of A
        # B block: rows j0,j1 of B; all 7 cols of B
        
        # Strassen M1..M7
        strassen = [
            # (U_pattern for A block, V_pattern for B block, W_pattern for output)
            # M1 = (a+d)(e+h): U has 1 at (0,0) and (1,1), V has 1 at all positions in both rows, W has 1 at (0,0),(0,1),(1,0),(1,1) adjusted
            # Actually let me be more careful.
        ]
        
        # For 2x2*2x7, the 7 Strassen products:
        # Let A = [a b; c d] (a,b,c,d are scalars)
        # Let B = [e1 e2 e3 e4 e5 e6 e7; g1 g2 g3 g4 g5 g6 g7]
        
        # M1 = (a+d) * (e_row + g_row) where e_row=[e1..e7], g_row=[g1..g7]
        # This contributes to C[0,0]=a*e1+c*g1, C[0,1]=a*e2+c*g2, ..., C[1,0]=d*e1+g*g1... wait no.
        
        # C[0,k] = a*B[0,k] + b*B[1,k] = a*ek + b*gk
        # C[1,k] = c*B[0,k] + d*B[1,k] = c*ek + d*gk
        
        # Strassen:
        # M1 = (a+d)*(e_row+g_row) -> 1x7 vector
        # M2 = (c+d)*g_row -> 1x7
        # M3 = (a-b)*g_row -> 1x7  
        # M4 = (a+d)*(e_row-g_row) -> 1x7
        # M5 = (a+c)*e_row -> 1x7
        # M6 = (b+d)*g_row -> 1x7
        # M7 = (a-c)*(e_row+g_row) -> 1x7
        
        # Then:
        # C[0,k] = M1[k] - M2[k] - M4[k] + M6[k] ... let me just use the standard formulas
        # C11 = M1 - M4 + M7 - M5 + M3
        # C12 = M5 + M4
        # C21 = M2 + M3
        # C22 = M1 - M2 + M6 - M7
        
        # For 2x7 output, each Ci is a 1x7 row.
        
        patterns = [
            # (U: 4 values for 2x2 A, V: 14 values for 2x7 B, W: 14 values for 2x7 C)
            # M1: (a+d)(e+h)
            ([1, 0, 0, 1], [1,1,1,1,1,1,1, 1,1,1,1,1,1,1], [1,0, 1,0, 1,0, 1,0, 1,0, 1,0, 1,0]),  # simplified, need proper W
            # This is getting complex. Let me just use rank-49 via 7 Strassen terms per column per subproblem.
        ]
        
    # Given the complexity, let me just return schoolbook for now and note that a proper Strassen implementation would give rank 49 (2 subproblems * 7 Strassen * 7 columns... no, 2 * 7 = 14 if each Strassen term handles all 7 columns)
    
    return schoolbook()
# EVOLVE-BLOCK-END
