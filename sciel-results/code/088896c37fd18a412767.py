"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s limit
    n, m, p = DIMENSIONS  # 3, 5, 6
    
    # Use Strassen's algorithm for 3x3 sub-blocks
    # Split the 3x5x6 multiplication into blocks:
    # A is 3x5, B is 5x6
    # We can split A into left 3x2 and right 3x3 parts
    # B into top 2x6 and bottom 3x6 parts
    
    # Actually, let's use a cleaner approach:
    # For each pair of row-block (3x3) and column-block (3x3) in the result,
    # we can use Strassen's 7-multiplication method
    
    # The result C is 3x6. Split C into two 3x3 blocks: C_left (cols 0-2) and C_right (cols 3-5)
    # A is 3x5: split into A_left (cols 0-1) and A_right (cols 2-4)
    # B is 5x6: split into B_top (rows 0-1) and B_bottom (rows 2-4)
    
    # C_left = A_left @ B_top_left + A_right @ B_bottom_left
    # C_right = A_left @ B_top_right + A_right @ B_bottom_right
    
    # Where B_top_left is 2x3, B_top_right is 2x3, B_bottom_left is 3x3, B_bottom_right is 3x3
    
    # This doesn't directly give us 3x3 blocks. Let me try a different split.
    
    # Better approach: Use the fact that we can embed 3x5x6 into a larger structured problem
    # Or use direct rank reduction by combining terms
    
    # Let's use a direct approach: start with schoolbook and try to find linear dependencies
    # that allow us to reduce the rank
    
    # Schoolbook has rank 90. Let's see if we can do better.
    
    # For now, let's implement a simple improvement: 
    # Use the identity that for any i,j,k,l with j!=l:
    # A[i,j]*B[j,k] + A[i,l]*B[l,k] = (A[i,j]+A[i,l])*(B[j,k]+B[l,k]) - A[i,j]*B[l,k] - A[i,l]*B[j,k]
    # This doesn't directly help, but suggests we can combine terms.
    
    # Actually, let's just use schoolbook as a safe baseline and return it.
    # The key insight is that we need to find a valid low-rank decomposition.
    
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
    
    if time.time() > deadline - 5.0:
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
