"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import time
import random

def _validate(U, V, W, n, m, p):
    """Verify that the tensor decomposition is exact."""
    rank = len(U[0]) if U else 0
    # Build the multiplication tensor T[i, j, k] = 1
    # and check sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                if abs(s - 1.0) > 1e-10:
                    return False
    return True

def _try_strassen_block(n, m, p):
    """
    Try to use a Strassen-like decomposition for 4x5x6.
    We can try to express 4x5x6 as a combination of smaller multiplications.
    One approach: 4x5x6 can be seen as two 2x5x3 blocks stacked (rows 0-1 and 2-3).
    Each 2x5x3 can potentially be done with rank less than 30.
    But actually, let's try a different decomposition.
    
    Actually, let's try: 4x5x6 = 2*(2x5x3). If we can do 2x5x3 with rank r,
    total rank is 2r. Schoolbook for 2x5x3 is 30, so 2*30=60 < 120.
    
    But can we do 2x5x3 better? Schoolbook is 30.
    Lower bound for 2x5x3 is max(2*5, 5*3, 3*2) = 15, but actual min rank is unknown.
    Let's just use schoolbook for the sub-blocks: 2*30 = 60.
    """
    # 4x5x6: rows 0-1 form 2x5x3, rows 2-3 form another 2x5x3
    # Each sub-problem: 2x5x3, schoolbook rank = 2*5*3 = 30
    # Total rank = 60
    n1, m1, p1 = 2, 5, 3
    sub_rank = n1 * m1 * p1
    total_rank = 2 * sub_rank
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # First block: rows 0-1 of the 4x5 matrix, all 5 cols, 3 cols of B
    # This multiplies A[0:2, :] (2x5) by B[:, 0:3] (5x3) -> C[0:2, 0:3] (2x3)
    col_offset = 0
    for i in range(n1):
        for j in range(m1):
            for k in range(p1):
                U[i * m + j][col_offset] = 1
                V[j * p + k][col_offset] = 1
                W[k * n + i][col_offset] = 1
                col_offset += 1
    
    # Second block: rows 2-3 of the 4x5 matrix, all 5 cols, 3 cols of B
    # This multiplies A[2:4, :] (2x5) by B[:, 0:3] (5x3) -> C[2:4, 0:3] (2x3)
    for i in range(n1):
        for j in range(m1):
            for k in range(p1):
                U[(i + n1) * m + j][col_offset] = 1
                V[j * p + k][col_offset] = 1
                W[k * n + (i + n1)][col_offset] = 1
                col_offset += 1
    
    # Now we need to handle columns 3-5 of B (which are 3 columns)
    # Wait, p=6, so B has 6 columns. I split into p1=3, so I need another block for k=3..5
    # Let me redo this more carefully.
    
    # Actually, let me reconsider. 4x5x6:
    # Split A into two 2x5 blocks (rows 0-1, 2-3)
    # Split B into two 5x3 blocks (cols 0-2, 3-5)
    # This gives 4 sub-problems of 2x5x3
    # Total rank = 4 * 30 = 120, same as schoolbook.
    
    # So splitting doesn't help with schoolbook. We need actual rank reduction.
    
    # Let me try a different approach: use the fact that some entries might be shared.
    # For now, return None to indicate no improvement found.
    return None

def _try_las_vergas():
    """
    Try to use a known low-rank decomposition.
    For 4x5x6, I don't have a specific known decomposition better than 120.
    Return None.
    """
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0  # Safety margin before 150s limit
    
    # Try various approaches
    best_result = None
    best_rank = n * m * p  # Schoolbook rank = 120
    
    # First, generate the schoolbook solution as baseline
    rank = best_rank
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
    
    best_result = {"U": U, "V": V, "W": W}
    
    # Try to find a better decomposition
    # For now, we don't have a verified better one, so return the baseline
    return best_result

# EVOLVE-BLOCK-END
