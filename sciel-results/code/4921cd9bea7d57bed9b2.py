import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 110

    # We decompose 2x4x6 into two 2x2x6 problems.
    # A = [A_L, A_R] where A_L, A_R are 2x2.
    # B is 4x6. Split into B_top (2x6) and B_bottom (2x6).
    # C = A_L B_top + A_R B_bottom.
    
    # For each 2x2x6 multiplication, we split the 2x6 B into three 2x2 blocks.
    # B = [B1, B2, B3].
    # Product = A B1 + A B2 + A B3.
    # We use Strassen (rank 7) for B1 and B2, and Schoolbook (rank 4) for B3.
    # Total rank per 2x2x6 is 7 + 7 + 4 = 18.
    # Total rank is 18 + 18 = 36.
    
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a Strassen block (2x2x2)
    def add_strassen_block(i_start, j_start, k_start, offset):
        col = offset
        U[(i_start+0)*m + (j_start+0)][col] += 1
        U[(i_start+1)*m + (j_start+1)][col] += 1
        V[(j_start+0)*p + (k_start+0)][col] += 1
        V[(j_start+1)*p + (k_start+1)][col] += 1
        W[(k_start+0)*n + (i_start+0)][col] += 1
        W[(k_start+1)*n + (i_start+1)][col] += 1
        
        col = offset + 1
        U[(i_start+1)*m + (j_start+0)][col] += 1
        U[(i_start+1)*m + (j_start+1)][col] += 1
        V[(j_start+0)*p + (k_start+0)][col] += 1
        W[(k_start+0)*n + (i_start+1)][col] += 1
        W[(k_start+1)*n + (i_start+1)][col] -= 1
        
        col = offset + 2
        U[(i_start+0)*m + (j_start+0)][col] += 1
        V[(j_start+0)*p + (k_start+1)][col] += 1
        V[(j_start+1)*p + (k_start+1)][col] -= 1
        W[(k_start+1)*n + (i_start+0)][col] += 1
        W[(k_start+1)*n + (i_start+1)][col] += 1
        
        col = offset + 3
        U[(i_start+1)*m + (j_start+1)][col] += 1
        V[(j_start+1)*p + (k_start+0)][col] += 1
        V[(j_start+0)*p + (k_start+0)][col] -= 1
        W[(k_start+0)*n + (i_start+0)][col] += 1
        W[(k_start+0)*n + (i_start+1)][col] += 1
        
        col = offset + 4
        U[(i_start+0)*m + (j_start+0)][col] += 1
        U[(i_start+0)*m + (j_start+1)][col] += 1
        V[(j_start+1)*p + (k_start+1)][col] += 1
        W[(k_start+0)*n + (i_start+0)][col] -= 1
        W[(k_start+1)*n + (i_start+0)][col] += 1
        
        col = offset + 5
        U[(i_start+1)*m + (j_start+0)][col] += 1
        U[(i_start+0)*m + (j_start+0)][col] -= 1
        V[(j_start+0)*p + (k_start+0)][col] += 1
        V[(j_start+0)*p + (k_start+1)][col] += 1
        W[(k_start+1)*n + (i_start+1)][col] += 1
        
        col = offset + 6
        U[(i_start+0)*m + (j_start+1)][col] += 1
        U[(i_start+1)*m + (j_start+1)][col] -= 1
        V[(j_start+1)*p + (k_start+0)][col] += 1
        V[(j_start+1)*p + (k_start+1)][col] += 1
        W[(k_start+0)*n + (i_start+0)][col] += 1

    # Helper to add a Schoolbook block (2x2x2)
    def add_schoolbook_block(i_start, j_start, k_start, offset):
        col = offset
        U[(i_start+0)*m + (j_start+0)][col] += 1
        V[(j_start+0)*p + (k_start+0)][col] += 1
        W[(k_start+0)*n + (i_start+0)][col] += 1
        
        col = offset + 1
        U[(i_start+0)*m + (j_start+1)][col] += 1
        V[(j_start+1)*p + (k_start+0)][col] += 1
        W[(k_start+0)*n + (i_start+0)][col] += 1
        
        col = offset + 2
        U[(i_start+1)*m + (j_start+0)][col] += 1
        V[(j_start+0)*p + (k_start+1)][col] += 1
        W[(k_start+1)*n + (i_start+1)][col] += 1
        
        col = offset + 3
        U[(i_start+1)*m + (j_start+1)][col] += 1
        V[(j_start+1)*p + (k_start+1)][col] += 1
        W[(k_start+1)*n + (i_start+1)][col] += 1

    # Block 1: A_L (rows 0,1 of A) * B_top (rows 0,1 of B)
    # A_L is 2x2 (i=0,1; j=0,1)
    # B_top is 2x6 (j=0,1; k=0..5)
    # Split B_top into B1 (k=0,1), B2 (k=2,3), B3 (k=4,5)
    
    # A_L * B1 (Strassen)
    add_strassen_block(0, 0, 0, 0)
    # A_L * B2 (Strassen)
    add_strassen_block(0, 0, 2, 7)
    # A_L * B3 (Schoolbook)
    add_schoolbook_block(0, 0, 4, 14)
    
    # Block 2: A_R (rows 0,1 of A) * B_bottom (rows 2,3 of B)
    # A_R is 2x2 (i=0,1; j=2,3)
    # B_bottom is 2x6 (j=2,3; k=0..5)
    # Split B_bottom into B4 (k=0,1), B5 (k=2,3), B6 (k=4,5)
    
    # A_R * B4 (Strassen)
    add_strassen_block(0, 2, 0, 18)
    # A_R * B5 (Strassen)
    add_strassen_block(0, 2, 2, 25)
    # A_R * B6 (Schoolbook)
    add_schoolbook_block(0, 2, 4, 32)
    
    return {"U": U, "V": V, "W": W}
