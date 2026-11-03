"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    def add_strassen_block(U, V, W, i0, j0, k0, col_offset):
        """Add a 2x2x2 Strassen decomposition starting at (i0, j0, k0)."""
        # M1 = (A00+A11)(B00+B11)
        c = col_offset + 0
        U[i0*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        # M2 = (A10+A11)B00
        c = col_offset + 1
        U[(i0+1)*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += -1
        
        # M3 = A00(B01-B11)
        c = col_offset + 2
        U[i0*m + j0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += -1
        W[(k0+1)*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        # M4 = A11(B10-B00)
        c = col_offset + 3
        U[(i0+1)*m + (j0+1)][c] += 1
        V[(j0+1)*p + k0][c] += 1
        V[j0*p + k0][c] += -1
        W[k0*n + i0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        
        # M5 = (A00+A01)B11
        c = col_offset + 4
        U[i0*m + j0][c] += 1
        U[i0*m + (j0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += -1
        W[(k0+1)*n + i0][c] += 1
        
        # M6 = (A10-A00)(B00+B01)
        c = col_offset + 5
        U[(i0+1)*m + j0][c] += 1
        U[i0*m + j0][c] += -1
        V[j0*p + k0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        # M7 = (A01-A11)(B10+B11)
        c = col_offset + 6
        U[i0*m + (j0+1)][c] += 1
        U[(i0+1)*m + (j0+1)][c] += -1
        V[(j0+1)*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1

    # Strategy: Use 4 independent 2x2x2 Strassen blocks on a 3x5x5 problem
    # We have i=3, j=5, k=5
    # We can place blocks at:
    # (i=0, j=0, k=0), (i=0, j=2, k=0), (i=0, j=0, k=2), (i=0, j=2, k=2)
    # These cover i=0,1, j=0,1,2,3, k=0,1,2,3
    # Remaining uncovered triples are those involving i=2, or j=4, or k=4
    # Specifically:
    # i=0, j=4, k=0..4 (5)
    # i=0, j=0..4, k=4 (5)
    # i=1, j=4, k=0..4 (5)
    # i=1, j=0..4, k=4 (5)
    # i=2, j=0..4, k=0..4 (25)
    # Total uncovered = 5+5+5+5+25 = 45
    # 4 Strassen blocks = 4 * 7 = 28
    # Total rank = 28 + 45 = 73
    
    # Let's check if we can do better.
    # What if we use a different set of blocks?
    # Can we cover i=2 as well?
    # Strassen blocks are 2x2x2.
    # We have height 3 (i). So blocks can be at i0=0 or i0=1.
    # If we place a block at i0=1, it covers i=1,2.
    # Let's try:
    # (i0=0, j0=0, k0=0), (i0=0, j0=2, k0=0), (i0=0, j0=0, k0=2), (i0=0, j0=2, k0=2)
    # This is the same as above.
    
    # Alternative:
    # (i0=0, j0=0, k0=0), (i0=0, j0=2, k0=0), (i0=0, j0=0, k0=2), (i0=1, j0=2, k0=2)
    # Blocks:
    # B1: i=0,1, j=0,1, k=0,1
    # B2: i=0,1, j=2,3, k=0,1
    # B3: i=0,1, j=0,1, k=2,3
    # B4: i=1,2, j=2,3, k=2,3
    
    # Covered i=0:
    # j=0,1; k=0,1,2,3 (4x2=8)
    # j=2,3; k=0,1 (2x2=4)
    # Total 12 for i=0.
    
    # Covered i=1:
    # j=0,1; k=0,1,2,3 (8)
    # j=2,3; k=0,1,2,3 (8)
    # Total 16 for i=1.
    
    # Covered i=2:
    # j=2,3; k=2,3 (4)
    # Total 4 for i=2.
    
    # Total covered = 12 + 16 + 4 = 32
    # Total triples = 3*5*5 = 75
    # Uncovered = 75 - 32 = 43
    # Rank = 4*7 + 43 = 28 + 43 = 71
    
    # This is better than 73!
    
    blocks = [
        (0, 0, 0), # i=0,1; j=0,1; k=0,1
        (0, 2, 0), # i=0,1; j=2,3; k=0,1
        (0, 0, 2), # i=0,1; j=0,1; k=2,3
        (1, 2, 2), # i=1,2; j=2,3; k=2,3
    ]
    
    # Check validity of blocks (no overlap, within bounds)
    covered = [[[False]*p for _ in range(m)] for _ in range(n)]
    valid = True
    for i0, j0, k0 in blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    if i0+di >= n or j0+dj >= m or k0+dk >= p:
                        valid = False
                    elif covered[i0+di][j0+dj][k0+dk]:
                        valid = False
                    else:
                        covered[i0+di][j0+dj][k0+dk] = True
        if not valid:
            break
            
    if not valid:
        # Fallback to previous block arrangement
        blocks = [
            (0, 0, 0),
            (0, 2, 0),
            (0, 0, 2),
            (0, 2, 2),
        ]
        covered = [[[False]*p for _ in range(m)] for _ in range(n)]
        for i0, j0, k0 in blocks:
            for di in range(2):
                for dj in range(2):
                    for dk in range(2):
                        covered[i0+di][j0+dj][k0+dk] = True

    total_rank = len(blocks) * 7
    # Add schoolbook for uncovered triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if not covered[i][j][k]:
                    total_rank += 1
    
    # Verify we found something better than or equal to 71
    if total_rank > 71:
        # Fallback to known good arrangement
        blocks = [
            (0, 0, 0),
            (0, 2, 0),
            (0, 0, 2),
            (1, 2, 2),
        ]
        covered = [[[False]*p for _ in range(m)] for _ in range(n)]
        for i0, j0, k0 in blocks:
            for di in range(2):
                for dj in range(2):
                    for dk in range(2):
                        covered[i0+di][j0+dj][k0+dk] = True
        total_rank = 4 * 7
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if not covered[i][j][k]:
                        total_rank += 1

    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    for i0, j0, k0 in blocks:
        add_strassen_block(U, V, W, i0, j0, k0, col)
        col += 7
        
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if not covered[i][j][k]:
                    U[i*m+j][col] += 1
                    V[j*p+k][col] += 1
                    W[k*n+i][col] += 1
                    col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
