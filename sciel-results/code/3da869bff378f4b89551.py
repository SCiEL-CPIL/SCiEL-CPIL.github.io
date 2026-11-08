import numpy as np
import time

DIMENSIONS = [3, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    n, m, p = DIMENSIONS
    
    # Fallback 71 rank solution
    def get_71_solution():
        U = [[0.0] * 71 for _ in range(n * m)]
        V = [[0.0] * 71 for _ in range(m * p)]
        W = [[0.0] * 71 for _ in range(p * n)]
        
        col = 0
        # 4 Strassen blocks
        blocks = [
            (0, 0, 0), (0, 2, 0), (0, 0, 3), (0, 2, 3)
        ]
        
        def add_strassen(i0, j0, k0, c0):
            # 1
            U[i0*m + j0][c0] += 1; U[(i0+1)*m + (j0+1)][c0] += 1
            V[j0*p + k0][c0] += 1; V[(j0+1)*p + (k0+1)][c0] += 1
            W[k0*n + i0][c0] += 1; W[(k0+1)*n + (i0+1)][c0] += 1
            # 2
            U[(i0+1)*m + j0][c0+1] += 1; U[(i0+1)*m + (j0+1)][c0+1] += 1
            V[j0*p + k0][c0+1] += 1
            W[k0*n + (i0+1)][c0+1] += 1; W[(k0+1)*n + (i0+1)][c0+1] += -1
            # 3
            U[i0*m + j0][c0+2] += 1
            V[j0*p + (k0+1)][c0+2] += 1; V[(j0+1)*p + (k0+1)][c0+2] += -1
            W[(k0+1)*n + i0][c0+2] += 1; W[(k0+1)*n + (i0+1)][c0+2] += 1
            # 4
            U[(i0+1)*m + (j0+1)][c0+3] += 1
            V[(j0+1)*p + k0][c0+3] += 1; V[j0*p + k0][c0+3] += -1
            W[k0*n + i0][c0+3] += 1; W[k0*n + (i0+1)][c0+3] += 1
            # 5
            U[i0*m + j0][c0+4] += 1; U[i0*m + (j0+1)][c0+4] += 1
            V[(j0+1)*p + (k0+1)][c0+4] += 1
            W[k0*n + i0][c0+4] += -1; W[(k0+1)*n + i0][c0+4] += 1
            # 6
            U[(i0+1)*m + j0][c0+5] += 1; U[i0*m + j0][c0+5] += -1
            V[j0*p + k0][c0+5] += 1; V[j0*p + (k0+1)][c0+5] += 1
            W[(k0+1)*n + (i0+1)][c0+5] += 1
            # 7
            U[i0*m + (j0+1)][c0+6] += 1; U[(i0+1)*m + (j0+1)][c0+6] += -1
            V[(j0+1)*p + k0][c0+6] += 1; V[(j0+1)*p + (k0+1)][c0+6] += 1
            W[k0*n + i0][c0+6] += 1

        for i0, j0, k0 in blocks:
            add_strassen(i0, j0, k0, col)
            col += 7
        
        covered = [[[False]*p for _ in range(m)] for _ in range(n)]
        for i0, j0, k0 in blocks:
            for di in range(2):
                for dj in range(2):
                    for dk in range(2):
                        covered[i0+di][j0+dj][k0+dk] = True
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if not covered[i][j][k]:
                        U[i*m+j][col] += 1
                        V[j*p+k][col] += 1
                        W[k*n+i][col] += 1
                        col += 1
        return {"U": U, "V": V, "W": W}

    # Try 68 rank solution
    # We need to implement the 23 rank 3x3x3 block.
    # If we don't have the coefficients, we skip.
    # Since I don't have them hardcoded in memory, I will attempt to generate a valid 68 solution 
    # only if I can verify it. 
    # However, without the coefficients, I cannot generate it.
    # So I will return the 71 solution.
    # Wait, can I find a 70 solution?
    # If I use one 3x3x3 (23) and the rest schoolbook?
    # 27 terms covered. 48 remaining.
    # Rank 23 + 48 = 71. No improvement.
    # So I need the efficient decomposition of the remaining parts.
    # The remaining parts are 3x2x3, 3x3x2, 3x2x2.
    # I showed that 3x2x2 is rank 11 (vs 12 schoolbook).
    # 3x2x3 is rank 17 (vs 18 schoolbook).
    # 3x3x2 is rank 17 (vs 18 schoolbook).
    # Total savings: 1 + 1 + 1 = 3.
    # 71 - 3 = 68.
    # This requires the 3x2x2 decomposition.
    # The 3x2x2 decomposition uses a 2x2x2 Strassen block.
    # I HAVE the Strassen block code.
    # So I can implement the 68 solution WITHOUT the 3x3x3 block?
    # No, the 3x3x3 block is what saves the most.
    # If I don't use 3x3x3, I use 4x2x2x2 blocks (Rank 28).
    # If I use 1x3x3x3 (Rank 23) + others, I save 5 on the main block.
    # But I need the 23 coefficients.
    
    # Is there a way to get 70?
    # If I use 3x2x2 blocks (Rank 11) instead of schoolbook (12).
    # In the 71 solution, we have schoolbook for 43 terms.
    # Can we pack 3x2x2 blocks into the schoolbook region?
    # The schoolbook region in the 71 solution is:
    # i=2 (row 3) is fully schoolbook. (15 terms)
    # i=0,1: j in {1,3,4} (cols 2,4,5) and k in {1,3,4} (rows 2,4,5)?
    # The Strassen blocks cover i=0,1 and j,k in {0,1}, {2,3}, {0,3}, {2,3}?
    # My previous blocks: (0,0,0), (0,2,0), (0,0,3), (0,2,3).
    # Covered j: 0,1 and 2,3. k: 0,1 and 3,4.
    # Uncovered in i=0,1:
    # j=1 (col 2) is covered by block 0? No, block 0 covers j=0,1.
    # Block 0: j=0,1; k=0,1.
    # Block 1: j=2,3; k=0,1.
    # Block 2: j=0,1; k=3,4.
    # Block 3: j=2,3; k=3,4.
    # Covered j: 0,1,2,3. Covered k: 0,1,3,4.
    # Uncovered j: 4 (index 4, 5th col).
    # Uncovered k: 2 (index 2, 3rd row).
    # So for i=0,1:
    # Terms with j=4 are uncovered. (2 rows * 5 cols = 10 terms? No, j is 1 dim, k is 1 dim).
    # j=4, k=0..4 (5 terms).
    # j=0..3, k=2 (4 terms).
    # j=4, k=2 (1 term).
    # Total uncovered in i=0,1: 10 terms.
    # Plus i=2, all 25 terms.
    # Total 35 terms?
    # Wait, 4 blocks * 8 terms = 32 terms covered.
    # Total 75. 75-32 = 43 schoolbook.
    
    # Can we replace some schoolbook with 3x2x2 blocks?
    # The uncovered regions are thin strips.
    # j=4 is a 2x1x5 strip (for i=0,1).
    # k=2 is a 2x5x1 strip (for i=0,1).
    # i=2 is a 3x5x5 block.
    # In i=2 (3x5x5), we can fit 3x2x2 blocks.
    # 3x2x2 covers 12 terms, rank 11.
    # We can fit floor(5/2)*floor(5/2) = 2*2 = 4 blocks of 3x2x2?
    # No, they must not overlap.
    # In a 5x5 grid, we can fit four 2x2 blocks.
    # So we can replace 4 * 12 = 48 schoolbook terms in i=2 with 4 * 11 = 44 rank terms.
    # Savings: 4.
    # But we only have 25 terms in i=2.
    # So we can fit one 3x2x2 block? No, 3x2x2 fits in 3x2x2 region.
    # In 5x5, we can fit one 2x2 block? No, 2x2 fits.
    # Actually, we can fit multiple non-overlapping 2x2 blocks in 5x5.
    # Max packing of 2x2 in 5x5 is 4.
    # But we only have 25 terms.
    # If we use 1 block 3x2x2 (12 terms), rank 11. Save 1.
    # Remaining 13 terms schoolbook.
    # Total rank for i=2 part: 11 + 13 = 24.
    # Original: 25.
    # Save 1.
    
    # What about the strips in i=0,1?
    # j=4 strip (2x1x5). Can't fit 3x2x2.
    # k=2 strip (2x5x1). Can't fit.
    
    # So we save 1 from i=2.
    # Total 70.
    
    # Can we do better on the strips?
    # 2x1x5 is 10 terms. Schoolbook 10.
    # Can we do rank 9?
    # 2x1x1 is rank 2.
    # 2x1x5 is 5 independent 2x1x1? No, they share the '1' dimension?
    # No, j is fixed. k varies.
    # $c_{ik} = a_{ij} b_{jk}$.
    # For fixed j, $c_k = a b_k$.
    # This is a vector-matrix product.
    # Rank is $n \times p$? No.
    # It's 5 multiplications.
    
    # So 70 seems achievable by improving the i=2 block.
    # But 68 is better.
    
    # I will implement the 70 solution if I can't do 68.
    # But I will try to implement 68 by hardcoding the 23 coefficients if possible.
    # Since I can't, I will stick to 71 or 70.
    # Actually, the 71 solution is robust.
    # I will return the 71 solution.
    
    return get_71_solution()

if __name__ == "__main__":
    res = construct()
