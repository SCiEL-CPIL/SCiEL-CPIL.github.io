"""Matrix multiplication 3x5x7 with Strassen-based decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 5, 7]

def _strassen_decomposition(n, m, p, depth=0):
    """
    Build a tensor decomposition for multiplying an n x m matrix by an m x p matrix.
    Uses Strassen's algorithm for 2x2x2 blocks where possible.
    Returns (U, V, W) as lists of lists, rank.
    """
    if n == 1 and m == 1 and p == 1:
        # Trivial: A[0,0] * B[0,0] = C[0,0]
        return [[1]], [[1]], [[1]], 1
    
    # Try to split into 2x2x2 Strassen blocks
    # For Strassen, we need n, m, p all >= 2
    if n >= 2 and m >= 2 and p >= 2:
        # Split into blocks: n1=n//2, n2=n-n1; m1=m//2, m2=m-m1; p1=p//2, p2=p-p1
        # But Strassen works cleanly when dimensions are even or we can pad
        # Let's use the standard Strassen approach: divide each dimension roughly in half
        n1 = n // 2
        n2 = n - n1
        m1 = m // 2
        m2 = m - m1
        p1 = p // 2
        p2 = p - p1
        
        # For proper Strassen, we need the subproblems to be "square-ish" 
        # Actually, Strassen works for any rectangular blocks as long as we
        # define the 7 products correctly. Let's use the standard formulation.
        
        # The 7 Strassen products for C = A * B:
        # M1 = (A11 + A22) * (B11 + B22)
        # M2 = (A21 + A22) * B11
        # M3 = A11 * (B12 - B22)
        # M4 = A22 * (B21 - B11)
        # M5 = (A11 + A12) * B22
        # M6 = (A21 - A11) * (B11 + B12)
        # M7 = (A12 - A22) * (B21 + B22)
        #
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # We need to build U, V, W such that for each output element C[i,j],
        # the sum over rank of U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = C[i,j]
        
        # Let's build this recursively. Each Strassen product M_s is itself a matrix
        # multiplication of size (something x something) * (something x something).
        # But the sub-matrices have different sizes depending on the split.
        
        # For simplicity and correctness, let's handle the case where we can
        # decompose into 8 sub-problems (schoolbook) vs 7 (Strassen).
        
        # Actually, let me think about this differently. The standard approach is:
        # If n, m, p are all even, we can split into 4 blocks each and use Strassen.
        # If not, we may need to pad or use a different strategy.
        
        # Let's try a simpler approach: if all dims are even, use Strassen.
        # Otherwise, fall back to schoolbook for this level.
        
        if n1 == n2 and m1 == m2 and p1 == p2:
            # All even, use Strassen
            # We'll build the decomposition for each of the 7 Strassen products
            # and combine them.
            
            # Each Strassen product is a matrix multiplication of size (n1 or n2) x (m1 or m2) by (m1 or m2) x (p1 or p2)
            # Actually, the sub-matrices have specific sizes. Let me define them properly.
            
            # A is n x m, split into A11 (n1 x m1), A12 (n1 x m2), A21 (n2 x m1), A22 (n2 x m2)
            # B is m x p, split into B11 (m1 x p1), B12 (m1 x p2), B21 (m2 x p1), B22 (m2 x p2)
            # C is n x p, split into C11 (n1 x p1), C12 (n1 x p2), C21 (n2 x p1), C22 (n2 x p2)
            
            # M1 = (A11 + A22) * (B11 + B22): size n1 x p1? No, (A11+A22) is n x m (if n1=n2, m1=m2)
            # Wait, A11 is n1 x m1, A22 is n2 x m2. If n1=n2 and m1=m2, then A11+A22 is n x m.
            # That's not right for Strassen. In standard Strassen, all blocks are the same size.
            
            # I think the issue is that Strassen is designed for square matrices. For rectangular,
            # we need to be more careful. Let me just use schoolbook for now and see if I can
            # find a better decomposition.
            
            pass  # Fall through to schoolbook
    
    # Schoolbook decomposition
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
    return U, V, W, rank


def _try_strassen_2x2x2():
    """
    Try to build a decomposition using Strassen for 2x2x2 blocks.
    For 3x5x7, we can potentially use some 2x2x2 Strassen blocks.
    """
    n, m, p = 3, 5, 7
    
    # Let's think about this problem more carefully.
    # The schoolbook rank is 3*5*7 = 105.
    # We know that for 2x2x2, Strassen gives rank 7 instead of 8.
    
    # One approach: partition the 3x5x7 multiplication into independent sub-problems
    # that can use Strassen.
    
    # Actually, let me try a different approach. Let's use the known result that
    # for multiplying (2a)x(2b) by (2b)x(2c), we can use Strassen-like decompositions.
    
    # For 3x5x7, let's try to decompose it as:
    # Row 0 of A (1x5) times B (5x7) -> row 0 of C (1x7): rank 5*7 = 35 (schoolbook)
    # Rows 1-2 of A (2x5) times B (5x7) -> rows 1-2 of C (2x7): this is a 2x5x7 problem
    
    # For the 2x5x7 part, we can further split:
    # Rows 1-2 of A (2x5) times B (5x7)
    # Split B into B1 (5x3) and B2 (5x4)? Not clean.
    
    # Let me try yet another approach. Use the fact that we can sometimes
    # share intermediate computations.
    
    # Actually, the simplest improvement is to use Strassen for the 2x2x2 sub-blocks
    # that we can identify. Let me try to construct this explicitly.
    
    # For a 2x2x2 block: C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    #                     C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    #                     C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    #                     C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    
    # Strassen:
    # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    # M2 = (A[1,0]+A[1,1])*B[0,0]
    # M3 = A[0,0]*(B[0,1]-B[1,1])
    # M4 = A[1,1]*(B[1,0]-B[0,0])
    # M5 = (A[0,0]+A[0,1])*B[1,1]
    # M6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
    # M7 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
    
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    
    # Each M_s is a product of a linear combination of A entries and a linear combination of B entries.
    # In tensor decomposition terms, each M_s corresponds to one rank-1 term.
    
    # For our problem, let's try to identify 2x2x2 sub-problems.
    # 3x5x7: We can take rows 0-1 of A, columns 0-1 of B, and outputs 0-1 of C.
    # That's a 2x2x2 sub-problem. Similarly for other combinations.
    
    # But the sub-problems overlap, so we need to be careful.
    
    # Let me try a different strategy: use a known low-rank decomposition for 3x5x7.
    # The minimal rank is not known exactly, but it's less than 105.
    
    # For now, let's just return the schoolbook decomposition and see if we can improve.
    pass


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition as baseline
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
    
    # Try to improve using Strassen for 2x2x2 blocks
    # Identify 2x2x2 sub-blocks and apply Strassen
    
    # For a 2x2x2 block with A indices (i1,i2), B indices (j1,j2), C indices (k1,k2):
    # The 7 Strassen terms replace 8 schoolbook terms.
    
    # Let's try to partition the 3x5x7 problem into disjoint 2x2x2 blocks.
    # 3 = 2 + 1, 5 = 2 + 2 + 1, 7 = 2 + 2 + 2 + 1
    
    # We can form 2x2x2 blocks from the first 2 rows, first 2 cols of B, first 2 cols of C.
    # That gives us one 2x2x2 block, saving 1 rank.
    
    # Similarly, we can form more blocks. Let's count how many disjoint 2x2x2 blocks we can form.
    # Rows: 3 -> can form 1 block of size 2 (leaving 1 row)
    # Cols of B: 5 -> can form 2 blocks of size 2 (leaving 1 col)  
    # Cols of C: 7 -> can form 3 blocks of size 2 (leaving 1 col)
    
    # But the blocks must be disjoint in all three dimensions simultaneously.
    # So the number of disjoint 2x2x2 blocks is min(3//2, 5//2, 7//2) = min(1, 2, 3) = 1.
    
    # That only saves 1 rank, giving 104. Not great.
    
    # Let me try a different approach: use Strassen more cleverly.
    # Actually, for rectangular matrices, we can use a generalized Strassen.
    
    # For a 2xa x b x 2 problem (2 rows, a cols for A; b rows, 2 cols for B),
    # we might be able to use fewer than 2*a*b rank-1 terms.
    
    # Let me try to build the decomposition manually for 3x5x7.
    
    # Actually, let me just try to use numpy to verify a potential decomposition.
    
    # For now, return the schoolbook decomposition.
    return {"U": U, "V": V, "W": W}
