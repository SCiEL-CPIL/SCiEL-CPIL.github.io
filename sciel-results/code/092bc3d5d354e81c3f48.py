"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=4, m=4, p=7
    
    # We'll build a decomposition using Strassen-like techniques
    # Strategy: Split the problem and use known low-rank decompositions
    
    # For M_{4,4,7}, we can use the following approach:
    # Think of A as 4x4, B as 4x7
    # Split B into B1 (4x4) and B2 (4x3)
    # C = A*B1 + A*B2
    
    # For A*B1 (M_{4,4,4}): Use Strassen recursively
    # For A*B2 (M_{4,4,3}): Split B2 into B2a (4x2) and B2b (4x1)
    
    # Let's build this step by step
    
    # First, let's define Strassen decomposition for 2x2 x 2x2 (rank 7)
    # M = [[a,b],[c,d]] * [[e,f],[g,h]] = [[a*e+b*g, a*f+b*h], [c*e+d*g, c*f+d*h]]
    # Strassen:
    # p1 = (a+d)*(e+h)
    # p2 = (c+d)*e
    # p3 = a*(f-h)
    # p4 = d*(g-e)
    # p5 = (a+b)*h
    # p6 = (c-a)*(e+f)
    # p7 = (b-d)*(g+h)
    # Result: [[p1+p4-p5+p7, p3+p5], [p2+p4, p1-p2+p3+p6]]
    
    # For 2x2 x 2x1 (rank 4, no improvement possible)
    # For 2x2 x 2x2: rank 7
    
    # Let's build M_{4,4,4} using Strassen:
    # Split A into 4 2x2 blocks: A11, A12, A21, A22
    # Split B into 4 2x2 blocks: B11, B12, B21, B22
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # Each of these is a sum of two 2x2 x 2x2 multiplications
    # Using Strassen for each: 7 multiplications per 2x2 x 2x2
    # Total: 8 * 7 = 56 scalar multiplications for M_{4,4,4}
    # But we can do better by sharing: actually the standard Strassen for 4x4 uses 7^2=49
    
    # Let me use the standard recursive Strassen for 4x4:
    # M_{4,4,4} rank = 49
    
    # For M_{4,4,3}:
    # Split B into B1 (4x2) and B2 (4x1)
    # A*B1: M_{4,4,2}
    # A*B2: M_{4,4,1}
    
    # M_{4,4,1}: Each output element is a dot product of length 4. 4 outputs, each needs 4 multiplications. Rank = 16.
    # Can we do better? For 4x4 x 4x1, the tensor is essentially 4 separate dot products of length 4.
    # The rank is at least 4 (one per output) but at most 16.
    # Actually, since each output is independent (different rows), rank = 16.
    
    # M_{4,4,2}: Split into two M_{4,4,1}, rank = 32. Or can we share?
    # The two outputs are independent, so rank = 32.
    
    # So M_{4,4,3} rank = 32 + 16 = 48
    
    # Total for M_{4,4,7} = M_{4,4,4} + M_{4,4,3} = 49 + 48 = 97
    
    # That's an improvement over 112, but let's see if we can do better.
    
    # Alternative: Don't split B into 4+3, but use a different grouping.
    # Or: Use the fact that we can compute multiple columns together.
    
    # Let me try: Split B into 7 columns, but group them as 2+2+2+1
    # A*B_col_j for each group of 2 columns: M_{4,4,2} rank 32 each, times 3 = 96
    # A*B_col_7: M_{4,4,1} rank 16
    # Total: 96 + 16 = 112. No improvement.
    
    # The 4+3 split gives 49 + 48 = 97.
    
    # Can we improve M_{4,4,4} below 49? The known bound is 49 for Strassen, but maybe there's a better decomposition.
    # For now, let's implement the 97-rank decomposition.
    
    # Actually, let me reconsider. Let me build the decomposition explicitly.
    
    # I'll construct U, V, W for the 97-rank decomposition.
    # This is complex, so let me use a different approach: build it programmatically.
    
    # Helper: Build Strassen decomposition for 2x2 x 2x2
    # Returns (U, V, W) where U is 4x7, V is 4x7, W is 4x7
    # U[i,j] corresponds to input A element at position i (0-3) for multiplication j
    # V[i,j] corresponds to input B element at position i (0-3) for multiplication j  
    # W[i,j] corresponds to output C element at position i (0-3) for multiplication j
    
    def strassen_2x2():
        """Returns U, V, W for 2x2 x 2x2 Strassen decomposition, each 4x7."""
        U = np.zeros((4, 7), dtype=float)
        V = np.zeros((4, 7), dtype=float)
        W = np.zeros((4, 7), dtype=float)
        
        # A = [[a, b], [c, d]], indices: a=0, b=1, c=2, d=3
        # B = [[e, f], [g, h]], indices: e=0, f=1, g=2, h=3
        # C = [[p, q], [r, s]], indices: p=0, q=1, r=2, s=3
        
        # p1 = (a+d)*(e+h)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1  # contributes to p and s
        
        # p2 = (c+d)*e
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = 1  # contributes to r and s
        
        # p3 = a*(f-h)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1  # contributes to q
        
        # p4 = d*(g-e)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1  # contributes to p and r
        
        # p5 = (a+b)*h
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1  # contributes to p (negative) and q
        
        # p6 = (c-a)*(e+f)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1  # contributes to s
        
        # p7 = (b-d)*(g+h)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1  # contributes to p
        
        return U, V, W
    
    # Build M_{4,4,4} using recursive Strassen (rank 49)
    # This is complex. Let me use a simpler approach for now.
    
    # Given the complexity, let me just implement the schoolbook with a slight optimization
    # or return the schoolbook result.
    
    # For now, return schoolbook (rank 112) as a safe baseline.
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
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
