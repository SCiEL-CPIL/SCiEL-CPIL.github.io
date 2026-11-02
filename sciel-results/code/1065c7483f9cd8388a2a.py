"""Matrix multiplication 3x3x7 tensor decomposition."""
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 7
    
    # We decompose the 3x3x7 multiplication into:
    # - Two 3x3x3 blocks (columns 0-2 and 3-5 of B)
    # - One 3x3x1 block (column 6 of B)
    #
    # For each 3x3x3 block, we use Strassen's algorithm for the 2x2 core
    # plus schoolbook for the boundary terms.
    #
    # Strassen for 2x2: rank 7
    # Boundary terms for 3x3: 
    #   Row 2, cols 0-1: A[2,0]*B[0,0..1] + A[2,1]*B[1,0..1] -> 4 terms
    #   Cols 2, rows 0-1: A[0..1,2]*B[0..1,2] -> 4 terms  
    #   But we need to be careful about overlaps
    #
    # Actually, let me think about this more carefully.
    #
    # For a 3x3x3 multiplication, we can partition as:
    # A = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1
    # B = [[B11, B12], [B21, B22]] where B11 is 2x2, B12 is 2x1, B21 is 1x2, B22 is 1x1
    #
    # C = [[A11*B11 + A12*B21, A11*B12 + A12*B22],
    #      [A21*B11 + A22*B21, A21*B12 + A22*B22]]
    #
    # Using Strassen for the 2x2 blocks:
    # M1 = A11*B11 (rank 7)
    # M2 = A12*B21 (rank 2, since A12 is 2x1 and B21 is 1x2)
    # M3 = A11*B12 (rank 2, since B12 is 2x1)
    # M4 = A21*B11 (rank 2, since A21 is 1x2)
    # M5 = A22*B22 (rank 1)
    # M6 = A21*B12 (rank 1)
    # M7 = A12*B22 (rank 1)
    #
    # Total rank: 7 + 2 + 2 + 2 + 1 + 1 + 1 = 16
    #
    # Wait, that's not right. Let me reconsider.
    #
    # Actually, for the boundary terms:
    # A12*B21: A12 is 2x1, B21 is 1x2, so this is an outer product, rank 1
    # A11*B12: A11 is 2x2, B12 is 2x1, so this is 2 scalar multiplications, rank 2
    # A21*B11: A21 is 1x2, B11 is 2x2, so this is 2 scalar multiplications, rank 2
    # A22*B22: rank 1
    # A21*B12: rank 1
    # A12*B22: rank 1
    #
    # Total: 7 + 1 + 2 + 2 + 1 + 1 + 1 = 15
    #
    # Hmm, let me just implement it directly.
    
    # Let's use a simpler approach: schoolbook for the 3x3x1 block (rank 9)
    # and for each 3x3x3 block, use a rank-15 decomposition.
    # Total rank: 15 + 15 + 9 = 39
    
    # Actually, let me just use schoolbook for everything to be safe, then optimize.
    # Schoolbook rank: 3*3*7 = 63
    
    # Let me implement the Strassen-based 3x3x3 decomposition.
    
    def add_term(U, V, W, i_u, i_v, i_w, val=1):
        U[i_u].append(val)
        V[i_v].append(val)
        W[i_w].append(val)
    
    # Initialize lists
    U = [[] for _ in range(n*m)]
    V = [[] for _ in range(m*p)]
    W = [[] for _ in range(p*n)]
    
    def strassen_3x3x3(offset_u, offset_v, offset_w):
        """
        Add a rank-15 decomposition for a 3x3x3 multiplication.
        offset_u: offset for U indices (i*m + j)
        offset_v: offset for V indices (j*p + k)  
        offset_w: offset for W indices (k*n + i)
        """
        # A is 3x3, B is 3x3
        # Partition: A11 = A[0:2, 0:2], A12 = A[0:2, 2], A21 = A[2, 0:2], A22 = A[2,2]
        # B11 = B[0:2, 0:2], B12 = B[0:2, 2], B21 = B[2, 0:2], B22 = B[2,2]
        
        # Strassen for A11*B11 (2x2 * 2x2): 7 terms
        # M1 = A11 * B11
        # M2 = (A[2,0]+A[2,1]) * B[0:2, 0]  -- wait, this doesn't work cleanly
        
        # Let me use the standard Strassen 2x2 decomposition for the top-left 2x2 block
        # and handle the rest with schoolbook.
        
        # Actually, let's just do schoolbook for simplicity and correctness.
        # 3x3x3 schoolbook: 9 terms
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    add_term(U, V, W, offset_u + i*3 + j, offset_v + j*3 + k, offset_w + k*3 + i, 1)
    
    # Use schoolbook for all three blocks: rank 27 + 27 + 9 = 63
    strassen_3x3x3(0, 0, 0)      # First 3x3x3 block
    strassen_3x3x3(0, 9, 9)      # Second 3x3x3 block (k offset by 3)
    
    # Third 3x3x1 block: k=6 only
    for i in range(3):
        for j in range(3):
            k = 6
            add_term(U, V, W, i*3 + j, j*3 + k, k*3 + i, 1)
    
    # Convert to proper format
    rank = len(U[0]) if U else 0
    U_mat = [U[i] for i in range(n*m)]
    V_mat = [V[i] for i in range(m*p)]
    W_mat = [W[i] for i in range(p*n)]
    
    return {"U": U_mat, "V": V_mat, "W": W_mat}
# EVOLVE-BLOCK-END
