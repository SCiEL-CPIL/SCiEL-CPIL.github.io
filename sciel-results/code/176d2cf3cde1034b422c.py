"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120s safety margin
    
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # Strategy: Use Strassen's algorithm to get rank-49 for 4x4x4
    # Then split the 4x8 matrix into two 4x4 matrices
    # Total rank: 2 * 49 = 98
    # This is better than schoolbook rank of 4*4*8 = 128
    
    # First, let's generate the Strassen rank-49 decomposition for 4x4x4
    # Strassen 2x2x2 has rank 7
    # For 4x4x4, we use 2x2 blocks, each computed with Strassen 2x2x2
    
    # The Strassen algorithm for 4x4x4:
    # Split A, B, C into 2x2 blocks of 2x2 matrices
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    # C = A*B
    # 
    # We compute 7 products of 2x2 matrices:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    #
    # Then:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # Each Mi is a product of two 2x2 matrices, which itself requires rank 7
    # So total rank = 7 * 7 = 49
    
    # Let's build the rank-49 decomposition for 4x4x4
    # Index convention:
    # U[i*m + j, c] for A (4x4)
    # V[j*p + k, c] for B (4x4) 
    # W[k*n + i, c] for C (4x4)
    
    # Strassen 2x2x2 rank-7 coefficients
    # For 2x2 matrices, the 7 multiplications are:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-f)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Now for 4x4x4, we have 7 such products, each of rank 7
    # Total rank = 49
    
    # Let's enumerate the 49 rank-1 tensors
    rank_49 = 49
    U44 = [[0] * rank_49 for _ in range(16)]  # 4*4 = 16
    V44 = [[0] * rank_49 for _ in range(16)]  # 4*4 = 16
    W44 = [[0] * rank_49 for _ in range(16)]  # 4*4 = 16
    
    # The 7 outer products (for the 4x4 block structure)
    # Each outer product corresponds to one of the 7 Strassen multiplications
    # For each outer multiplication, we have 7 inner Strassen multiplications
    
    # Let's define the outer Strassen patterns
    # For each of the 7 outer multiplications, we specify:
    # - Which 2x2 blocks of A are combined (with signs)
    # - Which 2x2 blocks of B are combined (with signs)
    # - How the result contributes to C blocks
    
    # Outer Strassen:
    # M1: A(A11+A22) * B(B11+B22) -> contributes to C11, C22
    # M2: A(A21+A22) * B(B11) -> contributes to C21, C22
    # M3: A(A11) * B(B12-B22) -> contributes to C12, C22
    # M4: A(A22) * B(B21-B11) -> contributes to C11, C21
    # M5: A(A11+A12) * B(B22) -> contributes to C11, C12
    # M6: A(A21-A11) * B(B11+B12) -> contributes to C22
    # M7: A(A12-A22) * B(B21+B22) -> contributes to C11
    
    # For each outer multiplication, we compute 7 inner Strassen multiplications
    # The inner Strassen for 2x2x2 has these patterns:
    # Let's index the 2x2 blocks:
    # Block 0 = top-left (indices 0,1,4,5 in 4x4)
    # Block 1 = top-right (indices 2,3,6,7)
    # Block 2 = bottom-left (indices 8,9,12,13)
    # Block 3 = bottom-right (indices 10,11,14,15)
    
    # Within each 2x2 block, the elements are:
    # For block b at position (row_block, col_block):
    # Element (i,j) in 2x2 has global index: row_block*4 + i, col_block*4 + j
    # So the 4 elements are at:
    # (row_block*4, col_block*4)
    # (row_block*4, col_block*4 + 1)
    # (row_block*4 + 1, col_block*4)
    # (row_block*4 + 1, col_block*4 + 1)
    
    # Let's define helper functions
    def block_indices(rb, cb):
        """Get the 4 indices for a 2x2 block at row_block=rb, col_block=cb"""
        return [rb*4 + cb*4, rb*4 + cb*4 + 1, rb*4 + 1 + cb*4, rb*4 + 1 + cb*4 + 1]
    
    # For A (4x4), block (rb, cb) has elements at indices:
    # A[rb*4 + i, cb*4 + j] for i,j in {0,1}
    # Linear index in 16-dim: (rb*4 + i) * 4 + (cb*4 + j)
    
    # Let's build the 49 rank-1 tensors
    # We'll iterate over outer_mult (0-6) and inner_mult (0-6)
    
    # Outer multiplication definitions:
    # Each is a list of (block_row, block_col, sign) for A and B
    # And a list of (block_row, block_col, sign) for C contribution
    
    # Let's be more explicit. For each outer multiplication:
    # We have A_combined = sum of certain 2x2 blocks with signs
    # We have B_combined = sum of certain 2x2 blocks with signs
    # The product contributes to certain C blocks with signs
    
    # Let's define the 7 outer multiplications more carefully
    
    # Actually, let me just directly construct the rank-49 decomposition
    # by explicitly listing all 49 rank-1 terms
    
    # This is getting complex. Let me use a different approach.
    # I'll use numpy to verify and construct the decomposition.
    
    # Actually, let me just implement it directly using the recursive structure
    
    # For now, let me use the simple schoolbook approach but with better constants
    # Or let me try to implement the Strassen recursion properly
    
    # Let me think about this differently. The 4x4x4 tensor can be decomposed
    # using 49 rank-1 terms. Let me construct them explicitly.
    
    # The 4x4 matrix A has elements A[i,j] for i,j in {0,1,2,3}
    # Linear index: i*4 + j
    # Similarly for B and C
    
    # Strassen 4x4x4:
    # Split into 2x2 blocks:
    # A11 = A[0:2, 0:2], A12 = A[0:2, 2:4]
    # A21 = A[2:4, 0:2], A22 = A[2:4, 2:4]
    # Similarly for B and C
    
    # The 7 outer multiplications:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # For each Mi, we compute the product of two 2x2 matrices using Strassen (rank 7)
    # So each Mi has 7 rank-1 terms
    # Total: 7 * 7 = 49 rank-1 terms
    
    # Let me enumerate them. For outer multiplication outer_m (0-6) and inner multiplication inner_m (0-6):
    # The rank-1 term is:
    # U[outer_m, inner_m], V[outer_m, inner_m], W[outer_m, inner_m]
    
    # Let's define the outer multiplication patterns:
    # For each outer_m, specify:
    # - A_blocks: list of (block_row, block_col, sign)
    # - B_blocks: list of (block_row, block_col, sign)
    # - C_blocks: list of (block_row, block_col, sign) for the contribution
    
    # Block positions: (0,0)=top-left, (0,1)=top-right, (1,0)=bottom-left, (1,1)=bottom-right
    
    outer_defs = [
        # M1: (A11 + A22)(B11 + B22) -> C11 + C22
        {'A': [(0,0,1), (1,1,1)], 'B': [(0,0,1), (1,1,1)], 'C': [(0,0,1), (1,1,1)]},
        # M2: (A21 + A22)B11 -> C21 + C22
        {'A': [(1,0,1), (1,1,1)], 'B': [(0,0,1)], 'C': [(1,0,1), (1,1,1)]},
        # M3: A11(B12 - B22) -> C12 + C22
        {'A': [(0,0,1)], 'B': [(0,1,1), (1,1,-1)], 'C': [(0,1,1), (1,1,1)]},
        # M4: A22(B21 - B11) -> C11 + C21
        {'A': [(1,1,1)], 'B': [(1,0,1), (0,0,-1)], 'C': [(0,0,1), (1,0,1)]},
        # M5: (A11 + A12)B22 -> C11 + C12
        {'A': [(0,0,1), (0,1,1)], 'B': [(1,1,1)], 'C': [(0,0,1), (0,1,1)]},
        # M6: (A21 - A11)(B11 + B12) -> C22
        {'A': [(1,0,1), (0,0,-1)], 'B': [(0,0,1), (0,1,1)], 'C': [(1,1,1)]},
        # M7: (A12 - A22)(B21 + B22) -> C11
        {'A': [(0,1,1), (1,1,-1)], 'B': [(1,0,1), (1,1,1)], 'C': [(0,0,1)]},
    ]
    
    # For the inner Strassen (2x2x2), the 7 multiplications are:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-f)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    #
    # Where A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For each inner multiplication, specify:
    # - A_elements: list of (i, j, sign) for the 2x2 A block
    # - B_elements: list of (i, j, sign) for the 2x2 B block
    # - C_elements: list of (i, j, sign) for the 2x2 C block contribution
    
    # Within a 2x2 block at position (br, bc) in the 4x4 matrix:
    # Element (i,j) in 2x2 has global index: (br*2 + i)*4 + (bc*2 + j)
    
    inner_defs = [
        # M1 = (a+d)(e+h)
        {'A': [(0,0,1), (1,1,1)], 'B': [(0,0,1), (1,1,1)], 'C': [(0,0,1), (1,1,1)]},
        # M2 = (c+d)e
        {'A': [(1,0,1), (1,1,1)], 'B': [(0,0,1)], 'C': [(1,0,1), (1,1,1)]},
        # M3 = a(f-h)
        {'A': [(0,0,1)], 'B': [(0,1,1), (1,1,-1)], 'C': [(0,1,1), (1,1,1)]},
        # M4 = d(g-f)
        {'A': [(1,1,1)], 'B': [(1,0,1), (0,0,-1)], 'C': [(0,0,1), (1,0,1)]},
        # M5 = (a+b)h
        {'A': [(0,0,1), (0,1,1)], 'B': [(1,1,1)], 'C': [(0,0,1), (0,1,1)]},
        # M6 = (c-a)(e+f)
        {'A': [(1,0,1), (0,0,-1)], 'B': [(0,0,1), (0,1,1)], 'C': [(1,1,1)]},
        # M7 = (b-d)(g+h)
        {'A': [(0,1,1), (1,1,-1)], 'B': [(1,0,1), (1,1,1)], 'C': [(0,0,1)]},
    ]
    
    # Now construct the 49 rank-1 terms
    col = 0
    for outer_m in range(7):
        outer_def = outer_defs[outer_m]
        for inner_m in range(7):
            inner_def = inner_defs[inner_m]
            
            # For A: for each block in outer_def['A'], for each element in inner_def['A']
            for (ar, ac, asgn) in outer_def['A']:
                for (ai, aj, asgn2) in inner_def['A']:
                    # Global row: ar*2 + ai, global col: ac*2 + aj
                    # Linear index: (ar*2 + ai)*4 + (ac*2 + aj)
                    idx = (ar*2 + ai)*4 + (ac*2 + aj)
                    U44[idx][col] = asgn * asgn2
            
            # For B: similar
            for (br, bc, bs) in outer_def['B']:
                for (bi, bj, bs2) in inner_def['B']:
                    idx = (br*2 + bi)*4 + (bc*2 + bj)
                    V44[idx][col] = bs * bs2
            
            # For C: similar
            for (cr, cc, cs) in outer_def['C']:
                for (ci, cj, cs2) in inner_def['C']:
                    idx = (cr*2 + ci)*4 + (cc*2 + cj)
                    W44[idx][col] = cs * cs2
            
            col += 1
    
    # Now for 4x4x8:
    # We split B (4x8) into B1 (4x4) and B2 (4x4)
    # C = A*B1 + A*B2
    # We use the rank-49 decomposition twice
    
    rank = 98
    U = [[0] * rank for _ in range(16)]  # 4*4 = 16
    V = [[0] * rank for _ in range(32)]  # 4*8 = 32
    W = [[0] * rank for _ in range(32)]  # 8*4 = 32
    
    # Copy U44 to U
    for i in range(16):
        for c in range(49):
            U[i][c] = U44[i][c]
            U[i][c + 49] = U44[i][c]
    
    # For V: first 49 columns use V44 for B1, next 49 for B2
    # B1 is B[0:4, 0:4], B2 is B[0:4, 4:8]
    for i in range(16):
        for c in range(49):
            V[i][c] = V44[i][c]  # B1
            V[i + 16][c + 49] = V44[i][c]  # B2 (offset by 16 in row, 49 in col)
    
    # For W: first 49 columns use W44 for C from B1, next 49 for C from B2
    # C is 8x4, with C[0:4, :] from B1 and C[4:8, :] from B2
    for i in range(16):
        for c in range(49):
            W[i][c] = W44[i][c]  # C from B1 (rows 0-3)
            W[i + 16][c + 49] = W44[i][c]  # C from B2 (rows 4-7)
    
    # Check time
    if time.time() > deadline - 5:
        # Not enough time, but we already have the result
        pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

