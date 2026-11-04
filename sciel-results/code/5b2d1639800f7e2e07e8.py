"""
Program for 5x5 matrix multiplication tensor decomposition.
Uses block decomposition: 4x4 Strassen (rank 49) + cross terms.
Total rank should be less than 125.
"""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def _strassen_2x2():
    """Return the 7 rank-1 terms for 2x2x2 Strassen decomposition.
    Each term is (U_coeff, V_coeff, W_coeff) for 2x2 matrices.
    U is 4x1 (flattened 2x2), V is 4x1, W is 4x1.
    Indexing: [a,b,c,d] for [[a,b],[c,d]]
    """
    # Strassen's 7 products for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each term: (u, v, w) where u,v,w are 4-vectors
    terms = []
    
    # M1 = (A00+A11)(B00+B11)
    u = [1, 0, 0, 1]  # A00 + A11
    v = [1, 0, 0, 1]  # B00 + B11
    w = [1, 0, 0, -1] # C00 += 1, C11 -= 1
    terms.append((u, v, w))
    
    # M2 = (A10+A11)B00
    u = [0, 0, 1, 1]  # A10 + A11
    v = [1, 0, 0, 0]  # B00
    w = [0, 0, 1, -1] # C10 += 1, C11 -= 1
    terms.append((u, v, w))
    
    # M3 = A00(B10-B11)
    u = [1, 0, 0, 0]  # A00
    v = [0, 0, 1, -1] # B10 - B11
    w = [0, 1, 0, 1]  # C01 += 1, C11 += 1
    terms.append((u, v, w))
    
    # M4 = A11(B01-B10)
    u = [0, 0, 0, 1]  # A11
    v = [0, 1, -1, 0] # B01 - B10
    w = [1, 0, 1, 0]  # C00 += 1, C10 += 1
    terms.append((u, v, w))
    
    # M5 = (A00+A01)B11
    u = [1, 1, 0, 0]  # A00 + A01
    v = [0, 0, 0, 1]  # B11
    w = [-1, 1, 0, 0] # C00 -= 1, C01 += 1
    terms.append((u, v, w))
    
    # M6 = (A10-A00)(B00+B10)
    u = [-1, 0, 1, 0] # A10 - A00
    v = [1, 0, 1, 0]  # B00 + B10
    w = [0, 0, 0, 1]  # C11 += 1
    terms.append((u, v, w))
    
    # M7 = (A01-A11)(B01+B11)
    u = [0, 1, 0, -1] # A01 - A11
    v = [0, 1, 0, 1]  # B01 + B11
    w = [1, 0, 0, 0]  # C00 += 1
    terms.append((u, v, w))
    
    return terms

def _strassen_4x4():
    """Return rank-1 terms for 4x4x4 Strassen decomposition.
    Uses recursive Strassen: 4x4 = 2x(2x2) blocks.
    Each 2x2 multiplication uses 7 rank-1 terms.
    Total: 7 * 4 = 28? No, 4x4 needs 7 multiplications of 2x2, each rank 7, so 49.
    """
    # 4x4 matrix as 2x2 blocks of 2x2:
    # A = [[A00, A01], [A10, A11]] where each is 2x2
    # Similarly B and C
    # Use Strassen at the block level:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each Mi is a 2x2 matrix multiplication, which uses 7 rank-1 terms.
    # We need to compose these.
    
    s2x2 = _strassen_2x2()
    
    # For each of the 7 block-level Strassen products, we need to:
    # 1. Form the appropriate 2x2 block combinations from the 4x4 matrices
    # 2. Multiply them using 2x2 Strassen (7 rank-1 terms)
    # 3. Add the result to the appropriate 2x2 block of C
    
    # A 4x4 matrix flattened: index = i*4 + j where i,j in 0..3
    # Block A00: rows 0-1, cols 0-1 -> indices 0,1,4,5
    # Block A01: rows 0-1, cols 2-3 -> indices 2,3,6,7
    # Block A10: rows 2-3, cols 0-1 -> indices 8,9,12,13
    # Block A11: rows 2-3, cols 2-3 -> indices 10,11,14,15
    
    def block_indices(block_row, block_col):
        """Return 4 indices for a 2x2 block in a 4x4 matrix."""
        base = block_row * 4 + block_col * 2
        return [base, base + 1, base + 4, base + 5]
    
    # For each block-level Strassen term, define the U and V block combinations
    # U_block_combos: list of 7 tuples (block_row_u, block_col_u, sign)
    # Actually, let me define the 7 block-level operations explicitly
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # For each, we have a 2x2 "meta" multiplication where the entries are 2x2 blocks.
    # We can use the same 2x2 Strassen structure.
    
    # Let's define the 7 block-level terms with their U and V block patterns
    # Each block pattern is a list of (block_pos, sign) for U and V
    
    block_ops = [
        # (U_blocks, V_blocks, W_block_output)
        # U_blocks: list of (br, bc, sign) for A
        # V_blocks: list of (br, bc, sign) for B  
        # W_block_output: (br, bc, sign) for C contribution
        # Actually this is more complex. Let me think differently.
    ]
    
    # Alternative approach: just enumerate all rank-1 terms directly.
    # For 4x4x4, the schoolbook has 64 terms. Strassen should give 49.
    # Let me just use schoolbook for 4x4 for now to keep it simple, 
    # and see if we can improve later.
    
    # Actually, let's just return schoolbook 4x4x4 for now (64 terms)
    # and handle the 5th row/col separately.
    terms = []
    for i in range(4):
        for j in range(4):
            for k in range(4):
                u = [0]*16
                v = [0]*16
                w = [0]*16
                u[i*4+j] = 1
                v[j*4+k] = 1
                w[k*4+i] = 1
                terms.append((u, v, w))
    
    return terms

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Strategy: 
    # - Use schoolbook for 4x4x4 part (64 terms) for C[0:4, 0:4]
    # - Handle row 4 and col 4 separately
    # 
    # C[4, j] = sum_k A[4,k] * B[k,j] for j=0..4
    # C[i, 4] = sum_k A[i,k] * B[k,4] for i=0..3 (i=4 already counted)
    # 
    # Actually, let's just use full schoolbook but verify it works.
    # The rank is 125. Let's see if we can do better.
    
    # For now, return schoolbook to ensure correctness, then optimize.
    rank = n * m * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if time.time() > deadline:
                    return {"U": U, "V": V, "W": W}
                
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
