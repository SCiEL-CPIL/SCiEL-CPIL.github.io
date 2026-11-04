"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 5x5 matrix multiplication.
    
    Strategy: Partition the 5x5 matrices into blocks and use Strassen's algorithm
    where beneficial, falling back to schoolbook for small blocks.
    
    The 5x5 multiplication can be decomposed using a 2+2+1 block partition:
    - 2x2 blocks: use Strassen (rank 7 instead of 8)
    - 2x1, 1x2, 1x1 blocks: use schoolbook
    
    This gives a total rank lower than the schoolbook 125.
    """
    n, m, p = DIMENSIONS  # all 5
    
    # We'll use a block decomposition approach
    # Partition indices into groups: [0,1], [2,3], [4]
    # So block sizes are 2, 2, 1
    
    block_sizes = [2, 2, 1]
    
    # Precompute Strassen's 7-term decomposition for 2x2 matrices
    # For 2x2 matrices A, B, C = A*B:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # We need to build U, V, W matrices for the full 5x5x5 multiplication
    # using these block decompositions.
    
    # Let's think about this differently. We'll enumerate all the "terms" in the
    # tensor decomposition. Each term corresponds to a rank-1 component.
    
    # For a block decomposition with block sizes [2,2,1] for both input and output:
    # We have 3x3 = 9 block positions in the output.
    # For each output block (i,j) with sizes ni x nj, and corresponding input blocks:
    #   - If both input blocks are 2x2: use Strassen (7 terms)
    #   - If one is 2x1 or 1x2 or 1x1: use schoolbook (ni*nj terms)
    
    # Let's build the decomposition term by term.
    
    terms = []  # Each term is (U_entry, V_entry, W_entry) where each is a dict of (index, value)
    
    # Block boundaries
    # Rows of A: [0,1], [2,3], [4]
    # Cols of A / Rows of B: [0,1], [2,3], [4]  
    # Cols of B: [0,1], [2,3], [4]
    # Rows of C: [0,1], [2,3], [4]
    # Cols of C: [0,1], [2,3], [4]
    
    # For output block at position (row_block, col_block):
    # C[row_block, col_block] = sum over k_block of A[row_block, k_block] * B[k_block, col_block]
    
    # Let's define the index ranges for each block
    def block_range(block_idx, sizes):
        start = sum(sizes[:block_idx])
        end = start + sizes[block_idx]
        return range(start, end)
    
    # For each combination of output row block, col block, and sum block
    for r_block in range(3):
        for c_block in range(3):
            for k_block in range(3):
                # A is [r_block, k_block], B is [k_block, c_block]
                a_rows = block_range(r_block, block_sizes)
                a_cols = block_range(k_block, block_sizes)
                b_rows = block_range(k_block, block_sizes)
                b_cols = block_range(c_block, block_sizes)
                
                # Output C is [r_block, c_block]
                c_rows = block_range(r_block, block_sizes)
                c_cols = block_range(c_block, block_sizes)
                
                # Determine the size of this sub-problem
                # A sub-block: len(a_rows) x len(a_cols)
                # B sub-block: len(b_rows) x len(b_cols)
                # C sub-block: len(c_rows) x len(c_cols)
                
                ar = len(a_rows)
                ac = len(a_cols)
                br = len(b_rows)
                bc = len(b_cols)
                cr = len(c_rows)
                cc = len(c_cols)
                
                # For the multiplication of these sub-blocks:
                # C_sub[i,j] = sum_k A_sub[i,k] * B_sub[k,j]
                
                if ar == 2 and ac == 2 and br == 2 and bc == 2:
                    # Use Strassen
                    # A_sub is 2x2, B_sub is 2x2, C_sub is 2x2
                    # Strassen gives 7 multiplications
                    # M1 = (A00+A11)(B00+B11)
                    # M2 = (A10+A11)B00
                    # M3 = A00(B10-B11)
                    # M4 = A11(B01-B10)
                    # M5 = (A00+A01)B11
                    # M6 = (A10-A00)(B00+B10)
                    # M7 = (A01-A11)(B01+B11)
                    
                    a00, a01 = a_rows[0], a_rows[1]
                    a10, a11 = a_cols[0], a_cols[1]
                    b00, b01 = b_rows[0], b_rows[1]
                    b10, b11 = b_cols[0], b_cols[1]
                    c00, c01 = c_rows[0], c_rows[1]
                    c10, c11 = c_cols[0], c_cols[1]
                    
                    # M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
                    # Contributes to C[0,0] and C[1,1]
                    term = {}
                    term['U'] = {a00: 1, a11: 1}  # Wait, I need to think about this more carefully
                    # Actually, let me reconsider the structure.
                    
                    # Let me use a different approach. I'll build the full U, V, W matrices
                    # by iterating over all possible rank-1 terms.
                    
                    pass  # This approach is getting complicated. Let me try a cleaner method.
    
    # Let me restart with a cleaner approach.
    # I'll directly construct the U, V, W matrices.
    
    # The tensor for matrix multiplication C = A*B has entries T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    # We want to decompose this as sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    # Let me use the block decomposition with Strassen for 2x2 blocks.
    
    # First, let's build a list of "operations" where each operation is a rank-1 term.
    # Each term has:
    #   - A set of (i,j) indices with coefficients for U
    #   - A set of (j,k) indices with coefficients for V  
    #   - A set of (k,i) indices with coefficients for W
    
    operations = []
    
    # For each block combination, add the appropriate terms
    for r_blk in range(3):
        for k_blk in range(3):
            for c_blk in range(3):
                ar = list(block_range(r_blk, block_sizes))
                ak = list(block_range(k_blk, block_sizes))
                bk = list(block_range(k_blk, block_sizes))
                bc = list(block_range(c_blk, block_sizes))
                cr = list(block_range(r_blk, block_sizes))
                cc = list(block_range(c_blk, block_sizes))
                
                # Check if this is a 2x2x2 problem
                if len(ar) == 2 and len(ak) == 2 and len(bk) == 2 and len(bc) == 2:
                    # Use Strassen
                    a0, a1 = ar[0], ar[1]
                    k0, k1 = ak[0], ak[1]
                    # bk should be same as ak since it's the same k_block
                    b0, b1 = bk[0], bk[1]
                    c0, c1 = bc[0], bc[1]
                    # cr should be same as ar
                    # cc should be same as bc
                    
                    # Strassen's 7 terms:
                    # M1 = (A[a0,k0]+A[a1,k1]) * (B[b0,c0]+B[b1,c1])
                    # M2 = (A[a1,k0]+A[a1,k1]) * B[b0,c0]
                    # M3 = A[a0,k0] * (B[b1,c0]-B[b1,c1])
                    # M4 = A[a1,k1] * (B[b0,c1]-B[b0,c0])
                    # M5 = (A[a0,k0]+A[a0,k1]) * B[b1,c1]
                    # M6 = (A[a1,k0]-A[a0,k0]) * (B[b0,c0]+B[b1,c0])
                    # M7 = (A[a0,k1]-A[a1,k1]) * (B[b0,c1]+B[b1,c1])
                    
                    # C[a0,c0] = M1 + M4 - M5 + M7
                    # C[a0,c1] = M3 + M5
                    # C[a1,c0] = M2 + M4
                    # C[a1,c1] = M1 - M2 + M3 + M6
                    
                    # M1
                    op = {'U': {(a0,k0): 1, (a1,k1): 1}, 
                          'V': {(b0,c0): 1, (b1,c1): 1},
                          'W': {(c0,a0): 1, (c1,a1): 1}}
                    operations.append(op)
                    
                    # M2
                    op = {'U': {(a1,k0): 1, (a1,k1): 1},
                          'V': {(b0,c0): 1},
                          'W': {(c0,a1): 1}}
                    operations.append(op)
                    
                    # M3
                    op = {'U': {(a0,k0): 1},
                          'V': {(b1,c0): 1, (b1,c1): -1},
                          'W': {(c0,a0): 1, (c1,a0): 1}}
                    operations.append(op)
                    
                    # M4
                    op = {'U': {(a1,k1): 1},
                          'V': {(b0,c1): 1, (b0,c0): -1},
                          'W': {(c0,a1): 1, (c1,a0): 1}}
                    operations.append(op)
                    
                    # M5
                    op = {'U': {(a0,k0): 1, (a0,k1): 1},
                          'V': {(b1,c1): 1},
                          'W': {(c0,a0): -1, (c1,a0): 1}}
                    operations.append(op)
                    
                    # M6
                    op = {'U': {(a1,k0): 1, (a0,k0): -1},
                          'V': {(b0,c0): 1, (b1,c0): 1},
                          'W': {(c1,a1): 1}}
                    operations.append(op)
                    
                    # M7
                    op = {'U': {(a0,k1): 1, (a1,k1): -1},
                          'V': {(b0,c1): 1, (b1,c1): 1},
                          'W': {(c0,a0): 1}}
                    operations.append(op)
                    
                else:
                    # Use schoolbook for this sub-block
                    for i in ar:
                        for j in ak:
                            for k in bc:
                                # C[i,k] += A[i,j] * B[j,k]
                                # But j and k here: j is from ak (cols of A), k is from bc (cols of B)
                                # Wait, I need to be more careful with indices.
                                # A is indexed by (row, col) = (i, j) where i in ar, j in ak
                                # B is indexed by (row, col) = (j, k) where j in bk, k in bc
                                # C is indexed by (row, col) = (i, k) where i in cr, k in cc
                                # But ak and bk should be the same since they're from the same k_block
                                # And cr = ar, cc = bc
                                
                                op = {'U': {(i, j): 1},
                                      'V': {(j, k): 1},
                                      'W': {(k, i): 1}}
                                operations.append(op)
    
    # Now build the U, V, W matrices
    rank = len(operations)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, op in enumerate(operations):
        for (i, j), val in op['U'].items():
            U[i * m + j][r] = val
        for (j, k), val in op['V'].items():
            V[j * p + k][r] = val
        for (k, i), val in op['W'].items():
            W[k * n + i][r] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
