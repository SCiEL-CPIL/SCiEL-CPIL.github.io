import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # We will compute rank for 4x4x4 using Strassen (rank 49)
    # and 4x4x2 using schoolbook (rank 32)
    # Total rank 81
    
    rank_444 = 49
    rank_442 = 32
    total_rank = rank_444 + rank_442
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Part 1: 4x4x4 (columns 0-3 of B)
    # Strassen for 4x4
    # Define the 7 Strassen products for 2x2
    # u, v, w are 4-vectors (flattened 2x2)
    # Indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    
    # Term 1: (A00+A11)(B00+B11) -> C00, C11
    # u: [1, 0, 0, 1], v: [1, 0, 0, 1], w: [1, 0, 0, 1]
    # Term 2: (A10+A11)B00 -> C10
    # u: [0, 0, 1, 1], v: [1, 0, 0, 0], w: [0, 0, 1, 0]
    # Term 3: A00(B01-B11) -> C01
    # u: [1, 0, 0, 0], v: [0, 1, 0, -1], w: [0, 1, 0, 0]
    # Term 4: A11(B10-B00) -> C00, C10
    # u: [0, 0, 0, 1], v: [0, 0, 1, -1], w: [1, 0, 1, 0]
    # Term 5: (A00+A01)B11 -> C00, C01
    # u: [1, 1, 0, 0], v: [0, 0, 0, 1], w: [1, 1, 0, 0]
    # Term 6: (A10-A00)(B00+B01) -> C11
    # u: [0, 0, 1, -1], v: [1, 1, 0, 0], w: [0, 0, 0, 1]
    # Term 7: (A01-A11)(B10+B11) -> C00, C11
    # u: [0, 1, 0, -1], v: [0, 0, 1, 1], w: [1, 0, 0, -1]
    
    # Check Term 7:
    # C00 += M7. C11 -= M7.
    # w7: [1, 0, 0, -1]. Correct.
    
    strassen_2x2 = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [1, 1, 0, 0]),
        ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, -1])
    ]
    
    # For 4x4, we have 7 block products.
    # Block s uses A blocks and B blocks.
    # A = [[A00, A01], [A10, A11]] (each 2x2)
    # B = [[B00, B01], [B10, B11]]
    
    # s=0: (A00+A11)(B00+B11)
    # A terms: A00, A11. B terms: B00, B11.
    # s=1: (A10+A11)B00
    # A terms: A10, A11. B terms: B00.
    # s=2: A00(B01-B11)
    # A terms: A00. B terms: B01, B11.
    # s=3: A11(B10-B00)
    # A terms: A11. B terms: B10, B00.
    # s=4: (A00+A01)B11
    # A terms: A00, A01. B terms: B11.
    # s=5: (A10-A00)(B00+B01)
    # A terms: A10, A00. B terms: B00, B01.
    # s=6: (A01-A11)(B10+B11)
    # A terms: A01, A11. B terms: B10, B11.
    
    # We need to map local 2x2 indices to global 4x4 indices.
    # A global index (i, j) where i,j in 0..3.
    # Block (bi, bj) corresponds to rows 2*bi..2*bi+1, cols 2*bj..2*bj+1.
    # Local index l in 0..3 corresponds to (li, lj) in 0..1.
    # Global index = (2*bi + li, 2*bj + lj).
    
    # For each s, we have a set of A blocks and B blocks involved.
    # Actually, the Strassen formula for 4x4 uses the 2x2 Strassen formula on the blocks.
    # So for each s, we iterate over the 7 terms of the 2x2 Strassen.
    # For each term t in 0..6:
    #   We have local u_t, v_t, w_t (length 4).
    #   We need to expand these to global U, V, W vectors (length 16).
    #   The expansion depends on which blocks are used in s.
    
    # This is tricky because the linear forms in s are sums of blocks.
    # e.g., s=0: A_part = A00 + A11.
    # The local u_t is a linear form on the 2x2 matrix A_part.
    # So if u_t has 1 at local index 0 (A_part_00), it corresponds to A00_00 and A11_00.
    
    # Let's define the block mapping for each s.
    # s=0: A_blocks = [(0,0), (1,1)], B_blocks = [(0,0), (1,1)]
    # s=1: A_blocks = [(1,0), (1,1)], B_blocks = [(0,0)]
    # s=2: A_blocks = [(0,0)], B_blocks = [(0,1), (1,1)]
    # s=3: A_blocks = [(1,1)], B_blocks = [(1,0), (0,0)]
    # s=4: A_blocks = [(0,0), (0,1)], B_blocks = [(1,1)]
    # s=5: A_blocks = [(1,0), (0,0)], B_blocks = [(0,0), (0,1)]
    # s=6: A_blocks = [(0,1), (1,1)], B_blocks = [(1,0), (1,1)]
    
    # Note: Signs are handled in the local u_t, v_t.
    # But the block selection is positive sum.
    # e.g., s=2: B_part = B01 - B11.
    # The local v_t is a linear form on B_part.
    # If v_t has 1 at local index 1 (B_part_01), it corresponds to B01_01 - B11_01?
    # No, the Strassen formula is (A00)(B01 - B11).
    # This is A00 * (B01 - B11).
    # So the linear form on B is B01 - B11.
    # In the 2x2 Strassen decomposition, the term is $A_{00} (B_{01} - B_{11})$.
    # This is not a single rank-1 term in the 2x2 tensor sense?
    # Wait, the 2x2 Strassen decomposition I listed above IS the rank-7 decomposition.
    # Term 3 is $A_{00} (B_{01} - B_{11})$.
    # This term contributes to $C_{01}$.
    # In the 2x2 tensor, this is one of the 7 rank-1 terms.
    # So for s=2, we use the 7 terms of the 2x2 Strassen, but applied to the "super-matrix" formed by the blocks?
    # No.
    # The recursion is:
    # To compute 4x4 product, we compute 7 products of 2x2 matrices.
    # Each 2x2 product is computed using 7 rank-1 terms.
    # So for each s, we have 7 rank-1 terms.
    # The "A" part of term (s,t) is the linear combination of A-blocks specified by s, combined with the local linear form u_t.
    # The "B" part is the linear combination of B-blocks specified by s, combined with the local linear form v_t.
    # The "C" part is the local linear form w_t, placed in the correct C-blocks.
    
    # C-blocks for s:
    # s=0: C00, C11
    # s=1: C10
    # s=2: C01
    # s=3: C00, C10
    # s=4: C00, C01
    # s=5: C11
    # s=6: C00, C11
    
    # This is getting too complex to code correctly in one shot.
    # I will use the schoolbook for 4x4x4 as well, but that's rank 64.
    # 64 + 32 = 96. No improvement.
    
    # Let's try a simpler improvement.
    # Use Strassen for 2x2 blocks directly on the 4x4x6 tensor?
    # No.
    
    # How about using the rank of 4x4x4 = 49 (Strassen) is correct.
    # I will implement the 49-rank decomposition for 4x4x4.
    
    # To save time, I will hardcode the 49 terms.
    
    # Part 2: 4x4x2 (columns 4-5 of B)
    # Schoolbook: 4*2*4 = 32 terms.
    
    # Total rank 81.
    
    # I will generate the 49 terms programmatically.
    
    col = 0
    for s in range(7):
        # Define A_blocks, B_blocks, C_blocks for this Strassen product
        # This is the standard Strassen recursion step
        pass
