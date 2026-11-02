"""
Strassen-based rank-49 decomposition for 4x4 matrix multiplication.

The key insight: 4x4 matrix multiplication can be decomposed using
Strassen's algorithm applied to 2x2 block matrices. Each 2x2 block
multiplication uses Strassen's 7-multiplication algorithm, and the
overall 4x4 multiplication requires 49 scalar multiplications total
(7 multiplications per 2x2 block * 7 blocks... actually it's more nuanced).

Actually, the standard result is that 4x4 matrix multiplication has
rank 49, achieved by applying Strassen's algorithm to the 2x2 blocks.
Let me think about this more carefully.

A 4x4 matrix product C = A*B where A and B are 4x4 can be viewed as
multiplying two 2x2 block matrices where each block is 2x2. The product
of two 2x2 block matrices (with 2x2 blocks) requires 7 block
multiplications via Strassen, and each block multiplication is a 2x2
matrix product which itself requires 7 scalar multiplications via
Strassen. So total = 7 * 7 = 49.

The tensor decomposition:
- U has shape (16, 49) - for the 16 entries of A (4*4)
- V has shape (16, 49) - for the 16 entries of B (4*4)  
- W has shape (16, 49) - for the 16 entries of C (4*4), indexed as W[k*n+i, r]

Let me construct this explicitly.

For 2x2 Strassen:
Given [[a,b],[c,d]] * [[e,f],[g,h]]:
M1 = (a+d)(e+h)
M2 = (c+d)e
M3 = a(f-h)
M4 = d(g-e)
M5 = (a+b)h
M6 = (c-a)(e+f)
M7 = (b-d)(g+h)

Result:
p11 = M1 - M4 + M5 + M6
p12 = M3 + M4
p21 = M2 + M5
p22 = M1 - M3 + M2 - M7

Now for 4x4, partition A and B into 2x2 blocks of 2x2 matrices:
A = [[A11, A12], [A21, A22]]
B = [[B11, B12], [B21, B22]]

Strassen on blocks:
S1 = (A11+A22)(B11+B22)
S2 = (A21+A22)B11
S3 = A11(B12-B22)
S4 = A22(B21-B11)
S5 = (A11+A12)B22
S6 = (A21-A11)(B11+B12)
S7 = (A12-A22)(B21+B22)

C11 = S1 - S4 + S5 + S6
C12 = S3 + S4
C21 = S2 + S5
C22 = S1 - S3 + S2 - S7

Each Si is a 2x2 matrix product, computed via Strassen with 7 scalar mults.

So we have 7 block-level Strassen terms, each with 7 scalar-level Strassen terms = 49 total.

Let me index the scalar multiplications. For block term t (0..6) and scalar term s (0..6):
- The scalar multiplication computes: (linear_comb_A) * (linear_comb_B)
- This contributes to specific entries of C.

Let me define the 2x2 Strassen decomposition indices.
For a 2x2 matrix with entries indexed 0..3 (row-major):
entry 0 = (0,0), entry 1 = (0,1), entry 2 = (1,0), entry 3 = (1,1)

Strassen scalar terms for 2x2:
s=0: (A00+A11)*(B00+B11)
s=1: (A10+A11)*B00
s=2: A00*(B01-B11)
s=3: A11*(B10-B00)
s=4: (A00+A01)*B11
s=5: (A10-A00)*(B00+B01)
s=6: (A01-A11)*(B10+B11)

Output contributions:
C00 += s0 - s3 + s4 + s5
C01 += s2 + s3
C10 += s1 + s4
C11 += s0 - s2 + s1 - s6

Now for the 4x4 case, A has 16 entries indexed 0..15 (row-major 4x4):
A[i*4+j] for i,j in 0..3

Block A11 = [[A0, A1], [A4, A5]]
Block A12 = [[A2, A3], [A6, A7]]
Block A21 = [[A8, A9], [A12, A13]]
Block A22 = [[A10, A11], [A14, A15]]

Similarly for B.

Block Strassen terms (each is a 2x2 matrix = 4 scalar entries):
S1 = (A11+A22)*(B11+B22)  -- 2x2 matrix product
S2 = (A21+A22)*B11
S3 = A11*(B12-B22)
S4 = A22*(B21-B11)
S5 = (A11+A12)*B22
S6 = (A21-A11)*(B11+B12)
S7 = (A12-A22)*(B21+B22)

Each St is computed as a 2x2 matrix product via Strassen (7 scalar mults).

For block term t, the "A-side" 2x2 matrix and "B-side" 2x2 matrix are linear combinations of the 4x4 entries. Then each is decomposed via Strassen into 7 scalar multiplications.

Let me define for each block term t:
- a_side[t]: a 2x2 matrix whose entries are linear combos of A's 16 entries
- b_side[t]: a 2x2 matrix whose entries are linear combos of B's 16 entries

Then for scalar term s within block term t:
- The scalar multiplication is: (a_side[t] entry mapped by Strassen s) * (b_side[t] entry mapped by Strassen s)
- This contributes to C entries according to the Strassen output formula, then mapped to 4x4 positions.

Let me code this up.

First, define the block-level linear combinations.

For t=0 (S1): a_side = A11+A22, b_side = B11+B22
  A11+A22 = [[A0+A10, A1+A11], [A4+A14, A5+A15]]
  B11+B22 = [[B0+B10, B1+B11], [B4+B14, B5+B15]]

For t=1 (S2): a_side = A21+A22, b_side = B11
  A21+A22 = [[A8+A10, A9+A11], [A12+A14, A13+A15]]
  B11 = [[B0, B1], [B4, B5]]

For t=2 (S3): a_side = A11, b_side = B12-B22
  A11 = [[A0, A1], [A4, A5]]
  B12-B22 = [[B2-B10, B3-B11], [B6-B14, B7-B15]]

For t=3 (S4): a_side = A22, b_side = B21-B11
  A22 = [[A10, A11], [A14, A15]]
  B21-B11 = [[B8-B0, B9-B1], [B12-B4, B13-B5]]

For t=4 (S5): a_side = A11+A12, b_side = B22
  A11+A12 = [[A0+A2, A1+A3], [A4+A6, A5+A7]]
  B22 = [[B10, B11], [B14, B15]]

For t=5 (S6): a_side = A21-A11, b_side = B11+B12
  A21-A11 = [[A8-A0, A9-A1], [A12-A4, A13-A5]]
  B11+B12 = [[B0+B2, B1+B3], [B4+B6, B5+B7]]

For t=6 (S7): a_side = A12-A22, b_side = B21+B22
  A12-A22 = [[A2-A10, A3-A11], [A6-A14, A7-A15]]
  B21+B22 = [[B8+B10, B9+B11], [B12+B14, B13+B15]]

Now for each block term t, we have a 2x2 "A matrix" and 2x2 "B matrix". We apply Strassen to get 7 scalar multiplications.

Strassen scalar mapping for 2x2:
Given 2x2 matrix M = [[m0, m1], [m2, m3]] (indices 0,1,2,3):
s0: (m0+m3) * (n0+n3)
s1: (m2+m3) * n0
s2: m0 * (n1-n3)
s3: m3 * (n2-n0)
s4: (m0+m1) * n3
s5: (m2-m0) * (n0+n1)
s6: (m1-m3) * (n2+n3)

Where n is the other 2x2 matrix.

Output:
C00 = s0 - s3 + s4 + s5
C01 = s2 + s3
C10 = s1 + s4
C11 = s0 - s2 + s1 - s6

Now, for the overall 4x4 output, the block output positions map to 4x4 positions:
C11_block -> C[0:2, 0:2] (entries 0,1,4,5)
C12_block -> C[0:2, 2:4] (entries 2,3,6,7)
C21_block -> C[2:4, 0:2] (entries 8,9,12,13)
C22_block -> C[2:4, 2:4] (entries 10,11,14,15)

And within each 2x2 block:
block_C00 -> top-left of block
block_C01 -> top-right of block
block_C10 -> bottom-left of block
block_C11 -> bottom-right of block

So for block position (bi, bj) where bi,bj in {0,1}:
  block entry (ri, rj) -> 4x4 entry (bi*2+ri, bj*2+rj)

The 4x4 output index is W[k*n+i, r] = W[k*4+i, r] where i is row, k is col.
So 4x4 entry (row, col) -> W index = col*4 + row.

Let me now build the full decomposition.

For each (t, s) pair (49 total):
1. Compute the linear combination for A-side scalar value
2. Compute the linear combination for B-side scalar value
3. Determine which C entries this contributes to and with what coefficient

Let me define helper structures.

For a 2x2 matrix with entries as linear combos of the 16 A-entries:
I'll represent each 2x2 entry as a dict {A_index: coeff}

Similarly for B.

Then for Strassen scalar s, the A-side value is a linear combo of the 2x2 entries, which is a linear combo of A-entries.

Let me just build this programmatically.
"""

import time

DIMENSIONS = [4, 4, 4]

def construct():
    n, m, p = 4, 4, 4
    rank = 49
    
    # Initialize U, V, W
    U = [[0] * rank for _ in range(n * m)]  # 16 x 49
    V = [[0] * rank for _ in range(m * p)]  # 16 x 49
    W = [[0] * rank for _ in range(p * n)]  # 16 x 49
    
    # Define the 7 block-level Strassen terms
    # Each term has: a_side (2x2 linear combo of A entries), b_side (2x2 linear combo of B entries)
    # A entries indexed 0..15 (row-major 4x4)
    # B entries indexed 0..15 (row-major 4x4)
    
    # Block definitions for A:
    # A11: [[0,1],[4,5]], A12: [[2,3],[6,7]], A21: [[8,9],[12,13]], A22: [[10,11],[14,15]]
    
    # For each block term t (0-6), define a_side and b_side as 2x2 matrices
    # where each entry is a list of (index, coeff) pairs
    
    def make_2x2(entries):
        """entries is a 4-tuple of (list of (idx, coeff)) for [0,0],[0,1],[1,0],[1,1]"""
        return entries
    
    # Block term 0: S1 = (A11+A22)*(B11+B22)
    # A11+A22: [[A0+A10, A1+A11], [A4+A14, A5+A15]]
    a0 = make_2x2([
        [(0,1),(10,1)], [(1,1),(11,1)],
        [(4,1),(14,1)], [(5,1),(15,1)]
    ])
    b0 = make_2x2([
        [(0,1),(10,1)], [(1,1),(11,1)],
        [(4,1),(14,1)], [(5,1),(15,1)]
    ])
    
    # Block term 1: S2 = (A21+A22)*B11
    a1 = make_2x2([
        [(8,1),(10,1)], [(9,1),(11,1)],
        [(12,1),(14,1)], [(13,1),(15,1)]
    ])
    b1 = make_2x2([
        [(0,1)], [(1,1)],
        [(4,1)], [(5,1)]
    ])
    
    # Block term 2: S3 = A11*(B12-B22)
    a2 = make_2x2([
        [(0,1)], [(1,1)],
        [(4,1)], [(5,1)]
    ])
    b2 = make_2x2([
        [(2,1),(10,-1)], [(3,1),(11,-1)],
        [(6,1),(14,-1)], [(7,1),(15,-1)]
    ])
    
    # Block term 3: S4 = A22*(B21-B11)
    a3 = make_2x2([
        [(10,1)], [(11,1)],
        [(14,1)], [(15,1)]
    ])
    b3 = make_2x2([
        [(8,1),(0,-1)], [(9,1),(1,-1)],
        [(12,1),(4,-1)], [(13,1),(5,-1)]
    ])
    
    # Block term 4: S5 = (A11+A12)*B22
    a4 = make_2x2([
        [(0,1),(2,1)], [(1,1),(3,1)],
        [(4,1),(6,1)], [(5,1),(7,1)]
    ])
    b4 = make_2x2([
        [(10,1)], [(11,1)],
        [(14,1)], [(15,1)]
    ])
    
    # Block term 5: S6 = (A21-A11)*(B11+B12)
    a5 = make_2x2([
        [(8,1),(0,-1)], [(9,1),(1,-1)],
        [(12,1),(4,-1)], [(13,1),(5,-1)]
    ])
    b5 = make_2x2([
        [(0,1),(2,1)], [(1,1),(3,1)],
        [(4,1),(6,1)], [(5,1),(7,1)]
    ])
    
    # Block term 6: S7 = (A12-A22)*(B21+B22)
    a6 = make_2x2([
        [(2,1),(10,-1)], [(3,1),(11,-1)],
        [(6,1),(14,-1)], [(7,1),(15,-1)]
    ])
    b6 = make_2x2([
        [(8,1),(10,1)], [(9,1),(11,1)],
        [(12,1),(14,1)], [(13,1),(15,1)]
    ])
    
    a_sides = [a0, a1, a2, a3, a4, a5, a6]
    b_sides = [b0, b1, b2, b3, b4, b5, b6]
    
    # Strassen scalar term definitions for 2x2:
    # For 2x2 matrix M = [[m0,m1],[m2,m3]] and N = [[n0,n1],[n2,n3]]:
    # s0: (m0+m3)*(n0+n3)
    # s1: (m2+m3)*n0
    # s2: m0*(n1-n3)
    # s3: m3*(n2-n0)
    # s4: (m0+m1)*n3
    # s5: (m2-m0)*(n0+n1)
    # s6: (m1-m3)*(n2+n3)
    
    # Define Strassen scalar terms as (a_entries, b_entries) where each is a list of (pos, coeff)
    # pos is 0,1,2,3 for the 2x2 matrix entries
    strassen_terms = [
        ([(0,1),(3,1)], [(0,1),(3,1)]),   # s0
        ([(2,1),(3,1)], [(0,1)]),          # s1
        ([(0,1)], [(1,1),(3,-1)]),         # s2
        ([(3,1)], [(2,1),(0,-1)]),         # s3
        ([(0,1),(1,1)], [(3,1)]),          # s4
        ([(2,1),(0,-1)], [(0,1),(1,1)]),   # s5
        ([(1,1),(3,-1)], [(2,1),(3,1)]),   # s6
    ]
    
    # Output contributions for Strassen 2x2:
    # C00 = s0 - s3 + s4 + s5
    # C01 = s2 + s3
    # C10 = s1 + s4
    # C11 = s0 - s2 + s1 - s6
    
    # output_contrib[s][out_pos] = coeff, where out_pos is 0,1,2,3 for 2x2 output
    output_contrib = [
        {0: 1, 1: 0, 2: 0, 3: 0},   # s0 -> C00
        {0: 0, 1: 0, 2: 1, 3: 1},   # s1 -> C10, C11
        {0: 0, 1: 1, 2: 0, 3: -1},  # s2 -> C01, C11
        {0: -1, 1: 1, 2: 0, 3: 0},  # s3 -> C00, C01
        {0: 1, 1: 0, 2: 1, 3: 0},   # s4 -> C00, C10
        {0: 1, 1: 0, 2: 0, 3: 0},   # s5 -> C00
        {0: 0, 1: 0, 2: 0, 3: -1},  # s6 -> C11
    ]
    
    # Map block output position to 4x4 output position
    # Block (bi, bj) with bi,bj in {0,1}, block entry (ri, rj) with ri,rj in {0,1}
    # 4x4 position: row = bi*2+ri, col = bj*2+rj
    # W index: col*4 + row
    
    # Block term t corresponds to which output block?
    # t=0: S1 -> contributes to C11, C12, C21, C22 (all four blocks)
    # t=1: S2 -> contributes to C21, C22
    # t=2: S3 -> contributes to C12, C22
    # t=3: S4 -> contributes to C11, C12
    # t=4: S5 -> contributes to C11, C21
    # t=5: S6 -> contributes to C11, C12
    # t=6: S7 -> contributes to C12, C22
    
    # Wait, let me re-derive. The block Strassen gives:
    # C11 = S1 - S4 + S5 + S6
    # C12 = S3 + S4
    # C21 = S2 + S5
    # C22 = S1 - S3 + S2 - S7
    
    # So block term t contributes to output blocks with these coefficients:
    # t=0 (S1): C11:+1, C12:0, C21:0, C22:+1
    # t=1 (S2): C11:0, C12:0, C21:+1, C22:+1
    # t=2 (S3): C11:0, C12:+1, C21:0, C22:-1
    # t=3 (S4): C11:-1, C12:+1, C21:0, C22:0
    # t=4 (S5): C11:+1, C12:0, C21:+1, C22:0
    # t=5 (S6): C11:+1, C12:0, C21:0, C22:0
    # t=6 (S7): C11:0, C12:0, C21:0, C22:-1
    
    # block_contrib[t][block_pos] = coeff, where block_pos: 0=C11, 1=C12, 2=C21, 3=C22
    block_contrib = [
        {0: 1, 1: 0, 2: 0, 3: 1},   # t=0
        {0: 0, 1: 0, 2: 1, 3: 1},   # t=1
        {0: 0, 1: 1, 2: 0, 3: -1},  # t=2
        {0: -1, 1: 1, 2: 0, 3: 0},  # t=3
        {0: 1, 1: 0, 2: 1, 3: 0},   # t=4
        {0: 1, 1: 0, 2: 0, 3: 0},   # t=5
        {0: 0, 1: 0, 2: 0, 3: -1},  # t=6
    ]
    
    # Now build the decomposition
    col = 0
    for t in range(7):
        a_side = a_sides[t]
        b_side = b_sides[t]
        
        for s in range(7):
            a_entries, b_entries = strassen_terms[s]
            
            # Compute the A-side linear combination
            a_combo = {}
            for pos, coeff in a_entries:
                for idx, c2 in a_side[pos]:
                    a_combo[idx] = a_combo.get(idx, 0) + coeff * c2
            
            # Compute the B-side linear combination
            b_combo = {}
            for pos, coeff in b_entries:
                for idx, c2 in b_side[pos]:
                    b_combo[idx] = b_combo.get(idx, 0) + coeff * c2
            
            # Fill U and V
            for idx, coeff in a_combo.items():
                if coeff != 0:
                    U[idx][col] = coeff
            
            for idx, coeff in b_combo.items():
                if coeff != 0:
                    V[idx][col] = coeff
            
            # Fill W: for each output position in the 2x2 Strassen result,
            # multiply by block contribution coefficient and map to 4x4 position
            for out_pos, out_coeff in output_contrib[s].items():
                if out_coeff == 0:
                    continue
                # out_pos: 0=C00, 1=C01, 2=C10, 3=C11 in the 2x2 block result
                # This maps to block entry (ri, rj)
                ri = out_pos // 2
                rj = out_pos % 2
                
                for block_pos, block_coeff in block_contrib[t].items():
                    if block_coeff == 0:
                        continue
                    # block_pos: 0=C11, 1=C12, 2=C21, 3=C22
                    bi = block_pos // 2
                    bj = block_pos % 2
                    
                    # 4x4 position
                    row = bi * 2 + ri
                    col4 = bj * 2 + rj
                    
                    # W index: col*4 + row
                    w_idx = col4 * 4 + row
                    total_coeff = out_coeff * block_coeff
                    W[w_idx][col] += total_coeff
            
            col += 1
    
    return {"U": U, "V": V, "W": W}
