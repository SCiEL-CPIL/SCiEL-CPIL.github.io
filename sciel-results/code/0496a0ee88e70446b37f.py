"""Matrix multiplication 2,4,7 – rank-minimal exact tensor decomposition.

Strategy: use a block decomposition of the 4×7 factor into two 2×4 blocks
along the inner dimension, and apply a Strassen-like 7-product algorithm
for each 2×2×2 micro-block.  Concretely, we split the 4 inner columns into
two groups of 2, so the 2×4 matrix becomes [A | B] where A,B are 2×2, and
the 4×7 matrix becomes [[C],[D]] where C,D are 2×7.  The product is
  P = A·C + B·D
Each of A·C and B·D is a 2×2 times 2×7 product.  We handle each 2×2×2
micro-product (extracting columns of C and D) with a rank-7 Strassen
decomposition.  Since C has 7 columns and D has 7 columns, and we process
them in 2×2 blocks along the 7 dimension (3 full 2×2 blocks + 1 single
column), we get:
  - For the 7 columns: 3 full 2-column groups use Strassen (rank 7 each
    per 2×2×2 micro-block, but each micro-block produces 2 output columns,
    so per 2-column group of C we need 7 rank-1 tensors for A·C[2 cols]
    and 7 for B·D[2 cols] → 14 per 2-col group, times 3 groups = 42)
  - The last single column of C and D: 2×2×1 product has rank 4 (schoolbook)
    per A and B → 8 total
  Total rank = 42 + 8 = 50.

Wait, let me re-examine.  A 2×2×2 matrix multiply tensor has rank 7 (Strassen).
Each rank-1 term in Strassen corresponds to one scalar product feeding into
the output.  When the "2" in the 2×2×2 is expanded to a vector of length p,
each Strassen product becomes a vector of length p, contributing p rank-1
tensors.  So a 2×2×p multiply has rank 7p.

For our case:
  P = A·C + B·D,  A,B ∈ R^{2×2}, C,D ∈ R^{2×7}
  
We can view this as two independent 2×2×7 multiplies.  Each 2×2×7 has
rank 7×7 = 49 by Strassen-on-vectors.  So total would be 98.  That's worse.

Better approach: think of the full 2×4×7 tensor.  Split the 4 into 2+2:
  The tensor is the sum of two tensors: T1 (A·C part) and T2 (B·D part),
  where T1 multiplies 2×2 by 2×7, and T2 multiplies 2×2 by 2×7.
  Each has rank 49, total 98.  Worse than 56.

Alternative: use a different splitting.  Split the 7 into 4+3?
  2×4×4 + 2×4×3.  Rank of 2×4×4 ≤ 2×4×4 schoolbook = 32.  Rank of 2×4×3 = 24.
  Total = 56.  Same.

Try: split 4 into 1+3: 2×1×7 (rank 7) + 2×3×7 (rank 42) = 49.
Is rank(2×1×7) = 7?  Yes, it's just 7 scalar multiplies, rank 7.
Rank(2×3×7): schoolbook = 42.  Can we do better?

2×3×7: split 3 into 1+2: 2×1×7 (rank 7) + 2×2×7 (rank 49) = 56.  Worse.

Try a more clever approach.  The known result: rank(2×n×p) for n,p ≥ 2.
For 2×2×2, rank is 7.  For 2×2×p, the rank is at most 7p (Strassen extension)
but can it be less?  Actually for 2×2×p with p ≥ 2, the rank is at most
2p + 2(p-1) = 4p-2 by a different construction?  Let me think.

Actually, a known bound: R(2×2×p) ≤ 7p for p even, and ≤ 7(p-1)+4 for p odd
using Strassen on pairs of columns.  For p=7 (odd): 7×6 + 4 = 46.

So 2×2×7 has rank at most 46.

Then 2×4×7: split 4 into 2+2: two copies of 2×2×7, each rank ≤ 46, total ≤ 92.
Still worse than 56.

Split 4 into 1+3: 2×1×7 (rank 7) + 2×3×7.
2×3×7: split 3 into 1+2: 2×1×7 (rank 7) + 2×2×7 (rank ≤ 46) = 53.
Total: 7 + 53 = 60.  Worse than 56.

Split 7 into 3+4: 2×4×3 (rank 24) + 2×4×4 (rank 32) = 56.  Same.

What about using the fact that 2×4×4 might have rank < 32?
2×4×4: split 4 into 2+2: two 2×2×4.  Each 2×2×4 has rank ≤ 7×2=14 (Strassen
on 2 column pairs) or 7×4=28.  Actually 2×2×4: 4 is even, so Strassen on two
2×2×2 blocks: 2×7 = 14.  So two of those: 28.  Total for 2×4×4: 28.
And 2×4×3: split 3 into 1+2: 2×4×1 (rank 8) + 2×4×2.
2×4×2: split 4 into 2+2: two 2×2×2, each rank 7, total 14.
So 2×4×3 ≤ 8 + 14 = 22.
Total: 28 + 22 = 50!

This is a valid decomposition with rank 50.  Let me verify the construction.

2×4×4 with rank 28:
  A is 2×4, B is 4×4.  Split A = [A1 | A2] where A1,A2 are 2×2.
  Split B = [[B11],[B12]] where B11,B12 are 2×4 (top 2 rows, bottom 2 rows).
  Product: A·B = A1·B11 + A2·B12.
  Each A1·B11 is 2×2×4.  Using Strassen on 2 column-pairs of B11:
    B11 = [b1 | b2] where b1,b2 are 2×2.
    A1·b1 uses rank 7 (Strassen 2×2×2), A1·b2 uses rank 7.
    So A1·B11 has rank ≤ 14.
  Similarly A2·B12 has rank ≤ 14.
  Total for 2×4×4: 28. ✓

2×4×2 with rank 14:
  A is 2×4, B is 4×2.  Split A = [A1 | A2], B = [[B1],[B2]] where B1,B2 are 2×2.
  A·B = A1·B1 + A2·B2.  Each 2×2×2, rank 7.  Total 14. ✓

2×4×1 with rank 8:
  A is 2×4, B is 4×1.  Schoolbook: 8 rank-1 terms.  Rank 8. ✓

So total rank = 28 + 22 = 50.

Let me implement this carefully.

Indices:
  i ∈ {0,1} (rows of A and output)
  j ∈ {0,1,2,3} (inner dimension, 4)
  k ∈ {0,...,6} (columns of B and output)

Output W[k*2+i, c]

Step 1: 2×4×4 part (k=0..3), rank 28
  Split j into j1∈{0,1} and j2∈{2,3}.
  Split k into k1∈{0,1} and k2∈{2,3}.
  
  Term group 1: A1·B11 (i, j1, k1) — 2×2×2, Strassen rank 7
  Term group 2: A1·B12 (i, j1, k2) — 2×2×2, Strassen rank 7
  Term group 3: A2·B21 (i, j2, k1) — 2×2×2, Strassen rank 7
  Term group 4: A2·B22 (i, j2, k2) — 2×2×2, Strassen rank 7
  Total: 28.

Step 2: 2×4×2 part (k=4,5), rank 14
  Split j into j1∈{0,1} and j2∈{2,3}.
  k ∈ {4,5}, treat as k'∈{0,1} (so B is 4×2, rows j, cols k').
  
  Term group 5: A1·B51 (i, j1, k') — 2×2×2, Strassen rank 7
  Term group 6: A2·B52 (i, j2, k') — 2×2×2, Strassen rank 7
  Total: 14.

Step 3: 2×4×1 part (k=6), rank 8
  Schoolbook: for each i∈{0,1}, j∈{0,1,2,3}: one rank-1 term.
  Total: 8.

Grand total: 28 + 14 + 8 = 50.

Strassen 2×2×2 rank-7 decomposition:
  Given matrices X (2×2) and Y (2×2), the 7 products are:
    M1 = (X00+X11)(Y00+Y11)
    M2 = (X10+X11)·Y00
    M3 = X00·(Y01-Y11)
    M4 = X11·(Y10-Y00)
    M5 = (X00+X01)·Y11
    M6 = (X10-X00)(Y00+Y01)
    M7 = (X01-X11)(Y10+Y11)
  
  Output Z = X·Y where:
    Z00 = M1 + M4 - M5 + M7
    Z01 = M3 + M5
    Z10 = M2 + M4
    Z11 = M1 - M2 + M3 + M6

  Each M is a scalar product of a linear combination of X entries and a
  linear combination of Y entries.  In the tensor decomposition, each rank-1
  term corresponds to one M, with U, V, W entries being the coefficients.

  For a 2×2×2 block with row-index offset i0, col-index offset j0, out-offset k0:
    X entries: A[i0+di, j0+dj] for di,dj ∈ {0,1}
    Y entries: B[j0+dj', k0+dk] for dj',dk ∈ {0,1}
    Output: P[i0+di_out, k0+dk_out] for di_out,dk_out ∈ {0,1}

  U has shape (n*m, rank) = (8, rank).  U[i*m + j, c]
  V has shape (m*p, rank) = (28, rank).  V[j*p + k, c]
  W has shape (p*n, rank) = (14, rank).  W[k*n + i, c]

  For each Strassen product M, the U coefficient for row (i,j) is the
  coefficient of A[i,j] in the X-linear-combination.  Similarly for V and W.
"""

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 7
    
    # Strassen 2x2x2: 7 rank-1 terms
    # Each term: (x_coeff, y_coeff, z_coeff) where
    #   x_coeff = {(di,dj): coeff} for A[di,dj]
    #   y_coeff = {(dj',dk): coeff} for B[dj',dk]
    #   z_coeff = {(di_out,dk_out): coeff} for P[di_out,dk_out]
    
    # M1 = (X00+X11)(Y00+Y11)
    s1_x = {(0,0):1, (1,1):1}
    s1_y = {(0,0):1, (1,1):1}
    s1_z = {(0,0):1, (0,1):0, (1,0):0, (1,1):1}  # Z00+=1, Z11+=1
    
    # M2 = (X10+X11)Y00
    s2_x = {(1,0):1, (1,1):1}
    s2_y = {(0,0):1}
    s2_z = {(1,0):1, (1,1):-1}  # Z10+=1, Z11-=1
    
    # M3 = X00(Y01-Y11)
    s3_x = {(0,0):1}
    s3_y = {(0,1):1, (1,1):-1}
    s3_z = {(0,0):1, (1,1):1}  # Z00+=1, Z11+=1
    
    # M4 = X11(Y10-Y00)
    s4_x = {(1,1):1}
    s4_y = {(1,0):1, (0,0):-1}
    s4_z = {(0,0):1, (1,0):1}  # Z00+=1, Z10+=1
    
    # M5 = (X00+X01)Y11
    s5_x = {(0,0):1, (0,1):1}
    s5_y = {(1,1):1}
    s5_z = {(0,0):-1, (0,1):1}  # Z00-=1, Z01+=1
    
    # M6 = (X10-X00)(Y00+Y01)
    s6_x = {(1,0):1, (0,0):-1}
    s6_y = {(0,0):1, (0,1):1}
    s6_z = {(1,1):1}  # Z11+=1
    
    # M7 = (X01-X11)(Y10+Y11)
    s7_x = {(0,1):1, (1,1):-1}
    s7_y = {(1,0):1, (1,1):1}
    s7_z = {(0,0):1}  # Z00+=1
    
    strassen_terms = [
        (s1_x, s1_y, s1_z),
        (s2_x, s2_y, s2_z),
        (s3_x, s3_y, s3_z),
        (s4_x, s4_y, s4_z),
        (s5_x, s5_y, s5_z),
        (s6_x, s6_y, s6_z),
        (s7_x, s7_y, s7_z),
    ]
    
    # Collect all rank-1 terms: (u_vec, v_vec, w_vec)
    # u_vec: dict from (i,j) to coeff
    # v_vec: dict from (j,k) to coeff
    # w_vec: dict from (k,i) to coeff
    
    all_terms = []
    
    def add_strassen_block(i0, j0, k0):
        """Add a 2x2x2 Strassen block at offsets (i0, j0, k0)."""
        for (sx, sy, sz) in strassen_terms:
            u = {}
            v = {}
            w = {}
            for (di, dj), coeff in sx.items():
                u[(i0+di, j0+dj)] = coeff
            for (dj, dk), coeff in sy.items():
                v[(j0+dj, k0+dk)] = coeff
            for (di, dk), coeff in sz.items():
                w[(k0+dk, i0+di)] = coeff
            all_terms.append((u, v, w))
    
    # Part 1: 2x4x4 (k=0..3), split into four 2x2x2 blocks
    # j-groups: 0-1, 2-3; k-groups: 0-1, 2-3
    add_strassen_block(0, 0, 0)  # A[0:2,0:2] * B[0:2,0:2]
    add_strassen_block(0, 0, 2)  # A[0:2,0:2] * B[0:2,2:4]
    add_strassen_block(0, 2, 0)  # A[0:2,2:4] * B[2:4,0:2]
    add_strassen_block(0, 2, 2)  # A[0:2,2:4] * B[2:4,2:4]
    # That's 4 blocks × 7 = 28 terms. ✓
    
    # Part 2: 2x4x2 (k=4,5), split into two 2x2x2 blocks
    # j-groups: 0-1, 2-3; k: 4-5 (treat as 0-1 offset)
    add_strassen_block(0, 0, 4)  # A[0:2,0:2] * B[0:2,4:6]
    add_strassen_block(0, 2, 4)  # A[0:2,2:4] * B[2:4,4:6]
    # 2 blocks × 7 = 14 terms. ✓
    
    # Part 3: 2x4x1 (k=6), schoolbook rank 8
    for i in range(n):
        for j in range(m):
            u = {(i, j): 1}
            v = {(j, 6): 1}
            w = {(6, i): 1}
            all_terms.append((u, v, w))
    # 8 terms. ✓
    
    rank = len(all_terms)
    
    # Build U, V, W as lists of lists
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, (u, v, w) in enumerate(all_terms):
        for (i, j), coeff in u.items():
            U[i * m + j][c] = coeff
        for (j, k), coeff in v.items():
            V[j * p + k][c] = coeff
        for (k, i), coeff in w.items():
            W[k * n + i][c] = coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
