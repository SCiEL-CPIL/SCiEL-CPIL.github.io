# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2():
    """
    Return the 7 rank-1 terms for 2x2x2 matrix multiplication.
    Each term is a tuple (U_row, V_col, W_elem, coeff) where:
      U_row: index in (0..3) = i*2+j
      V_col: index in (0..3) = j*2+k
      W_elem: index in (0..3) = k*2+i  (per trace convention W[k*n+i] for n=2)
      coeff: multiplier
    """
    # Strassen's 7 products:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # For each product, we need to express it as sum of rank-1 terms.
    # A product like (a_u * a_v)(b_u * b_v) expands to 4 rank-1 terms.
    # But Strassen combines them. Let me just enumerate the 7 products
    # and expand each into individual (i,j,k) contributions.
    
    # Actually, let me think differently. The 2x2x2 tensor has 8 entries.
    # Strassen gives 7 rank-1 decompositions. Let me construct them directly.
    
    # C11 = M1 - M4 + M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M2 - M6
    
    # Each Mk is a product of two 2x2 submatrix sums.
    # Let me just hardcode the 7 terms by expanding.
    
    # Term 1: M1 = (A11+A22)(B11+B22)
    # = A11*B11 + A11*B22 + A22*B11 + A22*B22
    # These contribute to C11 and C22 via M1.
    # C11 gets +M1, C22 gets +M1
    
    # This is getting complex. Let me use a different approach:
    # Just enumerate all 8 (i,j,k) pairs and use the known Strassen decomposition.
    
    # Actually, the simplest correct approach: use the trivial rank-8 decomposition
    # for 2x2x2, then apply Strassen to get rank 7.
    
    # Let me construct the Strassen decomposition properly.
    # The 7 multiplications are:
    # M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    # M2 = (A[1,0]+A[1,1]) * B[0,0]
    # M3 = A[0,0] * (B[0,1]-B[1,1])
    # M4 = A[1,1] * (B[1,0]-B[0,0])
    # M5 = (A[0,0]+A[0,1]) * B[1,1]
    # M6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
    # M7 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
    
    # C[0,0] = M1 - M4 + M5 - M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M3 + M2 - M6
    
    # Each Mk expands to rank-1 terms. Let's collect all rank-1 terms.
    
    terms = []
    
    # M1 = (A00+A11)(B00+B11) = A00*B00 + A00*B11 + A11*B00 + A11*B11
    # In terms of (i,j,k): 
    # A00*B00 -> (i=0,j=0,k=0)
    # A00*B11 -> (i=0,j=0,k=1)
    # A11*B00 -> (i=1,j=0,k=0)
    # A11*B11 -> (i=1,j=0,k=1)
    # Wait, that's not right. Let me be more careful.
    
    # A[i,j] * B[j,k] contributes to C[i,k].
    # M1 involves:
    # A[0,0]*B[0,0] -> C[0,0]
    # A[0,0]*B[0,1] -> C[0,1]  (no, B11 means B[1,1] in 0-indexed? Let me use 0-indexed)
    
    # Let me redo with 0-indexed:
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    
    # M1 = (A00+A11)(B00+B11)
    # = A00*B00 + A00*B11 + A11*B00 + A11*B11
    # A00*B00: i=0,j=0,k=0
    # A00*B11: i=0,j=1,k=1
    # A11*B00: i=1,j=0,k=0
    # A11*B11: i=1,j=1,k=1
    
    # M2 = (A10+A11)*B00
    # = A10*B00 + A11*B00
    # A10*B00: i=1,j=0,k=0
    # A11*B00: i=1,j=0,k=0  -- wait, A11 is at i=1,j=1, so A11*B00: i=1,j=1,k=0
    
    # M3 = A00*(B01-B11)
    # = A00*B01 - A00*B11
    # A00*B01: i=0,j=0,k=1
    # A00*B11: i=0,j=1,k=1
    
    # M4 = A11*(B10-B00)
    # = A11*B10 - A11*B00
    # A11*B10: i=1,j=1,k=0
    # A11*B00: i=1,j=1,k=0  -- no, A11 is i=1,j=1, B00 is j=0,k=0. But j must match!
    # A[i,j]*B[j,k]. A11 has j=1, B00 has j=0. Mismatch!
    
    # I think I'm confusing myself. Let me just use the standard result.
    # The rank of 2x2x2 is 7. Let me just use a known decomposition.
    
    # Actually, for the purpose of this problem, I can just use the trivial rank-8
    # decomposition for 2x2x2 and then compose for 4x4x4. That gives 64 for 4x4x4.
    # But Strassen gives 49. Let me try to implement Strassen properly.
    
    # Let me use a different approach: build the 4x4x4 decomposition by
    # applying Strassen at the 2x2 block level, where each 2x2 subproduct
    # uses the trivial rank-4 decomposition.
    
    # This gives 7 * 4 = 28 rank-1 terms per 4x4x4 block? No.
    # Each Strassen product is a 2x2 matrix multiplication, which has rank 4.
    # 7 products * 4 = 28. But Strassen for 4x4x4 should give 49.
    
    # Wait, Strassen for 4x4: split into 2x2 blocks. Each of the 7 products
    # is a 2x2 * 2x2 multiplication, which uses 4 scalar multiplications (trivial).
    # So 7 * 4 = 28. But the known rank of 4x4x4 is 49 (from applying Strassen
    # to 2x2 blocks where each 2x2 uses Strassen).
    
    # Actually, Strassen recursively: rank(4x4x4) <= 7 * rank(2x2x2) = 7*7 = 49.
    # Yes! So each of the 7 block products uses Strassen's 7 scalar multiplications.
    
    # So I need to compose: for each of the 7 Strassen block products,
    # generate 7 rank-1 terms for the 2x2 subproblem, then map to global indices.
    
    # Let me implement this carefully.
    
    # First, the 7 Strassen 2x2x2 terms (as (i,j,k) with coefficients in W):
    # From Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M2 - M6
    
    # Let me expand each M into rank-1 terms and collect.
    
    # I'll represent each term as (i, j, k, coeff_in_W)
    # where the term is coeff * A[i,j] * B[j,k] contributing to W[k*2+i]
    
    # M1: (A00+A11)(B00+B11)
    # = A00*B00 + A00*B11 + A11*B00 + A11*B11
    # But wait, A00*B00: i=0,j=0,k=0. A00*B11: i=0,j=1,k=1? No.
    # A[i,j]*B[j,k]. A00 has i=0,j=0. B00 has j=0,k=0. So A00*B00 -> i=0,j=0,k=0.
    # A00*B11: A00 has j=0, B11 has j=1. Mismatch! This product doesn't make sense
    # as a single (i,j,k) term. 
    
    # I see the issue. In Strassen, the products are of submatrices, not individual
    # elements. Let me think of it as: each M is a product of a 2x2 matrix (linear
    # combination of A's) and a 2x2 matrix (linear combination of B's). The result
    # is a 2x2 matrix, each entry of which is a sum of products A[i,j]*B[j,k].
    
    # This is getting too complex to hand-derive. Let me just use the trivial
    # rank-8 decomposition for 2x2x2 and build 4x4x4 with rank 64 (8*8).
    # Then 4x4x9 = two 4x4x4 blocks (rank 64 each) + one 4x4x1 (rank 4) = 132.
    # Score = 144/132 = 1.09. Still an improvement over 1.0.
    
    # But can I do better? Let me try to implement Strassen for 2x2x2 properly.
    
    # Actually, let me just use numpy to verify. But I can't run code here.
    
    # Let me use a simpler approach: just use the trivial decomposition but
    # with the block structure. For 4x4x9:
    # - Columns 0-3: 4x4x4, rank 64 (trivial)
    # - Columns 4-7: 4x4x4, rank 64 (trivial)  
    # - Column 8: 4x4x1, rank 4
    # Total: 132. Score 144/132 = 1.09.
    
    # Alternatively, if I can get Strassen working for 2x2x2 (rank 7),
    # then 4x4x4 would be 49, and total would be 49+49+4 = 102. Score 1.41.
    
    # Let me try to hardcode the Strassen 2x2x2 decomposition.
    
    # After careful thought, here are the 7 rank-1 terms for 2x2x2:
    # Using the standard Strassen decomposition, each "M" is not a single
    # rank-1 term but a set of rank-1 terms. The total is 7 multiplications,
    # but each multiplication of two linear forms gives multiple (i,j,k) terms.
    
    # I think the cleanest approach is to just return the trivial decomposition
    # for now and get a score improvement.
    
    pass  # placeholder, will implement construct directly

def construct():
    n, m, p = 4, 4, 9
    # Use block decomposition:
    # Split p=9 into 4+4+1
    # For each 4x4x4 block, use trivial rank 64
    # For 4x4x1, use rank 4
    # Total rank: 64 + 64 + 4 = 132
    
    rank = 132
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    # Block 1: columns 0-3 (k=0,1,2,3)
    for i in range(n):
        for j in range(m):
            for k in range(4):
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
                r += 1
    
    # Block 2: columns 4-7 (k=4,5,6,7)
    for i in range(n):
        for j in range(m):
            for k in range(4, 8):
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
                r += 1
    
    # Block 3: column 8 (k=8)
    for i in range(n):
        for j in range(m):
            U[i * m + j][r] = 1
            V[j * p + 8][r] = 1
            W[8 * n + i][r] = 1
            r += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
