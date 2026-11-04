"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0  # safety margin before 150s limit
    
    nm = n * m  # 20
    mp = m * p  # 30
    pn = p * n  # 24
    
    # Build the target tensor: T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some (i,j,k)
    # Otherwise T[a,b,c] = 0
    
    # Schoolbook decomposition (rank = n*m*p = 120)
    schoolbook_rank = n * m * p
    U_sb = [[0] * schoolbook_rank for _ in range(nm)]
    V_sb = [[0] * schoolbook_rank for _ in range(mp)]
    W_sb = [[0] * schoolbook_rank for _ in range(pn)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_sb[a][col] = 1
                V_sb[b][col] = 1
                W_sb[c][col] = 1
                col += 1
    
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = schoolbook_rank
    
    # Try Strassen-based decomposition for 2x2 blocks
    # For 4x5 * 5x6, we can partition into 2x2 blocks:
    # A is 4x5: split into A11(2x2), A12(2x1), A21(2x2), A22(2x1)
    # B is 5x6: split into B11(2x2), B12(2x3), B21(2x2), B22(2x3)
    # 
    # C11 = A11*B11 + A12*B21  (2x2)
    # C12 = A11*B12 + A12*B22  (2x3)
    # C21 = A21*B11 + A22*B21  (2x2)
    # C22 = A21*B12 + A22*B22  (2x3)
    #
    # For each 2x2 * 2x2 block multiplication, use Strassen (rank 7 instead of 8)
    # For 2x2 * 2x3 and 2x3 * 2x2, we can extend Strassen
    #
    # Let's decompose each block multiplication:
    #
    # For C11 = A11*B11 + A12*B21:
    #   A11 is 2x2, B11 is 2x2 -> Strassen rank 7
    #   A12 is 2x1, B21 is 2x2 -> this is (2x1)*(2x2) = 2x2
    #     Can be seen as multiplying a 2x1 by 2x2, which has rank 4 schoolbook
    #     But we can embed: treat A12 as [A12; 0] (2x2) and B21 as is
    #     Actually, (2x1)*(2x2): result is 2x2
    #     Schoolbook: 2*1*2 = 4 terms
    #     We can use a rank-3 decomposition? Let's think...
    #     Actually, let's just use schoolbook for these mixed cases
    #
    # Let me reconsider. The key insight from the cell findings is that using
    # Strassen for 2x2 blocks reduces rank from 120 to 108.
    #
    # 4x5 * 5x6:
    # Partition A (4x5) into 2x2 blocks:
    #   A = [[A11, A12], [A21, A22]]
    #   where A11 is 2x2, A12 is 2x1, A21 is 2x2, A22 is 2x1
    #
    # Partition B (5x6) into 2x2 blocks:
    #   B = [[B11, B12], [B21, B22]]
    #   where B11 is 2x2, B12 is 2x3, B21 is 2x2, B22 is 2x3
    #
    # C = A*B = [[C11, C12], [C21, C22]]
    #   C11 = A11*B11 + A12*B21  (2x2)
    #   C12 = A11*B12 + A12*B22  (2x3)
    #   C21 = A21*B11 + A22*B21  (2x2)
    #   C22 = A21*B12 + A22*B22  (2x3)
    #
    # For each 2x2 * 2x2 product, use Strassen (rank 7 vs 8)
    # For 2x2 * 2x3 and 2x1 * 2x2 products, use schoolbook
    #
    # Let's count the rank:
    # C11 = A11*B11 (Strassen, rank 7) + A12*B21 (schoolbook, rank 4) = 11
    # C12 = A11*B12 (schoolbook, rank 6) + A12*B22 (schoolbook, rank 6) = 12
    # C21 = A21*B11 (schoolbook, rank 4) + A22*B21 (schoolbook, rank 4) = 8
    # C22 = A21*B12 (schoolbook, rank 6) + A22*B22 (schoolbook, rank 6) = 12
    # Total = 11 + 12 + 8 + 12 = 43... that doesn't match 108
    #
    # Wait, I need to be more careful. Let me re-examine.
    #
    # Actually, the cell finding says rank 108. Let me think about what decomposition gives 108.
    #
    # Alternative: Maybe we should use Strassen more cleverly.
    #
    # Let's try: treat the 4x5*5x6 as a block matrix and apply Strassen at a higher level.
    #
    # Actually, let me just implement a straightforward Strassen for the 2x2 sub-blocks
    # where both operands are 2x2, and schoolbook otherwise.
    #
    # But wait - maybe the 108 comes from a different partitioning.
    #
    # Let me try: 4x5 * 5x6
    # Split into 2x2 blocks where possible:
    # A: rows 0-1, 2-3; cols 0-1, 2-3, 4
    # B: rows 0-1, 2-3, 4; cols 0-1, 2-3, 4-5
    #
    # This is getting complex. Let me just implement the Strassen-based approach
    # and verify it works.
    
    # Strassen's algorithm for 2x2 matrix multiplication:
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
    
    # For our tensor decomposition, each "multiplication" M = X*Y contributes
    # a set of terms to U, V, W.
    
    # Let's build the decomposition using Strassen for 2x2*2x2 blocks.
    
    # First, let's define helper functions to add a term to the decomposition
    U_terms = []  # list of (row, coeff) for each term
    V_terms = []  # list of (row, coeff) for each term
    W_terms = []  # list of (row, coeff) for each term
    
    def add_term(a, b, c, coeff=1):
        """Add a term U[a]*V[b]*W[c] with given coefficient."""
        U_terms.append((a, coeff))
        V_terms.append((b, coeff))
        W_terms.append((c, coeff))
    
    def add_scaled_term(a, b, c, coeff):
        """Add a scaled term."""
        if coeff != 0:
            U_terms.append((a, coeff))
            V_terms.append((b, 1))
            W_terms.append((c, 1))
    
    # Actually, let me think about this differently.
    # 
    # The tensor T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    # 
    # For Strassen, we need to express each Strassen multiplication as
    # a sum of rank-1 tensors.
    #
    # A 2x2 * 2x2 multiplication has 8 rank-1 terms (schoolbook) or 7 (Strassen).
    #
    # Let me define the Strassen decomposition for a generic 2x2*2x2 block.
    # 
    # Suppose we're multiplying A_block (2x2) by B_block (2x2).
    # Let A_block have entries A[0,0], A[0,1], A[1,0], A[1,1]
    # Let B_block have entries B[0,0], B[0,1], B[1,0], B[1,1]
    # 
    # The product C_block = A_block * B_block has entries:
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    #
    # Strassen:
    # M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    # M2 = (A[1,0]+A[1,1]) * B[0,0]
    # M3 = A[0,0] * (B[0,1]-B[1,1])
    # M4 = A[1,1] * (B[1,0]-B[0,0])
    # M5 = (A[0,0]+A[0,1]) * B[1,1]
    # M6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
    # M7 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
    #
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    #
    # Each M_i is a product of two linear combinations of A entries and B entries.
    # To express this as a rank-1 tensor decomposition, we need to expand each M_i.
    #
    # For example, M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    #   = A[0,0]*B[0,0] + A[0,0]*B[1,1] + A[1,1]*B[0,0] + A[1,1]*B[1,1]
    # This is 4 rank-1 terms! So Strassen doesn't directly give 7 rank-1 terms
    # in the tensor sense.
    #
    # Wait, I'm confusing two things. Strassen reduces the number of scalar
    # multiplications, but in the tensor decomposition sense, we need to express
    # the full tensor as a sum of rank-1 tensors.
    #
    # Actually, for the tensor decomposition, each "multiplication" in Strassen
    # corresponds to a bilinear form, which can be decomposed into rank-1 terms.
    #
    # Let me reconsider. The tensor for 2x2*2x2 matrix multiplication has
    # dimension 4*4*4 = 64, with 8 nonzero entries. The rank of this tensor
    # is 7 (by Strassen).
    #
    # So there EXISTS a rank-7 decomposition. Let me find it explicitly.
    #
    # The 7 rank-1 tensors for Strassen's decomposition of 2x2*2x2:
    # Each M_i = f_i(A) * g_i(B) where f_i and g_i are linear forms.
    # The contribution to C is: C = sum_i (coefficients) * M_i
    #
    # But M_i = f_i(A) * g_i(B) is not a single rank-1 tensor in the 3-way sense.
    # We need to express f_i(A) * g_i(B) as a sum of rank-1 tensors in the
    # (A-index, B-index, C-index) space.
    #
    # Hmm, this is getting complicated. Let me try a different approach.
    #
    # Let me just implement the schoolbook decomposition and then try to
    # verify if a rank-108 decomposition exists by construction.
    #
    # Actually, let me look at this from the perspective of the known result.
    # The cell finding says rank 108 is achievable. Let me try to construct it.
    #
    # 4*5*6 = 120. 120 - 108 = 12. So we're saving 12 terms.
    #
    # If we use Strassen for each 2x2*2x2 sub-block, we save 1 term per 2x2*2x2.
    # How many 2x2*2x2 sub-blocks are there?
    #
    # In 4x5 * 5x6:
    # A has 2x2 blocks: positions (0,0), (0,2), (2,0), (2,2) but (0,2) is 2x1, etc.
    #
    # Let me just try to construct the rank-108 decomposition directly.
    
    # I'll use a different approach: build the decomposition by partitioning
    # the indices into 2x2 blocks and applying Strassen where both operands
    # are 2x2.
    
    # Clear the terms
    U_terms = []
    V_terms = []
    W_terms = []
    
    # Define the Strassen rank-1 decomposition for 2x2*2x2
    # We need 7 rank-1 tensors (u, v, w) such that their sum gives the
    # 2x2*2x2 multiplication tensor.
    
    # Let me index A entries as (i,j) for i,j in {0,1}
    # B entries as (j,k) for j,k in {0,1}
    # C entries as (i,k) for i,k in {0,1}
    
    # The tensor T[i,j,k] = 1 if i,j,k are consistent (which they always are for 2x2*2x2)
    # Actually, T[i,j,k] = 1 for all i,j,k in {0,1} in the local 2x2*2x2 case.
    # Wait no. For 2x2*2x2, the tensor has T[i,j,k] = 1 for all 8 combinations.
    # The rank of this tensor is 7.
    
    # Strassen's 7 rank-1 decompositions:
    # Each is a triple (u, v, w) where u is in R^4 (for A), v in R^4 (for B), w in R^4 (for C)
    
    # Let me use the standard Strassen decomposition:
    # M1 = (A00+A11)(B00+B11) -> contributes to C00(+1), C11(+1)
    # M2 = (A10+A11)B00 -> contributes to C10(+1)
    # M3 = A00(B01-B11) -> contributes to C01(+1), C00(-1) [wait, let me be careful]
    
    # Actually, let me just look up the explicit rank-1 tensors for Strassen.
    
    # For 2x2*2x2, the 7 rank-1 tensors are:
    # (using 0-indexed, A[i,j], B[j,k], C[i,k])
    
    # M1: u = [1,0,0,1], v = [1,0,0,1], w = [1,0,0,-1]  (C00 += 1, C11 -= 1)
    # Wait, that's not right either.
    
    # Let me derive it properly.
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    
    # Strassen:
    # M1 = (A00+A11)(B00+B11) = A00*B00 + A00*B11 + A11*B00 + A11*B11
    # M2 = (A10+A11)B00 = A10*B00 + A11*B00
    # M3 = A00(B01-B11) = A00*B01 - A00*B11
    # M4 = A11(B10-B00) = A11*B10 - A11*B00
    # M5 = (A00+A01)B11 = A00*B11 + A01*B11
    # M6 = (A10-A00)(B00+B01) = A10*B00 + A10*B01 - A00*B00 - A00*B01
    # M7 = (A01-A11)(B10+B11) = A01*B10 + A01*B11 - A11*B10 - A11*B11
    
    # C00 = M1 + M4 - M5 + M7
    #   = (A00*B00 + A00*B11 + A11*B00 + A11*B11)
    #     + (A11*B10 - A11*B00)
    #     - (A00*B11 + A01*B11)
    #     + (A01*B10 + A01*B11 - A11*B10 - A11*B11)
    #   = A00*B00 + A01*B10 ✓
    
    # C01 = M3 + M5
    #   = (A00*B01 - A00*B11) + (A00*B11 + A01*B11)
    #   = A00*B01 + A01*B11 ✓
    
    # C10 = M2 + M4
    #   = (A10*B00 + A11*B00) + (A11*B10 - A11*B00)
    #   = A10*B00 + A11*B10 ✓
    
    # C11 = M1 - M2 + M3 + M6
    #   = (A00*B00 + A00*B11 + A11*B00 + A11*B11)
    #     - (A10*B00 + A11*B00)
    #     + (A00*B01 - A00*B11)
    #     + (A10*B00 + A10*B01 - A00*B00 - A00*B01)
    #   = A10*B01 + A11*B11 ✓
    
    # Great! Now, each M_i is a bilinear form. To express the whole thing as
    # a sum of rank-1 tensors in the (A, B, C) space, we need to expand each M_i.
    #
    # But M_i is not a single rank-1 tensor. M_i = f_i(A) * g_i(B) where f_i and g_i
    # are linear forms. This is a rank-1 tensor in the (A, B) space, but we also
    # have the C component.
    #
    # Actually, I think the way to think about it is:
    # The full tensor T is in R^{4x4x4} (for 2x2*2x2).
    # T[i,j,k] = delta_{i,k} ... no wait.
    #
    # For matrix multiplication C = A*B, the tensor is:
    # T[i,j,k] = 1 if C[i,k] = sum_j A[i,j]*B[j,k]
    # So T[i,j,k] = 1 for all i,j,k (in the 2x2 case, all 8 entries are 1).
    #
    # A rank-1 tensor is u[i]*v[j]*w[k].
    # We want sum_r u_r[i]*v_r[j]*w_r[k] = T[i,j,k] = 1 for all i,j,k.
    #
    # This is the tensor rank of the 2x2*2x2 matrix multiplication tensor, which is 7.
    
    # Let me find the explicit 7 rank-1 tensors.
    # Using the Strassen decomposition above, we can write:
    # T = sum_{i,k} (sum_j A[i,j]*B[j,k]) * E_{i,k}
    #   = sum_{i,k} sum_j A[i,j]*B[j,k]*E_{i,k}
    #
    # But this is the schoolbook with 8 terms.
    #
    # For Strassen, we use the 7 bilinear forms M_1,...,M_7 and express C in terms of them.
    # The key insight is that each M_i can be written as a single rank-1 tensor
    # in the (A, B, C) space by appropriate choice of u, v, w.
    #
    # Actually, I realize I need to be more careful. Let me look for the explicit
    # rank-1 decomposition.
    
    # After some research, the 7 rank-1 tensors for Strassen's 2x2*2x2 are:
    # (u_r, v_r, w_r) for r=1,...,7
    
    # Let me just implement it by expanding each M_i into rank-1 terms and
    # combining.
    
    # Actually, the simplest approach: since we know the rank is 7, and we have
    # the Strassen bilinear forms, we can construct the rank-1 tensors as follows:
    
    # For each M_i = f_i(A) * g_i(B), the contribution to C is:
    # C += alpha_i * M_i * (linear combination of C basis vectors)
    
    # This is getting too complex to derive by hand. Let me just use the
    # schoolbook decomposition for now and see if I can find a rank-108
    # decomposition through a different means.
    
    # Actually, let me just return the schoolbook decomposition. The cell finding
    # says rank 108 is achievable, but I need to construct it explicitly.
    
    # Let me try a simpler approach: use the known fact that the rank of
    # 2x2*2x2 is 7, and construct the decomposition for our 4x5*5x6 case.
    
    # I'll use a block decomposition where I apply Strassen to the 2x2 sub-blocks.
    
    # For now, let me just return the schoolbook.
    return best_result
# EVOLVE-BLOCK-END
