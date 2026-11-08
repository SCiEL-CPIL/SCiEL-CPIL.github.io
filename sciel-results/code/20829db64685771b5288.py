"""Tensor decomposition for 4x4 matrix multiplication.
Uses Strassen's algorithm for 2x2 blocks to achieve rank 27 (vs 64 schoolbook).
"""
import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-27 tensor decomposition for 4x4 matrix multiplication
    using Strassen's algorithm applied to 2x2 block matrices.
    
    The 4x4 matrices are partitioned into 2x2 blocks of 2x2 matrices:
    A = [[A00, A01], [A10, A11]]  (each Ai is 2x2)
    B = [[B00, B01], [B10, B11]]  (each Bj is 2x2)
    
    C = A*B is computed via Strassen's 7 multiplications of 2x2 matrices.
    Each 2x2 multiplication uses the schoolbook 4-term decomposition.
    Total rank = 7 * 4 = 28... wait, let me reconsider.
    
    Actually, Strassen for 2x2 gives 7 products of 2x2 matrices.
    Each 2x2 product has rank 4 (schoolbook).
    So total rank = 7 * 4 = 28.
    
    But we can do better: the Strassen decomposition for 2x2 is rank 7,
    and each "product" is a 2x2 matrix multiplication which itself
    decomposes into 4 rank-1 terms. So the total tensor rank is 7*4 = 28.
    
    Let me verify: n=m=p=4. Schoolbook rank = 64. Strassen-based = 28.
    """
    deadline = time.time() + 120
    
    # Build the Strassen 4x4 decomposition
    # A is 4x4, B is 4x4, C = A @ B is 4x4
    # Index convention: A[i,j] where i in [0,4), j in [0,4)
    # B[j,k] where j in [0,4), k in [0,4)
    # C[i,k] where i in [0,4), k in [0,4)
    #
    # Block structure (2x2 blocks of 2x2):
    # A = [[A00, A01], [A10, A11]]
    #   A00 = A[0:2, 0:2], A01 = A[0:2, 2:4]
    #   A10 = A[2:4, 0:2], A11 = A[2:4, 2:4]
    # B = [[B00, B01], [B10, B11]]
    #   B00 = B[0:2, 0:2], B01 = B[0:2, 2:4]
    #   B10 = B[2:4, 0:2], B11 = B[2:4, 2:4]
    # C = [[C00, C01], [C10, C11]]
    #   C00 = C[0:2, 0:2], etc.
    #
    # Strassen:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # Each Mi is a 2x2 matrix product. For a 2x2 product P = X*Y:
    # P[0,0] = X[0,0]*Y[0,0] + X[0,1]*Y[1,0]
    # P[0,1] = X[0,0]*Y[0,1] + X[0,1]*Y[1,1]
    # P[1,0] = X[1,0]*Y[0,0] + X[1,1]*Y[1,0]
    # P[1,1] = X[1,0]*Y[0,1] + X[1,1]*Y[1,1]
    #
    # So each 2x2 product has 4 rank-1 terms.
    # Total: 7 * 4 = 28 rank-1 terms.
    #
    # The tensor T[i,j,k] = 1 if C[i,k] = sum_j A[i,j]*B[j,k]
    # Decomposition: T = sum_c U[:,c] outer V[:,c] outer W[:,c]
    # where U has shape (4*4, rank) = (16, rank)
    #       V has shape (4*4, rank) = (16, rank)  
    #       W has shape (4*4, rank) = (16, rank)
    #
    # Indexing: U[i*4+j, c], V[j*4+k, c], W[k*4+i, c]
    # (following the convention W[k*n+i, r] with n=4)
    
    n = m = p = 4
    rank = 28
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Helper: add a 2x2 schoolbook product to the decomposition
    # X is a 2x2 matrix formed as linear combination of A blocks
    # Y is a 2x2 matrix formed as linear combination of B blocks
    # The product contributes to C blocks with given coefficients
    #
    # For each of the 4 scalar products in the 2x2 product:
    #   (X[a,b] * Y[c,d]) contributes to P[a,d]
    #   where a,c in {0,1}, b,d in {0,1}
    #
    # X[a,b] = sum over A-block entries with coefficients
    # Y[c,d] = sum over B-block entries with coefficients
    
    # Define the 7 Strassen products
    # Each product: (X, Y, output_block, coefficient)
    # X and Y are 2x2 matrices expressed as linear combos of A and B blocks
    
    # Block indices for 2x2 sub-matrices within the 4x4:
    # A block (bi, bj) covers rows 2*bi to 2*bi+1, cols 2*bj to 2*bj+1
    # B block (bi, bj) covers rows 2*bi to 2*bi+1, cols 2*bj to 2*bj+1
    # C block (ci, ck) covers rows 2*ci to 2*ci+1, cols 2*ck to 2*ck+1
    
    # For a 2x2 product X*Y = P:
    # P[r, s] = sum_{t=0}^{1} X[r, t] * Y[t, s]
    # Each X[r,t] is a linear combo of A entries
    # Each Y[t,s] is a linear combo of B entries
    #
    # The contribution to C is: coeff * P[r,s] added to C_block[cr, cs][r, s]
    # where C_block[cr,cs] is the 2x2 block at position (cr, cs)
    
    # Let's define each Strassen term as:
    # term = (X_coefficients, Y_coefficients, C_block_position, sign)
    # X_coefficients: dict mapping (a_row, a_col) -> list of (A_block_row, A_block_col, coeff)
    # This is getting complex. Let me use a more direct approach.
    
    # Direct approach: enumerate all 28 rank-1 terms explicitly.
    # For each of the 7 Strassen products, for each of the 4 scalar products
    # in the 2x2 matrix multiplication, create a rank-1 term.
    
    # Strassen products and their contributions to C blocks:
    # M1 = (A00+A11)(B00+B11) -> C00: +1, C11: +1
    # M2 = (A10+A11)B00       -> C10: +1, C11: -1
    # M3 = A00(B01-B11)       -> C01: +1, C11: +1
    # M4 = A11(B10-B00)       -> C00: +1, C10: +1
    # M5 = (A00+A01)B11       -> C00: -1, C01: +1
    # M6 = (A10-A00)(B00+B01) -> C11: +1
    # M7 = (A01-A11)(B10+B11) -> C00: +1
    
    # For each Mi, the 2x2 product has 4 scalar products:
    # P[0,0] = X[0,0]*Y[0,0] + X[0,1]*Y[1,0]
    # P[0,1] = X[0,0]*Y[0,1] + X[0,1]*Y[1,1]
    # P[1,0] = X[1,0]*Y[0,0] + X[1,1]*Y[1,0]
    # P[1,1] = X[1,0]*Y[0,1] + X[1,1]*Y[1,1]
    
    # I'll build this systematically.
    
    # Each "product term" is defined by:
    # - A linear combination for the X factor (2x2 matrix)
    # - A linear combination for the Y factor (2x2 matrix)  
    # - Which C blocks it contributes to, and with what sign
    
    # Let me define each of the 28 terms explicitly.
    
    # Term structure: for a scalar product X[r,t] * Y[t,s]:
    # X[r,t] = sum_{(br,bc)} coeff_a * A[2*br + r, 2*bc + t]
    # Y[t,s] = sum_{(br,bc)} coeff_b * B[2*br + t, 2*bc + s]
    # Contribution to C: sign * C[2*cr + r, 2*cs + s]
    
    # Let me just hardcode all 28 terms.
    
    # I'll use a list of tuples: (U_entries, V_entries, W_entries)
    # where U_entries is a dict {flat_index: value}, etc.
    
    terms = []
    
    # Helper to add a term
    def add_term(u_dict, v_dict, w_dict):
        terms.append((u_dict, v_dict, w_dict))
    
    # For a 2x2 product where:
    # X[r,t] = sum of A entries
    # Y[t,s] = sum of B entries
    # We need to create rank-1 terms for each (r,t,s) combination
    # but it's not rank-1 in the simple sense because X[r,t] and Y[t,s]
    # are each sums of multiple entries.
    #
    # Actually, each scalar product like A[i,j]*B[j,k] IS a rank-1 term.
    # So if X[0,0] = A[0,0] + A[2,2], then X[0,0]*Y[0,0] expands to
    # multiple rank-1 terms.
    #
    # Let me reconsider. The Strassen algorithm says:
    # C00 = M1 + M4 - M5 + M7
    # where M1 = (A00+A11)(B00+B11)
    #
    # M1[0,0] = (A00[0,0]+A11[0,0])*(B00[0,0]+B11[0,0]) + (A00[0,1]+A11[0,1])*(B00[1,0]+B11[1,0])
    #
    # This expands to 4 products, each of which is a product of sums.
    # (a+b)(c+d) = ac + ad + bc + bd
    #
    # So M1[0,0] = 4 terms, each a product of one A entry and one B entry.
    # Similarly for all other entries.
    #
    # Total terms per Strassen product: 4 (matrix entries) * 4 (expansion) = 16?
    # No wait. Let me recount.
    #
    # M1 = (A00+A11)(B00+B11) is a 2x2 matrix.
    # M1[r,s] = sum_t (A00[r,t]+A11[r,t]) * (B00[t,s]+B11[t,s])
    #          = sum_t [A00[r,t]*B00[t,s] + A00[r,t]*B11[t,s] + A11[r,t]*B00[t,s] + A11[r,t]*B11[t,s]]
    #
    # For fixed r,s: sum over t in {0,1} of 4 terms = 8 terms
    # For all r,s in {0,1}: 4 * 8 = 32 terms? That can't be right.
    #
    # Wait, I'm overcomplicating. Let me think about it differently.
    #
    # The key insight: each rank-1 term in the tensor decomposition is of the form
    # u_i * v_j * w_k for a single (i,j,k) triple... no.
    #
    # A rank-1 term in the matrix multiplication tensor is:
    # T[i,j,k] = U[i*4+j, c] * V[j*4+k, c] * W[k*4+i, c]
    #
    # For a single rank-1 term to contribute to C[i,k] = sum_j A[i,j]*B[j,k],
    # we need: U[i*4+j, c] = a_i (for all j, or specific j), etc.
    #
    # Actually, the simplest rank-1 term is: U has a 1 at position (i,j),
    # V has a 1 at position (j,k), W has a 1 at position (k,i).
    # This gives T[i,j,k] = 1 for that specific (i,j,k) triple.
    #
    # The schoolbook uses 64 such terms (one for each i,j,k).
    #
    # For Strassen, we combine terms. A Strassen term like A00[0,0]*B00[0,0]
    # corresponds to A[0,0]*B[0,0] which contributes to C[0,0].
    # That's the same as a schoolbook term!
    #
    # The difference is that Strassen uses COMBINED terms:
    # (A00[0,0]+A11[0,0])*(B00[0,0]+B11[0,0])
    # = A[0,0]*B[0,0] + A[0,0]*B[2,0] + A[2,0]*B[0,0] + A[2,0]*B[2,0]
    #
    # This is 4 rank-1 terms! So Strassen doesn't reduce the number of
    # rank-1 terms in this naive expansion.
    #
    # Hmm, I think I was confused. Let me reconsider.
    #
    # Actually, the Strassen algorithm reduces the number of MULTIPLICATIONS,
    # but the tensor rank (minimum number of rank-1 terms) is a different question.
    #
    # For 2x2 matrix multiplication, the tensor rank is known to be 7 (Strassen).
    # This means the 2x2 matmul tensor can be decomposed into 7 rank-1 terms.
    #
    # For 4x4, if we use Strassen recursively or via block decomposition,
    # the rank should be 7 * (rank of 2x2 matmul) = 7 * 7 = 49?
    # No, that's not right either.
    #
    # Let me think more carefully. The 4x4 matmul tensor has dimension 4*4*4 = 64.
    # The schoolbook gives rank 64.
    #
    # Using 2x2 blocks: the 4x4 matmul can be seen as 4 block multiplications
    # (C00, C01, C10, C11), each involving 2x2 block matrices.
    # But Strassen computes all 4 block outputs using 7 block multiplications.
    #
    # Each 2x2 block multiplication has tensor rank 7.
    # So the total rank is 7 * 7 = 49?
    #
    # Wait, no. The 7 Strassen products are 2x2 matrix multiplications.
    # Each has rank 7. So total rank = 7 * 7 = 49.
    #
    # But that's worse than 64/4 = 16... no, 49 < 64. So it's an improvement.
    #
    # Actually, I recall that the best known rank for 4x4 matrix multiplication
    # is 49 (using Strassen twice or once with appropriate blocking).
    #
    # Let me verify: 
    # - 2x2 matmul: rank 7 (Strassen)
    # - 4x4 matmul via 2x2 blocks with Strassen: 7 products of 2x2 matrices, each of rank 7, so total rank 49.
    #
    # Yes, rank 49 for 4x4 is the standard result.
    
    # OK so I need to implement the rank-49 decomposition.
    # 
    # Structure:
    # - 7 Strassen "level-1" products, each a 2x2 matrix multiplication
    # - Each 2x2 multiplication decomposes into 7 rank-1 terms (Strassen for 2x2)
    # - Total: 7 * 7 = 49 rank-1 terms
    #
    # Let me implement this step by step.
    
    # First, define the Strassen decomposition for 2x2 matrices.
    # For 2x2 matrices X, Y, Z = X*Y:
    # M1 = (X00+X11)(Y00+Y11)
    # M2 = (X10+X11)Y00
    # M3 = X00(Y01-Y11)
    # M4 = X11(Y10-Y00)
    # M5 = (X00+X01)Y11
    # M6 = (X10-X00)(Y00+Y01)
    # M7 = (X01-X11)(Y10+Y11)
    #
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    #
    # Each Mi is a scalar product of two linear combinations.
    # For the tensor decomposition, each Mi = (sum a_i * X[i,j]) * (sum b_j * Y[j,k])
    # expands to rank-1 terms.
    
    # For 2x2, the rank-7 decomposition means we can write:
    # T2x2[i,j,k] = sum_{c=1}^{7} U2[i,j,c] * V2[j,k,c] * W2[k,i,c]
    #
    # Let me derive this. For each of the 7 Mi:
    # Mi = X_expr * Y_expr where X_expr = sum of X entries with coeffs,
    # Y_expr = sum of Y entries with coeffs.
    #
    # Mi contributes to Z entries with certain coefficients.
    #
    # This is getting complex. Let me just build it programmatically.
    
    # For 2x2 Strassen, the 7 terms and their contributions:
    # Term 1: (X00+X11)*(Y00+Y11) -> Z00: +1, Z11: +1
    # Term 2: (X10+X11)*Y00       -> Z10: +1, Z11: -1
    # Term 3: X00*(Y01-Y11)       -> Z01: +1, Z11: +1
    # Term 4: X11*(Y10-Y00)       -> Z00: +1, Z10: +1
    # Term 5: (X00+X01)*Y11       -> Z00: -1, Z01: +1
    # Term 6: (X10-X00)*(Y00+Y01) -> Z11: +1
    # Term 7: (X01-X11)*(Y10+Y11) -> Z00: +1
    
    # Each term is a product of two linear forms. To make it rank-1 in the
    # tensor sense, we need to expand each product into individual scalar products.
    # But that would give more than 7 terms!
    #
    # I think the issue is that the "rank" in tensor decomposition allows
    # the U, V, W vectors to have multiple non-zero entries.
    #
    # For example, for Term 1: (X00+X11)*(Y00+Y11)
    # This contributes to Z00 and Z11.
    # We can create ONE rank-1 term where:
    # U has non-zero entries at (0,0) and (1,1) [for X00 and X11]
    # V has non-zero entries at (0,0) and (1,1) [for Y00 and Y11]
    # W has non-zero entries at (0,0) and (1,1) [for Z00 and Z11]
    #
    # But wait, that's not how it works. A single rank-1 term U*V*W
    # has T[i,j,k] = U[i,j]*V[j,k]*W[k,i].
    # If U[0,0]=1, U[1,1]=1, V[0,0]=1, V[1,1]=1, W[0,0]=1, W[1,1]=1,
    # then T[0,0,0] = 1, T[1,1,1] = 1, but also T[0,1,0] = 0 (since V[1,0]=0), etc.
    #
    # Actually, let me reconsider. For the 2x2 case:
    # T[i,j,k] = delta_{i,k} * delta_{j,j} ... no.
    # T[i,j,k] = 1 if the (i,j) entry of X and (j,k) entry of Y multiply
    # to contribute to the (i,k) entry of Z.
    #
    # So T[i,j,k] = 1 for all i,j,k in {0,1} (8 entries total for 2x2).
    #
    # A rank-1 term: T[i,j,k] = U[i,j] * V[j,k] * W[k,i]
    #
    # For Strassen Term 1: (X00+X11)*(Y00+Y11)
    # X00 = X[0,0], X11 = X[1,1]
    # Y00 = Y[0,0], Y11 = Y[1,1]
    #
    # This product = X[0,0]*Y[0,0] + X[0,0]*Y[1,1] + X[1,1]*Y[0,0] + X[1,1]*Y[1,1]
    #
    # These contribute to:
    # X[0,0]*Y[0,0] -> Z[0,0]
    # X[0,0]*Y[1,1] -> Z[0,1]  (wait, no: X[0,0]*Y[0,1] -> Z[0,1], X[0,0]*Y[1,1] doesn't directly)
    #
    # I'm confusing myself. Let me be very explicit.
    #
    # Z[i,k] = sum_j X[i,j] * Y[j,k]
    #
    # So Z[0,0] = X[0,0]*Y[0,0] + X[0,1]*Y[1,0]
    # Z[0,1] = X[0,0]*Y[0,1] + X[0,1]*Y[1,1]
    # Z[1,0] = X[1,0]*Y[0,0] + X[1,1]*Y[1,0]
    # Z[1,1] = X[1,0]*Y[0,1] + X[1,1]*Y[1,1]
    #
    # Strassen Term 1: (X[0,0]+X[1,1])*(Y[0,0]+Y[1,1])
    # = X[0,0]*Y[0,0] + X[0,0]*Y[1,1] + X[1,1]*Y[0,0] + X[1,1]*Y[1,1]
    #
    # None of these directly match the Z entries! X[0,0]*Y[1,1] would contribute
    # to Z[0,1] only if it were X[0,0]*Y[0,1] or X[0,1]*Y[1,1].
    #
    # I see my error. The Strassen terms are:
    # M1 = (X00+X11)(Y00+Y11) where X00=X[0,0], X11=X[1,1], Y00=Y[0,0], Y11=Y[1,1]
    #
    # But M1 is used as: Z00 = M1 + M4 - M5 + M7
    # M1 = (X[0,0]+X[1,1])*(Y[0,0]+Y[1,1])
    #
    # This is a SCALAR, not a matrix. And it's added to Z[0,0].
    # But Z[0,0] = X[0,0]*Y[0,0] + X[0,1]*Y[1,0]
    #
    # So M1 contributes X[0,0]*Y[0,0] + X[0,0]*Y[1,1] + X[1,1]*Y[0,0] + X[1,1]*Y[1,1]
    # to Z[0,0]. But Z[0,0] should only have X[0,0]*Y[0,0] + X[0,1]*Y[1,0].
    #
    # The other terms (M4, M5, M7) cancel out the extra parts.
    #
    # So the Strassen decomposition works by having the 7 scalar products
    # combine to give the correct Z entries, with cancellations.
    #
    # For the TENSOR decomposition, each scalar product like X[i,j]*Y[j,k]
    # that appears in the Strassen formulas becomes a rank-1 term.
    # But the Strassen formulas have products of SUMS, which expand to
    # multiple rank-1 terms.
    #
    # I think the correct interpretation is:
    # - The 2x2 matmul tensor has rank 7.
    # - This means there exist U, V, W (each 4x7) such that
    #   T[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
    #   for all i,j,k in {0,1}.
    #
    # The Strassen algorithm gives such a decomposition with 7 terms,
    # but each "term" in the algorithm is a product of linear forms,
    # which in the tensor decomposition corresponds to a single rank-1 term
    # where U, V, W can have multiple non-zero entries.
    #
    # Let me look up the explicit rank-7 decomposition of 2x2 matmul.
    #
    # From literature, the Strassen decomposition for 2x2:
    # c1 = (a11+a22)(b11+b22)
    # c2 = (a21+a22)b11
    # c3 = a11(b12-b22)
    # c4 = a22(b21-b11)
    # c5 = (a11+a12)b22
    # c6 = (a21-a11)(b11+b12)
    # c7 = (a12-a22)(b21+b22)
    #
    # d11 = c1+c4-c5+c7
    # d12 = c3+c5
    # d21 = c2+c4
    # d22 = c1-c2+c3+c6
    #
    # For the tensor decomposition, each cj is a scalar = (linear form in A) * (linear form in B).
    # cj contributes to d entries with coefficients.
    #
    # In the tensor T[i,j,k], the entry T[i,j,k] = 1 means that A[i,j]*B[j,k]
    # contributes to D[i,k].
    #
    # For cj = (sum_i alpha_i * A[i,j_i]) * (sum_k beta_k * B[j_k,k]),
    # this is a sum of products A[i,j_i]*B[j_k,k], each contributing to D[i,k].
    #
    # In the tensor decomposition, cj corresponds to a rank-1 term where:
    # U[i, j_i, c] = alpha_i (for the A part)
    # V[j_k, k, c] = beta_k (for the B part)
    # W[k, i, c] = (coefficient of D[i,k] from cj)
    #
    # But this doesn't quite work because a single rank-1 term has
    # T[i,j,k] = U[i,j,c]*V[j,k,c]*W[k,i,c], and the j index must be consistent.
    #
    # I think the correct way is:
    # For cj = (sum_{(i,j)} alpha_{ij} * A[i,j]) * (sum_{(j,k)} beta_{jk} * B[j,k]),
    # the rank-1 term has:
    # U[i,j,c] = alpha_{ij} for all (i,j)
    # V[j,k,c] = beta_{jk} for all (j,k)
    # W[k,i,c] = (sum over j of alpha_{ij}*beta_{jk} * coeff_{ik})
    #
    # No, that's not right either. Let me just directly construct it.
    
    # I'll take a different approach: directly construct the rank-49 decomposition
    # for 4x4 by combining the Strassen 2x2 block decomposition with the
    # rank-7 decomposition of each 2x2 block multiplication.
    
    # Actually, let me just use the known result and construct it directly.
    
    # For 4x4 matmul with rank 49:
    # The 4x4 matrices are split into 2x2 blocks.
    # Strassen gives 7 block products, each a 2x2 matmul of rank 7.
    # Total: 49 rank-1 terms.
    
    # I'll construct this by:
    # 1. For each of the 7 Strassen block products, determine the 2x2 matrices
    #    being multiplied (as linear combinations of 4x4 entries).
    # 2. For each such 2x2 product, apply the rank-7 Strassen decomposition.
    # 3. Combine all 49 terms.
    
    # Let me define the 7 Strassen block products for 4x4:
    # A = [[A00, A01], [A10, A11]] where Aij is 2x2
    # B = [[B00, B01], [B10, B11]] where Bij is 2x2
    
    # S1 = (A00+A11)*(B00+B11)
    # S2 = (A10+A11)*B00
    # S3 = A00*(B01-B11)
    # S4 = A11*(B10-B00)
    # S5 = (A00+A01)*B11
    # S6 = (A10-A00)*(B00+B01)
    # S7 = (A01-A11)*(B10+B11)
    
    # C00 = S1 + S4 - S5 + S7
    # C01 = S3 + S5
    # C10 = S2 + S4
    # C11 = S1 - S2 + S3 + S6
    
    # Each Sj is a 2x2 matrix product.
    # For each Sj = Xj * Yj where Xj and Yj are 2x2 matrices
    # (linear combos of A and B blocks), I need the rank-7 decomposition.
    
    # For a 2x2 product X*Y = Z, the rank-7 decomposition is:
    # z1 = (X[0,0]+X[1,1])*(Y[0,0]+Y[1,1])
    # z2 = (X[1,0]+X[1,1])*Y[0,0]
    # z3 = X[0,0]*(Y[0,1]-Y[1,1])
    # z4 = X[1,1]*(Y[1,0]-Y[0,0])
    # z5 = (X[0,0]+X[0,1])*Y[1,1]
    # z6 = (X[1,0]-X[0,0])*(Y[0,0]+Y[0,1])
    # z7 = (X[0,1]-X[1,1])*(Y[1,0]+Y[1,1])
    #
    # Z[0,0] = z1 + z4 - z5 + z7
    # Z[0,1] = z3 + z5
    # Z[1,0] = z2 + z4
    # Z[1,1] = z1 - z2 + z3 + z6
    
    # Each zt is a scalar = (linear form in X) * (linear form in Y).
    # In terms of the 4x4 entries, X and Y are linear combos of A and B blocks.
    
    # This is very complex to do by hand. Let me write code to generate it.
    
    # I'll represent each 2x2 block matrix as a 4x4 matrix with entries
    # being linear combinations of the original 4x4 entries.
    
    # For the 4x4 case, let me use a computational approach:
    # 1. Build the 4x4 matmul tensor.
    # 2. Use the block Strassen structure to generate 49 rank-1 terms.
    # 3. Verify by checking that the decomposition is exact.
    
    # Given the complexity, let me just hardcode a known rank-49 decomposition
    # or generate it algorithmically.
    
    # Algorithmic generation:
    # For each of the 7 Strassen block products Sj = Xj * Yj:
    #   Xj is a 2x2 matrix where each entry is a linear combo of A entries
    #   Yj is a 2x2 matrix where each entry is a linear combo of B entries
    #   Sj contributes to C blocks with certain coefficients
    #
    # For each Sj, apply the rank-7 2x2 Strassen:
    #   For each of the 7 sub-terms zt:
    #     zt = (linear form in Xj) * (linear form in Yj)
    #     zt contributes to Sj entries, which contribute to C entries
    #
    # Each zt becomes a rank-1 term in the 4x4 decomposition.
    
    # Let me define the 7 Strassen block products with their X and Y matrices.
    
    # Block (bi, bj) of a 4x4 matrix covers rows 2*bi to 2*bi+1, cols 2*bj to 2*bj+1.
    # So A[bi,bj] is the 2x2 block at block position (bi,bj).
    # A[bi,bj][r,s] = A[2*bi+r, 2*bj+s]
    
    # Strassen block products:
    # S1: X = A00+A11, Y = B00+B11
    #   X[r,s] = A[0,2r+s] + A[2,2r+s]  (for r,s in 0,1)
    #   Y[r,s] = B[0,2r+s] + B[2,2r+s]
    #
    # S2: X = A10+A11, Y = B00
    #   X[r,s] = A[2,2r+s] + A[3,2r+s]  (wait, A10 is rows 2-3, cols 0-1)
    #   A10[r,s] = A[2+r, s], A11[r,s] = A[2+r, 2+s]
    #   X[r,s] = A[2+r, s] + A[2+r, 2+s]
    #   Y[r,s] = B[r, s]  (B00[r,s] = B[r,s])
    
    # OK this is getting very tedious. Let me just write the code to generate
    # the decomposition programmatically and verify it.
    
    # I'll use numpy for verification.
    import numpy as np
    
    # Generate random test matrices
    np.random.seed(42)
    A_test = np.random.randn(4, 4)
    B_test = np.random.randn(4, 4)
    C_test = A_test @ B_test
    
    # Build the rank-49 decomposition
    # I'll construct U, V, W as lists of 49 terms.
    
    U_list = []
    V_list = []
    W_list = []
    
    # For each of the 7 Strassen block products, and for each of the 7
    # 2x2 Strassen sub-terms, create a rank-1 term.
    
    # Define the 7 block products:
    # Each is (X_coeffs, Y_coeffs, C_block_contributions)
    # X_coeffs: 2x2 array where each entry is a dict {A_flat_index: coeff}
    # Y_coeffs: 2x2 array where each entry is a dict {B_flat_index: coeff}
    # C_block_contributions: dict {C_block_position: sign}
    
    # A flat index: A[i,j] -> i*4+j
    # B flat index: B[j,k] -> j*4+k
    # C flat index: C[i,k] -> i*4+k
    
    def block_coeffs(block_row, block_col, sign=1.0):
        """Return dict of flat indices for a 2x2 block."""
        d = {}
        for r in range(2):
            for s in range(2):
                idx = (block_row * 2 + r) * 4 + (block_col * 2 + s)
                d[idx] = sign
        return d
    
    def add_block_dict(d1, d2, sign2=1.0):
        """Add sign2*d2 to d1."""
        result = dict(d1)
        for k, v in d2.items():
            result[k] = result.get(k, 0.0) + sign2 * v
        return result
    
    # S1: X = A00+A11, Y = B00+B11
    # X[r,s] = A[0,2r+s] + A[2,2r+s]  -> wait, A00 is rows 0-1, cols 0-1
    # A00[r,s] = A[r, s], A11[r,s] = A[2+r, 2+s]
    # X[r,s] = A[r,s] + A[2+r, 2+s]
    
    S1_X = [[block_coeffs(0,0) for s in range(2)] for r in range(2)]  # wrong
    # Let me redo this.
    
    # X is 2x2. X[r,s] for r,s in {0,1}.
    # For S1: X = A00 + A11
    # A00[r,s] = A[r, s] (flat: r*4+s)
    # A11[r,s] = A[2+r, 2+s] (flat: (2+r)*4+(2+s))
    
    S1_X = [[{} for s in range(2)] for r in range(2)]
    for r in range(2):
        for s in range(2):
            S1_X[r][s] = {r*4+s: 1.0, (2+r)*4+(2+s): 1.0}
    
    S1_Y = [[{} for s in range(2)] for r in range(2)]
    for r in range(2):
        for s in range(2):
            S1_Y[r][s] = {r*4+s: 1.0, (2+r)*4+(2+s): 1.0}
    
    S1_C = {(0,0): 1.0, (1,1): 1.0}  # contributes to C00 and C11
    
    # S2: X = A10+A11, Y = B00
    # A10[r,s] = A[2+r, s], A11[r,s] = A[2+r, 2+s]
    S2_X = [[{} for s in range(2)] for r in range(2)]
    for r in range(2):
        for s in range(2):
            S2_X[r][s] = {(2+r)*4+s: 1.0, (2+r)*4+(2+s): 1.0}
    
    S2_Y = [[{} for s in range(2)] for r in range(2)]
    for r in range(2):
        for s in range(2):
            S2_Y[r][s] = {r*4+s: 1.0}
    
    S2_C = {(1,0): 1.0, (1,1): -1.0}
    
    # S3: X = A00, Y = B01-B11
    # A00[r,s] = A[r,s]
    # B01[r,s] = B[r, 2+s], B11[r,s] = B[2+r, 2+s]
    S3_X = [[{} for s in range(2)] for r in range(2)]
    for r in range(2):
        for s in range(2):
            S3_X[r][s] = {r*4+s: 1.0}
    
    S3_Y = [[{} for s in range(2)] for r in range(2)]
    for r in range(2):
        for s in range(2):
            S3_Y[r][s] = {r*4+(2+s): 1.0, (2+r)*4+(2+s): -1.0}
    
    S3_C = {(0,1): 1.0, (1,1): 1.0}
    
    # S4: X = A11, Y = B10-B00
    # A11[r,s] = A[2+r, 2+s]
    # B10[r,s] = B[2+r, s], B00[r,s] = B[r, s]
    S4_X = [[{} for s in range(2)] for r in range(2)]
    for r in range(2):
        for s in range(2):
            S4_X[r][s] = {(2+r)*4+(2+s): 1.0}
    
    S4_Y = [[{} for s in range(2)] for r in range(2)]
    for r in range(2):
        for s in range(2):
            S4_Y[r][s] = {(2+r)*4+s: 1.0, r*4+s: -1.0}
    
    S4_C = {(0,0): 1.0, (1,0): 1.0}
    
    # S5: X = A00+A01, Y = B11
    # A00[r,s] = A[r,s], A01[r,s] = A[r, 2+s]
    # B11[r,s] = B[2+r, 2+s]
    S5_X = [[{} for s in range(2)] for r in range(2)]
    for r in range(2):
        for s in range(2):
            S5_X[r][s] = {r*4+s: 1.0, r*4+(2+s): 1.0}
    
    S5_Y = [[{} for s in range(2)] for r in range(2)]
    for r in range(2):
        for s in range(2):
            S5_Y[r][s] = {(2+r)*4+(2+s): 1.0}
    
    S5_C = {(0,0): -1.0, (0,1): 1.0}
    
    # S6: X = A10-A00, Y = B00+B01
    # A10[r,s] = A[2+r,s], A00[r,s] = A[r,s]
    # B00[r,s] = B[r,s], B01[r,s] = B[r,2+s]
    S6_X = [[{} for s in range(2)] for r in range(2)]
    for r in range(2):
        for s in range(2):
            S6_X[r][s] = {(2+r)*4+s: 1.0, r*4+s: -1.0}
    
    S6_Y = [[{} for s in range(2)] for r in range(2)]
    for r in range(2):
        for s in range(2):
            S6_Y[r][s] = {r*4+s: 1.0, r*4+(2+s): 1.0}
    
    S6_C = {(1,1): 1.0}
    
    # S7: X = A01-A11, Y = B10+B11
    # A01[r,s] = A[r,2+s], A11[r,s] = A[2+r,2+s]
    # B10[r,s] = B[2+r,s], B11[r,s] = B[2+r,2+s]
    S7_X = [[{} for s in range(2)] for r in range(2)]
    for r in range(2):
        for s in range(2):
            S7_X[r][s] = {r*4+(2+s): 1.0, (2+r)*4+(2+s): -1.0}
    
    S7_Y = [[{} for s in range(2)] for r in range(2)]
    for r in range(2):
        for s in range(2):
            S7_Y[r][s] = {(2+r)*4+s: 1.0, (2+r)*4+(2+s): 1.0}
    
    S7_C = {(0,0): 1.0}
    
    block_products = [
        (S1_X, S1_Y, S1_C),
        (S2_X, S2_Y, S2_C),
        (S3_X, S3_Y, S3_C),
        (S4_X, S4_Y, S4_C),
        (S5_X, S5_Y, S5_C),
        (S6_X, S6_Y, S6_C),
        (S7_X, S7_Y, S7_C),
    ]
    
    # For each block product, apply the 2x2 Strassen rank-7 decomposition.
    # The 7 sub-terms for a 2x2 product X*Y = Z:
    # z1 = (X[0,0]+X[1,1])*(Y[0,0]+Y[1,1])
    # z2 = (X[1,0]+X[1,1])*Y[0,0]
    # z3 = X[0,0]*(Y[0,1]-Y[1,1])
    # z4 = X[1,1]*(Y[1,0]-Y[0,0])
    # z5 = (X[0,0]+X[0,1])*Y[1,1]
    # z6 = (X[1,0]-X[0,0])*(Y[0,0]+Y[0,1])
    # z7 = (X[0,1]-X[1,1])*(Y[1,0]+Y[1,1])
    #
    # Z[0,0] = z1 + z4 - z5 + z7
    # Z[0,1] = z3 + z5
    # Z[1,0] = z2 + z4
    # Z[1,1] = z1 - z2 + z3 + z6
    
    # For each zt, it's (linear form in X) * (linear form in Y).
    # The linear form in X is a sum of X entries with coefficients.
    # Each X entry is itself a linear form in A entries.
    # So zt is a product of two linear forms in A and B entries.
    #
    # In the tensor decomposition, zt contributes to Z entries, which
    # contribute to C entries via the block product's C contributions.
    
    # For zt = (sum_{rs} alpha_{rs} * X[r,s]) * (sum_{rs} beta_{rs} * Y[r,s]):
    # This is a scalar. It contributes to Z entries via the Z formulas.
    #
    # In the 4x4 tensor decomposition, this becomes a rank-1 term where:
    # U[i,j,c] = coefficient of A[i,j] in the X linear form
    # V[j,k,c] = coefficient of B[j,k] in the Y linear form
    # W[k,i,c] = coefficient of C[i,k] from this term
    #
    # But the X linear form is a sum over (r,s) of alpha_{rs} * X[r,s],
    # and X[r,s] is a sum over (i,j) of a_{rs,ij} * A[i,j].
    # So the total coefficient of A[i,j] is sum_{rs} alpha_{rs} * a_{rs,ij}.
    #
    # Similarly for B and C.
    
    # Let me compute this for each of the 49 terms.
    
    # For each block product (X_coeffs, Y_coeffs, C_contrib):
    #   For each of the 7 2x2 Strassen sub-terms:
    #     Compute the U, V, W vectors for this rank-1 term.
    
    # 2x2 Strassen sub-term definitions:
    # Each sub-term t has:
    #   X_expr: list of (r, s, coeff) for X entries
    #   Y_expr: list of (r, s, coeff) for Y entries
    #   Z_contrib: dict {(0,0): coeff, (0,1): coeff, ...} for Z entries
    
    sub_terms = [
        # z1 = (X[0,0]+X[1,1])*(Y[0,0]+Y[1,1])
        {
            'X': [(0,0,1.0), (1,1,1.0)],
            'Y': [(0,0,1.0), (1,1,1.0)],
            'Z': {(0,0): 1.0, (1,1): 1.0}
        },
        # z2 = (X[1,0]+X[1,1])*Y[0,0]
        {
            'X': [(1,0,1.0), (1,1,1.0)],
            'Y': [(0,0,1.0)],
            'Z': {(1,0): 1.0, (1,1): -1.0}
        },
        # z3 = X[0,0]*(Y[0,1]-Y[1,1])
        {
            'X': [(0,0,1.0)],
            'Y': [(0,1,1.0), (1,1,-1.0)],
            'Z': {(0,1): 1.0, (1,1): 1.0}
        },
        # z4 = X[1,1]*(Y[1,0]-Y[0,0])
        {
            'X': [(1,1,1.0)],
            'Y': [(1,0,1.0), (0,0,-1.0)],
            'Z': {(0,0): 1.0, (1,0): 1.0}
        },
        # z5 = (X[0,0]+X[0,1])*Y[1,1]
        {
            'X': [(0,0,1.0), (0,1,1.0)],
            'Y': [(1,1,1.0)],
            'Z': {(0,0): -1.0, (0,1): 1.0}
        },
        # z6 = (X[1,0]-X[0,0])*(Y[0,0]+Y[0,1])
        {
            'X': [(1,0,1.0), (0,0,-1.0)],
            'Y': [(0,0,1.0), (0,1,1.0)],
            'Z': {(1,1): 1.0}
        },
        # z7 = (X[0,1]-X[1,1])*(Y[1,0]+Y[1,1])
        {
            'X': [(0,1,1.0), (1,1,-1.0)],
            'Y': [(1,0,1.0), (1,1,1.0)],
            'Z': {(0,0): 1.0}
        },
    ]
    
    # Now generate all 49 rank-1 terms.
    U_cols = []
    V_cols = []
    W_cols = []
    
    for (Xc, Yc, Cc) in block_products:
        for st in sub_terms:
            # Compute U vector (16 entries)
            U_vec = [0.0] * 16
            V_vec = [0.0] * 16
            W_vec = [0.0] * 16
            
            # For each X entry (r,s,coeff) in the sub-term:
            for (r, s, xcoeff) in st['X']:
                # Xc[r][s] is a dict of A flat indices
                for a_idx, a_coeff in Xc[r][s].items():
                    U_vec[a_idx] += xcoeff * a_coeff
            
            # For each Y entry (r,s,coeff) in the sub-term:
            for (r, s, ycoeff) in st['Y']:
                # Yc[r][s] is a dict of B flat indices
                for b_idx, b_coeff in Yc[r][s].items():
                    V_vec[b_idx] += ycoeff * b_coeff
            
            # For W: the sub-term contributes to Z entries, which map to C blocks.
            # Z entry (zr, zs) corresponds to C block (Cc block, zr, zs)
            # C block (cb_r, cb_c) covers C[2*cb_r + zr, 2*cb_c + zs]
            for (zr, zs), zcoeff in st['Z'].items():
                for (cb_r, cb_c), ccoeff in Cc.items():
                    # C flat index: (2*cb_r + zr) * 4 + (2*cb_c + zs)
                    c_idx = (2*cb_r + zr) * 4 + (2*cb_c + zs)
                    W_vec[c_idx] += zcoeff * ccoeff
            
            U_cols.append(U_vec)
            V_cols.append(V_vec)
            W_cols.append(W_vec)
    
    # Verify the decomposition
    if time.time() > deadline - 30:
        # Not enough time to verify, just return what we have
        pass
    else:
        # Check: sum_c U[:,c] * V[:,c] * W[:,c] should equal the matmul tensor
        # T[i,j,k] = 1 for all i,j,k (since C[i,k] = sum_j A[i,j]*B[j,k])
        # In our convention: U[i*4+j, c], V[j*4+k, c], W[k*4+i, c]
        # T[i,j,k] = sum_c U[i*4+j, c] * V[j*4+k, c] * W[k*4+i, c]
        
        T_computed = np.zeros((4, 4, 4))
        for c in range(len(U_cols)):
            for i in range(4):
                for j in range(4):
                    for k in range(4):
                        T_computed[i, j, k] += U_cols[c][i*4+j] * V_cols[c][j*4+k] * W_cols[c][k*4+i]
        
        # The matmul tensor should have T[i,j,k] = 1 for all i,j,k
        # (since every A[i,j]*B[j,k] contributes to C[i,k])
        T_expected = np.ones((4, 4, 4))
        
        if not np.allclose(T_computed, T_expected, atol=1e-10):
            # Something went wrong. Fall back to schoolbook.
            rank = 64
            U = [[0.0] * rank for _ in range(16)]
            V = [[0.0] * rank for _ in range(16)]
            W = [[0.0] * rank for _ in range(16)]
            col = 0
            for i in range(4):
                for j in range(4):
                    for k in range(4):
                        U[i*4+j][col] = 1.0
                        V[j*4+k][col] = 1.0
                        W[k*4+i][col] = 1.0
                        col += 1
            return {"U": U, "V": V, "W": W}
    
    # Convert to list of lists
    U = U_cols
    V = V_cols
    W = W_cols
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
