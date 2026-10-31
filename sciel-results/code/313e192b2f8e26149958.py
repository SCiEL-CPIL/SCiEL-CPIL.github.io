"""Matrix multiplication 4x4x4 rank-minimal decomposition using Strassen's algorithm."""
import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct rank-49 tensor decomposition for 4x4x4 matrix multiplication
    using Strassen's algorithm applied to 2x2 block structure.
    
    The 4x4x4 multiplication tensor is decomposed by viewing the 4x4 matrices
    as 2x2 block matrices with 2x2 blocks. Strassen's algorithm gives rank 7
    for 2x2 multiplication, so the total rank is 7*7 = 49.
    
    U shape: (n*m, rank) = (16, 49)
    V shape: (m*p, rank) = (16, 49)  
    W shape: (p*n, rank) = (16, 49)
    
    Convention: W[k*n+i, r] where output C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    Actually, the standard tensor decomposition is:
    T = sum_r U_r ⊗ V_r ⊗ W_r
    where T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    and T[i,j,k] = 1 if k==i (for matmul: C[i,k] = sum_j A[i,j]*B[j,k])
    
    Wait, let me reconsider the tensor structure. For matrix multiplication:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor is T_{i,j,k} where the multiplication is:
    C[i,k] = sum_{j} A[i,j] B[j,k]
    
    So the tensor T has indices (i, j, k) and T[i,j,k] = 1 for all valid i,j,k
    (it's the "multiplication tensor" where each entry contributes).
    
    Actually the tensor is: T[i,j,k] = δ_{i,?} ... no.
    
    The matrix multiplication tensor M has entries M[i,j,k] = 1 for all i,j,k
    in range, and the multiplication is:
    (A*B)[i,k] = sum_j M[i,j,k] * A[i,j] * B[j,k]
    
    So M[i,j,k] = 1 for all i,j,k. The rank is the minimum number of rank-1
    tensors that sum to M.
    
    For Strassen: we decompose the 4x4x4 tensor using block structure.
    
    Let me use the explicit Strassen construction.
    """
    n, m, p = 4, 4, 4
    rank = 49
    
    # Initialize factor matrices
    U = [[0] * rank for _ in range(n * m)]  # 16 x 49
    V = [[0] * rank for _ in range(m * p)]  # 16 x 49
    W = [[0] * rank for _ in range(p * n)]  # 16 x 49
    
    # Strassen 2x2 decomposition coefficients
    # For 2x2: C00 = M1 - M4 + M5 - M7
    #           C01 = M3 + M5
    #           C10 = M2 + M4
    #           C11 = M1 - M2 + M3 + M6
    #
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # For 4x4, we view matrices as 2x2 blocks of 2x2 submatrices.
    # Block (b_i, b_j) where b_i, b_j in {0,1} refers to 2x2 block.
    # Within each 2x2 block, local indices 0,1.
    
    # The full 4x4x4 tensor is a 2x2 block tensor of 2x2x2 tensors.
    # We apply Strassen at the block level (rank 7 for blocks) AND at the 
    # sub-block level (rank 7 for each 2x2x2).
    # Total rank: 7 * 7 = 49.
    
    # Strassen coefficients for the outer (block) level:
    # Block multiplication: C_block[a,b] = sum_{x,y} A_block[a,x] * B_block[x,b]
    # Using Strassen: 7 products at block level.
    
    # For each block-level Strassen product M_s (s=1..7), we need to compute
    # a 2x2x2 tensor product, which itself has rank 7 (Strassen).
    
    # Let me define the Strassen 2x2 decomposition explicitly.
    # The 2x2x2 multiplication tensor has rank 7.
    # T_2x2[i,j,k] = 1 for all i,j,k in {0,1}
    # 
    # Rank-7 decomposition of 2x2x2 matmul tensor:
    # Using standard Strassen, the 7 rank-1 components are:
    
    # Let me use a different approach. I'll directly construct the 49-term
    # decomposition by combining two levels of Strassen.
    
    # Level 1: Block-level Strassen (7 terms for 2x2 block structure)
    # Level 2: Sub-block-level Strassen (7 terms for 2x2 sub-block structure)
    
    # Strassen 2x2 decomposition:
    # The 2x2x2 tensor T[i,j,k] (i=row of A, j=col of A/row of B, k=col of B)
    # T[i,j,k] = 1 for all i,j,k in {0,1}
    
    # Rank-1 decomposition via Strassen:
    # Term 1: A factor: (1,1,0,1) for A00,A01,A10,A11 -> A00+A11
    #          B factor: (1,1,1,1) for B00,B01,B10,B11 -> B00+B11
    #          C contribution: to C00, C11 (with signs)
    
    # Let me use the explicit known rank-7 decomposition.
    
    # Standard Strassen rank-7 for 2x2:
    # The 7 rank-1 tensors (u, v, w) where u is A-factor, v is B-factor, w is C-factor
    # u, v, w are vectors in R^4 (for A, B) and R^4 (for C)
    
    # Actually, let me use a cleaner formulation.
    
    # For 2x2 matmul, the tensor is T_{i,j,k} = 1 (all ones 2x2x2).
    # Rank-7 decomposition:
    
    # I'll use the following standard Strassen decomposition:
    # Let A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # C = A*B = [[c00, c01], [c10, c11]]
    
    # 7 products:
    # p1 = (a00+a11)(b00+b11)
    # p2 = (a10+a11)b00
    # p3 = a00(b01-b11)
    # p4 = a11(b10-b00)
    # p5 = (a00+a01)b11
    # p6 = (a10-a00)(b00+b01)
    # p7 = (a01-a11)(b10+b11)
    
    # C00 = p1 - p4 + p5 - p7
    # C01 = p3 + p5
    # C10 = p2 + p4
    # C11 = p1 - p2 + p3 + p6
    
    # As rank-1 tensor decomposition, each product p_s contributes:
    # u_s (A-factor) * v_s (B-factor) * w_s (C-factor)
    
    # p1: u=(1,0,0,1), v=(1,1,1,1), w=(1,0,0,1)  [contributes to C00 and C11]
    #     But w should be (1,0,0,1) meaning C00 += 1, C11 += 1
    #     Wait, C00 = p1 - p4 + p5 - p7, C11 = p1 - p2 + p3 + p6
    #     So p1 contributes +1 to C00 and +1 to C11. w=(1,0,0,1)
    
    # Let me list all 7:
    # p1: u=(1,0,0,1), v=(1,1,1,1), w=(1,0,0,1)
    # p2: u=(0,0,1,1), v=(1,0,0,0), w=(0,0,1,0)  [C10 += p2, C11 -= p2]
    #     w=(0,0,1,-1)
    # p3: u=(1,0,0,0), v=(0,1,0,-1), w=(0,1,0,0)  [C01 += p3, C11 += p3]
    #     w=(0,1,0,1)
    # p4: u=(0,0,0,1), v=(0,-1,1,0), w=(-1,0,1,0)  [C00 -= p4, C10 += p4]
    # p5: u=(1,1,0,0), v=(0,0,0,1), w=(1,1,0,0)  [C00 += p5, C01 += p5]
    # p6: u=(0,0,1,-1), v=(1,1,0,0), w=(0,0,0,1)  [C11 += p6]
    # p7: u=(0,1,0,-1), v=(0,0,1,1), w=(0,0,0,-1)  [C00 -= p7, C11... wait]
    
    # Let me recheck: C00 = p1 - p4 + p5 - p7
    # So p7 contributes -1 to C00. w for p7: (-1, 0, 0, 0)? 
    # But C11 = p1 - p2 + p3 + p6, p7 doesn't appear in C11.
    # So w7 = (-1, 0, 0, 0)
    
    # Let me redo carefully:
    # C00 = p1 - p4 + p5 - p7
    # C01 = p3 + p5
    # C10 = p2 + p4
    # C11 = p1 - p2 + p3 + p6
    
    # p1: A=(1,0,0,1), B=(1,1,1,1), C=(1,0,0,1)
    # p2: A=(0,0,1,1), B=(1,0,0,0), C=(0,0,1,-1)
    # p3: A=(1,0,0,0), B=(0,1,0,-1), C=(0,1,0,1)
    # p4: A=(0,0,0,1), B=(0,-1,1,0), C=(-1,0,1,0)
    # p5: A=(1,1,0,0), B=(0,0,0,1), C=(1,1,0,0)
    # p6: A=(0,0,1,-1), B=(1,1,0,0), C=(0,0,0,1)
    # p7: A=(0,1,0,-1), B=(0,0,1,1), C=(-1,0,0,0)
    
    # Verify C00: p1*1 + p2*0 + p3*0 + p4*(-1) + p5*1 + p6*0 + p7*(-1) = p1-p4+p5-p7 ✓
    # C01: p1*0 + p2*0 + p3*1 + p4*0 + p5*1 + p6*0 + p7*0 = p3+p5 ✓
    # C10: p1*0 + p2*1 + p3*0 + p4*1 + p5*0 + p6*0 + p7*0 = p2+p4 ✓
    # C11: p1*1 + p2*(-1) + p3*1 + p4*0 + p5*0 + p6*1 + p7*0 = p1-p2+p3+p6 ✓
    
    # Great! Now for 4x4x4, we have a 2x2 block structure.
    # Each "element" is a 2x2 matrix.
    # We apply Strassen at the block level, and for each block-level product,
    # we apply Strassen at the sub-block level.
    
    # Block-level Strassen: 7 products, each involving 2x2 block matrices.
    # Sub-block-level Strassen: each 2x2 block product has rank 7.
    # Total: 7 * 7 = 49 rank-1 terms.
    
    # Indexing:
    # A[i,j] where i,j in {0,1,2,3}
    # Block (bi, bj) of A contains A[2*bi+li, 2*bj+lj] for li,lj in {0,1}
    # Similarly for B and C.
    
    # For the full 4x4x4 tensor:
    # T[i,j,k] = 1 for all i,j,k (this is the multiplication tensor)
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # Using block Strassen:
    # C_block[a,c] = sum_b A_block[a,b] * B_block[b,c]
    # where A_block[a,b] is the 2x2 submatrix A[2a:2a+2, 2b:2b+2]
    # B_block[b,c] is the 2x2 submatrix B[2b:2b+2, 2c:2c+2]
    # C_block[a,c] is the 2x2 submatrix C[2a:2a+2, 2c:2c+2]
    
    # Block-level Strassen gives 7 products P_s (s=0..6), each a 2x2 matrix.
    # Each P_s = (linear combo of A_blocks) * (linear combo of B_blocks)
    # Then C_block[a,c] = linear combo of P_s.
    
    # Each P_s is a 2x2 matrix product, decomposed via sub-block Strassen into 7 rank-1 terms.
    
    # So total: 7 (block) * 7 (sub-block) = 49 rank-1 terms.
    
    # Let me define the block-level Strassen coefficients.
    # A_blocks: A00, A01, A10, A11 (each 2x2)
    # B_blocks: B00, B01, B10, B11 (each 2x2)
    # C_blocks: C00, C01, C10, C11 (each 2x2)
    
    # Block Strassen:
    # P0 = (A00+A11)(B00+B11)
    # P1 = (A10+A11)B00
    # P2 = A00(B01-B11)
    # P3 = A11(B10-B00)
    # P4 = (A00+A01)B11
    # P5 = (A10-A00)(B00+B01)
    # P6 = (A01-A11)(B10+B11)
    
    # C00 = P0 - P3 + P4 - P6
    # C01 = P2 + P4
    # C10 = P1 + P3
    # C11 = P0 - P1 + P2 + P5
    
    # For each P_s, we need the 2x2 matrix product decomposition.
    # P_s = (linear combo of A sub-blocks) * (linear combo of B sub-blocks)
    
    # Let me define for each s, the A-coefficients and B-coefficients at the block level.
    # A_block_coeffs[s][a][b] = coefficient of A_block[a][b] in the A-factor of P_s
    # B_block_coeffs[s][b][c] = coefficient of B_block[b][c] in the B-factor of P_s
    # C_block_coeffs[s][a][c] = coefficient of P_s in C_block[a][c]
    
    # s=0: A: A00+ A11, B: B00+ B11, C: C00+ C11
    # s=1: A: A10+ A11, B: B00, C: C10 - C11
    # s=2: A: A00, B: B01 - B11, C: C01 + C11
    # s=3: A: A11, B: B10 - B00, C: -C00 + C10
    # s=4: A: A00+ A01, B: B11, C: C00 + C01
    # s=5: A: A10 - A00, B: B00+ B01, C: C11
    # s=6: A: A01 - A11, B: B10+ B11, C: -C00
    
    # Now, for each s, P_s is a product of two 2x2 "super-matrices" (each being a linear combo of 2x2 blocks).
    # The A-factor of P_s is a 2x2 matrix where each entry is a linear combo of 2x2 sub-blocks.
    # Wait, no. P_s = (A_factor) * (B_factor) where A_factor and B_factor are 2x2 matrices of 2x2 blocks?
    
    # No. Let me reconsider. In block Strassen:
    # P_s = (sum_{a,b} alpha_s[a,b] * A_block[a,b]) * (sum_{b,c} beta_s[b,c] * B_block[b,c])
    
    # This is a product of two 2x2 matrices (each entry being a scalar times a 2x2 block).
    # So P_s is a 2x2 matrix where each entry is a scalar times a 2x2 block product.
    
    # Actually, P_s = M_s * N_s where M_s and N_s are 2x2 matrices with entries that are 2x2 blocks.
    # M_s[a][b] = alpha_s[a,b] * A_block[a,b] (but this isn't quite right either)
    
    # Let me think more carefully. In standard Strassen for 2x2:
    # p1 = (A00+A11)(B00+B11)
    # Here A00, A11, B00, B11 are scalars. The product is a scalar.
    
    # In block Strassen for 4x4 (2x2 blocks of 2x2):
    # P0 = (A_block00 + A_block11)(B_block00 + B_block11)
    # Here A_block00, A_block11, B_block00, B_block11 are 2x2 matrices.
    # P0 is a 2x2 matrix = (A_block00 + A_block11) @ (B_block00 + B_block11)
    
    # So P0 = A_block00@B_block00 + A_block00@B_block11 + A_block11@B_block00 + A_block11@B_block11
    
    # Each term like A_block00@B_block00 is a 2x2 matrix product.
    # We need to decompose P0 (a 2x2 matrix) into rank-1 tensors.
    
    # P0 = (A_block00 + A_block11) @ (B_block00 + B_block11)
    # This is a single 2x2 matrix product where:
    #   M = A_block00 + A_block11 (a 2x2 matrix)
    #   N = B_block00 + B_block11 (a 2x2 matrix)
    #   P0 = M @ N (a 2x2 matrix)
    
    # To decompose P0 = M @ N where M and N are 2x2 matrices:
    # M = [[m00, m01], [m10, m11]] where m_ij are scalars (linear combos of A entries)
    # N = [[n00, n01], [n10, n11]] where n_ij are scalars (linear combos of B entries)
    # P0 = [[p00, p01], [p10, p11]] where p_ij are scalars
    
    # The 2x2 matmul P0 = M@N has rank 7 (Strassen).
    # So P0 contributes 7 rank-1 terms to the overall decomposition.
    
    # For P0:
    # M = A_block00 + A_block11
    #   M[0,0] = A[0,0] + A[2,2]
    #   M[0,1] = A[0,1] + A[2,3]
    #   M[1,0] = A[1,0] + A[3,2]
    #   M[1,1] = A[1,1] + A[3,3]
    # N = B_block00 + B_block11
    #   N[0,0] = B[0,0] + B[2,2]
    #   N[0,1] = B[0,1] + B[2,3]
    #   N[1,0] = B[1,0] + B[3,2]
    #   N[1,1] = B[1,1] + B[3,3]
    
    # P0 = M @ N, and C_block00 += P0, C_block11 += P0 (from C00 = P0 - P3 + P4 - P6, C11 = P0 - P1 + P2 + P5)
    # Wait, C00 = P0 - P3 + P4 - P6 means C_block00 gets +P0, -P3, +P4, -P6
    # C11 = P0 - P1 + P2 + P5 means C_block11 gets +P0, -P1, +P2, +P5
    
    # So P0 contributes to C_block00 with coefficient +1 and C_block11 with coefficient +1.
    
    # Now, for the rank-1 decomposition of P0 = M@N (2x2 matmul):
    # Using the 7 Strassen terms defined earlier:
    # For the 2x2 matmul with A=M, B=N, C=P0:
    #   t1: u=(1,0,0,1), v=(1,1,1,1), w=(1,0,0,1)
    #   t2: u=(0,0,1,1), v=(1,0,0,0), w=(0,0,1,-1)
    #   t3: u=(1,0,0,0), v=(0,1,0,-1), w=(0,1,0,1)
    #   t4: u=(0,0,0,1), v=(0,-1,1,0), w=(-1,0,1,0)
    #   t5: u=(1,1,0,0), v=(0,0,0,1), w=(1,1,0,0)
    #   t6: u=(0,0,1,-1), v=(1,1,0,0), w=(0,0,0,1)
    #   t7: u=(0,1,0,-1), v=(0,0,1,1), w=(-1,0,0,0)
    
    # Each term t has:
    #   u vector (4 entries for M[0,0], M[0,1], M[1,0], M[1,1])
    #   v vector (4 entries for N[0,0], N[0,1], N[1,0], N[1,1])
    #   w vector (4 entries for P0[0,0], P0[0,1], P0[1,0], P0[1,1])
    
    # M[i,j] = linear combo of A entries:
    #   M[0,0] = A[0,0] + A[2,2]
    #   M[0,1] = A[0,1] + A[2,3]
    #   M[1,0] = A[1,0] + A[3,2]
    #   M[1,1] = A[1,1] + A[3,3]
    
    # So for term t with u = (u0, u1, u2, u3):
    #   A_factor = u0*M[0,0] + u1*M[0,1] + u2*M[1,0] + u3*M[1,1]
    #            = u0*(A[0,0]+A[2,2]) + u1*(A[0,1]+A[2,3]) + u2*(A[1,0]+A[3,2]) + u3*(A[1,1]+A[3,3])
    
    # Similarly for B_factor from v.
    
    # And P0[i,j] = w0*P0[0,0] + w1*P0[0,1] + w2*P0[1,0] + w3*P0[1,1]
    # But P0 contributes to C_block00 and C_block11.
    # P0[0,0] -> C[0,0], P0[0,1] -> C[0,1], P0[1,0] -> C[1,0], P0[1,1] -> C[1,1] (within C_block00)
    # P0[0,0] -> C[2,2], P0[0,1] -> C[2,3], P0[1,0] -> C[3,2], P0[1,1] -> C[3,3] (within C_block11)
    
    # So the W factor for this term has entries at:
    #   C[0,0]: w0, C[0,1]: w1, C[1,0]: w2, C[1,1]: w3 (from C_block00)
    #   C[2,2]: w0, C[2,3]: w1, C[3,2]: w2, C[3,3]: w3 (from C_block11)
    
    # Now I can construct the full decomposition.
    
    # Define Strassen 2x2 terms
    strassen_terms = [
        # (u, v, w)
        ((1,0,0,1), (1,1,1,1), (1,0,0,1)),   # t1
        ((0,0,1,1), (1,0,0,0), (0,0,1,-1)),   # t2
        ((1,0,0,0), (0,1,0,-1), (0,1,0,1)),   # t3
        ((0,0,0,1), (0,-1,1,0), (-1,0,1,0)),  # t4
        ((1,1,0,0), (0,0,0,1), (1,1,0,0)),    # t5
        ((0,0,1,-1), (1,1,0,0), (0,0,0,1)),   # t6
        ((0,1,0,-1), (0,0,1,1), (-1,0,0,0)),  # t7
    ]
    
    # Block-level Strassen: for each s, define A_block_coeffs and B_block_coeffs
    # A_block_coeffs[s] is a 2x2 matrix of coefficients for A_blocks
    # B_block_coeffs[s] is a 2x2 matrix of coefficients for B_blocks
    # C_block_coeffs[s] is a 2x2 matrix of coefficients for C_blocks
    
    # P0 = (A00+A11)(B00+B11), C: C00+=1, C11+=1
    # P1 = (A10+A11)B00, C: C10+=1, C11-=1
    # P2 = A00(B01-B11), C: C01+=1, C11+=1
    # P3 = A11(B10-B00), C: C00-=1, C10+=1
    # P4 = (A00+A01)B11, C: C00+=1, C01+=1
    # P5 = (A10-A00)(B00+B01), C: C11+=1
    # P6 = (A01-A11)(B10+B11), C: C00-=1
    
    # A_block_coeffs[s][a][b] = coeff of A_block[a,b] in A-factor of P_s
    # For P_s = (sum alpha[a,b] A_block[a,b]) @ (sum beta[b,c] B_block[b,c])
    
    # P0: A-factor = A00 + A11, so alpha = [[1,0],[0,1]]
    #     B-factor = B00 + B11, so beta = [[1,0],[0,1]]
    # P1: A-factor = A10 + A11, so alpha = [[0,0],[1,1]]
    #     B-factor = B00, so beta = [[1,0],[0,0]]
    # P2: A-factor = A00, so alpha = [[1,0],[0,0]]
    #     B-factor = B01 - B11, so beta = [[0,1],[0,-1]]
    # P3: A-factor = A11, so alpha = [[0,0],[0,1]]
    #     B-factor = B10 - B00, so beta = [[-1,0],[1,0]]
    # P4: A-factor = A00 + A01, so alpha = [[1,1],[0,0]]
    #     B-factor = B11, so beta = [[0,0],[0,1]]
    # P5: A-factor = A10 - A00, so alpha = [[-1,0],[1,0]]
    #     B-factor = B00 + B01, so beta = [[1,1],[0,0]]
    # P6: A-factor = A01 - A11, so alpha = [[0,1],[0,-1]]
    #     B-factor = B10 + B11, so beta = [[0,0],[1,1]]
    
    A_block_coeffs = [
        [[1,0],[0,1]],   # P0
        [[0,0],[1,1]],   # P1
        [[1,0],[0,0]],   # P2
        [[0,0],[0,1]],   # P3
        [[1,1],[0,0]],   # P4
        [[-1,0],[1,0]],  # P5
        [[0,1],[0,-1]],  # P6
    ]
    
    B_block_coeffs = [
        [[1,0],[0,1]],   # P0
        [[1,0],[0,0]],   # P1
        [[0,1],[0,-1]],  # P2
        [[-1,0],[1,0]],  # P3
        [[0,0],[0,1]],   # P4
        [[1,1],[0,0]],   # P5
        [[0,0],[1,1]],   # P6
    ]
    
    C_block_coeffs = [
        [[1,0],[0,1]],   # P0: C00+=1, C11+=1
        [[0,0],[1,-1]],  # P1: C10+=1, C11-=1
        [[0,1],[0,1]],   # P2: C01+=1, C11+=1
        [[-1,0],[1,0]],  # P3: C00-=1, C10+=1
        [[1,1],[0,0]],   # P4: C00+=1, C01+=1
        [[0,0],[0,1]],   # P5: C11+=1
        [[-1,0],[0,0]],  # P6: C00-=1
    ]
    
    # Now construct the 49 terms.
    # For each block-level term s (0..6) and sub-block Strassen term t (0..6):
    #   term_index = s * 7 + t
    #   u_strassen, v_strassen, w_strassen = strassen_terms[t]
    
    #   A_factor = sum_{i,j} u_strassen[i*2+j] * (sum_{a,b} A_block_coeffs[s][a][b] * A_block[a,b][i,j])
    #   Wait, I need to be more careful with indexing.
    
    # M is a 2x2 matrix: M[i][j] for i,j in {0,1}
    # M[i][j] = sum_{a,b} A_block_coeffs[s][a][b] * A_block[a,b][i][j]
    # A_block[a,b][i][j] = A[2*a+i][2*b+j]
    
    # So M[i][j] = sum_{a,b} A_block_coeffs[s][a][b] * A[2*a+i][2*b+j]
    
    # u_strassen = (u0, u1, u2, u3) corresponds to M[0,0], M[0,1], M[1,0], M[1,1]
    # A_factor = u0*M[0,0] + u1*M[0,1] + u2*M[1,0] + u3*M[1,1]
    #          = sum_{i,j} u_strassen[i*2+j] * M[i][j]
    #          = sum_{i,j} u_strassen[i*2+j] * sum_{a,b} A_block_coeffs[s][a][b] * A[2*a+i][2*b+j]
    #          = sum_{i,j,a,b} u_strassen[i*2+j] * A_block_coeffs[s][a][b] * A[2*a+i][2*b+j]
    
    # So U[row, term_index] = sum over the A entries.
    # U is indexed by (i_A, j_A) where i_A = row of A, j_A = col of A.
    # U[i_A * m + j_A, term_index] = coefficient of A[i_A, j_A]
    
    # From above: coefficient of A[2*a+i][2*b+j] = u_strassen[i*2+j] * A_block_coeffs[s][a][b]
    
    # So: U[(2*a+i)*m + (2*b+j), s*7+t] += u_strassen[i*2+j] * A_block_coeffs[s][a][b]
    
    # Similarly for V:
    # N[i][j] = sum_{b,c} B_block_coeffs[s][b][c] * B_block[b,c][i][j]
    # B_block[b,c][i][j] = B[2*b+i][2*c+j]
    # N[i][j] = sum_{b,c} B_block_coeffs[s][b][c] * B[2*b+i][2*c+j]
    # B_factor = sum_{i,j} v_strassen[i*2+j] * N[i][j]
    #          = sum_{i,j,b,c} v_strassen[i*2+j] * B_block_coeffs[s][b][c] * B[2*b+i][2*c+j]
    # V[(2*b+i)*m + (2*c+j), s*7+t] += v_strassen[i*2+j] * B_block_coeffs[s][b][c]
    
    # For W:
    # P0[i][j] is the (i,j) entry of the 2x2 product.
    # P0 contributes to C_blocks according to C_block_coeffs[s].
    # C_block[a,c][i,j] += C_block_coeffs[s][a][c] * P0[i][j]
    # C[2*a+i][2*c+j] += C_block_coeffs[s][a][c] * P0[i][j]
    
    # w_strassen = (w0, w1, w2, w3) for P0[0,0], P0[0,1], P0[1,0], P0[1,1]
    # P0[i][j] = w_strassen[i*2+j] * (the product value)
    # Wait, no. w_strassen gives the contribution to each entry of P0.
    # The term contributes w_strassen[i*2+j] to P0[i,j].
    # Then C[2*a+i][2*c+j] += C_block_coeffs[s][a][c] * w_strassen[i*2+j]
    
    # W is indexed by (k, i) where k = col of B/col of C, i = row of C.
    # W[k*n + i, term_index] = coefficient for C[i, k]
    # C[i, k] where i = 2*a+i_local, k = 2*c+j_local
    # W[(2*c+j_local)*n + (2*a+i_local), s*7+t] += C_block_coeffs[s][a][c] * w_strassen[i_local*2+j_local]
    
    # Let me code this up.
    
    for s in range(7):
        for t in range(7):
            term_idx = s * 7 + t
            u_s, v_s, w_s = strassen_terms[t]
            
            # A_factor coefficients
            for i in range(2):
                for j in range(2):
                    u_val = u_s[i*2+j]
                    if u_val == 0:
                        continue
                    for a in range(2):
                        for b in range(2):
                            alpha = A_block_coeffs[s][a][b]
                            if alpha == 0:
                                continue
                            row_A = 2*a + i
                            col_A = 2*b + j
                            U[row_A * m + col_A][term_idx] += u_val * alpha
            
            # B_factor coefficients
            for i in range(2):
                for j in range(2):
                    v_val = v_s[i*2+j]
                    if v_val == 0:
                        continue
                    for b in range(2):
                        for c in range(2):
                            beta = B_block_coeffs[s][b][c]
                            if beta == 0:
                                continue
                            row_B = 2*b + i
                            col_B = 2*c + j
                            V[row_B * m + col_B][term_idx] += v_val * beta
            
            # C_factor coefficients
            for i in range(2):
                for j in range(2):
                    w_val = w_s[i*2+j]
                    if w_val == 0:
                        continue
                    for a in range(2):
                        for c in range(2):
                            gamma = C_block_coeffs[s][a][c]
                            if gamma == 0:
                                continue
                            row_C = 2*a + i
                            col_C = 2*c + j
                            W[col_C * n + row_C][term_idx] += gamma * w_val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

