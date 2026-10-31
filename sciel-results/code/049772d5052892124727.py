"""Matrix multiplication 4x4x4 - rank 49 Strassen decomposition."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """Return rank-49 Strassen decomposition of 4x4x4 matrix multiplication.
    
    Uses recursive Strassen: split 4x4 into 2x2 blocks, apply Strassen's 7-multiplication
    formula at each level, yielding 7^2 = 49 scalar multiplications.
    
    Returns dict with 'U', 'V', 'W' matrices of shapes (16, 49), (16, 49), (16, 49)
    where W[k*4+i, r] follows the trace convention.
    """
    n, m, p = DIMENSIONS  # 4, 4, 4
    rank = 49
    
    U = [[0] * rank for _ in range(n * m)]  # 16 x 49
    V = [[0] * rank for _ in range(m * p)]  # 16 x 49
    W = [[0] * rank for _ in range(p * n)]  # 16 x 49
    
    # Strassen's 7 multiplications for 2x2 matrices:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    #
    # Result:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For 4x4 = 2x2 blocks of 2x2:
    # A = [[A00, A01], [A10, A11]] where each Aij is 2x2
    # B = [[B00, B01], [B10, B11]]
    #
    # First level Strassen gives 7 products of 2x2 matrices.
    # Each 2x2 product uses another 7 multiplications = 49 total.
    
    # Index mapping:
    # A[i,j] where i in {0,1,2,3}, j in {0,1,2,3}
    # Block Aij = A[2i:2i+2, 2j:2j+2]
    # Similarly for B and C.
    
    # Level 1: Define the 7 "super" multiplications at the 2x2 block level
    # For each of the 7 Strassen multiplications, we need to form
    # linear combinations of 2x2 blocks, multiply them (using Strassen again),
    # and combine results.
    
    # Let's define the structure more carefully.
    
    # At the top level (4x4), Strassen gives:
    # P1 = (A00 + A11) * (B00 + B11)
    # P2 = (A10 + A11) * B00
    # P3 = A00 * (B10 - B11)
    # P4 = A11 * (B01 - B00)
    # P5 = (A00 + A01) * B11
    # P6 = (A10 - A00) * (B00 + B01)
    # P7 = (A01 - A11) * (B10 + B11)
    #
    # Where Aij, Bij are 2x2 submatrices.
    #
    # Result:
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # Each Pk is a 2x2 matrix = product of two 2x2 matrices (linear combos of 2x2 blocks).
    # Each 2x2 * 2x2 uses Strassen's 7 multiplications.
    
    # So for each of the 7 top-level multiplications, we have 7 scalar mults = 49 total.
    
    # Let's enumerate:
    # Top-level multiplication s (0..6), inner Strassen multiplication t (0..6)
    # Column index = s * 7 + t
    
    # For each top-level mult s, we have:
    #   Left operand L_s = linear combination of 2x2 blocks of A
    #   Right operand R_s = linear combination of 2x2 blocks of B
    #
    # Then P_s = L_s * R_s, computed via Strassen on the 2x2 matrices L_s, R_s.
    
    # Strassen on 2x2 matrices L, R:
    # L = [[L00, L01], [L10, L11]], R = [[R00, R01], [R10, R11]]
    # S1 = (L00 + L11)(R00 + R11)
    # S2 = (L10 + L11)R00
    # S3 = L00(R10 - R11)
    # S4 = L11(R01 - R00)
    # S5 = (L00 + L01)R11
    # S6 = (L10 - L00)(R00 + R01)
    # S7 = (L01 - L11)(R10 + R11)
    #
    # Result:
    # C00 = S1 + S4 - S5 + S7
    # C01 = S3 + S5
    # C10 = S2 + S4
    # C11 = S1 - S2 + S3 + S6
    
    # Now, L_s and R_s are themselves linear combinations of 2x2 blocks.
    # For example, for s=0:
    #   L_0 = A00 + A11 (element-wise sum of the two 2x2 blocks)
    #   R_0 = B00 + B11
    #
    # So L_0[0,0] = A[0,0] + A[2,2]
    #    L_0[0,1] = A[0,1] + A[2,3]
    #    L_0[1,0] = A[1,0] + A[3,2]
    #    L_0[1,1] = A[1,1] + A[3,3]
    #
    # Similarly for R_0.
    
    # Let's define the coefficients for each top-level multiplication.
    # For top-level s, L_s = sum_{i,j} alpha_s[i,j] * A_{ij} (2x2 blocks)
    # where alpha_s[i,j] are coefficients (0, 1, or -1).
    
    # Top-level Strassen coefficients for left operand (which blocks of A):
    # s=0: A00 + A11  -> {(0,0):1, (1,1):1}
    # s=1: A10 + A11  -> {(1,0):1, (1,1):1}
    # s=2: A00         -> {(0,0):1}
    # s=3: A11         -> {(1,1):1}
    # s=4: A00 + A01  -> {(0,0):1, (0,1):1}
    # s=5: A10 - A00  -> {(1,0):1, (0,0):-1}
    # s=6: A01 - A11  -> {(0,1):1, (1,1):-1}
    
    left_coeffs = [
        {(0,0):1, (1,1):1},      # s=0
        {(1,0):1, (1,1):1},      # s=1
        {(0,0):1},                # s=2
        {(1,1):1},                # s=3
        {(0,0):1, (0,1):1},      # s=4
        {(1,0):1, (0,0):-1},     # s=5
        {(0,1):1, (1,1):-1},     # s=6
    ]
    
    right_coeffs = [
        {(0,0):1, (1,1):1},      # s=0
        {(0,0):1},                # s=1
        {(1,0):1, (1,1):-1},     # s=2
        {(0,1):1, (0,0):-1},     # s=3
        {(1,1):1},                # s=4
        {(0,0):1, (0,1):1},      # s=5
        {(1,0):1, (1,1):1},      # s=6
    ]
    
    # Now for each top-level s and inner Strassen t (0..6):
    # The inner Strassen computes P_s = L_s * R_s
    # where L_s[br, bc] and R_s[br, bc] are elements of the 2x2 matrices L_s, R_s.
    
    # For inner Strassen multiplication t:
    # t=0: (L00+L11)(R00+R11)
    # t=1: (L10+L11)R00
    # t=2: L00(R10-R11)
    # t=3: L11(R01-R00)
    # t=4: (L00+L01)R11
    # t=5: (L10-L00)(R00+R01)
    # t=6: (L01-L11)(R10+R11)
    
    # For each t, we need to express:
    #   left_factor = sum of L_s elements with coefficients
    #   right_factor = sum of R_s elements with coefficients
    #
    # Each L_s element is a linear combination of A entries.
    # Each R_s element is a linear combination of B entries.
    
    # Let's build the U and V matrices.
    # U[row, col] where row = i*m+j (i,j in 0..3), col = s*7+t
    # V[row, col] where row = j*p+k (j,k in 0..3), col = s*7+t
    # W[row, col] where row = k*n+i (k,i in 0..3), col = s*7+t
    
    # For inner Strassen t, the left factor involves specific L_s elements:
    # Let's define for each t which L_s elements are used and with what coefficients.
    # L_s is 2x2: L_s[0,0], L_s[0,1], L_s[1,0], L_s[1,1]
    
    inner_left_coeffs = [
        {(0,0):1, (1,1):1},      # t=0
        {(1,0):1, (1,1):1},      # t=1
        {(0,0):1},                # t=2
        {(1,1):1},                # t=3
        {(0,0):1, (0,1):1},      # t=4
        {(1,0):1, (0,0):-1},     # t=5
        {(0,1):1, (1,1):-1},     # t=6
    ]
    
    inner_right_coeffs = [
        {(0,0):1, (1,1):1},      # t=0
        {(0,0):1},                # t=1
        {(1,0):1, (1,1):-1},     # t=2
        {(0,1):1, (0,0):-1},     # t=3
        {(1,1):1},                # t=4
        {(0,0):1, (0,1):1},      # t=5
        {(1,0):1, (1,1):1},      # t=6
    ]
    
    # For each s, t:
    # col = s*7 + t
    # 
    # Left factor (for U):
    #   sum over (br,bc) of inner_left_coeffs[t][(br,bc)] * L_s[br,bc]
    #   where L_s[br,bc] = sum over (bi,bj) of left_coeffs[s][(bi,bj)] * A[2*bi+br, 2*bj+bc]
    #
    # So U[2*bi+br, 2*bj+bc, col] += inner_left_coeffs[t][(br,bc)] * left_coeffs[s][(bi,bj)]
    #
    # Similarly for V:
    #   V[2*bj+bc, 2*bk+bd, col] += inner_right_coeffs[t][(bd,be)] * right_coeffs[s][(bk,bl)]
    #   Wait, let me be more careful.
    
    # Let me redefine more carefully.
    # 
    # A is 4x4, indexed A[i,j] for i,j in 0..3.
    # Block A_{ij} (i,j in 0..1) = A[2i:2i+2, 2j:2j+2]
    # So A_{ij}[r,c] = A[2i+r, 2j+c] for r,c in 0..1.
    #
    # For top-level s:
    #   L_s is a 2x2 matrix where L_s[r,c] = sum_{(bi,bj) in left_coeffs[s]} left_coeffs[s][(bi,bj)] * A_{bi,bj}[r,c]
    #   = sum_{(bi,bj)} left_coeffs[s][(bi,bj)] * A[2*bi+r, 2*bj+c]
    #
    # For inner Strassen t on L_s * R_s:
    #   The left factor for multiplication t is:
    #     LF = sum_{(br,bc) in inner_left_coeffs[t]} inner_left_coeffs[t][(br,bc)] * L_s[br,bc]
    #     = sum_{(br,bc)} inner_left_coeffs[t][(br,bc)] * sum_{(bi,bj)} left_coeffs[s][(bi,bj)] * A[2*bi+br, 2*bj+bc]
    #
    # So U[A[2*bi+br, 2*bj+bc], col] += inner_left_coeffs[t][(br,bc)] * left_coeffs[s][(bi,bj)]
    #
    # Similarly for V:
    #   R_s[r,c] = sum_{(bi,bj) in right_coeffs[s]} right_coeffs[s][(bi,bj)] * B[2*bi+r, 2*bj+c]
    #   RF = sum_{(br,bc) in inner_right_coeffs[t]} inner_right_coeffs[t][(br,bc)] * R_s[br,bc]
    #   V[B[2*bi+br, 2*bj+bc], col] += inner_right_coeffs[t][(br,bc)] * right_coeffs[s][(bi,bj)]
    
    # For W:
    #   P_s = L_s * R_s is a 2x2 matrix.
    #   P_s[r,c] = sum_{u,v in 0..1} L_s[r,u] * R_s[u,c]
    #
    # The result C = sum_s W_s * P_s where W_s are the top-level combination coefficients.
    # Actually, C_{ij} = sum_s top_result_coeffs[(i,j,s)] * P_s
    #
    # top_result_coeffs:
    # C00 = P0 + P3 - P4 + P6  -> {(0,0):1, (0,3):1, (0,4):-1, (0,6):1}
    # C01 = P2 + P4            -> {(0,1):1, (0,4):1}
    # C10 = P1 + P3            -> {(1,0):1, (1,3):1}
    # C11 = P0 - P1 + P2 + P5  -> {(1,1):1, (0,0):1, (1,0):-1, (1,2):1, (1,5):1}
    
    top_result_coeffs = [
        {(0,0):1, (0,3):1, (0,4):-1, (0,6):1},   # for C00 (block (0,0))
        {(0,1):1, (0,4):1},                        # for C01 (block (0,1))
        {(1,0):1, (1,3):1},                        # for C10 (block (1,0))
        {(1,1):1, (0,0):1, (1,0):-1, (1,2):1, (1,5):1},  # for C11 (block (1,1))
    ]
    # Wait, let me re-index. top_result_coeffs[ci,cj] should give the coeffs for C_{ci,cj}.
    # Let me redefine:
    # top_comb[(ci,cj)] = dict mapping s -> coeff
    
    top_comb = {
        (0,0): {0:1, 3:1, 4:-1, 6:1},
        (0,1): {2:1, 4:1},
        (1,0): {1:1, 3:1},
        (1,1): {0:1, 1:-1, 2:1, 5:1},
    }
    
    # P_s is 2x2. C_{ci,cj} = sum_s top_comb[(ci,cj)][s] * P_s
    # C[2*ci+r, 2*cj+c] = sum_s top_comb[(ci,cj)][s] * P_s[r,c]
    # P_s[r,c] = sum_{u,v} L_s[r,u] * R_s[u,c]
    #
    # But in the tensor decomposition, we have:
    # C[i,j] = sum_r U[i,r] * V[j,r] * W[r_out, r]
    #
    # Wait, the convention is:
    # (A*B)[i,j] = sum_k A[i,k] * B[k,j]
    # In tensor form: T[i,j,k] = delta_{i,k}... no.
    #
    # The matrix multiplication tensor has T[a,b,c] such that
    # (A*B)[a,c] = sum_b T[a,b,c] * A[a,b] * B[b,c]
    # So T[a,b,c] = 1 if the indices match appropriately, which they always do in standard mult.
    #
    # In the rank-r decomposition:
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    #
    # Where:
    # U has shape (n*m, r) = (16, r), indexed by (a, b) where a=row of A, b=col of A
    # V has shape (m*p, r) = (16, r), indexed by (b, c) where b=row of B, c=col of B
    # W has shape (p*n, r) = (16, r), indexed by (c, a) where c=row of result, a=col of result
    #
    # Wait, the problem says W[k*n+i, r] follows the trace convention.
    # Let me re-read: "The output factor follows the official trace convention W[k*n+i,r]."
    #
    # Standard: (A*B)[i,j] = sum_k A[i,k] * B[k,j]
    # Tensor: M[i,j,k] = A[i,k] * B[k,j] contributes to (A*B)[i,j]
    # 
    # In the factorization:
    # U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
    #
    # So:
    # U is (n*m, r) with index (i, k) -> i*m+k
    # V is (m*p, r) with index (k, j) -> k*p+j
    # W is (p*n, r) with index (j, i) -> j*n+i
    #
    # And sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] should equal (A*B)[i,j] when
    # we contract appropriately... Actually, the full tensor identity is:
    # sum_k A[i,k] * B[k,j] = sum_r (sum_k U[i*m+k,r] * A[i,k]) * (sum_k V[k*p+j,r] * B[k,j]) * W[j*n+i,r]
    #
    # No wait, that's not right either. Let me think again.
    #
    # The matrix multiplication tensor is:
    # T[i, j, k] where (A*B)[i,j] = sum_k A[i,k] * B[k,j]
    #
    # In the rank-r decomposition:
    # T[i,j,k] = sum_r U[i*k, r] ... 
    #
    # Actually, the standard formulation:
    # We want: (A*B)[i,j] = sum_k A[i,k] B[k,j]
    #
    # Define intermediates:
    # P_r = (sum_k U[i*m+k, r] A[i,k]) * (sum_k V[k*p+j, r] B[k,j])
    # Then (A*B)[i,j] = sum_r W[j*n+i, r] * P_r
    #
    # But this is not quite right because P_r depends on i,j through U and V.
    #
    # The correct tensor decomposition:
    # The multiplication tensor M has components M[i,j,k] = delta_{i,k} (in a sense, it's the identity in the appropriate indices).
    # Actually, M[i,j,k] such that <M, A ⊗ B> = (A*B)
    # 
    # Let me just use the standard form:
    # (A*B)[i,j] = sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] ... no, that's not how it works.
    
    # Let me look at the seed program:
    # U[i*m+j][column] = 1  (so U is indexed by (i, j) where i=row of A, j=col of A)
    # V[j*p+k][column] = 1  (V indexed by (j, k) where j=row of B, k=col of B)
    # W[k*n+i][column] = 1  (W indexed by (k, i) where k=row of result, i=col of result)
    #
    # And the decomposition is: for each (i,j,k), there's a rank-1 term
    # U[i*m+j] * V[j*p+k] * W[k*n+i]
    #
    # This means: sum_{(i,j,k)} A[i,j] * B[j,k] * (contribution to C[k,i])
    # which gives (A*B)[k,i] = sum_j A[i,j] B[j,k]... 
    #
    # Hmm, there's an index convention issue. Let me just follow the seed program's convention.
    #
    # In the seed:
    # U[i*m + j, c] = 1 means U is (n*m) x rank, row index = i*m+j
    # V[j*p + k, c] = 1 means V is (m*p) x rank, row index = j*p+k
    # W[k*n + i, c] = 1 means W is (p*n) x rank, row index = k*n+i
    #
    # The rank-1 tensor is U[i*m+j] * V[j*p+k] * W[k*n+i]
    # Contracted with A[i,j], B[j,k], this contributes to C[k,i].
    #
    # So the output C is indexed as C[k,i] = sum_j A[i,j] * B[j,k]
    # Which means C^T = A * B, or C = B * A^T... 
    #
    # Actually, I think the convention is that the "result" matrix R has R[k,i] = sum_j A[i,j]B[j,k],
    # which is just (A^T * B^T)^T or something. The key point is to follow the same convention.
    #
    # Let me just verify: for standard (A*B)[i,j] = sum_k A[i,k]B[k,j],
    # if we set i->k, j->i in the above, we get:
    # R[k,i] = sum_j A[i,j] B[j,k]
    # This is (A * B^T)[k,i]... 
    #
    # I'll just follow the seed program's indexing exactly and make sure my decomposition
    # satisfies: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] 
    # correctly represents the multiplication.
    
    # Given the complexity, let me just directly construct the rank-49 decomposition
    # by thinking of it as 49 scalar multiplications.
    
    # Let me restart with a cleaner approach.
    
    # I'll build U, V, W directly.
    
    # For the recursive Strassen approach:
    # Level 0 (top): 4x4 = 2x2 blocks of 2x2
    #   7 multiplications of 2x2 matrices
    # Level 1: each 2x2 * 2x2 uses 7 scalar multiplications
    #   Total: 49 scalar multiplications
    
    # Let me define everything explicitly.
    
    # A is 4x4: A[i,j] for i,j in 0..3
    # B is 4x4: B[i,j] for i,j in 0..3
    # C = A*B (in the appropriate convention)
    
    # Following seed convention:
    # C[k,i] = sum_j A[i,j] * B[j,k]
    
    # So C is also 4x4, and C[k,i] = (A * B^T)[k,i] if we think in standard terms.
    # Or equivalently, if we define C' = C^T, then C'[i,k] = sum_j A[i,j] B[j,k] = (A*B)[i,k]
    # So C' = A*B in standard convention.
    
    # I'll compute A*B in standard convention and then transpose to get C.
    # But for the factorization, I just need to match the seed's indexing.
    
    # Let me define:
    # Standard: (A*B_std)[i,k] = sum_j A[i,j] * B[j,k]
    # Seed: C[k,i] = (A*B_std)[i,k]
    
    # So I'll compute the decomposition for (A*B_std)[i,k] = sum_j A[i,j] B[j,k]
    # and then map to the seed's indexing.
    
    # In seed indexing:
    # U row: i*m+j = i*4+j (i=row of A, j=col of A)
    # V row: j*p+k = j*4+k (j=row of B, k=col of B)  
    # W row: k*n+i = k*4+i (k=col of result = col of A*B_std, i=row of result)
    
    # And (A*B_std)[i,k] = sum_r U[i*4+j, r] * V[j*4+k, r] * W[k*4+i, r] ... 
    # No, that's not right. The contraction is over the "multiplication" index r.
    
    # Actually, the way tensor decomposition works:
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # where T[i,j,k] = 1 (for the multiplication tensor, T is just the identity in the appropriate sense)
    #
    # No, T[i,j,k] represents the coefficient such that:
    # (A*B)[i,k] = sum_{j,r} T[i,j,k,r]... 
    #
    # I'm overcomplicating this. Let me just use the fact that the seed program works
    # and follow its exact indexing. The seed says:
    #
    # For each (i,j,k): U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1
    #
    # This means the rank-1 tensor is the outer product of:
    # e_{i*m+j} (in R^{16}) ⊗ e_{j*p+k} (in R^{16}) ⊗ e_{k*n+i} (in R^{16})
    #
    # And the sum over all (i,j,k) of these gives the multiplication tensor.
    #
    # For my Strassen decomposition, I need to find 49 vectors u_r, v_r, w_r in R^{16}
    # such that sum_r u_r ⊗ v_r ⊗ w_r = multiplication tensor.
    #
    # Let me just directly compute what U, V, W should be for each of the 49 scalar mults.
    
    # For scalar multiplication s (0..48), let s = S*7 + T where S is top-level (0..6), T is inner (0..6).
    
    # The scalar mult computes:
    # P_{S,T} = (left_linear_combo) * (right_linear_combo)
    # where left_linear_combo = sum of some A[i,j] with coefficients
    #       right_linear_combo = sum of some B[j,k] with coefficients
    #
    # And this contributes to C[k,i] with some coefficient.
    
    # So:
    # U[i*4+j, S*7+T] += coeff_left[i,j]
    # V[j*4+k, S*7+T] += coeff_right[j,k]  
    # W[k*4+i, S*7+T] += coeff_result[k,i]
    
    # Where coeff_left, coeff_right, coeff_result are determined by the Strassen structure.
    
    # Let me now carefully derive these.
    
    # Standard matrix multiplication: (A*B)[i,k] = sum_j A[i,j] B[j,k]
    
    # Strassen at top level (treating A, B as 2x2 blocks of 2x2):
    # A = [[A00, A01], [A10, A11]] where Aij is 2x2
    # B = [[B00, B01], [B10, B11]]
    
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B10-B11)
    # M3 = A11(B01-B00)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    
    # Now each M_s = L_s * R_s where L_s, R_s are 2x2 matrices (linear combos of 2x2 blocks).
    # Each L_s * R_s uses Strassen's 7 multiplications.
    
    # For M_s, let L_s = [[l00, l01], [l10, l11]], R_s = [[r00, r01], [r10, r11]]
    # where each element is a linear combination of A or B entries.
    
    # Inner Strassen for M_s:
    # N_{s,0} = (l00+l11)(r00+r11)
    # N_{s,1} = (l10+l11)r00
    # N_{s,2} = l00(r10-r11)
    # N_{s,3} = l11(r01-r00)
    # N_{s,4} = (l00+l01)r11
    # N_{s,5} = (l10-l00)(r00+r01)
    # N_{s,6} = (l01-l11)(r10+r11)
    
    # M_s[0,0] = N_{s,0} + N_{s,3} - N_{s,4} + N_{s,6}
    # M_s[0,1] = N_{s,2} + N_{s,4}
    # M_s[1,0] = N_{s,1} + N_{s,3}
    # M_s[1,1] = N_{s,0} - N_{s,1} + N_{s,2} + N_{s,5}
    
    # Now I need to express each N_{s,t} as a product of two linear forms in the A and B entries.
    
    # Let me define for each (S, T) the left linear form, right linear form, and result coefficients.
    
    # This is getting very detailed. Let me just code it up with the coefficient tables.
    
    # I'll define helper functions to build the coefficient arrays.
    
    import time
    start_time = time.time()
    
    # Build U, V, W as 16x49 arrays
    U_arr = [[0] * rank for _ in range(16)]
    V_arr = [[0] * rank for _ in range(16)]
    W_arr = [[0] * rank for _ in range(16)]
    
    # Top-level left coefficients: for each S, dict from (bi,bj) to coeff
    # where A_{bi,bj}[r,c] = A[2*bi+r, 2*bj+c]
    top_left = [
        {(0,0):1, (1,1):1},      # S=0
        {(1,0):1, (1,1):1},      # S=1
        {(0,0):1},                # S=2
        {(1,1):1},                # S=3
        {(0,0):1, (0,1):1},      # S=4
        {(1,0):1, (0,0):-1},     # S=5
        {(0,1):1, (1,1):-1},     # S=6
    ]
    
    top_right = [
        {(0,0):1, (1,1):1},      # S=0
        {(0,0):1},                # S=1
        {(1,0):1, (1,1):-1},     # S=2
        {(0,1):1, (0,0):-1},     # S=3
        {(1,1):1},                # S=4
        {(0,0):1, (0,1):1},      # S=5
        {(1,0):1, (1,1):1},      # S=6
    ]
    
    # Inner Strassen left coeffs: for each T, dict from (br,bc) to coeff
    # referring to elements of the 2x2 matrix L_S
    inner_left = [
        {(0,0):1, (1,1):1},      # T=0
        {(1,0):1, (1,1):1},      # T=1
        {(0,0):1},                # T=2
        {(1,1):1},                # T=3
        {(0,0):1, (0,1):1},      # T=4
        {(1,0):1, (0,0):-1},     # T=5
        {(0,1):1, (1,1):-1},     # T=6
    ]
    
    inner_right = [
        {(0,0):1, (1,1):1},      # T=0
        {(0,0):1},                # T=1
        {(1,0):1, (1,1):-1},     # T=2
        {(0,1):1, (0,0):-1},     # T=3
        {(1,1):1},                # T=4
        {(0,0):1, (0,1):1},      # T=5
        {(1,0):1, (1,1):1},      # T=6
    ]
    
    # For each S, T:
    for S in range(7):
        for T in range(7):
            col = S * 7 + T
            
            # Left linear form:
            # sum_{(br,bc) in inner_left[T]} inner_left[T][(br,bc)] * L_S[br,bc]
            # L_S[br,bc] = sum_{(bi,bj) in top_left[S]} top_left[S][(bi,bj)] * A[2*bi+br, 2*bj+bc]
            #
            # So coeff for A[2*bi+br, 2*bj+bc] is:
            # inner_left[T][(br,bc)] * top_left[S][(bi,bj)]
            
            for (br, bc), inner_coeff in inner_left[T].items():
                for (bi, bj), top_coeff in top_left[S].items():
                    i = 2 * bi + br
                    j = 2 * bj + bc
                    U_arr[i * 4 + j][col] += inner_coeff * top_coeff
            
            # Right linear form:
            # Similar for B
            for (br, bc), inner_coeff in inner_right[T].items():
                for (bi, bj), top_coeff in top_right[S].items():
                    j = 2 * bi + br  # row of B
                    k = 2 * bj + bc  # col of B
                    V_arr[j * 4 + k][col] += inner_coeff * top_coeff
            
            # Result coefficients:
            # N_{S,T} contributes to M_S[r,c] with certain coefficients,
            # and M_S contributes to C[ci,cj] with certain coefficients.
            
            # Inner result: M_S[r,c] = sum_T inner_result[T][(r,c)] * N_{S,T}
            inner_result = {
                0: {(0,0):1, (1,0):1, (1,1):1},  # N0: C00, C10, C11
                1: {(1,0):1, (1,1):-1},           # N1: C10, C11
                2: {(0,1):1, (1,1):1},            # N2: C01, C11
                3: {(0,0):1, (1,0):1},            # N3: C00, C10
                4: {(0,0):-1, (0,1):1},           # N4: C00, C01
                5: {(1,1):1},                     # N5: C11
                6: {(0,0):1},                     # N6: C00
            }
            
            # Top result: C[ci,cj] = sum_S top_comb[(ci,cj)][S] * M_S
            top_comb = {
                (0,0): {0:1, 3:1, 4:-1, 6:1},
                (0,1): {2:1, 4:1},
                (1,0): {1:1, 3:1},
                (1,1): {0:1, 1:-1, 2:1, 5:1},
            }
            
            # N_{S,T} contributes to M_S[r,c] with inner_result[T][(r,c)]
            # M_S contributes to C[ci,cj] with top_comb[(ci,cj)][S]
            #
            # C[ci,cj] is a 2x2 block: C[2*ci+r, 2*cj+c]
            #
            # So for each (r,c) in inner_result[T]:
            #   for each (ci,cj) in top_comb:
            #     if S in top_comb[(ci,cj)]:
            #       W[(2*cj+c)*4 + (2*ci+r)][col] += inner_result[T][(r,c)] * top_comb[(ci,cj)][S]
            
            for (r, c), inner_res_coeff in inner_result[T].items():
                for (ci, cj), top_coeffs in top_comb.items():
                    if S in top_coeffs:
                        top_res_coeff = top_coeffs[S]
                        # C index: row = 2*cj+c, col = 2*ci+r
                        # W index: k*n+i where k = row of C = 2*cj+c, i = col of C = 2*ci+r
                        k_idx = 2 * cj + c
                        i_idx = 2 * ci + r
                        W_arr[k_idx * 4 + i_idx][col] += inner_res_coeff * top_res_coeff
    
    return {"U": U_arr, "V": V_arr, "W": W_arr}
# EVOLVE-BLOCK-END
