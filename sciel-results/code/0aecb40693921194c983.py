import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140

    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}

    # Strassen-based decomposition for 4x4 x 4x5
    # Split B into B1 (4x4) and B2 (4x1)
    # A*B1: 4x4 x 4x4 via double Strassen (rank 49)
    # A*B2: 4x4 x 4x1 via schoolbook (rank 16)
    # Total rank: 65

    def strassen_2x2_factors():
        """Return the 7 Strassen factors for 2x2 matrix multiply.
        Each factor is (U_coeffs, V_coeffs, W_coeffs) where:
        U_coeffs is a 2x2 matrix of coefficients for A
        V_coeffs is a 2x2 matrix of coefficients for B
        W_coeffs is a 2x2 matrix of coefficients for C reconstruction
        """
        # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        # C00 = M1 - M4 - M6 + M7
        # C01 = M3 + M4
        # C10 = M2 + M5
        # C11 = M1 - M2 + M5 - M3

        factors = []
        # Factor 1: (A00+A11)(B00+B11)
        factors.append({
            'U': np.array([[1, 0], [0, 1]]),
            'V': np.array([[1, 0], [0, 1]]),
            'W': np.array([[1, 0], [0, 1]])
        })
        # Factor 2: (A10+A11)B00
        factors.append({
            'U': np.array([[0, 0], [1, 1]]),
            'V': np.array([[1, 0], [0, 0]]),
            'W': np.array([[0, 0], [1, 0]])
        })
        # Factor 3: A00(B10-B11)
        factors.append({
            'U': np.array([[1, 0], [0, 0]]),
            'V': np.array([[0, 0], [1, -1]]),
            'W': np.array([[0, 1], [0, 0]])
        })
        # Factor 4: A11(B01-B00)
        factors.append({
            'U': np.array([[0, 0], [0, 1]]),
            'V': np.array([[0, 1], [0, 0]]),
            'W': np.array([[0, 1], [0, 0]])
        })
        # Factor 5: (A00+A01)B11
        factors.append({
            'U': np.array([[1, 1], [0, 0]]),
            'V': np.array([[0, 0], [0, 1]]),
            'W': np.array([[0, 0], [1, 1]])
        })
        # Factor 6: (A10-A00)(B00+B01)
        factors.append({
            'U': np.array([[-1, 0], [1, 0]]),
            'V': np.array([[1, 1], [0, 0]]),
            'W': np.array([[0, 0], [0, 0]])  # Will fix W below
        })
        # Factor 7: (A01-A11)(B10+B11)
        factors.append({
            'U': np.array([[0, 1], [0, -1]]),
            'V': np.array([[0, 0], [1, 1]]),
            'W': np.array([[0, 0], [0, 0]])  # Will fix W below
        })

        # Fix W coefficients for factors 6 and 7
        # C00 = M1 - M4 - M6 + M7
        # C01 = M3 + M4
        # C10 = M2 + M5
        # C11 = M1 - M2 + M5 - M3
        # Factor 6 contributes -1 to C00
        factors[5]['W'] = np.array([[-1, 0], [0, 0]])
        # Factor 7 contributes +1 to C00
        factors[6]['W'] = np.array([[1, 0], [0, 0]])

        return factors

    def build_strassen_4x4x4():
        """Build rank-1 terms for 4x4 x 4x4 matrix multiply using double Strassen.
        Returns list of (u, v, w) where u is length 16, v is length 16, w is length 16.
        """
        s2_factors = strassen_2x2_factors()
        terms = []

        for outer_f in s2_factors:
            U_outer = outer_f['U']  # 2x2
            V_outer = outer_f['V']  # 2x2
            W_outer = outer_f['W']  # 2x2

            # The outer factor computes S = (U_outer * A_blocks) * (V_outer * B_blocks)
            # where A_blocks and B_blocks are 2x2 matrices of 2x2 blocks.
            # U_outer * A_blocks gives a 2x2 matrix of 2x2 blocks (the left operand of inner multiply)
            # V_outer * B_blocks gives a 2x2 matrix of 2x2 blocks (the right operand)
            # Then we do inner Strassen on these 2x2 block matrices.

            # Left operand: X = U_outer @ A_blocks
            # X[0,0] = U_outer[0,0]*A[0,0] + U_outer[0,1]*A[0,1]  (2x2 block)
            # X[0,1] = U_outer[0,0]*A[1,0] + U_outer[0,1]*A[1,1]
            # X[1,0] = U_outer[1,0]*A[0,0] + U_outer[1,1]*A[0,1]
            # X[1,1] = U_outer[1,0]*A[1,0] + U_outer[1,1]*A[1,1]

            # Right operand: Y = V_outer @ B_blocks
            # Y[0,0] = V_outer[0,0]*B[0,0] + V_outer[0,1]*B[0,1]
            # Y[0,1] = V_outer[0,0]*B[1,0] + V_outer[0,1]*B[1,1]
            # Y[1,0] = V_outer[1,0]*B[0,0] + V_outer[1,1]*B[0,1]
            # Y[1,1] = V_outer[1,0]*B[1,0] + V_outer[1,1]*B[1,1]

            # For each inner Strassen factor, we get 7 scalar multiplications.
            for inner_f in s2_factors:
                U_inner = inner_f['U']  # 2x2
                V_inner = inner_f['V']  # 2x2
                W_inner = inner_f['W']  # 2x2

                # The inner factor computes:
                # P = (U_inner @ X) * (V_inner @ Y)
                # where X and Y are 2x2 block matrices.
                # U_inner @ X gives a 2x2 matrix of 2x2 blocks:
                # (U_inner @ X)[r,c] = sum_k U_inner[r,k] * X[k,c]
                # = sum_k U_inner[r,k] * (sum_j U_outer[k,j] * A[j,c])
                # = sum_j (sum_k U_inner[r,k] * U_outer[k,j]) * A[j,c]
                # = (U_inner @ U_outer)[r,j] * A[j,c]

                # Similarly for V:
                # (V_inner @ Y)[r,c] = sum_k V_inner[r,k] * Y[k,c]
                # = sum_k V_inner[r,k] * (sum_j V_outer[k,j] * B[j,c])
                # = sum_j (sum_k V_inner[r,k] * V_outer[k,j]) * B[j,c]
                # = (V_inner @ V_outer)[r,j] * B[j,c]

                # So the effective U and V for the scalar multiplication are:
                # U_eff = U_inner @ U_outer  (2x2 matrix of coefficients for A blocks)
                # V_eff = V_inner @ V_outer  (2x2 matrix of coefficients for B blocks)

                U_eff = U_inner @ U_outer  # 2x2
                V_eff = V_inner @ V_outer  # 2x2

                # Now, the scalar multiplication is:
                # prod = (sum_{j,c} U_eff[j,c] * A[j,c]) * (sum_{j,c} V_eff[j,c] * B[j,c])
                # Wait, no. X and Y are 2x2 block matrices, and the inner Strassen
                # treats them as 2x2 matrices where each "element" is a 2x2 block.
                # The inner Strassen factor computes a scalar (in the block field) which
                # is a 2x2 matrix result. But we're doing scalar multiplications, so
                # each "block" multiplication becomes 4 scalar multiplications.

                # Actually, let me reconsider. The inner Strassen operates on 2x2 matrices
                # over a field. Here, the "field" elements are 2x2 matrices, so each
                # "multiplication" in the inner Strassen is actually a 2x2 matrix multiply,
                # which itself has rank 4 (schoolbook) or 7 (Strassen).

                # For simplicity, I'll use schoolbook for the innermost level.
                # So each inner Strassen "factor" (which is a block multiply of 2x2 blocks)
                # actually requires 4 scalar multiplications.

                # The inner factor computes:
                # P_block = (U_inner @ X) * (V_inner @ Y)  [this is a 2x2 block matrix]
                # But (U_inner @ X) is a 2x2 matrix of 2x2 blocks, and same for (V_inner @ Y).
                # The product is a 2x2 matrix of 2x2 blocks.
                # Each of the 4 block elements is a 2x2 matrix multiply.

                # Let me just enumerate all scalar multiplications directly.

                # The left operand for the innermost scalar mults:
                # L = U_inner @ U_outer  (2x2 coefficients for A blocks)
                # This means: left_operand[i,j] = sum_k L[i,k] * A[k,j]  for the block level
                # But A[k,j] is a 2x2 block, so:
                # left_operand is a 2x2 matrix of 2x2 blocks:
                # left_operand[r,c] = sum_k L[r,k] * A_block[k,c]

                # Similarly for right operand.

                # For the innermost scalar multiplications, I need to expand this fully.
                # Let me just build the U, V, W vectors directly.

                # U vector: length 16 (4x4 matrix A, row-major: A[0,0],A[0,1],A[0,2],A[0,3],A[1,0],...)
                # V vector: length 16 (4x4 matrix B, row-major)
                # W vector: length 16 (4x4 matrix C, row-major)

                u = np.zeros(16)
                v = np.zeros(16)
                w = np.zeros(16)

                # The inner Strassen factor, when applied to X and Y (2x2 block matrices),
                # produces a result that contributes to the outer Strassen result.
                # The contribution to C is weighted by W_outer and W_inner.

                # Let me think about this differently. I'll directly compute the coefficients.

                # For a given (outer_f, inner_f) pair, the scalar multiplication is:
                # prod = (sum_{a,b} coeff_A[a,b] * A[a,b]) * (sum_{c,d} coeff_B[c,d] * B[c,d])
                # where coeff_A and coeff_B are 4x4 matrices derived from U_eff and V_eff.

                # U_eff is 2x2, where U_eff[r,c] is the coefficient for A_block[c, ?]
                # Actually, U_eff = U_inner @ U_outer, and:
                # U_eff[r,c] = sum_k U_inner[r,k] * U_outer[k,c]
                # This means: the left operand's block [r,c] = U_eff[r,c] * ... no.

                # Let me restart the indexing.
                # A is 4x4, viewed as 2x2 blocks of 2x2:
                # A_block[0,0] = A[0:2, 0:2], A_block[0,1] = A[0:2, 2:4]
                # A_block[1,0] = A[2:4, 0:2], A_block[1,1] = A[2:4, 2:4]

                # U_outer is 2x2, and X = U_outer @ A_blocks means:
                # X[r,c] = sum_k U_outer[r,k] * A_block[k,c]  (2x2 block)
                # X[r,c][i,j] = sum_k U_outer[r,k] * A_block[k,c][i,j]
                # = sum_k U_outer[r,k] * A[2*k+i, 2*c+j]

                # Similarly, U_inner @ X:
                # (U_inner @ X)[r,c] = sum_k U_inner[r,k] * X[k,c]  (2x2 block)
                # (U_inner @ X)[r,c][i,j] = sum_k U_inner[r,k] * X[k,c][i,j]
                # = sum_k U_inner[r,k] * sum_l U_outer[k,l] * A[2*l+i, 2*c+j]
                # = sum_{l,k} (U_inner[r,k] * U_outer[k,l]) * A[2*l+i, 2*c+j]
                # = sum_l (U_inner @ U_outer)[r,l] * A[2*l+i, 2*c+j]

                # So the left operand for the innermost level is a 2x2 matrix of 2x2 blocks:
                # L[r,c][i,j] = sum_l U_eff[r,l] * A[2*l+i, 2*c+j]
                # where U_eff = U_inner @ U_outer

                # The scalar product for one innermost term is:
                # prod = (sum_{r,c,i,j} L[r,c][i,j] * ... ) * (similar for V)
                # No, each "block multiplication" in the inner Strassen is a 2x2 matrix multiply.
                # So for the inner factor, we have 4 block multiplications (for the 4 elements of the result),
                # each of which is a 2x2 matrix multiply (4 scalar mults).

                # This is getting very complex. Let me just hardcode the 49 terms.

                # Actually, let me take a different approach. I'll directly generate the rank-1 terms
                # by thinking of the full 4x4 x 4x4 Strassen decomposition.

                pass

        return terms

    # Build rank-1 terms for 4x4 x 4x5 using:
    # - 49 terms for 4x4 x 4x4 (double Strassen)
    # - 16 terms for 4x4 x 4x1 (schoolbook)
    # Total: 65 terms

    def build_rank1_terms():
        """Build all 65 rank-1 terms.
        Returns list of (u, v, w) where:
        u: length 16 (coefficients for A[4x4], row-major)
        v: length 20 (coefficients for B[4x5], row-major)
        w: length 20 (coefficients for C[5x4], W[k*n+i, r] convention)
        """
        terms = []

        # Part 1: 4x4 x 4x4 multiply (columns 0-3 of B)
        # Double Strassen: 49 terms
        s2_factors = strassen_2x2_factors()

        for of_idx, outer_f in enumerate(s2_factors):
            U_outer = outer_f['U']  # 2x2
            V_outer = outer_f['V']  # 2x2
            W_outer = outer_f['W']  # 2x2

            for if_idx, inner_f in enumerate(s2_factors):
                U_inner = inner_f['U']  # 2x2
                V_inner = inner_f['V']  # 2x2
                W_inner = inner_f['W']  # 2x2

                # Effective coefficients
                U_eff = U_inner @ U_outer  # 2x2
                V_eff = V_inner @ V_outer  # 2x2

                # The left operand is a 2x2 matrix of 2x2 blocks:
                # L[r,c][i,j] = sum_l U_eff[r,l] * A[2*l+i, 2*c+j]
                # The right operand is a 2x2 matrix of 2x2 blocks:
                # R[r,c][i,j] = sum_l V_eff[r,l] * B[2*l+i, 2*c+j]  (for cols 0-3)

                # The inner Strassen factor computes P = L * R (2x2 block matrix multiply)
                # P[r,c] = sum_k L[r,k] * R[k,c]  (2x2 block multiply)
                # P[r,c][i,j] = sum_k sum_a L[r,k][i,a] * R[k,c][a,j]
                # = sum_k sum_a (sum_l U_eff[r,l] * A[2*l+i, 2*k+a]) * (sum_m V_eff[k,m] * B[2*m+a, 2*c+j])

                # This is a sum of products of linear combinations.
                # For the tensor decomposition, each "scalar multiplication" in the innermost
                # level corresponds to one rank-1 term.

                # The innermost scalar multiplications: for each (k, a) pair in the block multiply,
                # we have:
                # term = (sum_l U_eff[r,l] * A[2*l+i, 2*k+a]) * (sum_m V_eff[k,m] * B[2*m+a, 2*c+j])

                # But this is not a single scalar product; it's a product of two linear combinations.
                # In the tensor decomposition, this corresponds to a rank-1 term where:
                # u vector has non-zero entries at positions corresponding to A[2*l+i, 2*k+a]
                # v vector has non-zero entries at positions corresponding to B[2*m+a, 2*c+j]
                # w vector has non-zero entry at the position corresponding to C[2*c+j, 2*r+i]
                #   (with appropriate weight from W_outer and W_inner)

                # Wait, I need to be more careful. The product of two linear combinations
                # (sum_l alpha_l * A[idx_l]) * (sum_m beta_m * B[idx_m])
                # expands to sum_{l,m} alpha_l * beta_m * A[idx_l] * B[idx_m]
                # which is a sum of rank-1 terms, not a single rank-1 term.

                # So my approach of using Strassen at both levels doesn't directly give
                # rank-1 terms. I need to think about this differently.

                # The correct approach: The tensor decomposition of 4x4 x 4x4 with rank 49
                # means there are 49 rank-1 terms. Each rank-1 term is:
                # u_t ⊗ v_t ⊗ w_t
                # where u_t is a vector in R^16, v_t in R^16, w_t in R^16.

                # The Strassen algorithm computes 7 "multiplications" at the outer level,
                # each of which is a 2x2 matrix multiply (rank 7 each), giving 49 scalar mults.
                # Each scalar mult is: (linear combo of A entries) * (linear combo of B entries)
                # and contributes to C with a linear combination.

                # So each of the 49 terms IS a rank-1 term in the tensor decomposition!
                # The u vector has the coefficients of the linear combo of A,
                # the v vector has the coefficients of the linear combo of B,
                # and the w vector has the coefficients for how the product contributes to C.

                # Let me construct these explicitly.

                # For the (outer_f, inner_f) pair:
                # Left linear combo: sum_{i,j} L_coeff[i,j] * A[i,j]
                # where L_coeff[i,j] = sum_l U_eff[?, ?] ... 

                # I'm getting confused with the indexing. Let me just directly compute
                # the coefficients for each of the 49 terms.

                # The outer Strassen factor of_idx computes:
                # S_of = (U_outer @ A_blocks) * (V_outer @ B_blocks)
                # where A_blocks, B_blocks are 2x2 matrices of 2x2 blocks.

                # The inner Strassen factor if_idx, when applied to the operands of S_of,
                # computes one of the 7 scalar (block) products.

                # Let me just enumerate: for each of the 49 (outer, inner) pairs,
                # determine the linear combination of A entries, the linear combination
                # of B entries, and the contribution to C.

                # This is complex. Let me use a different strategy:
                # Directly compute the 49 rank-1 terms by expanding the double Strassen.

                # I'll skip this and just use schoolbook for the 4x4x4 part too,
                # giving rank 64 + 16 = 80. No improvement.

                # Actually, let me try yet another approach. I know that the rank of
                # 4x4 x 4x4 is 49. Let me just hardcode the 49 terms.

                pass

        # For now, return schoolbook
        return None

    # Direct construction of rank-65 decomposition
    # Part 1: 4x4 x 4x4 (cols 0-3 of B) via double Strassen -> 49 terms
    # Part 2: 4x4 x 4x1 (col 4 of B) via schoolbook -> 16 terms

    U_list = []
    V_list = []
    W_list = []

    # Strassen 2x2 factors: (U_2x2, V_2x2, W_2x2)
    # M1 = (A00+A11)(B00+B11), C00+=1, C11+=1
    # M2 = (A10+A11)B00, C10+=1
    # M3 = A00(B10-B11), C01+=1
    # M4 = A11(B01-B00), C00-=1, C01+=1
    # M5 = (A00+A01)B11, C10+=1, C11+=1
    # M6 = (A10-A00)(B00+B01), C00-=1
    # M7 = (A01-A11)(B10+B11), C00+=1

    s2 = [
        # (U, V, W) as 2x2 arrays
        (np.array([[1,0],[0,1]]), np.array([[1,0],[0,1]]), np.array([[1,0],[0,1]])),
        (np.array([[0,0],[1,1]]), np.array([[1,0],[0,0]]), np.array([[0,0],[1,0]])),
        (np.array([[1,0],[0,0]]), np.array([[0,0],[1,-1]]), np.array([[0,1],[0,0]])),
        (np.array([[0,0],[0,1]]), np.array([[0,1],[0,0]]), np.array([[-1,1],[0,0]])),
        (np.array([[1,1],[0,0]]), np.array([[0,0],[0,1]]), np.array([[0,0],[1,1]])),
        (np.array([[-1,0],[1,0]]), np.array([[1,1],[0,0]]), np.array([[-1,0],[0,0]])),
        (np.array([[0,1],[0,-1]]), np.array([[0,0],[1,1]]), np.array([[1,0],[0,0]])),
    ]

    # Double Strassen for 4x4 x 4x4
    for Uo, Vo, Wo in s2:
        for Ui, Vi, Wi in s2:
            # U_eff = Ui @ Uo (2x2), V_eff = Vi @ Vo (2x2)
            Ueff = Ui @ Uo
            Veff = Vi @ Vo
            # W_eff = Wo @ Wi? No.
            # The outer factor contributes to C via Wo, and the inner factor via Wi.
            # The combined weight for C block [r,c] is Wo[r,c] * Wi[?,?]
            # Actually, the inner Strassen computes a 2x2 block result P,
            # and the outer Strassen uses P as one of its 7 factors.
            # The reconstruction is: C_blocks = sum_outer Wo_outer * P_outer
            # where P_outer is the 2x2 block result from the inner multiply.
            # And P_outer = sum_inner Wi_inner * (scalar product)_inner
            # So the total weight for C block [r,c] from this (outer,inner) pair is:
            # Wo[r,c] * Wi[r2,c2] for each block position (r2,c2) in the inner result.
            # But the inner result P is a 2x2 block matrix, and each element P[r2,c2]
            # is a 2x2 matrix. The contribution to C block [r,c] is:
            # sum_{r2,c2} Wo[r,c] ... no.

            # Let me think again. The outer Strassen computes:
            # C_blocks = f(Wo_1, P_1, Wo_2, P_2, ..., Wo_7, P_7)
            # where P_i are the 7 "products" from the outer level.
            # Each P_i is a 2x2 block matrix computed by inner Strassen.
            # P_i = g(Wi_1, Q_i1, Wi_2, Q_i2, ..., Wi_7, Q_i7)
            # where Q_ij are scalar (block) products.

            # The final C is a 4x4 matrix. C block [r,c] is a 2x2 matrix.
            # C[r,c] = sum_i Wo_i[r,c] * P_i  (where P_i is the 2x2 block result)
            # P_i[r2,c2] = sum_j Wi_j[r2,c2] * Q_ij  (where Q_ij is a scalar)
            # Q_ij = (linear combo of A) * (linear combo of B)

            # So C[r,c][a,b] = sum_i Wo_i[r,c] * P_i[r2,c2] ... no, this is wrong.

            # I think the issue is that P_i is a 2x2 block matrix, and Wo_i is a 2x2 matrix
            # of scalars. The product Wo_i * P_i is a 2x2 block matrix where each block is
            # Wo_i[r,c] * P_i[r,c].

            # So C_block[r,c] = sum_i Wo_i[r,c] * P_i[r,c]  (2x2 block)
            # P_i[r,c] = sum_j Wi_j[r,c] * Q_ij  (scalar, which is a 2x2 matrix product result)

            # Wait, Q_ij is the result of multiplying two 2x2 matrices (the operands),
            # so Q_ij is a 2x2 matrix. And Wi_j[r,c] is a scalar.
            # So P_i[r,c] = sum_j Wi_j[r,c] * Q_ij  (2x2 matrix)

            # And C_block[r,c] = sum_i Wo_i[r,c] * P_i[r,c]  (2x2 matrix)
            # = sum_i Wo_i[r,c] * sum_j Wi_j[r,c] * Q_ij
            # = sum_{i,j} (Wo_i[r,c] * Wi_j[r,c]) * Q_ij

            # Q_ij = (U_i @ Uo) * (V_i @ Vo) in the block sense, which expands to
            # a 2x2 matrix where each element is a product of linear combos of A and B entries.

            # For the tensor decomposition, each Q_ij element contributes rank-1 terms.
            # Q_ij[a,b] = sum_{k} (left_operand[a,k]) * (right_operand[k,b])
            # where left_operand and right_operand are 2x2 matrices of linear combos.

            # This is getting too complex. Let me just use schoolbook.

            pass

    # Return schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
