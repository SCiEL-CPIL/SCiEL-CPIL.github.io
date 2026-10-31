      def construct():
          n, m, p = 4, 4, 4
          rank = 49
          U = [[0] * rank for _ in range(n * m)]
          V = [[0] * rank for _ in range(m * p)]
          W = [[0] * rank for _ in range(p * n)]
          
          # Strassen coefficients for 2x2x2
          # u_s: coeffs for A (4 entries)
          # v_s: coeffs for B (4 entries)
          # w_s: coeffs for C (4 entries)
          # Note: C is reconstructed as sum of w_s * M_s
          # M_s is the product of the linear forms.
          
          # We define the 7 steps
          # Step 1: (A00+A11)(B00+B11)
          # Step 2: (A10+A11)B00
          # Step 3: A00(B01-B11)
          # Step 4: A11(B10-B00)
          # Step 5: (A00+A01)B11
          # Step 6: (A10-A00)(B00+B01)
          # Step 7: (A01-A11)(B10+B11)
          
          # We need to map these to the 4x4x4 tensor.
          # The 4x4 matrix is divided into 2x2 blocks.
          # Block (i1, j1) contains entries A[2*i1+i0, 2*j1+j0]
          
          # For each outer step alpha, we define X_alpha and Y_alpha
          # X_alpha is a 2x2 matrix of linear forms on A
          # Y_alpha is a 2x2 matrix of linear forms on B
          
          # Then we compute the product Z_alpha = X_alpha Y_alpha
          # Z_alpha is a 2x2 matrix of linear forms on C?
          # No, Z_alpha is a 2x2 matrix of "products" of linear forms.
          # Each entry Z_alpha[i0, k0] is a sum of products of linear forms.
          # We decompose each entry using the inner Strassen steps.
          
          # This is effectively:
          # For each alpha:
          #   For each beta (inner Strassen step):
          #     The term is U_outer_alpha[beta] * V_outer_alpha[beta] * W_outer_alpha[beta]?
          #     No, the inner Strassen applies to the multiplication of the 2x2 matrices X_alpha and Y_alpha.
          
          # Let's define the inner Strassen U, V, W vectors for 2x2x2.
          # u_s[i], v_s[j], w_s[k]
          
          # For a fixed alpha:
          # X_alpha is 2x2. Y_alpha is 2x2.
          # The product Z_alpha = X_alpha Y_alpha.
          # Z_alpha[i0, k0] = sum_j0 X_alpha[i0, j0] Y_alpha[j0, k0]
          # We want to express this sum using the rank-7 decomposition of the 2x2x2 tensor.
          # The 2x2x2 tensor T_inner has T_inner[i0, j0, k0] = 1 if i0, j0, k0 match the multiplication.
          # Decomposition: T_inner = sum_s u_s[i0] v_s[j0] w_s[k0]
          # So Z_alpha[i0, k0] = sum_s (sum_j0 u_s[i0] X_alpha[i0, j0] v_s[j0]) (sum_j0 ...)?
          # No.
          # The multiplication is (X Y)[i, k] = sum_j X[i, j] Y[j, k].
          # Using Strassen:
          # (X Y)[i, k] = sum_s w_s[i, k] M_s
          # Where M_s is the "scalar" product for step s.
          # M_s = (sum_i u_s[i] X[i, ?]) ...
          # Actually, Strassen computes 7 scalars M_s.
          # M_s = L^A_s(X) * L^B_s(Y)
          # Where L^A_s and L^B_s are linear forms on the entries of X and Y.
          # Then (X Y)[i, k] = sum_s c_{s, i, k} M_s.
          
          # So for fixed alpha:
          # We have X_alpha and Y_alpha (2x2 matrices of linear forms on A and B).
          # We apply Strassen to the multiplication X_alpha * Y_alpha.
          # For each inner step s (1..7):
          #   Compute M_s = L^A_s(X_alpha) * L^B_s(Y_alpha)
          #   L^A_s(X_alpha) is a linear form on the entries of X_alpha.
          #   Since entries of X_alpha are linear forms on A, M_s is a product of two linear forms on A and B.
          #   This corresponds to a rank-1 tensor term.
          #   The contribution to C is sum_{i0, k0} c_{s, i0, k0} M_s placed in block (i0, k0) of C?
          #   No, Z_alpha is the result of the block multiplication.
          #   The final C is sum_alpha (contribution of alpha).
          #   Contribution of alpha to C is:
          #   For each block (i0, k0) of C:
          #     c_{alpha, i0, k0} * Z_alpha[i0, k0]
          #   Where c_{alpha, i0, k0} are the outer Strassen reconstruction coefficients.
          
          # So, for fixed alpha and inner step s:
          # Term = (Product of linear forms for M_s)
          # Contribution to C block (i0, k0) is c_{alpha, i0, k0} * c_{s, i0, k0} * M_s
          # So the total coefficient for C entry (2*i0+i, 2*k0+k) is c_{alpha, i0, k0} * c_{s, i0, k0}.
          
          # We need to construct U, V, W for this term.
          # U part: Linear form on A from L^A_s(X_alpha).
          # V part: Linear form on B from L^B_s(Y_alpha).
          # W part: Coefficient for C entry.
          
          # This is implementable.
          
          # 1. Define outer Strassen coefficients.
          # 2. Define inner Strassen coefficients.
          # 3. Loop alpha, s.
          # 4. Compute U, V, W vectors.
      