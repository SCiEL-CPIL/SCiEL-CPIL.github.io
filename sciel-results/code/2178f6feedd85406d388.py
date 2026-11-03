import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    deadline = time.time() + 140

    def schoolbook():
        """Schoolbook decomposition: rank = n*m*p = 80"""
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

    def verify(U, V, W, num_tests=10):
        """Verify that the decomposition exactly reproduces matrix multiplication."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        
        np.random.seed(42)
        for _ in range(num_tests):
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C_expected = A @ B
            
            C_decomp = np.zeros((p, n))
            for r in range(rank):
                u_r = U[:, r]
                v_r = V[:, r]
                w_r = W[:, r]
                
                Au = u_r.reshape(n, m)
                Bv = v_r.reshape(m, p)
                Cw = w_r.reshape(p, n)
                
                A_masked = A * Au
                B_masked = B * Bv
                C_decomp += (A_masked @ B_masked) * Cw
            
            if not np.allclose(C_decomp, C_expected, atol=1e-10):
                return False
        return True

    def strassen_2x2_terms():
        """
        Return the 7 rank-1 terms for 2x2 matrix multiplication.
        Each term is (u, v, w) where:
          u: 4-vector (coefficients for A flattened row-major)
          v: 4-vector (coefficients for B flattened row-major)
          w: 4-vector (coefficients for C flattened row-major)
        
        The 7 Strassen products:
        M1 = (A00+A11)(B00+B11)
        M2 = (A10+A11)B00
        M3 = A00(B10-B11)
        M4 = A11(B01-B00)
        M5 = (A00+A01)B11
        M6 = (A10-A00)(B00+B01)
        M7 = (A01-A11)(B10+B11)
        
        C00 = M1 - M4 - M6 + M7
        C01 = M3 + M4
        C10 = M2 + M5
        C11 = M1 - M2 + M5 - M3
        """
        # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
        # C = [[C00, C01], [C10, C11]]
        # Flatten row-major: A = [A00, A01, A10, A11], B = [B00, B01, B10, B11], C = [C00, C01, C10, C11]
        
        terms = []
        
        # M1 = (A00+A11)(B00+B11)
        # u: [1, 0, 0, 1] (A00 + A11)
        # v: [1, 0, 0, 1] (B00 + B11)
        # w: [1, 0, 0, 1] (C00 + C11)
        terms.append((np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1])))
        
        # M2 = (A10+A11)B00
        # u: [0, 0, 1, 1] (A10 + A11)
        # v: [1, 0, 0, 0] (B00)
        # w: [0, 0, 1, 0] (C10)
        terms.append((np.array([0, 0, 1, 1]), np.array([1, 0, 0, 0]), np.array([0, 0, 1, 0])))
        
        # M3 = A00(B10-B11)
        # u: [1, 0, 0, 0] (A00)
        # v: [0, 0, 1, -1] (B10 - B11)
        # w: [0, 1, 0, 0] (C01)
        terms.append((np.array([1, 0, 0, 0]), np.array([0, 0, 1, -1]), np.array([0, 1, 0, 0])))
        
        # M4 = A11(B01-B00)
        # u: [0, 0, 0, 1] (A11)
        # v: [0, 1, 0, 0] (B01) ... wait, B01 - B00 = [0-1, 1-0, 0, 0] = [-1, 1, 0, 0]
        # Let me redo: B = [B00, B01, B10, B11]
        # B01 - B00: coefficient of B00 is -1, B01 is 1, B10 is 0, B11 is 0
        # v: [-1, 1, 0, 0]
        # C01 = M3 + M4, C11 = M1 - M2 + M5 - M3, so M4 contributes to C01 and C11
        # C01 gets +M4, C11 gets -M4
        # w: [0, 1, 0, -1]
        terms.append((np.array([0, 0, 0, 1]), np.array([-1, 1, 0, 0]), np.array([0, 1, 0, -1])))
        
        # M5 = (A00+A01)B11
        # u: [1, 1, 0, 0] (A00 + A01)
        # v: [0, 0, 0, 1] (B11)
        # C10 = M2 + M5, C11 = M1 - M2 + M5 - M3, so M5 contributes to C10 and C11
        # w: [0, 0, 1, 1]
        terms.append((np.array([1, 1, 0, 0]), np.array([0, 0, 0, 1]), np.array([0, 0, 1, 1])))
        
        # M6 = (A10-A00)(B00+B01)
        # u: [-1, 0, 1, 0] (A10 - A00)
        # v: [1, 1, 0, 0] (B00 + B01)
        # C00 = M1 - M4 - M6 + M7, so M6 contributes to C00 with -1
        # w: [-1, 0, 0, 0]
        terms.append((np.array([-1, 0, 1, 0]), np.array([1, 1, 0, 0]), np.array([-1, 0, 0, 0])))
        
        # M7 = (A01-A11)(B10+B11)
        # u: [0, 1, 0, -1] (A01 - A11)
        # v: [0, 0, 1, 1] (B10 + B11)
        # C00 = M1 - M4 - M6 + M7, so M7 contributes to C00 with +1
        # w: [1, 0, 0, 0]
        terms.append((np.array([0, 1, 0, -1]), np.array([0, 0, 1, 1]), np.array([1, 0, 0, 0])))
        
        return terms

    def build_decomposition():
        """
        Build a rank-65 decomposition:
        - First 49 terms: double Strassen for the 4x4 x 4x4 part (first 4 columns of B)
        - Next 16 terms: schoolbook for the 4x4 x 1 part (5th column of B)
        
        For the double Strassen part:
        A is 4x4, B_left is 4x4 (first 4 columns), C_left is 4x4 (first 4 columns of C)
        
        We partition A into 2x2 blocks of 2x2:
        A = [[A00, A01], [A10, A11]] where each block is 2x2
        
        Similarly B_left = [[B00, B01], [B10, B11]]
        
        The double Strassen decomposition has 7*7 = 49 terms.
        For outer term (o) and inner term (i):
        - The left linear combination involves: U_outer[o] applied to the 2x2 blocks of A,
          then U_inner[i] applied to the 2x2 entries within each block.
        - Similarly for the right side.
        - The output coefficient is W_outer[o] * W_inner[i] applied to the 2x2 blocks of C,
          then to the 2x2 entries within each block.
        
        Let's construct this carefully.
        
        For the 4x4 matrix A, flattened row-major:
        A = [A[0,0], A[0,1], A[0,2], A[0,3],
             A[1,0], A[1,1], A[1,2], A[1,3],
             A[2,0], A[2,1], A[2,2], A[2,3],
             A[3,0], A[3,1], A[3,2], A[3,3]]
        
        Block A00 = A[0:2, 0:2] = [[A[0,0], A[0,1]], [A[1,0], A[1,1]]]
        Block A01 = A[0:2, 2:4] = [[A[0,2], A[0,3]], [A[1,2], A[1,3]]]
        Block A10 = A[2:4, 0:2] = [[A[2,0], A[2,1]], [A[3,0], A[3,1]]]
        Block A11 = A[2:4, 2:4] = [[A[2,2], A[2,3]], [A[3,2], A[3,3]]]
        
        For each outer term o with 2x2 coefficient matrix U_o:
        L_o = U_o @ A_blocks (matrix multiplication of 2x2 matrices of 2x2 blocks)
        L_o[r,c] = sum_l U_o[r,l] * A_block[l,c]
        
        Then for each inner term i with 2x2 coefficient matrix U_i:
        The effective left combination for the (r,c) block is:
        L_o[r,c] @ ... no, this isn't right.
        
        Actually, the double Strassen works as follows:
        For the outer level, we have 7 terms. For each outer term o:
        - Form L_o (a 4x4 matrix) by applying the outer Strassen to the 2x2 blocks of A
        - Form R_o (a 4x4 matrix) by applying the outer Strassen to the 2x2 blocks of B
        - Compute M_o = L_o * R_o (4x4 matrix product)
        - Add W_o * M_o to C (where W_o is the 2x2 output coefficient matrix)
        
        But this is not a rank-1 decomposition! The product L_o * R_o is a full matrix product.
        
        The key insight: the rank-49 decomposition comes from the tensor product structure.
        Each of the 49 terms corresponds to a pair (o, i) where o is an outer Strassen term
        and i is an inner Strassen term.
        
        For term (o, i):
        - u vector: For each entry A[a,b] of the 4x4 matrix, the coefficient is
          U_o[outer_block_row, outer_block_col] * U_i[inner_row, inner_col]
          where outer_block_row = a // 2, outer_block_col = b // 2,
          inner_row = a % 2, inner_col = b % 2.
        - v vector: Similarly for B.
        - w vector: Similarly for C.
        
        Let me verify this with a simple case.
        
        Actually, I think the correct construction is:
        The 4x4 matrix multiplication tensor can be viewed as a tensor product of two
        2x2 matrix multiplication tensors, with appropriate identifications.
        
        The rank of the tensor product is the product of the ranks: 7 * 7 = 49.
        
        For the tensor product of two rank-r decompositions:
        If T1 = sum_{i=1}^{r} u1_i ⊗ v1_i ⊗ w1_i and T2 = sum_{j=1}^{s} u2_j ⊗ v2_j ⊗ w2_j,
        then T1 ⊗ T2 = sum_{i,j} (u1_i ⊗ u2_j) ⊗ (v1_i ⊗ v2_j) ⊗ (w1_i ⊗ w2_j)
        
        So for our case, each of the 49 terms has:
        u = u_outer ⊗ u_inner (Kronecker product of two 4-vectors = 16-vector)
        v = v_outer ⊗ v_inner
        w = w_outer ⊗ w_inner
        
        But we need to be careful about the ordering. The 4x4 matrix A has entries
        A[a,b] where a = 2*ar + ai, b = 2*bc + bi.
        
        The Kronecker product u_outer ⊗ u_inner, where u_outer has entries indexed by
        (ar, bc) and u_inner has entries indexed by (ai, bi), gives a vector indexed by
        (ar, bc, ai, bi). But we need it indexed by (a, b) = (2*ar+ai, 2*bc+bi).
        
        So we need to reorder: the Kronecker product gives [u_outer[0,0]*u_inner[0,0],
        u_outer[0,0]*u_inner[0,1], u_outer[0,0]*u_inner[1,0], u_outer[0,0]*u_inner[1,1],
        u_outer[0,1]*u_inner[0,0], ...]
        
        But we want [u for (0,0), u for (0,1), u for (0,2), u for (0,3), u for (1,0), ...]
        = [u for (ar=0,ai=0,bc=0,bi=0), u for (ar=0,ai=0,bc=1,bi=0), u for (ar=0,ai=0,bc=0,bi=1), u for (ar=0,ai=0,bc=1,bi=1), ...]
        
        Hmm, this is getting complicated. Let me just construct the u, v, w vectors directly.
        
        For a given outer term o (with 2x2 matrices U_o, V_o, W_o) and inner term i
        (with 2x2 matrices U_i, V_i, W_i):
        
        For each entry A[a,b] of the 4x4 matrix:
        a = 2*ar + ai, b = 2*bc + bi
        The coefficient is U_o[ar, bc] * U_i[ai, bi]
        
        Similarly for B and C.
        
        Let me implement this.
        """
        s2 = strassen_2x2_terms()
        
        U = np.zeros((n * m, 49))
        V = np.zeros((m * p, 49))
        W = np.zeros((p * n, 49))
        
        for oi, (u_o, v_o, w_o) in enumerate(s2):
            for ii, (u_i, v_i, w_i) in enumerate(s2):
                col = oi * 7 + ii
                
                # Construct u vector (length 16) for A
                # u[a,b] = U_o[ar, bc] * U_i[ai, bi]
                # where a = 2*ar + ai, b = 2*bc + bi
                # U_o is flattened row-major: [U_o[0,0], U_o[0,1], U_o[1,0], U_o[1,1]]
                # U_i is flattened row-major: [U_i[0,0], U_i[0,1], U_i[1,0], U_i[1,1]]
                
                Uo = u_o.reshape(2, 2)
                Ui = u_i.reshape(2, 2)
                Vo = v_o.reshape(2, 2)
                Vi = v_i.reshape(2, 2)
                Wo = w_o.reshape(2, 2)
                Wi = w_i.reshape(2, 2)
                
                for a in range(4):
                    for b in range(4):
                        ar = a // 2
                        ai = a % 2
                        bc = b // 2
                        bi = b % 2
                        U[a * m + b, col] = Uo[ar, bc] * Ui[ai, bi]
                
                # Construct v vector (length 20) for B
                # B is 4x5, but for the first 4 columns we use the double Strassen
                # For b in 0..3: same as above
                # For b = 4: this is handled by the schoolbook terms, so v[j*5+4, col] = 0
                for j in range(4):
                    for k in range(4):
                        jr = j // 2
                        ji = j % 2
                        kc = k // 2
                        ki = k % 2
                        V[j * p + k, col] = Vo[jr, kc] * Vi[ji, ki]
                # V[j*p+4, col] remains 0
                
                # Construct w vector (length 20) for C
                # C is 5x4, but for the first 4 rows we use the double Strassen
                # For k in 0..3: same as above
                # For k = 4: this is handled by the schoolbook terms, so W[4*4+i, col] = 0
                for k in range(4):
                    for i in range(4):
                        kr = k // 2
                        ki = k % 2
                        ic = i // 2
                        ii_ = i % 2
                        W[k * n + i, col] = Wo[kr, ic] * Wi[ki, ii_]
                # W[4*n+i, col] remains 0
        
        # Add 16 schoolbook terms for the 5th column of B (k=4)
        for i in range(n):
            for j in range(m):
                col = 49 + i * m + j
                U[i * m + j, col] = 1
                V[j * p + 4, col] = 1
                W[4 * n + i, col] = 1
        
        # Extend U, V, W to have 65 columns
        U_full = np.zeros((n * m, 65))
        V_full = np.zeros((m * p, 65))
        W_full = np.zeros((p * n, 65))
        
        U_full[:, :49] = U
        V_full[:, :49] = V
        W_full[:, :49] = W
        
        U_full[:, 49:] = U[49:65, :] if U.shape[1] >= 65 else np.zeros((n*m, 16))
        # Actually, U only has 49 columns from the Strassen part. The schoolbook terms
        # are in columns 49..64 of the full matrices.
        
        # Let me redo this more carefully.
        U_full = np.zeros((n * m, 65))
        V_full = np.zeros((m * p, 65))
        W_full = np.zeros((p * n, 65))
        
        # First 49 columns: double Strassen
        for oi, (u_o, v_o, w_o) in enumerate(s2):
            for ii, (u_i, v_i, w_i) in enumerate(s2):
                col = oi * 7 + ii
                Uo = u_o.reshape(2, 2)
                Ui = u_i.reshape(2, 2)
                Vo = v_o.reshape(2, 2)
                Vi = v_i.reshape(2, 2)
                Wo = w_o.reshape(2, 2)
                Wi = w_i.reshape(2, 2)
                
                for a in range(4):
                    for b in range(4):
                        ar = a // 2
                        ai = a % 2
                        bc = b // 2
                        bi = b % 2
                        U_full[a * m + b, col] = Uo[ar, bc] * Ui[ai, bi]
                
                for j in range(4):
                    for k in range(4):
                        jr = j // 2
                        ji = j % 2
                        kc = k // 2
                        ki = k % 2
                        V_full[j * p + k, col] = Vo[jr, kc] * Vi[ji, ki]
                
                for k in range(4):
                    for i in range(4):
                        kr = k // 2
                        ki = k % 2
                        ic = i // 2
                        ii_ = i % 2
                        W_full[k * n + i, col] = Wo[kr, ic] * Wi[ki, ii_]
        
        # Next 16 columns: schoolbook for k=4
        for i in range(n):
            for j in range(m):
                col = 49 + i * m + j
                U_full[i * m + j, col] = 1
                V_full[j * p + 4, col] = 1
                W_full[4 * n + i, col] = 1
        
        return {'U': U_full.tolist(), 'V': V_full.tolist(), 'W': W_full.tolist()}

    # Try the Strassen-based decomposition
    try:
        result = build_decomposition()
        if verify(result['U'], result['V'], result['W']):
            return result
    except Exception:
        pass
    
    # Fallback to schoolbook
    result = schoolbook()
    return result
# EVOLVE-BLOCK-END
