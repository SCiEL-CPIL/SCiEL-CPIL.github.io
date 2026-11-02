import numpy as np

def construct():
    n, m, p = 2, 4, 8
    
    # Step 1: Construct 2x2 x 2x4 decomposition (Strassen per 2x2 block)
    # A is 2x2, B is 2x4, C is 2x4
    # U shape: (4, 14), V shape: (8, 14), W shape: (8, 14)
    
    # Strassen for 2x2 x 2x2 (block 1, columns 0,1)
    # M1 = (a+d)(e+j)
    # M2 = (c+d)e
    # M3 = a(f-j)
    # M4 = d(i-e)
    # M5 = (a+b)j
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(i+j)
    
    # For 2x2 x 2x4, we have two 2x2 blocks of B.
    # Block 1: columns 0,1 of B
    # Block 2: columns 2,3 of B
    
    # Let me define the 14 products for 2x2 x 2x4.
    # A = [[a, b], [c, d]]
    # B = [[e, f, g, h], [i, j, k, l]]
    
    # Block 1 (e,f,i,j):
    # P1 = (a+d)(e+j)
    # P2 = (c+d)e
    # P3 = a(f-j)
    # P4 = d(i-e)
    # P5 = (a+b)j
    # P6 = (c-a)(e+f)
    # P7 = (b-d)(i+j)
    
    # Block 2 (g,h,k,l):
    # P8 = (a+d)(g+l)
    # P9 = (c+d)g
    # P10 = a(h-l)
    # P11 = d(k-g)
    # P12 = (a+b)l
    # P13 = (c-a)(g+h)
    # P14 = (b-d)(k+l)
    
    # C[0,0] = P1 - P4 - P5 + P7
    # C[0,1] = P3 + P5
    # C[1,0] = P2 + P4
    # C[1,1] = P1 - P2 + P3 - P6
    # C[0,2] = P8 - P11 - P12 + P14
    # C[0,3] = P10 + P12
    # C[1,2] = P9 + P11
    # C[1,3] = P8 - P9 + P10 - P13
    
    # Now construct U, V, W for 2x2 x 2x4.
    # U has shape (4, 14) where rows are [a, b, c, d]
    # V has shape (8, 14) where rows are [e, f, g, h, i, j, k, l]
    # W has shape (8, 14) where rows are [C[0,0], C[0,1], C[0,2], C[0,3], C[1,0], C[1,1], C[1,2], C[1,3]]
    
    # Wait, the W convention for the sub-problem. The sub-problem is 2x2 x 2x4, so n_sub=2, p_sub=4.
    # W[k*n_sub+i, r] = W[k*2+i, r]. So W rows are indexed by (k, i) where k is column, i is row.
    # W[0] = C[0,0], W[1] = C[1,0], W[2] = C[0,1], W[3] = C[1,1], W[4] = C[0,2], W[5] = C[1,2], W[6] = C[0,3], W[7] = C[1,3]
    
    U_sub = np.zeros((4, 14))
    V_sub = np.zeros((8, 14))
    W_sub = np.zeros((8, 14))
    
    # P1 = (a+d)(e+j)
    U_sub[0, 0] = 1  # a
    U_sub[2, 0] = 1  # d
    V_sub[0, 0] = 1  # e
    V_sub[5, 0] = 1  # j
    # W for P1: contributes to C[0,0] (+1), C[1,1] (+1)
    # C[0,0] is W[0], C[1,1] is W[3]
    W_sub[0, 0] = 1
    W_sub[3, 0] = 1
    
    # P2 = (c+d)e
    U_sub[1, 1] = 1  # c  -- wait, c is index 2 in [a,b,c,d]
    # Let me redo: U rows are [a, b, c, d] = indices 0,1,2,3
    U_sub[2, 1] = 1  # c
    U_sub[3, 1] = 1  # d
    V_sub[0, 1] = 1  # e
    # W for P2: contributes to C[1,0] (+1), C[1,1] (-1)
    # C[1,0] is W[1], C[1,1] is W[3]
    W_sub[1, 1] = 1
    W_sub[3, 1] = -1
    
    # P3 = a(f-j)
    U_sub[0, 2] = 1  # a
    V_sub[1, 2] = 1  # f
    V_sub[5, 2] = -1  # j
    # W for P3: contributes to C[0,1] (+1), C[1,1] (+1)
    # C[0,1] is W[2], C[1,1] is W[3]
    W_sub[2, 2] = 1
    W_sub[3, 2] = 1
    
    # P4 = d(i-e)
    U_sub[3, 3] = 1  # d
    V_sub[4, 3] = 1  # i
    V_sub[0, 3] = -1  # e
    # W for P4: contributes to C[0,0] (-1), C[1,0] (+1)
    W_sub[0, 3] = -1
    W_sub[1, 3] = 1
    
    # P5 = (a+b)j
    U_sub[0, 4] = 1  # a
    U_sub[1, 4] = 1  # b
    V_sub[5, 4] = 1  # j
    # W for P5: contributes to C[0,0] (-1), C[0,1] (+1)
    W_sub[0, 4] = -1
    W_sub[2, 4] = 1
    
    # P6 = (c-a)(e+f)
    U_sub[2, 5] = 1  # c
    U_sub[0, 5] = -1  # a
    V_sub[0, 5] = 1  # e
    V_sub[1, 5] = 1  # f
    # W for P6: contributes to C[1,1] (-1)
    W_sub[3, 5] = -1
    
    # P7 = (b-d)(i+j)
    U_sub[1, 6] = 1  # b
    U_sub[3, 6] = -1  # d
    V_sub[4, 6] = 1  # i
    V_sub[5, 6] = 1  # j
    # W for P7: contributes to C[0,0] (+1)
    W_sub[0, 6] = 1
    
    # P8 = (a+d)(g+l)
    U_sub[0, 7] = 1  # a
    U_sub[3, 7] = 1  # d
    V_sub[2, 7] = 1  # g
    V_sub[7, 7] = 1  # l
    # W for P8: contributes to C[0,2] (+1), C[1,3] (+1)
    # C[0,2] is W[4], C[1,3] is W[7]
    W_sub[4, 7] = 1
    W_sub[7, 7] = 1
    
    # P9 = (c+d)g
    U_sub[2, 8] = 1  # c
    U_sub[3, 8] = 1  # d
    V_sub[2, 8] = 1  # g
    # W for P9: contributes to C[1,2] (+1), C[1,3] (-1)
    # C[1,2] is W[5], C[1,3] is W[7]
    W_sub[5, 8] = 1
    W_sub[7, 8] = -1
    
    # P10 = a(h-l)
    U_sub[0, 9] = 1  # a
    V_sub[3, 9] = 1  # h
    V_sub[7, 9] = -1  # l
    # W for P10: contributes to C[0,3] (+1), C[1,3] (+1)
    # C[0,3] is W[6], C[1,3] is W[7]
    W_sub[6, 9] = 1
    W_sub[7, 9] = 1
    
    # P11 = d(k-g)
    U_sub[3, 10] = 1  # d
    V_sub[6, 10] = 1  # k
    V_sub[2, 10] = -1  # g
    # W for P11: contributes to C[0,2] (-1), C[1,2] (+1)
    # C[0,2] is W[4], C[1,2] is W[5]
    W_sub[4, 10] = -1
    W_sub[5, 10] = 1
    
    # P12 = (a+b)l
    U_sub[0, 11] = 1  # a
    U_sub[1, 11] = 1  # b
    V_sub[7, 11] = 1  # l
    # W for P12: contributes to C[0,2] (-1), C[0,3] (+1)
    # C[0,2] is W[4], C[0,3] is W[6]
    W_sub[4, 11] = -1
    W_sub[6, 11] = 1
    
    # P13 = (c-a)(g+h)
    U_sub[2, 12] = 1  # c
    U_sub[0, 12] = -1  # a
    V_sub[2, 12] = 1  # g
    V_sub[3, 12] = 1  # h
    # W for P13: contributes to C[1,3] (-1)
    W_sub[7, 12] = -1
    
    # P14 = (b-d)(k+l)
    U_sub[1, 13] = 1  # b
    U_sub[3, 13] = -1  # d
    V_sub[6, 13] = 1  # k
    V_sub[7, 13] = 1  # l
    # W for P14: contributes to C[0,2] (+1)
    W_sub[4, 13] = 1
    
    # Now construct 1x2 x 2x2 decomposition (rank 4)
    # A is 1x2: [a, b]
    # B is 2x2: [[e, f], [g, h]]
    # C is 1x2: [c0, c1] where c0 = a*e + b*g, c1 = a*f + b*h
    # W convention: W[k*n+i, r] where n=1, so W[k*1+i, r] = W[k, r]. So W[0] = c0, W[1] = c1.
    
    U_t = np.zeros((2, 4))
    V_t = np.zeros((4, 4))
    W_t = np.zeros((2, 4))
    
    # Term 1: a*e -> c0
    U_t[0, 0] = 1  # a
    V_t[0, 0] = 1  # e
    W_t[0, 0] = 1  # c0
    
    # Term 2: a*f -> c1
    U_t[0, 1] = 1  # a
    V_t[1, 1] = 1  # f
    W_t[1, 1] = 1  # c1
    
    # Term 3: b*g -> c0
    U_t[1, 2] = 1  # b
    V_t[2, 2] = 1  # g
    W_t[0, 2] = 1  # c0
    
    # Term 4: b*h -> c1
    U_t[1, 3] = 1  # b
    V_t[3, 3] = 1  # h
    W_t[1, 3] = 1  # c1
    
    # Now take Kronecker product of the two decompositions.
    # T = T_sub ⊗ T_t
    # U = U_sub ⊗ U_t (Kronecker product of matrices)
    # V = V_sub ⊗ V_t
    # W = W_sub ⊗ W_t
    
    # But wait, the Kronecker product of the tensors: if T_sub = Σ_r u_sub_r ⊗ v_sub_r ⊗ w_sub_r and T_t = Σ_s u_t_s ⊗ v_t_s ⊗ w_t_s, then T = Σ_{r,s} (u_sub_r ⊗ u_t_s) ⊗ (v_sub_r ⊗ v_t_s) ⊗ (w_sub_r ⊗ w_t_s).
    
    # The Kronecker product of matrices U_sub (4x14) and U_t (2x4) gives a matrix of shape (4*2, 14*4) = (8, 56).
    # Similarly for V and W.
    
    U = np.kron(U_sub, U_t)
    V = np.kron(V_sub, V_t)
    W = np.kron(W_sub, W_t)
    
    # Now we need to verify that this gives the correct 2x4 x 4x8 multiplication.
    # The Kronecker product structure:
    # A (2x4) = A_sub (2x2) ⊗ A_t (1x2)? Let's check: (2x2)⊗(1x2) = 2x4 ✓
    # B (4x8) = B_sub (2x4) ⊗ B_t (2x2)? (2x4)⊗(2x2) = 4x8 ✓
    # C (2x8) = C_sub (2x4) ⊗ C_t (1x2)? (2x4)⊗(1x2) = 2x8 ✓
    
    # The Kronecker product of matrices: (A_sub ⊗ A_t) · (B_sub ⊗ B_t) = (A_sub · B_sub) ⊗ (A_t · B_t) = C_sub ⊗ C_t = C.
    
    # So the decomposition is correct IF the Kronecker product structure matches.
    
    # But we need to make sure the indexing is correct. The Kronecker product U_sub ⊗ U_t has entries:
    # (U_sub ⊗ U_t)[i1*2 + i2, r1*4 + r2] = U_sub[i1, r1] * U_t[i2, r2]
    
    # This corresponds to the Kronecker product of the index spaces.
    
    # Let me verify with a random example.
    
    np.random.seed(42)
    A = np.random.randn(2, 4)
    B = np.random.randn(4, 8)
    C_expected = A @ B
    
    # Reshape A into Kronecker form: A = A_sub ⊗ A_t
    # A is 2x4. A_sub is 2x2, A_t is 1x2.
    # A[i, j] where i in {0,1}, j in {0,1,2,3}.
    # j = j1*2 + j2 where j1 in {0,1}, j2 in {0,1}.
    # A[i, j1*2+j2] = A_sub[i, j1] * A_t[0, j2]
    
    # Actually, the Kronecker product (A_sub ⊗ A_t) has entries:
    # (A_sub ⊗ A_t)[i1*1 + i2, j1*2 + j2] = A_sub[i1, j1] * A_t[i2, j2]
    # Since A_t is 1x2, i2 is always 0.
    # (A_sub ⊗ A_t)[i1, j1*2 + j2] = A_sub[i1, j1] * A_t[0, j2]
    
    # So A[i, j] = A_sub[i, j//2] * A_t[0, j%2]
    
    # Let's decompose A:
    A_sub = np.zeros((2, 2))
    A_t = np.zeros((1, 2))
    for i in range(2):
        for j in range(4):
            j1, j2 = j // 2, j % 2
            A_sub[i, j1] = A[i, j]
            A_t[0, j2] = 1  # This doesn't work in general
    
    # Hmm, the Kronecker product decomposition of A into A_sub ⊗ A_t is not always possible with A_t = [1, 1].
    # Actually, A = A_sub ⊗ A_t means A[i, j1*2+j2] = A_sub[i, j1] * A_t[0, j2].
    # For this to hold, A must have a specific structure. In general, A does NOT have this structure.
    
    # So the Kronecker product approach doesn't directly work for arbitrary A and B.
    # The Kronecker product of tensors works for the tensor itself, but the input matrices A and B don't need to be Kronecker products.
    
    # Actually, I think I'm confusing myself. The tensor decomposition is for the multiplication tensor, not for the matrices A and B.
    
    # The multiplication tensor M has entries M[i, j, k] = 1 if C[i, k] depends on A[i, j] and B[j, k].
    # For 2x4 x 4x8: M is a tensor of shape (2*4, 4*8, 2*8) = (8, 32, 16) but with the trace convention, the output index is k*n+i.
    
    # The Kronecker product of two multiplication tensors: if M1 is the tensor for 2x2 x 2x4 and M2 is the tensor for 1x2 x 2x2, then M1 ⊗ M2 is the tensor for (2x2 x 2x4) ⊗ (1x2 x 2x2) = 2x4 x 4x8.
    
    # The key insight: the Kronecker product of the tensors gives the correct multiplication tensor, regardless of whether A and B are Kronecker products.
    
    # So the decomposition U = U_sub ⊗ U_t, V = V_sub ⊗ V_t, W = W_sub ⊗ W_t should be correct.
    
    # Let me verify by checking that U @ diag(intermediate) @ V^T gives the correct C.
    
    # Actually, the decomposition means: C = Σ_r (U[:, r] * V[:, r]) * W[:, r] (outer product).
    # More precisely: C_flat = Σ_r U_flat[:, r] * V_flat[:, r] * W_flat[:, r] (element-wise).
    # Wait, no. The tensor decomposition is: M = Σ_r u_r ⊗ v_r ⊗ w_r.
    # This means M[a, b, c] = Σ_r u_r[a] * v_r[b] * w_r[c].
    # And C[c] = Σ_{a,b} M[a, b, c] * A_flat[a] * B_flat[b].
    # = Σ_c Σ_r w_r[c] * (Σ_a u_r[a] * A_flat[a]) * (Σ_b v_r[b] * B_flat[b])
    # = Σ_r (U @ A_flat) * (V @ B_flat) * W[:, r] (where the last term is broadcast).
    
    # So C_flat = (U @ A_flat) * (V @ B_flat) @ W^T (element-wise product, then sum over r).
    
    # Let me verify this.
    
    A_flat = A.flatten()  # shape (8,)
    B_flat = B.flatten()  # shape (32,)
    
    # U @ A_flat: shape (56,)
    # V @ B_flat: shape (56,)
    # Element-wise product: shape (56,)
    # @ W^T: shape (16,)
    
    C_flat = ((U @ A_flat) * (V @ B_flat)) @ W.T
    C_computed = C_flat.reshape(2, 8, order='F')  # trace convention: k*n+i
    
    # Wait, the trace convention W[k*n+i, r] means the output index is k*n+i where k is column, i is row.
    # So C_flat[k*n+i] = C[i, k].
    # C_computed[k*n+i] should equal C_expected[i, k].
    
    # Let me check:
    C_expected_flat = np.zeros(16)
    for k in range(8):
        for i in range(2):
            C_expected_flat[k*2+i] = C_expected[i, k]
    
    print("Max error:", np.max(np.abs(C_flat - C_expected_flat)))
    
    # Convert to lists for the return value
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
