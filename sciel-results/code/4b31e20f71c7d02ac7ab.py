import numpy as np

def _strassen_2x2_decomp():
    """
    Returns U, V, W for 2x2x2 matrix multiplication with rank 7.
    U shape (4, 7), V shape (4, 7), W shape (4, 7).
    Indices: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r].
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Strassen coefficients for M1..M7
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # We need to expand each Ms = Xs Ys into rank-1 tensors.
    # Xs and Ys are 2x2 matrices.
    # Xs = sum_{i,j} x_{ij} A_{ij}
    # Ys = sum_{j,k} y_{jk} B_{jk}
    # Product Xs Ys = sum_{i,j,k} x_{ij} y_{jk} A_{ij} B_{jk}
    # This is a linear combination of the 8 scalar multiplications A_{ij}B_{jk}.
    # Wait, the tensor is sum_{i,j,k} A_{ij} B_{jk} C_{ki}.
    # The rank-1 decomposition of M_{2,2,2} is known.
    # Let's just use the standard rank-7 decomposition vectors.
    
    # The 7 rank-1 tensors for 2x2x2 are given by:
    # u_r, v_r, w_r vectors in R^4.
    # C = sum_r (u_r . A^T) (v_r . B) w_r^T ? No.
    # C_{ki} = sum_r U_{i, r} V_{j, r} W_{k, r} A_{ij} B_{jk}?
    # The convention is U[i*m+j, r], V[j*p+k, r], W[k*n+i, r].
    # For 2x2x2: n=2, m=2, p=2.
    # U shape (4, 7), V shape (4, 7), W shape (4, 7).
    
    # Standard Strassen decomposition vectors:
    # r=0 (M1):
    # A part: A00 + A11 -> [1, 0, 0, 1] (i,j: 00, 01, 10, 11)
    # B part: B00 + B11 -> [1, 0, 0, 1] (j,k: 00, 01, 10, 11)
    # C part: C00 + C11 - ... wait, the C part is just the output of the multiplication.
    # The rank-1 tensor is U \otimes V \otimes W.
    # The contribution to C is (U . A) (V . B) W^T?
    # Actually, the tensor is T = sum_r U_r \otimes V_r \otimes W_r.
    # T_{ijk} = sum_r U_{ij, r} V_{jk, r} W_{ki, r}.
    # We need T_{ijk} = delta_{i,i} delta_{j,j} delta_{k,k}? No.
    # The multiplication tensor is T_{ijk} = 1 if the indices match the multiplication, 0 otherwise?
    # No, the tensor for matrix multiplication M_{n,m,p} is defined by T_{i,j,k} = 1 if the term A_{ij} B_{jk} contributes to C_{ki}.
    # Actually, T_{i,j,k} is the coefficient of A_{ij} B_{jk} in C_{ki}.
    # For matrix multiplication, C_{ki} = sum_j A_{ij} B_{jk}.
    # So T_{i,j,k} = 1 for all i,j,k? No.
    # The tensor is a 3-tensor. T_{i,j,k} = 1 if the indices are valid?
    # Yes, the matrix multiplication tensor is the sum of n*m*p rank-1 tensors e_i \otimes e_j \otimes e_k.
    # Wait, that's the schoolbook.
    # Strassen gives a different decomposition.
    
    # Let's find the explicit U, V, W for Strassen 2x2.
    # Reference: Strassen (1969).
    # The 7 terms are:
    # 1. (A11+A22)(B11+B22) -> C11 += ..., C22 += ...
    # This is tricky to map directly to U,V,W vectors without deriving.
    
    # Alternative: Use the known explicit vectors.
    # From literature, the rank-7 decomposition of 2x2x2 is:
    # U, V, W are 4x7 matrices.
    
    # Let's construct them from the formulas.
    # Each Ms is a product of two 2x2 matrices.
    # M_s = X_s Y_s.
    # X_s = sum_{i,j} x_{s,ij} A_{ij}
    # Y_s = sum_{j,k} y_{s,jk} B_{jk}
    # M_s = sum_{i,j,k} x_{s,ij} y_{s,jk} A_{ij} B_{jk}
    # This sum is a linear combination of the 8 scalar products A_{ij} B_{jk}.
    # But we need rank-1 tensors for the whole 2x2x2 multiplication.
    # The multiplication C = A B is:
    # C_{ki} = sum_j A_{ij} B_{jk}
    # We can write C = sum_{r=1}^7 (U_r^T A V_r) W_r ? No.
    
    # Let's use the property that Strassen's algorithm computes 7 scalar multiplications.
    # Each scalar multiplication is of the form (linear form in A) * (linear form in B).
    # Let L_A(s) and L_B(s) be the linear forms.
    # The result is added to C with coefficients.
    # This is exactly the tensor decomposition.
    # T = sum_s L_A(s) \otimes L_B(s) \otimes L_C(s)
    # Where L_C(s) is the linear form in C that collects the result.
    
    # For 2x2:
    # s=1: L_A = A00+A11, L_B = B00+B11, L_C = C00+C11 (coeff 1) ... wait.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # So for each s, we have a term L_A(s) \otimes L_B(s) \otimes L_C(s).
    # But L_A(s) is a vector in R^4 (coeffs of A00, A01, A10, A11).
    # L_B(s) is a vector in R^4 (coeffs of B00, B01, B10, B11).
    # L_C(s) is a vector in R^4 (coeffs of C00, C01, C10, C11).
    
    # Let's list them:
    # s=1:
    # L_A: [1, 0, 0, 1] (A00, A01, A10, A11)
    # L_B: [1, 0, 0, 1] (B00, B01, B10, B11)
    # L_C: C00 += 1, C11 += 1. So [1, 0, 0, 1] (C00, C01, C10, C11)
    
    # s=2:
    # L_A: [0, 0, 1, 1] (A10, A11)
    # L_B: [1, 0, 0, 0] (B00)
    # L_C: C10 += 1, C11 -= 1. So [0, 0, 1, -1]
    
    # s=3:
    # L_A: [1, 0, 0, 0] (A00)
    # L_B: [0, 1, 0, -1] (B01 - B11)
    # L_C: C01 += 1, C11 += 1. So [0, 1, 0, 1]
    
    # s=4:
    # L_A: [0, 0, 0, 1] (A11)
    # L_B: [0, 0, 1, -1] (B10 - B00) -> [ -1, 0, 1, 0 ]?
    # B10 is index 2, B00 is index 0. So [ -1, 0, 1, 0 ].
    # L_C: C00 += 1, C10 += 1. So [1, 0, 1, 0]
    
    # s=5:
    # L_A: [1, 1, 0, 0] (A00 + A01)
    # L_B: [0, 0, 0, 1] (B11)
    # L_C: C00 -= 1, C01 += 1. So [-1, 1, 0, 0]
    
    # s=6:
    # L_A: [-1, 0, 1, 0] (A10 - A00)
    # L_B: [1, 1, 0, 0] (B00 + B01)
    # L_C: C11 += 1. So [0, 0, 0, 1]
    
    # s=7:
    # L_A: [0, 1, 0, -1] (A01 - A11)
    # L_B: [0, 0, 1, 1] (B10 + B11)
    # L_C: C00 += 1. So [1, 0, 0, 0]

    # Check:
    # C00: s1(1) + s4(1) - s5(1) + s7(1) = 1+1-1+1 = 2?
    # Wait, C00 = M1 + M4 - M5 + M7.
    # My L_C for s5 was [-1, 1, 0, 0]. So coeff for C00 is -1.
    # C00 coeff: 1 (s1) + 1 (s4) - 1 (s5) + 1 (s7) = 2.
    # But C00 should be sum_j A0j Bj0 = A00 B00 + A01 B10.
    # Let's check the expansion of M1..M7.
    # M1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11
    # M4 = A11(B10-B00) = A11B10 - A11B00
    # M5 = (A00+A01)B11 = A00B11 + A01B11
    # M7 = (A01-A11)(B10+B11) = A01B10 + A01B11 - A11B10 - A11B11
    # Sum for C00: M1 + M4 - M5 + M7
    # = (A00B00 + A00B11 + A11B00 + A11B11) + (A11B10 - A11B00) - (A00B11 + A01B11) + (A01B10 + A01B11 - A11B10 - A11B11)
    # A00B00: 1
    # A00B11: 1 - 1 = 0
    # A11B00: 1 - 1 = 0
    # A11B11: 1 - 1 = 0
    # A11B10: 1 - 1 = 0
    # A01B11: -1 + 1 = 0
    # A01B10: 1
    # Result: A00B00 + A01B10. Correct.
    
    # So the vectors are correct.
    U = np.array([
        [1, 0, 0, 1], # s1
        [0, 0, 1, 1], # s2
        [1, 0, 0, 0], # s3
        [0, 0, 0, 1], # s4
        [1, 1, 0, 0], # s5
        [-1, 0, 1, 0],# s6
        [0, 1, 0, -1] # s7
    ]).T # Shape (4, 7)
    
    V = np.array([
        [1, 0, 0, 1], # s1
        [1, 0, 0, 0], # s2
        [0, 1, 0, -1],# s3
        [-1, 0, 1, 0],# s4
        [0, 0, 0, 1], # s5
        [1, 1, 0, 0], # s6
        [0, 0, 1, 1]  # s7
    ]).T # Shape (4, 7)
    
    W = np.array([
        [1, 0, 0, 1], # s1
        [0, 0, 1, -1],# s2
        [0, 1, 0, 1], # s3
        [1, 0, 1, 0], # s4
        [-1, 1, 0, 0],# s5
        [0, 0, 0, 1], # s6
        [1, 0, 0, 0]  # s7
    ]).T # Shape (4, 7)
    
    return U, V, W

def construct():
    n, m, p = 4, 4, 8
    # We use two 4x4x4 multiplications.
    # Each 4x4x4 is decomposed using Strassen 2x2 blocks.
    # Rank of 4x4x4 is 49.
    # Total rank 98.
    
    U_222, V_222, W_222 = _strassen_2x2_decomp()
    
    # Build 4x4x4 decomposition
    # U_444 shape (16, 49)
    # V_444 shape (16, 49)
    # W_444 shape (16, 49)
    
    U_444 = np.zeros((16, 49))
    V_444 = np.zeros((16, 49))
    W_444 = np.zeros((16, 49))
    
    # Strassen 4x4 formulas
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Coefficients alpha[s, I, K]
    # s in 1..7, I in 0..1, K in 0..1
    # C_{KI} = sum_s alpha[s, I, K] M_s
    
    alpha = np.zeros((7, 2, 2))
    # s1: C00(1), C11(1)
    alpha[0, 0, 0] = 1
    alpha[0, 1, 1] = 1
    # s2: C10(1), C11(-1)
    alpha[1, 0, 1] = 1
    alpha[1, 1, 1] = -1
    # s3: C01(1), C11(1)
    alpha[2, 1, 0] = 1
    alpha[2, 1, 1] = 1
    # s4: C00(1), C10(1)
    alpha[3, 0, 0] = 1
    alpha[3, 0, 1] = 1
    # s5: C00(-1), C01(1)
    alpha[4, 0, 0] = -1
    alpha[4, 1, 0] = 1
    # s6: C11(1)
    alpha[5, 1, 1] = 1
    # s7: C00(1)
    alpha[6, 0, 0] = 1
    
    # X_s, Y_s coefficients
    # X_s = sum_{I,J} x_{s, I, J} A_{IJ}
    # Y_s = sum_{J,K} y_{s, J, K} B_{JK}
    
    x_coeff = np.zeros((7, 2, 2))
    y_coeff = np.zeros((7, 2, 2))
    
    # s1: X = A00+A11, Y = B00+B11
    x_coeff[0, 0, 0] = 1; x_coeff[0, 1, 1] = 1
    y_coeff[0, 0, 0] = 1; y_coeff[0, 1, 1] = 1
    # s2: X = A10+A11, Y = B00
    x_coeff[1, 0, 1] = 1; x_coeff[1, 1, 1] = 1
    y_coeff[1, 0, 0] = 1
    # s3: X = A00, Y = B01-B11
    x_coeff[2, 0, 0] = 1
    y_coeff[2, 0, 1] = 1; y_coeff[2, 1, 1] = -1
    # s4: X = A11, Y = B10-B00
    x_coeff[3, 1, 1] = 1
    y_coeff[3, 1, 0] = 1; y_coeff[3, 0, 0] = -1
    # s5: X = A00+A01, Y = B11
    x_coeff[4, 0, 0] = 1; x_coeff[4, 0, 1] = 1
    y_coeff[4, 1, 1] = 1
    # s6: X = A10-A00, Y = B00+B01
    x_coeff[5, 0, 1] = 1; x_coeff[5, 0, 0] = -1
    y_coeff[5, 0, 0] = 1; y_coeff[5, 0, 1] = 1
    # s7: X = A01-A11, Y = B10+B11
    x_coeff[6, 0, 1] = 1; x_coeff[6, 1, 1] = -1
    y_coeff[6, 1, 0] = 1; y_coeff[6, 1, 1] = 1
    
    # Construct U_444, V_444, W_444
    # Index mapping:
    # A_{IJ} has local indices u, v. Global i = 2I+u, j = 2J+v.
    # B_{JK} has local indices v, w. Global j = 2J+v, k = 2K+w.
    # C_{KI} has local indices w, u. Global k = 2K+w, i = 2I+u.
    
    # U_444[i, r] where i = 2I+u, r = 7*s + t (t in 0..6)
    # V_444[j, r] where j = 2J+v, r = 7*s + t
    # W_444[k, r] where k = 2K+w, r = 7*s + t
    
    for s in range(7):
        for t in range(7):
            r = s * 7 + t
            # u, v, w from U_222, V_222, W_222
            # U_222[u*2+v, t] is coeff for A_uv in L_A(s)
            # V_222[v*2+w, t] is coeff for B_vw in L_B(s)
            # W_222[w*2+u, t] is coeff for C_wu in L_C(s)
            
            for I in range(2):
                for J in range(2):
                    for K in range(2):
                        # Check if x_coeff[s, I, J] and y_coeff[s, J, K] and alpha[s, I, K] are non-zero
                        if x_coeff[s, I, J] != 0 and y_coeff[s, J, K] != 0 and alpha[s, I, K] != 0:
                            for u in range(2):
                                for v in range(2):
                                    for w in range(2):
                                        u_val = U_222[u*2+v, t]
                                        v_val = V_222[v*2+w, t]
                                        w_val = W_222[w*2+u, t]
                                        
                                        if u_val != 0 and v_val != 0 and w_val != 0:
                                            i = 2*I + u
                                            j = 2*J + v
                                            k = 2*K + w
                                            
                                            coeff = x_coeff[s, I, J] * y_coeff[s, J, K] * alpha[s, I, K] * u_val * v_val * w_val
                                            
                                            U_444[i, r] += coeff
                                            V_444[j, r] += coeff
                                            W_444[k, r] += coeff
    
    # Now build U, V, W for 4x4x8
    # U shape (16, 98)
    # V shape (32, 98)
    # W shape (32, 98)
    
    U = np.zeros((16, 98))
    V = np.zeros((32, 98))
    W = np.zeros((32, 98))
    
    # Part 1: B[0:4, 0:4], C[0:4, :]
    U[:, 0:49] = U_444
    V[0:16, 0:49] = V_444
    W[0:16, 0:49] = W_444
    
    # Part 2: B[0:4, 4:8], C[4:8, :]
    U[:, 49:98] = U_444
    V[16:32, 49:98] = V_444
    W[16:32, 49:98] = W_444
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
