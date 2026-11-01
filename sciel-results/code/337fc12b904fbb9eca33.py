"""Strassen-based decomposition for 2x2 * 2x14 matrix multiplication."""
DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 14
    # 7 blocks of 2x2 * 2x2, each using Strassen (rank 7)
    num_blocks = p // 2  # 7 blocks
    assert n == 2 and m == 2, "This solution is specific to 2x2 * 2x14"
    
    rank = num_blocks * 7  # 49
    U = [[0] * rank for _ in range(n * m)]  # 4 x 49
    V = [[0] * rank for _ in range(m * p)]  # 28 x 49
    W = [[0] * rank for _ in range(p * n)]  # 28 x 49
    
    # Strassen's 7 multiplications for 2x2 * 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # For each block b (0..6), we have a 2x2 * 2x2 subproblem
    # A indices: i in {0,1}, j in {0,1} -> A[i,j] at position i*m+j in U
    # B indices: j in {0,1}, k in {0,1} within block -> B[j,k] at position j*p + (2*b+k) in V
    # C indices: k in {0,1}, i in {0,1} within block -> C[k,i] at position (2*b+k)*n + i in W
    
    # Strassen factors for one 2x2 block (7 rank-1 terms):
    # Each term has (U_factor, V_factor, W_factor) where:
    # U_factor[i,j] for i,j in {0,1}
    # V_factor[j,k] for j,k in {0,1}  
    # W_factor[k,i] for k,i in {0,1}
    
    # M1: U = [[1,0],[0,1]], V = [[1,1],[1,1]], W = [[1,1],[1,0]] (C11 gets M1, C22 gets M1)
    # Actually let me be more careful. The rank-1 decomposition for C = A*B:
    # C[i,k] = sum_r U[i,j_r] * V[j_r,k] ... no wait.
    # 
    # The tensor decomposition: C[i,k] = sum_r U[i,j]_r * V[j,k]_r where we sum over j too?
    # No. The standard form: C[i,k] = sum_{r} U[i, r] * V[r, k] where U is (n*m) x rank and V is (m*p) x rank.
    # So U[i*m+j, r] and V[j*p+k, r], and C[i,k] = sum_r sum_j U[i*m+j, r] * V[j*p+k, r].
    # But W[k*n+i, r] is the output factor.
    # 
    # Actually the convention is: the multiplication tensor T[i,j,k] = delta (i.e., C[i,k] = sum_j A[i,j]*B[j,k])
    # decomposes as T = sum_r U_r ⊗ V_r ⊗ W_r where U_r is n*m vector, V_r is m*p vector, W_r is p*n vector.
    # And C[i,k] = sum_r W_r[k*n+i] * (sum_j U_r[i*m+j] * A[i,j]) * (sum_j V_r[j*p+k] * B[j,k])
    # Hmm, that's not quite right either. Let me think again.
    #
    # The matrix multiplication tensor: for A (n×m), B (m×p), C = A*B (n×p)
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # A rank-r decomposition: C[i,k] = sum_{r=1}^{R} sum_j U[i*m+j, r] * A[i,j] * V[j*p+k, r] * ... 
    # No, that's not right. The correct form is:
    # C[i,k] = sum_{r=1}^{R} (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
    # But this only works if the sums over j in U and V are independent, which they're not in general.
    #
    # Actually the standard tensor decomposition for matrix multiplication is:
    # T_{i,j,k} = sum_r U_{i,j}^{(r)} * V_{j,k}^{(r)} * W_{k,i}^{(r)}  -- no, this isn't standard either.
    #
    # Let me look at the schoolbook decomposition in the seed:
    # U[i*m+j][column] = 1, V[j*p+k][column] = 1, W[k*n+i][column] = 1
    # So for each (i,j,k) triple, we have one rank-1 term.
    # The reconstruction: C[i,k] = sum_{r} sum_j U[i*m+j, r] * A[i,j] * V[j*p+k, r] * ... 
    # 
    # Wait, I think the correct interpretation is:
    # C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
    # 
    # No, that's a product of three things which doesn't make sense dimensionally.
    #
    # Let me reconsider. The matrix multiplication can be seen as a bilinear map:
    # (A, B) -> C where C[i,k] = sum_j A[i,j] B[j,k]
    # 
    # A rank-r decomposition of this bilinear map means:
    # C[i,k] = sum_{r=1}^{R} f_r(A) * g_r(B) * h_r[i,k]
    # where f_r(A) = sum_j U[i*m+j, r] * A[i,j] ... but this depends on i which is in the output.
    #
    # I think the correct form based on the seed is:
    # For each rank-1 component r:
    #   f_r(A) = sum_{i,j} U[i*m+j, r] * A[i,j]  (a scalar)
    #   g_r(B) = sum_{j,k} V[j*p+k, r] * B[j,k]  (a scalar)  
    #   h_r[i,k] = W[k*n+i, r]  (the output pattern)
    # And C[i,k] = sum_r f_r(A) * g_r(B) * W[k*n+i, r]
    #
    # Let's verify with schoolbook: for term (i0,j0,k0):
    # f(A) = A[i0,j0], g(B) = B[j0,k0], W[k0*n+i0, r] = 1, W[other] = 0
    # So C[i,k] gets contribution A[i0,j0]*B[j0,k0] only when i=i0 and k=k0. ✓
    
    # So for Strassen, I need to find f_r(A), g_r(B), and W_r such that:
    # C[i,k] = sum_r f_r(A) * g_r(B) * W_r[k*n+i]
    
    # For a 2x2 block (indices local to the block, i,j,k in {0,1}):
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    
    # Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # So for each M_r, f_r(A) and g_r(B) are bilinear forms, and W_r gives the output coefficients.
    
    # M1: f = A00+A11, g = B00+B11
    #   W: C00 gets +1, C11 gets +1
    # M2: f = A10+A11, g = B00
    #   W: C10 gets +1, C11 gets -1
    # M3: f = A00, g = B10-B11
    #   W: C01 gets +1, C11 gets +1
    # M4: f = A11, g = B01-B00
    #   W: C00 gets +1, C10 gets +1
    # M5: f = A00+A01, g = B11
    #   W: C00 gets -1, C01 gets +1
    # M6: f = A10-A00, g = B00+B01
    #   W: C11 gets +1
    # M7: f = A01-A11, g = B10+B11
    #   W: C00 gets +1
    
    # Now map to global indices for block b:
    # A[i,j] global: i in {0,1}, j in {0,1} -> position i*m + j = i*2 + j in U (rows 0..3)
    # B[j,k] global: j in {0,1}, k_local in {0,1}, k_global = 2*b + k_local -> position j*p + k_global = j*14 + 2*b + k_local in V (rows 0..27)
    # C[k,i] global: k_local in {0,1}, k_global = 2*b + k_local, i in {0,1} -> position k_global*n + i = (2*b+k_local)*2 + i in W (rows 0..27)
    
    # Strassen terms (f_coeffs, g_coeffs, w_coeffs):
    # f_coeffs: dict from (i,j) to coefficient for A[i,j]
    # g_coeffs: dict from (j,k_local) to coefficient for B[j,k_local]
    # w_coeffs: dict from (k_local, i) to coefficient for C[k_local, i]
    
    strassen_terms = [
        # M1
        ([(0,0,1),(1,1,1)], [(0,0,1),(0,1,1),(1,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        # M2
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
        # M3
        ([(0,0,1)], [(1,0,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
        # M4
        ([(1,1,1)], [(0,1,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
        # M5
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
        # M6
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        # M7
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]
    
    for b in range(num_blocks):
        for t_idx, (f_coeffs, g_coeffs, w_coeffs) in enumerate(strassen_terms):
            r = b * 7 + t_idx  # rank index
            # Set U
            for (i, j, coeff) in f_coeffs:
                U[i * m + j][r] = coeff
            # Set V
            for (j, k_local, coeff) in g_coeffs:
                k_global = 2 * b + k_local
                V[j * p + k_global][r] = coeff
            # Set W
            for (k_local, i, coeff) in w_coeffs:
                k_global = 2 * b + k_local
                W[k_global * n + i][r] = coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
