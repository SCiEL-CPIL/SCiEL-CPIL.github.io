"""Hardcoded rank-18 exact tensor decomposition for 2x2 * 2x5 matrix multiplication."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 2, 5
    # Rank 18: Strassen(7) for cols 0-1, Strassen(7) for cols 3-4, Schoolbook(4) for col 2.
    rank = 18
    # U has shape (n*m, rank) = (4, 18)
    # V has shape (m*p, rank) = (10, 18)
    # W has shape (p*n, rank) = (10, 18)
    # Indexing:
    # U: A[i,j] at i*m + j
    # V: B[j,k] at j*p + k
    # W: C[k,i] at k*n + i  (Official trace convention W[k*n+i, r])

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # --- Part 1: Strassen for columns k=0, 1 ---
    # Let B01 be the 2x2 submatrix of B for columns 0,1.
    # B01[0,0]=B[0,0], B01[0,1]=B[0,1]
    # B01[1,0]=B[1,0], B01[1,1]=B[1,1]
    # Strassen variables:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    #
    # Results for C = A * B01:
    # C00 = M1 + M4 - M6 + M7
    # C01 = M3 + M5 + M6 - M7
    # C10 = M2 + M4 - M3 + M5
    # C11 = M1 + M2 - M4 + M6
    
    # Column 0 corresponds to C[:,0] which involves B[:,0]
    # Wait, Strassen computes a full 2x2 multiplication.
    # Here we multiply 2x2 A by 2x2 B01 to get 2x2 C01.
    # C01[:,0] is the first column of the result, C01[:,1] is the second.
    
    # Let's map the Strassen terms to U, V, W.
    # Each term is a rank-1 update: U_col * V_col^T * W_col^T? No, sum of outer products.
    # Term t: U_t (vector of A coeffs), V_t (vector of B coeffs), W_t (vector of C coeffs).
    # The contribution to C is sum_t (W_t) * (U_t^T A) * (V_t^T B)? 
    # Actually, the tensor decomposition is T = sum_t U_t \otimes V_t \otimes W_t.
    # C_{ki} = sum_{j,l} A_{ij} B_{jl} T_{k,i,j,l}? No.
    # The standard form is C_{ki} = sum_{j,l} A_{ij} B_{jl}.
    # The decomposition is:
    # sum_t U_t[i,j] V_t[j,l] W_t[k,i] = T_{i,j,k,l} (where T is the identity tensor for multiplication)
    # Wait, the task defines W[k*n+i, r]. So W index is (k, i).
    # U index is (i, j). V index is (j, k).
    # The equation is: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = delta_{i,i'} delta_{k,k'}? No.
    # It must reproduce the multiplication.
    # Let's just use the known formulas for Strassen.
    # For 2x2 * 2x2 -> 2x2:
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # C = [[a*e+b*g, a*f+b*h], [c*e+d*g, c*f+d*h]]
    # Strassen:
    # p1 = (a+d)(e+h)
    # p2 = (c+d)e
    # p3 = a(f-h)
    # p4 = d(g-e)
    # p5 = (a+b)h
    # p6 = (c-a)(e+f)
    # p7 = (b-d)(g+h)
    # C00 = p1 + p4 - p6 + p7
    # C01 = p3 + p5 + p6 - p7
    # C10 = p2 + p4 - p3 + p5
    # C11 = p1 + p2 - p4 + p6

    def add_strassen(U, V, W, base_col, b_col_offset):
        # b_col_offset is the column index in B for the start of the 2x2 block.
        # Here we are processing columns k0 and k1.
        # B entries involved: B[0, k0], B[0, k1], B[1, k0], B[1, k1]
        # Indices in V:
        # B[0, k0] is at V[0*p + k0, r]
        # B[0, k1] is at V[0*p + k1, r]
        # B[1, k0] is at V[1*p + k0, r]
        # B[1, k1] is at V[1*p + k1, r]
        
        # A entries:
        # A[0,0] at U[0, r]
        # A[0,1] at U[1, r]
        # A[1,0] at U[2, r]
        # A[1,1] at U[3, r]
        
        # C entries (result of A * B_block):
        # The block is columns k0, k1 of B.
        # Result C_block is 2x2.
        # C_block[0,0] corresponds to C[0, k0] -> W[k0*n + 0, r]
        # C_block[0,1] corresponds to C[0, k1] -> W[k1*n + 0, r]
        # C_block[1,0] corresponds to C[1, k0] -> W[k0*n + 1, r]
        # C_block[1,1] corresponds to C[1, k1] -> W[k1*n + 1, r]
        
        # p1 = (a+d)(e+h)
        # U: a=1, d=1
        # V: e=1, h=1
        # W: C00, C11 contribute
        # C00 = ... + p1 ...
        # C11 = ... + p1 ...
        r = base_col + 0
        U[0][r] += 1
        U[3][r] += 1
        V[0*p + b_col_offset][r] += 1
        V[1*p + b_col_offset + 1][r] += 1
        W[b_col_offset*n + 0][r] += 1
        W[(b_col_offset+1)*n + 1][r] += 1
        
        # p2 = (c+d)e
        r = base_col + 1
        U[2][r] += 1
        U[3][r] += 1
        V[0*p + b_col_offset][r] += 1
        # C10, C11
        W[b_col_offset*n + 1][r] += 1
        W[(b_col_offset+1)*n + 1][r] += 1

        # p3 = a(f-h)
        r = base_col + 2
        U[0][r] += 1
        V[0*p + b_col_offset + 1][r] += 1
        V[1*p + b_col_offset + 1][r] += -1
        # C01, C10
        W[(b_col_offset+1)*n + 0][r] += 1
        W[b_col_offset*n + 1][r] += -1

        # p4 = d(g-e)
        r = base_col + 3
        U[3][r] += 1
        V[1*p + b_col_offset][r] += 1
        V[0*p + b_col_offset][r] += -1
        # C00, C10, C11
        W[b_col_offset*n + 0][r] += 1
        W[b_col_offset*n + 1][r] += 1
        W[(b_col_offset+1)*n + 1][r] += -1

        # p5 = (a+b)h
        r = base_col + 4
        U[0][r] += 1
        U[1][r] += 1
        V[1*p + b_col_offset + 1][r] += 1
        # C01, C10
        W[(b_col_offset+1)*n + 0][r] += 1
        W[b_col_offset*n + 1][r] += 1

        # p6 = (c-a)(e+f)
        r = base_col + 5
        U[2][r] += 1
        U[0][r] += -1
        V[0*p + b_col_offset][r] += 1
        V[0*p + b_col_offset + 1][r] += 1
        # C00, C01
        W[b_col_offset*n + 0][r] += -1
        W[(b_col_offset+1)*n + 0][r] += 1

        # p7 = (b-d)(g+h)
        r = base_col + 6
        U[1][r] += 1
        U[3][r] += -1
        V[1*p + b_col_offset][r] += 1
        V[1*p + b_col_offset + 1][r] += 1
        # C00, C01
        W[b_col_offset*n + 0][r] += 1
        W[(b_col_offset+1)*n + 0][r] += -1

    # Apply Strassen for k=0,1 (offset 0)
    add_strassen(U, V, W, 0, 0)
    
    # Apply Strassen for k=3,4 (offset 3)
    add_strassen(U, V, W, 7, 3)

    # --- Part 3: Schoolbook for column k=2 ---
    # Terms:
    # s0 = A00 B02
    # s1 = A01 B12
    # s2 = A10 B02
    # s3 = A11 B12
    # C02 = s0 + s1
    # C12 = s2 + s3
    
    base_col = 14
    # s0
    r = base_col + 0
    U[0][r] = 1
    V[0*p + 2][r] = 1
    W[2*n + 0][r] = 1
    
    # s1
    r = base_col + 1
    U[1][r] = 1
    V[1*p + 2][r] = 1
    W[2*n + 0][r] = 1
    
    # s2
    r = base_col + 2
    U[2][r] = 1
    V[0*p + 2][r] = 1
    W[2*n + 1][r] = 1
    
    # s3
    r = base_col + 3
    U[3][r] = 1
    V[1*p + 2][r] = 1
    W[2*n + 1][r] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
