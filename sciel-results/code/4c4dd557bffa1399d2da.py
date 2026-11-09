DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=3
    rank = 17
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a rank-1 term
    # a_idx: index in U (i*m + j)
    # b_idx: index in V (j*p + k)
    # c_idx: index in W (k*n + i)
    # coeff: coefficient
    def add_term(a_idx, b_idx, c_idx, coeff=1):
        nonlocal col
        U[a_idx][col] += coeff
        V[b_idx][col] += coeff
        W[c_idx][col] += coeff
        col += 1

    # Part 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # Local indices i, j, k in 0..1
    # Global A: i, j
    # Global B: j, k
    # Global C: i, k
    
    # Strassen M1: (A00+A11)(B00+B11) -> C00, C11
    # u: A00, A11
    # v: B00, B11
    # w: C00, C11
    # We need to represent this as sum of rank-1 tensors?
    # No, Strassen's algorithm decomposes the tensor into 7 terms.
    # Each term r has u_r, v_r, w_r.
    # Term 1:
    # u: A00, A11
    # v: B00, B11
    # w: C00, C11
    # This is NOT a single rank-1 tensor. A rank-1 tensor is u_scalar * v_scalar * w_scalar.
    # Strassen's algorithm says T = sum_{r=1}^7 u_r \otimes v_r \otimes w_r.
    # Here u_r is a vector in A-space, v_r in B-space, w_r in C-space.
    # So for r=1:
    # u_1 has 1 at A00, 1 at A11.
    # v_1 has 1 at B00, 1 at B11.
    # w_1 has 1 at C00, 1 at C11.
    # This corresponds to ONE column in U, V, W.
    
    # Let's define the 7 Strassen terms for 2x2x2.
    # Indices: A(i,j), B(j,k), C(i,k)
    # M1: (A00+A11)(B00+B11). C00+=1, C11+=1.
    # M2: (A10+A11)B00. C10+=1.
    # M3: A00(B10-B11). C01+=1.
    # M4: A11(B01-B00). C00+=1, C10+=1.
    # M5: (A00+A10)B11. C01+=1, C00-=1.
    # M6: (A11-A00)(B00+B10). C11+=1.
    # M7: (A01-A11)(B11+B01). C00+=1, C01-=1.
    
    # Wait, let's verify the reconstruction.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Term 1:
    # u: A00(1), A11(1)
    # v: B00(1), B11(1)
    # w: C00(1), C11(1)
    add_term(0*3+0, 0*3+0, 0*2+0, 1) # A00, B00, C00
    add_term(1*3+1, 0*3+0, 0*2+0, 1) # A11, B00, C00 -- Wait, this is not how it works.
    
    # Correction:
    # A rank-1 tensor component is defined by:
    # U[row_a][col] = coeff_a
    # V[row_b][col] = coeff_b
    # W[row_c][col] = coeff_c
    # The contribution to C[c] is U[a]*V[b]*W[c]? No.
    # The tensor T = sum_r U_r \otimes V_r \otimes W_r.
    # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r].
    # So for one column r, we set U[a,r], V[b,r], W[c,r].
    # If u_r has 1 at A00 and A11, then U[0, r]=1 and U[3, r]=1 (assuming A11 is index 1*3+1=4? No, A is 2x3. A11 is i=1, j=1. Index 1*3+1=4).
    # So for Term 1:
    # U[0, r] = 1 (A00)
    # U[4, r] = 1 (A11)
    # V[0, r] = 1 (B00)
    # V[4, r] = 1 (B11) -- B is 3x3. B11 is j=1, k=1. Index 1*3+1=4.
    # W[0, r] = 1 (C00) -- C is 2x3. C00 is i=0, k=0. Index 0*2+0=0.
    # W[2, r] = 1 (C11) -- C11 is i=1, k=1. Index 1*2+1=3? No.
    # W index is k*n + i.
    # C00: k=0, i=0 -> 0.
    # C11: k=1, i=1 -> 1*2+1=3.
    
    # So we just need to fill the matrices for each of the 7 terms.
    
    # Term 1
    r = col
    U[0][r] = 1; U[4][r] = 1
    V[0][r] = 1; V[4][r] = 1
    W[0][r] = 1; W[3][r] = 1
    col += 1
    
    # Term 2: (A10+A11)B00 -> C10
    # u: A10(1), A11(1)
    # v: B00(1)
    # w: C10(1)
    # A10: i=1, j=0 -> 3. A11: i=1, j=1 -> 4.
    # B00: j=0, k=0 -> 0.
    # C10: i=1, k=0 -> 1*2+0=2.
    r = col
    U[3][r] = 1; U[4][r] = 1
    V[0][r] = 1
    W[2][r] = 1
    col += 1
    
    # Term 3: A00(B10-B11) -> C01
    # u: A00(1)
    # v: B10(1), B11(-1)
    # w: C01(1)
    # A00: 0.
    # B10: j=1, k=0 -> 3. B11: j=1, k=1 -> 4.
    # C01: i=0, k=1 -> 1*2+0=2.
    r = col
    U[0][r] = 1
    V[3][r] = 1; V[4][r] = -1
    W[2][r] = 1
    col += 1
    
    # Term 4: A11(B01-B00) -> C00, C10
    # u: A11(1)
    # v: B01(1), B00(-1)
    # w: C00(1), C10(1)
    # A11: 4.
    # B01: j=0, k=1 -> 1. B00: 0.
    # C00: 0. C10: 2.
    r = col
    U[4][r] = 1
    V[1][r] = 1; V[0][r] = -1
    W[0][r] = 1; W[2][r] = 1
    col += 1
    
    # Term 5: (A00+A10)B11 -> C01, C00(-1)
    # u: A00(1), A10(1)
    # v: B11(1)
    # w: C01(1), C00(-1)
    # A00: 0. A10: 3.
    # B11: 4.
    # C01: 2. C00: 0.
    r = col
    U[0][r] = 1; U[3][r] = 1
    V[4][r] = 1
    W[2][r] = 1; W[0][r] = -1
    col += 1
    
    # Term 6: (A11-A00)(B00+B10) -> C11
    # u: A11(1), A00(-1)
    # v: B00(1), B10(1)
    # w: C11(1)
    # A11: 4. A00: 0.
    # B00: 0. B10: 3.
    # C11: 3.
    r = col
    U[4][r] = 1; U[0][r] = -1
    V[0][r] = 1; V[3][r] = 1
    W[3][r] = 1
    col += 1
    
    # Term 7: (A01-A11)(B11+B01) -> C00, C01(-1)
    # u: A01(1), A11(-1)
    # v: B11(1), B01(1)
    # w: C00(1), C01(-1)
    # A01: i=0, j=1 -> 1. A11: 4.
    # B11: 4. B01: 1.
    # C00: 0. C01: 2.
    r = col
    U[1][r] = 1; U[4][r] = -1
    V[4][r] = 1; V[1][r] = 1
    W[0][r] = 1; W[2][r] = -1
    col += 1
    
    # Part 2: Outer Product A[0:2, 2] * B[2, 0:2] -> C[0:2, 0:2]
    # A_i2 * B_2k -> C_ik
    # i in 0,1; k in 0,1
    # A indices: i*3+2.
    # B indices: 2*3+k.
    # C indices: k*2+i.
    
    for i in range(2):
        for k in range(2):
            r = col
            a_idx = i*3+2
            b_idx = 2*3+k
            c_idx = k*2+i
            U[a_idx][r] = 1
            V[b_idx][r] = 1
            W[c_idx][r] = 1
            col += 1
            
    # Part 3: A[0:2, 0:2] * B[0:2, 2] -> C[0:2, 2]
    # A_ij * B_j2 -> C_i2
    # i in 0,1; j in 0,1
    # A indices: i*3+j.
    # B indices: j*3+2.
    # C indices: 2*2+i.
    
    for i in range(2):
        for j in range(2):
            r = col
            a_idx = i*3+j
            b_idx = j*3+2
            c_idx = 2*2+i
            U[a_idx][r] = 1
            V[b_idx][r] = 1
            W[c_idx][r] = 1
            col += 1
            
    # Part 4: A[0:2, 2] * B[2, 2] -> C[0:2, 2]
    # A_i2 * B_22 -> C_i2
    # i in 0,1
    # A indices: i*3+2.
    # B indices: 2*3+2.
    # C indices: 2*2+i.
    
    for i in range(2):
        r = col
        a_idx = i*3+2
        b_idx = 2*3+2
        c_idx = 2*2+i
        U[a_idx][r] = 1
        V[b_idx][r] = 1
        W[c_idx][r] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
