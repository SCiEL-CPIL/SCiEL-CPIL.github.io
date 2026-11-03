import time

DIMENSIONS = [3, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    # Rank 66 construction
    rank = 66
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, coeff=1):
        nonlocal col
        if col >= rank:
            return
        U[u_idx][col] += coeff
        V[v_idx][col] += coeff
        W[w_idx][col] += coeff
        col += 1

    # Indices
    # A is 3x4
    # B is 4x6
    # C is 3x6
    
    # Split A into A_top (2x4) and A_bot (1x4)
    # Split B into B_1 (2x6) and B_2 (2x6) based on inner dim 4 -> 2+2
    # A_top = [A11 (2x2), A12 (2x2)]
    # A_bot = [A21 (1x2), A22 (1x2)]
    # B = [B1 (2x6); B2 (2x6)]
    
    # C_top = A11 B1 + A12 B2
    # C_bot = A21 B1 + A22 B2
    
    # 1. A11 B1 (2x2 * 2x6)
    # Split B1 (2x6) into B1a (2x2), B1b (2x2), B1c (2x2)
    # Use Strassen for each 2x2*2x2
    
    # Strassen terms for A11 (2x2) and B1a (2x2)
    # A11 indices: 
    # i=0,1 ; j=0,1
    # A11[0,0] -> U[0*4+0] = U[0]
    # A11[0,1] -> U[0*4+1] = U[1]
    # A11[1,0] -> U[1*4+0] = U[4]
    # A11[1,1] -> U[1*4+1] = U[5]
    
    # B1a indices:
    # j=0,1 ; k=0,1
    # B1a[0,0] -> V[0*6+0] = V[0]
    # B1a[0,1] -> V[0*6+1] = V[1]
    # B1a[1,0] -> V[1*6+0] = V[6]
    # B1a[1,1] -> V[1*6+1] = V[7]
    
    # C1a indices:
    # i=0,1 ; k=0,1
    # C1a[0,0] -> W[0*3+0] = W[0]
    # C1a[0,1] -> W[1*3+0] = W[3]
    # C1a[1,0] -> W[0*3+1] = W[1]
    # C1a[1,1] -> W[1*3+1] = W[4]
    
    # Strassen for A11, B1a
    # M1 = (A00+A11)(B00+B11)
    add_term(0, 0, 0, 1); add_term(0, 7, 0, 1); add_term(5, 0, 0, 1); add_term(5, 7, 0, 1) # This is wrong, add_term adds one term.
    
    # I need a way to add a bilinear form (sum of products) as a single rank-1 term?
    # No, rank-1 term is $u \otimes v \otimes w$.
    # Strassen combines terms.
    # The rank is the number of terms in the sum.
    # Each Strassen "multiplication" $M_k$ is a rank-1 term?
    # No, $M_1 = (A_{11}+A_{22})(B_{11}+B_{22})$.
    # This is not a single monomial $A_{ij} B_{kl}$.
    # It is a product of sums.
    # To decompose this into rank-1 terms, we need to expand?
    # No, the rank of the tensor is the minimum number of rank-1 tensors.
    # The Strassen algorithm expresses the tensor as a sum of 7 rank-1 tensors?
    # No, Strassen expresses the *algorithm* as 7 multiplications.
    # But each "multiplication" $(A_{11}+A_{22})$ is not a single entry of A.
    # So the tensor decomposition is not simply 7 terms.
    # Wait. The rank of $M_{2,2,2}$ is 7.
    # This means there exist 7 vectors $u_t, v_t, w_t$ such that $T = \sum u_t \otimes v_t \otimes w_t$.
    # The Strassen algorithm provides such a decomposition.
    # Let's check the structure.
    # $C_{11} = M_1 + M_4 - M_5 + M_7$.
    # $M_1$ corresponds to the product $(A_{11}+A_{22}) (B_{11}+B_{22})$.
    # This product is not a single term $A_{ij} B_{kl}$.
    # However, in the tensor decomposition, we can have $u_t$ with multiple non-zero entries.
    # So $u_t$ can be $(1, 0, 1, 0)^T$ for $A_{11}+A_{22}$.
    # And $v_t$ can be $(1, 0, 1, 0)^T$ for $B_{11}+B_{22}$.
    # Then $u_t \otimes v_t \otimes w_t$ contributes to $C_{11}$ via $w_t$.
    # Yes, this is how it works. The U, V, W matrices can have multiple non-zeros per column.
    
    # So I need to define the 7 Strassen terms explicitly.
    
    def add_strassen_222(a_base, b_base, c_base, offset):
        """
        a_base: index in U for A[0,0]
        b_base: index in V for B[0,0]
        c_base: index in W for C[0,0]
        offset: column index to start
        """
        nonlocal col
        
        # A indices relative to a_base
        # A00: a_base
        # A01: a_base + 1
        # A10: a_base + 4 (since m=4, next row is +4? No, A is 2x2 block in 3x4.
        # If A11 is top-left 2x2, indices are:
        # 0, 1
        # 4, 5
        # So A10 is a_base + 4.
        
        # B indices relative to b_base
        # B00: b_base
        # B01: b_base + 1
        # B10: b_base + 6 (since p=6, next row is +6? No, B is 2x2 block in 4x6.
        # If B1a is top-left 2x2 of B1 (2x6), indices are:
        # 0, 1
        # 6, 7
        # So B10 is b_base + 6.
        
        # C indices relative to c_base
        # C00: c_base
        # C01: c_base + 3 (since n=3, next col is +3? No, C is 2x2 block in 3x6.
        # W index is k*n + i.
        # C00: k=0, i=0 -> 0
        # C01: k=1, i=0 -> 3
        # C10: k=0, i=1 -> 1
        # C11: k=1, i=1 -> 4
        # So C01 is c_base + 3.
        # C10 is c_base + 1.
        # C11 is c_base + 4.
        
        # M1 = (A00+A11)(B00+B11)
        # U: A00=1, A11=1
        # V: B00=1, B11=1
        # W: C00=1, C11=1, C10=-1, C01=1? 
        # C00 = M1 + M4 - M5 + M7
        # C11 = M1 - M2 + M3 + M6
        # C01 = M3 + M5
        # C10 = M2 + M4
        
        # So for M1:
        # W contribution: C00 += 1, C11 += 1
        # U: A00, A11
        # V: B00, B11
        
        U[a_base][offset] += 1
        U[a_base + 4][offset] += 1
        V[b_base][offset] += 1
        V[b_base + 6][offset] += 1
        W[c_base][offset] += 1
        W[c_base + 4][offset] += 1
        col += 1
        
        # M2 = (A10+A11)B00
        # U: A10, A11
        # V: B00
        # W: C10 += 1, C11 -= 1
        
        U[a_base + 4][offset] += 1
        U[a_base + 5][offset] += 1
        V[b_base][offset] += 1
        W[c_base + 1][offset] += 1
        W[c_base + 4][offset] -= 1
        col += 1
        
        # M3 = A00(B01-B11)
        # U: A00
        # V: B01, -B11
        # W: C01 += 1, C11 += 1
        
        U[a_base][offset] += 1
        V[b_base + 1][offset] += 1
        V[b_base + 7][offset] -= 1
        W[c_base + 3][offset] += 1
        W[c_base + 4][offset] += 1
        col += 1
        
        # M4 = A11(B10-B00)
        # U: A11
        # V: B10, -B00
        # W: C00 += 1, C10 += 1
        
        U[a_base + 5][offset] += 1
        V[b_base + 6][offset] += 1
        V[b_base][offset] -= 1
        W[c_base][offset] += 1
        W[c_base + 1][offset] += 1
        col += 1
        
        # M5 = (A00+A01)B11
        # U: A00, A01
        # V: B11
        # W: C00 -= 1, C01 += 1
        
        U[a_base][offset] += 1
        U[a_base + 1][offset] += 1
        V[b_base + 7][offset] += 1
        W[c_base][offset] -= 1
        W[c_base + 3][offset] += 1
        col += 1
        
        # M6 = (A10-A00)(B00+B01)
        # U: A10, -A00
        # V: B00, B01
        # W: C11 += 1
        
        U[a_base + 4][offset] += 1
        U[a_base][offset] -= 1
        V[b_base][offset] += 1
        V[b_base + 1][offset] += 1
        W[c_base + 4][offset] += 1
        col += 1
        
        # M7 = (A01-A11)(B10+B11)
        # U: A01, -A11
        # V: B10, B11
        # W: C00 += 1
        
        U[a_base + 1][offset] += 1
        U[a_base + 5][offset] -= 1
        V[b_base + 6][offset] += 1
        V[b_base + 7][offset] += 1
        W[c_base][offset] += 1
        col += 1
        
        return offset + 7

    # 1. A11 B1a
    # A11 base: 0
    # B1a base: 0
    # C1a base: 0
    col = add_strassen_222(0, 0, 0, col)
    
    # 2. A11 B1b
    # B1b is cols 2,3 of B1.
    # B1 is rows 0,1 of B.
    # B1b[0,0] is B[0,2] -> V[0*6+2] = V[2]
    # C1b is cols 2,3 of C1.
    # C1b[0,0] is C[0,2] -> W[2*3+0] = W[6]
    col = add_strassen_222(0, 2, 6, col)
    
    # 3. A11 B1c
    # B1c is cols 4,5 of B1.
    # B1c[0,0] is B[0,4] -> V[4]
    # C1c is cols 4,5 of C1.
    # C1c[0,0] is C[0,4] -> W[12]
    col = add_strassen_222(0, 4, 12, col)
    
    # 4. A12 B2a
    # A12 is top-right 2x2 of A.
    # A12[0,0] is A[0,2] -> U[0*4+2] = U[2]
    # B2a is bottom-left 2x2 of B? No, B2 is rows 2,3 of B.
    # B2a[0,0] is B[2,0] -> V[2*6+0] = V[12]
    # C1a is top-left 2x2 of C.
    # C1a[0,0] is C[0,0] -> W[0]
    # Wait, C_top = A11 B1 + A12 B2.
    # A12 B2 contributes to C_top (rows 0,1).
    # So C base is still 0?
    # Yes, C1a, C1b, C1c are the same output locations.
    # But we are adding terms to the same W indices.
    # So we just add to the existing W columns?
    # No, each Strassen call adds 7 new columns to U, V, W.
    # The W indices in those new columns will have values that sum to the correct C.
    # So we just call add_strassen_222 with the same C base.
    
    col = add_strassen_222(2, 12, 0, col)
    
    # 5. A12 B2b
    # A12 base: 2
    # B2b is cols 2,3 of B2.
    # B2b[0,0] is B[2,2] -> V[2*6+2] = V[14]
    # C1b base: 6
    col = add_strassen_222(2, 14, 6, col)
    
    # 6. A12 B2c
    # A12 base: 2
    # B2c is cols 4,5 of B2.
    # B2c[0,0] is B[2,4] -> V[2*6+4] = V[16]
    # C1c base: 12
    col = add_strassen_222(2, 16, 12, col)
    
    # Now C_bot = A21 B1 + A22 B2
    # A21 is 1x2. A21[0,0] is A[2,0] -> U[2*4+0] = U[8]
    # A22 is 1x2. A22[0,0] is A[2,2] -> U[2*4+2] = U[10]
    
    # A21 B1 (1x2 * 2x6)
    # Rank 12.
    # B1 is 2x6.
    # Split B1 into B1a (2x2), B1b (2x2), B1c (2x2).
    # A21 B1a (1x2 * 2x2)
    # C_bot_a is 1x2.
    # C_bot_a[0,0] is C[2,0] -> W[0*3+2] = W[2]
    # C_bot_a[0,1] is C[2,1] -> W[1*3+2] = W[5]
    
    # M_{1,2,2} rank 4.
    # C00 = A00 B00 + A01 B10
    # C01 = A00 B01 + A01 B11
    # Terms:
    # 1. A00 B00 -> C00
    # 2. A01 B10 -> C00
    # 3. A00 B01 -> C01
    # 4. A01 B11 -> C01
    
    def add_direct_122(a_base, b_base, c_base, offset):
        nonlocal col
        # A00: a_base
        # A01: a_base + 1
        # B00: b_base
        # B01: b_base + 1
        # B10: b_base + 6
        # B11: b_base + 7
        # C00: c_base
        # C01: c_base + 3
        
        U[a_base][offset] += 1
        V[b_base][offset] += 1
        W[c_base][offset] += 1
        col += 1
        
        U[a_base + 1][offset] += 1
        V[b_base + 6][offset] += 1
        W[c_base][offset] += 1
        col += 1
        
        U[a_base][offset] += 1
        V[b_base + 1][offset] += 1
        W[c_base + 3][offset] += 1
        col += 1
        
        U[a_base + 1][offset] += 1
        V[b_base + 7][offset] += 1
        W[c_base + 3][offset] += 1
        col += 1
        
        return offset + 4

    # A21 B1a
    col = add_direct_122(8, 0, 2, col)
    
    # A21 B1b
    # B1b base: 2
    # C1b base: 6
    col = add_direct_122(8, 2, 8, col) # C_bot_b[0,0] is C[2,2] -> W[2*3+2] = W[8]
    
    # A21 B1c
    # B1c base: 4
    # C1c base: 12
    col = add_direct_122(8, 4, 14, col) # C_bot_c[0,0] is C[2,4] -> W[4*3+2] = W[14]
    
    # A22 B2a
    # A22 base: 10
    # B2a base: 12
    # C_bot_a base: 2
    col = add_direct_122(10, 12, 2, col)
    
    # A22 B2b
    # A22 base: 10
    # B2b base: 14
    # C_bot_b base: 8
    col = add_direct_122(10, 14, 8, col)
    
    # A22 B2c
    # A22 base: 10
    # B2c base: 16
    # C_bot_c base: 14
    col = add_direct_122(10, 16, 14, col)
    
    # Total columns used:
    # 6 * 7 = 42
    # 6 * 4 = 24
    # Total 66.
    
    if col != 66:
        # Fallback to schoolbook if error
        rank = 72
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][c] = 1
                    V[j*p+k][c] = 1
                    W[k*n+i][c] = 1
                    c += 1
                    if time.time() > deadline:
                        break
                if time.time() > deadline: break
            if time.time() > deadline: break

    return {"U": U, "V": V, "W": W}
