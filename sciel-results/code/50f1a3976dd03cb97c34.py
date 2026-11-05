import time

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 4, 7
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # We don't know rank in advance, so we collect terms first.
    terms = []
    
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        terms.append((u_idx, v_idx, w_idx, u_val, v_val, w_val))

    def add_strassen_2x2(a_offsets, b_offsets, c_offsets):
        # a_offsets: (top_left, top_right, bot_left, bot_right) flattened A indices
        # b_offsets: (top_left, top_right, bot_left, bot_right) flattened B indices
        # c_offsets: (top_left, top_right, bot_left, bot_right) flattened C indices
        a_tl, a_tr, a_bl, a_br = a_offsets
        b_tl, b_tr, b_bl, b_br = b_offsets
        c_tl, c_tr, c_bl, c_br = c_offsets
        
        # M1 = (A11+A22)(B11+B22)
        # A part: a_tl, a_br. B part: b_tl, b_br.
        # This requires 4 rank-1 terms if we expand? No, Strassen combines them.
        # The decomposition U, V, W works such that:
        # U has the linear form in A. V has linear form in B. W has linear form in C.
        # For M1:
        # L_A = A11 + A22. L_B = B11 + B22.
        # U: a_tl:1, a_br:1
        # V: b_tl:1, b_br:1
        # W: c_tl:1, c_tr:1, c_bl:1, c_br:-1 (Wait, C = M1 - M2 - M4 + M7? No.)
        
        # Standard Strassen:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # C11 = M1 - M4 + M6
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M3 + M7
        
        # We add 7 terms.
        
        # Term 1: M1
        # U: a_tl(1), a_br(1)
        # V: b_tl(1), b_br(1)
        # W: c_tl(1), c_tr(0), c_bl(0), c_br(0) -- Wait, W distributes to C.
        # C11 gets +M1. C22 gets +M1.
        # So W: c_tl(1), c_br(1).
        add_term(a_tl, b_tl, c_tl, 1, 1, 1)
        add_term(a_tl, b_br, c_tl, 1, 1, 1)
        add_term(a_br, b_tl, c_tl, 1, 1, 1)
        add_term(a_br, b_br, c_tl, 1, 1, 1)
        add_term(a_tl, b_tl, c_br, 1, 1, 1)
        add_term(a_tl, b_br, c_br, 1, 1, 1)
        add_term(a_br, b_tl, c_br, 1, 1, 1)
        add_term(a_br, b_br, c_br, 1, 1, 1)
        # This is 8 terms for M1? No.
        # The rank-1 tensor decomposition allows U to be a vector, V to be a vector.
        # M1 = (a_tl+a_br)(b_tl+b_br).
        # This is not rank 1 in the U,V,W sense unless we expand.
        # (a_tl+a_br)(b_tl+b_br) = a_tl b_tl + a_tl b_br + a_br b_tl + a_br b_br.
        # Each product is a rank-1 tensor?
        # Yes.
        # So M1 contributes 4 rank-1 terms.
        # This defeats the purpose of Strassen?
        # No, Strassen reduces the number of *multiplications* of linear forms.
        # In the U,V,W decomposition, each column is a multiplication of a linear form in A and a linear form in B.
        # If we use a column for M1, U is (1,1) at a_tl, a_br. V is (1,1) at b_tl, b_br.
        # Then U * V is a scalar.
        # But W must map this scalar to C.
        # C11 += 1 * (a_tl+a_br)(b_tl+b_br).
        # C22 += 1 * (a_tl+a_br)(b_tl+b_br).
        # So W is 1 at c_tl and c_br.
        # This is ONE column (rank 1) that adds to C11 and C22.
        # Yes! This is how it works.
        
        # Let's restart the term addition logic.
        terms.clear()
        
        def add_strassen_term(u_map, v_map, w_map):
            # u_map: dict {idx: val}, v_map: dict {idx: val}, w_map: dict {idx: val}
            # This adds ONE column to the decomposition.
            u_vec = [0] * (n*m)
            v_vec = [0] * (m*p)
            w_vec = [0] * (p*n)
            for idx, val in u_map.items():
                u_vec[idx] = val
            for idx, val in v_map.items():
                v_vec[idx] = val
            for idx, val in w_map.items():
                w_vec[idx] = val
            terms.append((u_vec, v_vec, w_vec))

        # M1: (A11+A22)(B11+B22)
        # U: {a_tl: 1, a_br: 1}
        # V: {b_tl: 1, b_br: 1}
        # W: {c_tl: 1, c_br: 1}
        add_strassen_term({a_tl: 1, a_br: 1}, {b_tl: 1, b_br: 1}, {c_tl: 1, c_br: 1})
        
        # M2: (A21+A22)B11
        # U: {a_bl: 1, a_br: 1}
        # V: {b_tl: 1}
        # W: {c_bl: 1} (C21 gets M2)
        add_strassen_term({a_bl: 1, a_br: 1}, {b_tl: 1}, {c_bl: 1})
        
        # M3: A11(B12-B22)
        # U: {a_tl: 1}
        # V: {b_tr: 1, b_br: -1}
        # W: {c_tr: 1} (C12 gets M3)
        add_strassen_term({a_tl: 1}, {b_tr: 1, b_br: -1}, {c_tr: 1})
        
        # M4: A22(B21-B11)
        # U: {a_br: 1}
        # V: {b_bl: 1, b_tl: -1}
        # W: {c_bl: 1, c_tl: -1} (C21 gets M4, C11 gets -M4)
        add_strassen_term({a_br: 1}, {b_bl: 1, b_tl: -1}, {c_bl: 1, c_tl: -1})
        
        # M5: (A11+A12)B22
        # U: {a_tl: 1, a_tr: 1}
        # V: {b_br: 1}
        # W: {c_tr: 1} (C12 gets M5)
        add_strassen_term({a_tl: 1, a_tr: 1}, {b_br: 1}, {c_tr: 1})
        
        # M6: (A21-A11)(B11+B12)
        # U: {a_bl: 1, a_tl: -1}
        # V: {b_tl: 1, b_tr: 1}
        # W: {c_tl: 1} (C11 gets M6)
        add_strassen_term({a_bl: 1, a_tl: -1}, {b_tl: 1, b_tr: 1}, {c_tl: 1})
        
        # M7: (A12-A22)(B21+B22)
        # U: {a_tr: 1, a_br: -1}
        # V: {b_bl: 1, b_br: 1}
        # W: {c_br: 1} (C22 gets M7)
        add_strassen_term({a_tr: 1, a_br: -1}, {b_bl: 1, b_br: 1}, {c_br: 1})
        
        def add_schoolbook_row(a_row, b_cols, c_row):
            # a_row: list of 4 A indices
            # b_cols: list of B indices for a specific column? No.
            # For M_{1, 4, k}, we compute 1 row of C from 1 row of A and 4xk of B.
            # C_k = sum_j A_j B_{jk}
            # For each output column k, we have 4 terms.
            # If we process k separately, rank is 4*k.
            # Can we combine?
            # C1 = A1 B11 + A2 B21 + A3 B31 + A4 B41
            # C2 = A1 B12 + A2 B22 + A3 B32 + A4 B42
            # Rank 8.
            # Is it reducible?
            # The tensor is sum_{j,k} A_j B_{jk} E_{k}.
            # Rank is 4*k for M_{1, 4, k}.
            
            # So for M_{1, 4, 4}, rank 16?
            # Wait, earlier I said rank 4.
            # Let's check M_{1, 2, 2}.
            # C1 = A1 B11 + A2 B21
            # C2 = A1 B12 + A2 B22
            # Rank 4.
            # Is it 2?
            # If rank 2, then C1, C2 are linear combos of 2 products.
            # It is known that R(1, n, n) = n.
            # Wait. R(1, n, n) = n?
            # For n=2, R=4?
            # Let's check the literature.
            # The rank of M_{1, n, n} is n.
            # Counter-example?
            # C1 = A1 B11 + A2 B21
            # C2 = A1 B12 + A2 B22
            # If R=2, we can write:
            # C1 = L1 M1 + L2 M2
            # C2 = L3 M1 + L4 M2
            # This seems hard.
            # Actually, R(1, n, n) = n is correct for the *bilinear* rank?
            # No, R(1, n, n) = n is the rank of the matrix multiplication *operation* as a bilinear map?
            # Yes.
            
            # So M_{1, 4, 4} rank 16?
            # No, R(1, 4, 4) = 4?
            # Let's assume R(1, n, n) = n.
            # Then M_{1, 4, 4} rank 4.
            # How?
            # We can use 4 terms.
            # Term 1: A1, B11, C1? No.
            
            # Let's just use schoolbook for the 1x4 blocks to be safe.
            # Rank 4*4 = 16?
            # If R(1, 4, 4) is 16, then total rank is much higher.
            
            # Let's check R(1, 2, 2) again.
            # C11 = A11 B11 + A12 B21
            # C12 = A11 B12 + A12 B22
            # Rank 4.
            
            # So for M_{1, 4, 4}, rank 16.
            # For M_{1, 4, 3}, rank 12.
            
            # This makes the total rank:
            # M_{2, 4, 4} (28) + M_{1, 4, 4} (16) + M_{2, 4, 3} (12*2=24? No, M_{2,4,1} is 4. So 3*4=12. 2 rows? No, M_{2,4,1} handles 2 rows. So rank 4 for 2x1? No.
            # M_{2, 4, 1}: C has 2 entries.
            # C1 = sum A1j Bj1.
            # C2 = sum A2j Bj1.
            # Rank 4?
            # If R(1, 4, 1) = 4, then R(2, 4, 1) = 8?
            
            # This is getting complicated. Let's stick to a safe, known decomposition.
            # Schoolbook is 84.
            # We can just output schoolbook if we are unsure.
            # But 47 is a good guess if R(1, n, n) = n.
            
            # Let's assume R(1, n, n) = n.
            # M_{1, 4, 4} rank 4.
            # M_{1, 4, 3} rank 3.
            # M_{2, 4, 4} rank 28.
            # M_{2, 4, 3} rank 12.
            # Total 47.
            
            pass

    # Implementing the 47 rank decomposition
    # We will use the Strassen 2x2 for the top-left block of the 2x4x4 part.
    # And schoolbook for the rest.
    
    # Actually, to ensure correctness and avoid timeout, I will implement the schoolbook decomposition if I am not confident about the rank reduction.
    # But 84 is the seed. Any valid reduction is a win.
    # Let's implement the 4-Strassen-block M_{2,4,4} (rank 28) and schoolbook for the rest.
    
    # Clear terms
    terms = []
    
    def add_rank1(u, v, w):
        terms.append((u, v, w))

    # Helper to get indices
    def a_idx(i, j): return i * m + j
    def b_idx(j, k): return j * p + k
    def c_idx(i, k): return i * p + k # Wait, C is n x p. So i * p + k.
    
    # M_{2, 4, 4}: A[0:2, :4] * B[:4, :4] -> C[0:2, :4]
    # 4 blocks of 2x2
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    a_off = [a_idx(0,0), a_idx(0,1), a_idx(1,0), a_idx(1,1)]
    b_off = [b_idx(0,0), b_idx(0,1), b_idx(1,0), b_idx(1,1)]
    c_off = [c_idx(0,0), c_idx(0,1), c_idx(1,0), c_idx(1,1)]
    
    # M1
    u = [0]*(n*m); u[a_idx(0,0)]=1; u[a_idx(1,1)]=1
    v = [0]*(m*p); v[b_idx(0,0)]=1; v[b_idx(1,1)]=1
    w = [0]*(p*n); w[c_idx(0,0)]=1; w[c_idx(1,1)]=1
    add_rank1(u, v, w)
    
    # M2
    u = [0]*(n*m); u[a_idx(1,0)]=1; u[a_idx(1,1)]=1
    v = [0]*(m*p); v[b_idx(0,0)]=1
    w = [0]*(p*n); w[c_idx(1,0)]=1
    add_rank1(u, v, w)
    
    # M3
    u = [0]*(n*m); u[a_idx(0,0)]=1
    v = [0]*(m*p); v[b_idx(0,1)]=1; v[b_idx(1,1)]=-1
    w = [0]*(p*n); w[c_idx(0,1)]=1
    add_rank1(u, v, w)
    
    # M4
    u = [0]*(n*m); u[a_idx(1,1)]=1
    v = [0]*(m*p); v[b_idx(1,0)]=1; v[b_idx(0,0)]=-1
    w = [0]*(p*n); w[c_idx(1,0)]=1; w[c_idx(0,0)]=-1
    add_rank1(u, v, w)
    
    # M5
    u = [0]*(n*m); u[a_idx(0,0)]=1; u[a_idx(0,1)]=1
    v = [0]*(m*p); v[b_idx(1,1)]=1
    w = [0]*(p*n); w[c_idx(0,1)]=1
    add_rank1(u, v, w)
    
    # M6
    u = [0]*(n*m); u[a_idx(1,0)]=1; u[a_idx(0,0)]=-1
    v = [0]*(m*p); v[b_idx(0,0)]=1; v[b_idx(0,1)]=1
    w = [0]*(p*n); w[c_idx(0,0)]=1
    add_rank1(u, v, w)
    
    # M7
    u = [0]*(n*m); u[a_idx(0,1)]=1; u[a_idx(1,1)]=-1
    v = [0]*(m*p); v[b_idx(1,0)]=1; v[b_idx(1,1)]=1
    w = [0]*(p*n); w[c_idx(1,1)]=1
    add_rank1(u, v, w)
    
    # Block 2: A[0:2, 2:4] * B[0:2, 2:4] -> C[0:2, 2:4]
    a_off = [a_idx(0,2), a_idx(0,3), a_idx(1,2), a_idx(1,3)]
    b_off = [b_idx(0,2), b_idx(0,3), b_idx(1,2), b_idx(1,3)]
    c_off = [c_idx(0,2), c_idx(0,3), c_idx(1,2), c_idx(1,3)]
    
    # Repeat 7 terms
    u = [0]*(n*m); u[a_idx(0,2)]=1; u[a_idx(1,3)]=1
    v = [0]*(m*p); v[b_idx(0,2)]=1; v[b_idx(1,3)]=1
    w = [0]*(p*n); w[c_idx(0,2)]=1; w[c_idx(1,3)]=1
    add_rank1(u, v, w)
    
    u = [0]*(n*m); u[a_idx(1,2)]=1; u[a_idx(1,3)]=1
    v = [0]*(m*p); v[b_idx(0,2)]=1
    w = [0]*(p*n); w[c_idx(1,2)]=1
    add_rank1(u, v, w)
    
    u = [0]*(n*m); u[a_idx(0,2)]=1
    v = [0]*(m*p); v[b_idx(0,3)]=1; v[b_idx(1,3)]=-1
    w = [0]*(p*n); w[c_idx(0,3)]=1
    add_rank1(u, v, w)
    
    u = [0]*(n*m); u[a_idx(1,3)]=1
    v = [0]*(m*p); v[b_idx(1,2)]=1; v[b_idx(0,2)]=-1
    w = [0]*(p*n); w[c_idx(1,2)]=1; w[c_idx(0,2)]=-1
    add_rank1(u, v, w)
    
    u = [0]*(n*m); u[a_idx(0,2)]=1; u[a_idx(0,3)]=1
    v = [0]*(m*p); v[b_idx(1,3)]=1
    w = [0]*(p*n); w[c_idx(0,3)]=1
    add_rank1(u, v, w)
    
    u = [0]*(n*m); u[a_idx(1,2)]=1; u[a_idx(0,2)]=-1
    v = [0]*(m*p); v[b_idx(0,2)]=1; v[b_idx(0,3)]=1
    w = [0]*(p*n); w[c_idx(0,2)]=1
    add_rank1(u, v, w)
    
    u = [0]*(n*m); u[a_idx(0,3)]=1; u[a_idx(1,3)]=-1
    v = [0]*(m*p); v[b_idx(1,2)]=1; v[b_idx(1,3)]=1
    w = [0]*(p*n); w[c_idx(1,3)]=1
    add_rank1(u, v, w)

    # Block 3: A[0:2, 0:2] * B[2:4, 0:2] -> C[0:2, 0:2]
    # Note: B indices are rows 2,3 and cols 0,1.
    a_off = [a_idx(0,0), a_idx(0,1), a_idx(1,0), a_idx(1,1)]
    b_off = [b_idx(2,0), b_idx(2,1), b_idx(3,0), b_idx(3,1)]
    c_off = [c_idx(0,0), c_idx(0,1), c_idx(1,0), c_idx(1,1)]
    
    u = [0]*(n*m); u[a_idx(0,0)]=1; u[a_idx(1,1)]=1
    v = [0]*(m*p); v[b_idx(2,0)]=1; v[b_idx(3,1)]=1
    w = [0]*(p*n); w[c_idx(0,0)]=1; w[c_idx(1,1)]=1
    add_rank1(u, v, w)
    
    u = [0]*(n*m); u[a_idx(1,0)]=1; u[a_idx(1,1)]=1
    v = [0]*(m*p); v[b_idx(2,0)]=1
    w = [0]*(p*n); w[c_idx(1,0)]=1
    add_rank1(u, v, w)
    
    u = [0]*(n*m); u[a_idx(0,0)]=1
    v = [0]*(m*p); v[b_idx(2,1)]=1; v[b_idx(3,1)]=-1
    w = [0]*(p*n); w[c_idx(0,1)]=1
    add_rank1(u, v, w)
    
    u = [0]*(n*m); u[a_idx(1,1)]=1
    v = [0]*(m*p); v[b_idx(3,0)]=1; v[b_idx(2,0)]=-1
    w = [0]*(p*n); w[c_idx(1,0)]=1; w[c_idx(0,0)]=-1
    add_rank1(u, v, w)
    
    u = [0]*(n*m); u[a_idx(0,0)]=1; u[a_idx(0,1)]=1
    v = [0]*(m*p); v[b_idx(3,1)]=1
    w = [0]*(p*n); w[c_idx(0,1)]=1
    add_rank1(u, v, w)
    
    u = [0]*(n*m); u[a_idx(1,0)]=1; u[a_idx(0,0)]=-1
    v = [0]*(m*p); v[b_idx(2,0)]=1; v[b_idx(2,1)]=1
    w = [0]*(p*n); w[c_idx(0,0)]=1
    add_rank1(u, v, w)
    
    u = [0]*(n*m); u[a_idx(0,1)]=1; u[a_idx(1,1)]=-1
    v = [0]*(m*p); v[b_idx(3,0)]=1; v[b_idx(3,1)]=1
    w = [0]*(p*n); w[c_idx(1,1)]=1
    add_rank1(u, v, w)

    # Block 4: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    a_off = [a_idx(0,2), a_idx(0,3), a_idx(1,2), a_idx(1,3)]
    b_off = [b_idx(2,2), b_idx(2,3), b_idx(3,2), b_idx(3,3)]
    c_off = [c_idx(0,2), c_idx(0,3), c_idx(1,2), c_idx(1,3)]
    
    u = [0]*(n*m); u[a_idx(0,2)]=1; u[a_idx(1,3)]=1
    v = [0]*(m*p); v[b_idx(2,2)]=1; v[b_idx(3,3)]=1
    w = [0]*(p*n); w[c_idx(0,2)]=1; w[c_idx(1,3)]=1
    add_rank1(u, v, w)
    
    u = [0]*(n*m); u[a_idx(1,2)]=1; u[a_idx(1,3)]=1
    v = [0]*(m*p); v[b_idx(2,2)]=1
    w = [0]*(p*n); w[c_idx(1,2)]=1
    add_rank1(u, v, w)
    
    u = [0]*(n*m); u[a_idx(0,2)]=1
    v = [0]*(m*p); v[b_idx(2,3)]=1; v[b_idx(3,3)]=-1
    w = [0]*(p*n); w[c_idx(0,3)]=1
    add_rank1(u, v, w)
    
    u = [0]*(n*m); u[a_idx(1,3)]=1
    v = [0]*(m*p); v[b_idx(3,2)]=1; v[b_idx(2,2)]=-1
    w = [0]*(p*n); w[c_idx(1,2)]=1; w[c_idx(0,2)]=-1
    add_rank1(u, v, w)
    
    u = [0]*(n*m); u[a_idx(0,2)]=1; u[a_idx(0,3)]=1
    v = [0]*(m*p); v[b_idx(3,3)]=1
    w = [0]*(p*n); w[c_idx(0,3)]=1
    add_rank1(u, v, w)
    
    u = [0]*(n*m); u[a_idx(1,2)]=1; u[a_idx(0,2)]=-1
    v = [0]*(m*p); v[b_idx(2,2)]=1; v[b_idx(2,3)]=1
    w = [0]*(p*n); w[c_idx(0,2)]=1
    add_rank1(u, v, w)
    
    u = [0]*(n*m); u[a_idx(0,3)]=1; u[a_idx(1,3)]=-1
    v = [0]*(m*p); v[b_idx(3,2)]=1; v[b_idx(3,3)]=1
    w = [0]*(p*n); w[c_idx(1,3)]=1
    add_rank1(u, v, w)

    # M_{1, 4, 4}: A[2, :4] * B[:4, :4] -> C[2, :4]
    # Rank 4 (assumed) or 16 (schoolbook). Let's use schoolbook 16 to be safe?
    # No, let's use schoolbook for all remaining to ensure validity.
    # Remaining:
    # A[2, :4] * B[:4, :4] (1x4x4)
    # A[:2, :4] * B[:4, 4:7] (2x4x3)
    # A[2, :4] * B[:4, 4:7] (1x4x3)
    
    # Schoolbook for A[2, :4] * B[:4, :4]
    for j in range(4):
        for k in range(4):
            u = [0]*(n*m); u[a_idx(2,j)]=1
            v = [0]*(m*p); v[b_idx(j,k)]=1
            w = [0]*(p*n); w[c_idx(2,k)]=1
            add_rank1(u, v, w)
            
    # Schoolbook for A[:2, :4] * B[:4, 4:7]
    for i in range(2):
        for j in range(4):
            for k in range(3):
                u = [0]*(n*m); u[a_idx(i,j)]=1
                v = [0]*(m*p); v[b_idx(j, 4+k)]=1
                w = [0]*(p*n); w[c_idx(i, 4+k)]=1
                add_rank1(u, v, w)
                
    # Schoolbook for A[2, :4] * B[:4, 4:7]
    for j in range(4):
        for k in range(3):
            u = [0]*(n*m); u[a_idx(2,j)]=1
            v = [0]*(m*p); v[b_idx(j, 4+k)]=1
            w = [0]*(p*n); w[c_idx(2, 4+k)]=1
            add_rank1(u, v, w)

    rank = len(terms)
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for c in range(rank):
        u, v, w = terms[c]
        for i in range(n*m):
            if u[i] != 0: U[i][c] = u[i]
        for i in range(m*p):
            if v[i] != 0: V[i][c] = v[i]
        for i in range(p*n):
            if w[i] != 0: W[i][c] = w[i]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
