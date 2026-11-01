import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 54 decomposition
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_idx = 0
    
    # Helper to add a term
    def add_term(u_vec, v_vec, w_vec):
        nonlocal col_idx
        for i in range(len(u_vec)):
            if u_vec[i] != 0:
                U[i][col_idx] += u_vec[i]
        for i in range(len(v_vec)):
            if v_vec[i] != 0:
                V[i][col_idx] += v_vec[i]
        for i in range(len(w_vec)):
            if w_vec[i] != 0:
                W[i][col_idx] += w_vec[i]
        col_idx += 1

    # Strassen 2x2x2 coefficients
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    def get_strassen_terms(a_base, b_base, c_base):
        # a_base: index in U for a00. a01 is a_base+1, a10 is a_base+m, a11 is a_base+m+1
        # b_base: index in V for b00. b01 is b_base+1, b10 is b_base+p, b11 is b_base+p+1
        # c_base: index in W for c00. c01 is c_base+n, c10 is c_base+n+1? 
        # W index is k*n + i.
        # C is n x p. i in [0,n), k in [0,p).
        # C[i,k] index in W: k*n + i.
        # So c00 (i=0,k=0) -> 0. c01 (i=0,k=1) -> 1*n+0 = n.
        # c10 (i=1,k=0) -> 0*n+1 = 1. c11 (i=1,k=1) -> 1*n+1 = n+1.
        
        terms = []
        
        # M1
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[a_base] = 1; u[a_base+m] = 1
        v[b_base] = 1; v[b_base+p] = 1
        w[c_base] = 1; w[c_base+n+1] = 1
        terms.append((u,v,w))
        
        # M2
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[a_base+m] = 1; u[a_base+m+1] = 1
        v[b_base] = 1
        w[c_base+1] = 1; w[c_base+n+1] = -1
        terms.append((u,v,w))
        
        # M3
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[a_base] = 1
        v[b_base+1] = 1; v[b_base+p+1] = -1
        w[c_base+n] = 1; w[c_base+n+1] = 1
        terms.append((u,v,w))
        
        # M4
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[a_base+m+1] = 1
        v[b_base+p] = 1; v[b_base] = -1
        w[c_base] = 1; w[c_base+1] = 1
        terms.append((u,v,w))
        
        # M5
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[a_base] = 1; u[a_base+1] = 1
        v[b_base+p+1] = 1
        w[c_base] = -1; w[c_base+n] = 1
        terms.append((u,v,w))
        
        # M6
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[a_base+m] = 1; u[a_base] = -1
        v[b_base] = 1; v[b_base+1] = 1
        w[c_base+n+1] = 1
        terms.append((u,v,w))
        
        # M7
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[a_base+1] = 1; u[a_base+m+1] = -1
        v[b_base+p] = 1; v[b_base+p+1] = 1
        w[c_base] = 1
        terms.append((u,v,w))
        
        return terms

    # Rank 11 for 2x3x2
    # A is 2x3, B is 3x2
    # Split into A1(2x1)B1(1x2) + A2(2x2)B2(2x2)
    # Part 1: 4 terms
    # Part 2: 7 terms (Strassen on A2, B2)
    
    def get_rank11_terms(a_base, b_base, c_base):
        # a_base: index for A[0,0]. A is 2x3.
        # A[0,0] at a_base. A[0,1] at a_base+1. A[0,2] at a_base+2.
        # A[1,0] at a_base+3. A[1,1] at a_base+4. A[1,2] at a_base+5.
        
        # b_base: index for B[0,0]. B is 3x2.
        # B[0,0] at b_base. B[0,1] at b_base+1.
        # B[1,0] at b_base+2. B[1,1] at b_base+3.
        # B[2,0] at b_base+4. B[2,1] at b_base+5.
        
        # c_base: index for C[0,0]. C is 2x2.
        # C[0,0] at c_base. C[0,1] at c_base+n.
        # C[1,0] at c_base+1. C[1,1] at c_base+n+1.
        
        terms = []
        
        # Part 1: A1 B1
        # m1 = a00 b00 -> c00
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[a_base] = 1
        v[b_base] = 1
        w[c_base] = 1
        terms.append((u,v,w))
        
        # m2 = a00 b01 -> c01
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[a_base] = 1
        v[b_base+1] = 1
        w[c_base+n] = 1
        terms.append((u,v,w))
        
        # m3 = a10 b00 -> c10
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[a_base+3] = 1
        v[b_base] = 1
        w[c_base+1] = 1
        terms.append((u,v,w))
        
        # m4 = a10 b01 -> c11
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[a_base+3] = 1
        v[b_base+1] = 1
        w[c_base+n+1] = 1
        terms.append((u,v,w))
        
        # Part 2: A2 B2 (Strassen)
        # A2 is A[0,1], A[0,2]; A[1,1], A[1,2]
        # Indices: a_base+1, a_base+2; a_base+4, a_base+5
        # This is a 2x2 matrix. 
        # Local a00 = a_base+1, a01 = a_base+2
        # Local a10 = a_base+4, a11 = a_base+5
        # Note: The spacing in U is m=5. 
        # A[0,1] is index 1. A[1,1] is index 4.
        # So local a10 is at a_base+4, local a11 is at a_base+5.
        # The "m" offset in Strassen helper assumes contiguous rows?
        # My Strassen helper used a_base+m for a10.
        # Here a10 is at a_base+4, a00 at a_base+1. Offset is 3, not m=5.
        # So I cannot reuse the helper directly with just a_base.
        # I need to pass the specific indices or adjust the helper.
        
        # Let's create a more flexible Strassen term generator
        def add_strassen(a00, a01, a10, a11, b00, b01, b10, b11, c00, c01, c10, c11):
            # M1
            u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
            u[a00] = 1; u[a11] = 1
            v[b00] = 1; v[b11] = 1
            w[c00] = 1; w[c11] = 1
            terms.append((u,v,w))
            
            # M2
            u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
            u[a10] = 1; u[a11] = 1
            v[b00] = 1
            w[c10] = 1; w[c11] = -1
            terms.append((u,v,w))
            
            # M3
            u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
            u[a00] = 1
            v[b01] = 1; v[b11] = -1
            w[c01] = 1; w[c11] = 1
            terms.append((u,v,w))
            
            # M4
            u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
            u[a11] = 1
            v[b10] = 1; v[b00] = -1
            w[c00] = 1; w[c10] = 1
            terms.append((u,v,w))
            
            # M5
            u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
            u[a00] = 1; u[a01] = 1
            v[b11] = 1
            w[c00] = -1; w[c01] = 1
            terms.append((u,v,w))
            
            # M6
            u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
            u[a10] = 1; u[a00] = -1
            v[b00] = 1; v[b01] = 1
            w[c11] = 1
            terms.append((u,v,w))
            
            # M7
            u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
            u[a01] = 1; u[a11] = -1
            v[b10] = 1; v[b11] = 1
            w[c00] = 1
            terms.append((u,v,w))

        # A2 indices
        a2_00 = a_base + 1
        a2_01 = a_base + 2
        a2_10 = a_base + 4
        a2_11 = a_base + 5
        
        # B2 indices
        b2_00 = b_base + 2
        b2_01 = b_base + 3
        b2_10 = b_base + 4
        b2_11 = b_base + 5
        
        # C indices are same c_base...
        add_strassen(a2_00, a2_01, a2_10, a2_11, b2_00, b2_01, b2_10, b2_11, c_base, c_base+n, c_base+1, c_base+n+1)
        
        return terms

    # 1. Sub-problem 1: A1 (2x2) * B1 (2x6)
    # A1 is A[0:2, 0:2]. Base index 0.
    # B1 is B[0:2, 0:6].
    # C1 is C[0:2, 0:6].
    
    for b in range(3):
        # B block b: cols 2b, 2b+1
        # B indices:
        # b00: row 0, col 2b -> index 0*p + 2b = 2b
        # b01: row 0, col 2b+1 -> index 2b+1
        # b10: row 1, col 2b -> index 1*p + 2b = 6+2b
        # b11: row 1, col 2b+1 -> index 6+2b+1
        
        b_base = 2*b
        
        # A base is 0.
        a_base = 0
        
        # C base:
        # C[0, 2b] -> index 2b*n + 0 = 2b*2 = 4b
        # C[0, 2b+1] -> index (2b+1)*2 + 0 = 4b+2
        # C[1, 2b] -> index 2b*2 + 1 = 4b+1
        # C[1, 2b+1] -> index (2b+1)*2 + 1 = 4b+3
        
        c_base = 4*b
        
        # Get Strassen terms
        terms = get_strassen_terms(a_base, b_base, c_base)
        # Wait, get_strassen_terms assumes specific offsets.
        # In sub-problem 1, A is 2x2, m=5.
        # A[0,0] idx 0. A[0,1] idx 1. A[1,0] idx 5. A[1,1] idx 6.
        # My helper used a_base+m for a10. 0+5=5. Correct.
        # B is 2x2 block of B.
        # B[0,0] idx 2b. B[0,1] idx 2b+1.
        # B[1,0] idx 6+2b. B[1,1] idx 6+2b+1.
        # My helper used b_base+p for b10. 2b+6. Correct.
        
        for u,v,w in terms:
            add_term(u,v,w)

    # 2. Sub-problem 2: A2 (2x3) * B2 (3x6)
    # A2 is A[0:2, 2:5].
    # A[0,2] idx 2. A[0,3] idx 3. A[0,4] idx 4.
    # A[1,2] idx 7. A[1,3] idx 8. A[1,4] idx 9.
    # a_base for A2 (local 0,0) is global 2.
    
    for b in range(3):
        # B block b: cols 2b, 2b+1.
        # B2 is B[2:5, 0:6].
        # B[2, 2b] -> idx 2*p + 2b = 12+2b
        # B[2, 2b+1] -> idx 12+2b+1
        # B[3, 2b] -> idx 3*p + 2b = 18+2b
        # B[3, 2b+1] -> idx 18+2b+1
        # B[4, 2b] -> idx 4*p + 2b = 24+2b
        # B[4, 2b+1] -> idx 24+2b+1
        
        b_base = 12 + 2*b
        
        a_base = 2
        
        c_base = 4*b
        
        terms = get_rank11_terms(a_base, b_base, c_base)
        for u,v,w in terms:
            add_term(u,v,w)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
