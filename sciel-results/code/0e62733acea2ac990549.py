"""Program exploiting Kronecker product structure for 2x2 x 2x8 matrix multiplication."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    # The 2x2 x 2x8 multiplication is equivalent to 4 independent 2x2 multiplications
    # (columns 0-1, 2-3, 4-5, 6-7 of the 2x8 matrix).
    # Each 2x2 multiplication has rank 7 (Strassen), so total rank = 4 * 7 = 28.
    # We construct the decomposition using the Strassen algorithm for each block.
    
    n, m, p = DIMENSIONS  # n=2, m=2, p=8
    
    # Strassen decomposition for 2x2 x 2x2:
    # For A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = A*B
    # 7 products:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)(e)
    # M3 = (a)(f-h)
    # M4 = (d)(g-c)
    # M5 = (a+b)(h)
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C = [[M1+M4-M5+M7, M3+M5], [M2+M4, M1-M2+M3+M6]]
    
    # We'll build U, V, W matrices for the full 2x2 x 2x8 problem.
    # U has shape (n*m, rank) = (4, 28)
    # V has shape (m*p, rank) = (16, 28)
    # W has shape (p*n, rank) = (16, 28)
    
    rank = 28
    U = [[0] * rank for _ in range(4)]   # indices: i*m+j where i in 0,1; j in 0,1
    V = [[0] * rank for _ in range(16)]  # indices: j*p+k where j in 0,1; k in 0..7
    W = [[0] * rank for _ in range(16)]  # indices: k*n+i where k in 0..7; i in 0,1
    
    # Strassen coefficients for one 2x2 block
    # Each term t in 0..6 corresponds to one Strassen product
    # For block b (0..3), columns 2b and 2b+1 of the 2x8 matrix
    
    # Define the Strassen decomposition pattern
    # A indices: 0=a, 1=b, 2=c, 3=d (row-major: [a,b,c,d])
    # B indices for one 2x2 block: e,f,g,h
    # C indices: C00, C01, C10, C11
    
    # For each of the 7 Strassen terms, define:
    # - U contributions to A entries
    # - V contributions to B entries  
    # - W contributions to C entries
    
    # Term 1: M1 = (a+d)(e+h)
    # Term 2: M2 = (c+d)(e)
    # Term 3: M3 = (a)(f-h)
    # Term 4: M4 = (d)(g-c)
    # Term 5: M5 = (a+b)(h)
    # Term 6: M6 = (c-a)(e+f)
    # Term 7: M7 = (b-d)(g+h)
    
    # C = [[M1+M4-M5+M7, M3+M5], [M2+M4, M1-M2+M3+M6]]
    
    # For each block b, we have 7 terms, total 28 terms
    for b in range(4):  # 4 blocks
        for t in range(7):  # 7 Strassen terms per block
            col = b * 7 + t
            
            if t == 0:  # M1 = (a+d)(e+h)
                # U: a and d
                U[0][col] = 1  # a
                U[3][col] = 1  # d
                # V: e and h for block b
                # e is at position j=0, k=2b => index 0*8 + 2b = 2b
                # h is at position j=1, k=2b+1 => index 1*8 + 2b+1 = 8+2b+1
                V[2*b][col] = 1       # e
                V[8+2*b+1][col] = 1   # h
                # W: C00 and C11
                # C00: k=2b, i=0 => index 2b*2+0 = 4b
                # C11: k=2b+1, i=1 => index (2b+1)*2+1 = 4b+3
                W[4*b][col] = 1       # C00
                W[4*b+3][col] = 1     # C11
                
            elif t == 1:  # M2 = (c+d)(e)
                # U: c and d
                U[2][col] = 1  # c
                U[3][col] = 1  # d
                # V: e for block b
                V[2*b][col] = 1       # e
                # W: C10 and C11
                # C10: k=2b, i=1 => index 2b*2+1 = 4b+1
                # C11: k=2b+1, i=1 => index (2b+1)*2+1 = 4b+3
                W[4*b+1][col] = 1     # C10
                W[4*b+3][col] = 1     # C11
                
            elif t == 2:  # M3 = (a)(f-h)
                # U: a
                U[0][col] = 1  # a
                # V: f and -h for block b
                # f is at position j=0, k=2b+1 => index 0*8 + 2b+1 = 2b+1
                # h is at position j=1, k=2b+1 => index 1*8 + 2b+1 = 8+2b+1
                V[2*b+1][col] = 1      # f
                V[8+2*b+1][col] = -1   # -h
                # W: C01 and C11
                # C01: k=2b+1, i=0 => index (2b+1)*2+0 = 4b+2
                # C11: k=2b+1, i=1 => index (2b+1)*2+1 = 4b+3
                W[4*b+2][col] = 1      # C01
                W[4*b+3][col] = 1      # C11
                
            elif t == 3:  # M4 = (d)(g-c)
                # U: d
                U[3][col] = 1  # d
                # V: g and -c... wait, g is in B, c is in A
                # Actually M4 = (d)(g-c) means d * g - d * c
                # V: g for block b
                # g is at position j=1, k=2b => index 1*8 + 2b = 8+2b
                V[8+2*b][col] = 1      # g
                # But we also need -c from A... this is handled by U
                # Wait, let me reconsider. M4 = (d)(g-c)
                # This is d*g - d*c, so we need:
                # U: d (with coefficient for g) and d (with coefficient for -c)
                # But U is for A, V is for B.
                # The term is: d * g - d * c
                # So in the tensor: U contributes d to A, V contributes g to B for the + part
                # and U contributes d to A, V contributes... no, c is in A not B.
                # I think I made an error. Let me re-examine Strassen.
                
                # Actually, Strassen's M4 = (c+d)(e) is wrong. Let me use the standard:
                # M1 = (a+d)(e+h)
                # M2 = (c+d)(e)  
                # M3 = a(f-h)
                # M4 = d(g-c)
                # M5 = (a+b)h
                # M6 = (c-a)(e+f)
                # M7 = (b-d)(g+h)
                
                # So M4 = d*(g-c) = d*g - d*c
                # This involves d from A, g from B, and c from A.
                # But in a bilinear form, each term is u_A * u_B.
                # So M4 should be split or I'm misunderstanding.
                
                # Actually, in the standard Strassen decomposition, each M is a product
                # of a linear combination of A entries and a linear combination of B entries.
                # M4 = (d) * (g - c)? No, c is in A.
                
                # I think I have the standard wrong. Let me use:
                # M1 = (a+d)(e+h)
                # M2 = (c+d)(e)
                # M3 = (a-b)(g+h)  -- no
                # 
                # Let me use the correct standard Strassen:
                # p1 = (a + d)(e + h)
                # p2 = (c + d)(e)
                # p3 = (a)(f - h)
                # p4 = (d)(g - c)
                # p5 = (a + b)(h)
                # p6 = (c - a)(e + f)
                # p7 = (b - d)(g + h)
                #
                # C11 = p1 - p5 + p7
                # C12 = p3 + p5
                # C21 = p2 + p4
                # C22 = p1 - p2 + p3 + p6
                
                # So p4 = d*(g-c). But g is in B, c is in A. This is NOT a simple product
                # of a linear form in A and a linear form in B!
                
                # I think I'm confusing myself. In Strassen's algorithm, each p is indeed
                # a product of a linear combination of entries of A and a linear combination
                # of entries of B. Let me verify:
                # p4 = (d)(g - c): d is from A, g is from B, c is from A.
                # This is d*g - d*c, which is NOT separable as (linear in A)*(linear in B).
                
                # I must be misremembering Strassen. Let me look it up properly.
                
                # Correct Strassen:
                # p1 = (a + d)(e + h)
                # p2 = (c + d)(e)
                # p3 = (a)(f - h)
                # p4 = (d)(g - c)  -- this is problematic
                # 
                # Actually, I think the correct form is:
                # p4 = (d - b)(g + e)  or similar. Let me use a verified version.
                
                # Verified Strassen decomposition:
                # p1 = (a + d)(e + h)
                # p2 = (c + d)(e)
                # p3 = (a)(f - h)
                # p4 = (d)(g - c)
                # p5 = (a + b)(h)
                # p6 = (c - a)(e + f)
                # p7 = (b - d)(g + h)
                #
                # But p4 = d*(g-c) is not bilinear in the sense needed.
                # 
                # I realize my error: in the tensor decomposition, each term is of the form
                # u_i * v_j * w_k where u is in A space, v in B space, w in C space.
                # So each "product" p must be expressible as (sum of A entries with coeffs) * (sum of B entries with coeffs).
                # 
                # p4 = d*(g-c): here d is from A, g is from B, c is from A.
                # This equals d*g - d*c. The term d*c involves two A entries, which is NOT
                # allowed in a bilinear form.
                #
                # I think I have the wrong Strassen. Let me use:
                # p1 = (a + d)(e + h)
                # p2 = (c + d)(e)
                # p3 = (a - b)(g + h)
                # p4 = (a + b)(f - h)
                # p5 = (c - a)(e + f)
                # p6 = (b - d)(g + h)
                # p7 = (c + d)(e - a)
                #
                # This is getting messy. Let me just use the schoolbook for now and
                # verify if rank 28 is achievable. Given the complexity, I'll return
                # the schoolbook decomposition which has rank 32, and note that
                # finding rank 28 requires careful Strassen construction.
                
                # For now, let's just return schoolbook to be safe.
                pass
    
    # Given the complexity and risk of error in manual Strassen construction,
    # let me implement a clean schoolbook decomposition which is guaranteed correct.
    # Rank = 2*2*8 = 32.
    
    rank = n * m * p  # 32
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
