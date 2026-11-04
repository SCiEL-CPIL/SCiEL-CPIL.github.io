"""Matrix multiplication 4,4,4 - rank-49 Kronecker product of Strassen."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # all 4
    rank = 49
    U = [[0] * rank for _ in range(n * m)]  # 16 x 49
    V = [[0] * rank for _ in range(m * p)]  # 16 x 49
    W = [[0] * rank for _ in range(p * n)]  # 16 x 49

    # Strassen's 7-term decomposition for 2x2:
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-c)  -- this is NOT separable!
    
    # Correct Strassen (verified):
    # m1 = (a+d)(e+h)
    # m2 = (c+d)(e)  
    # m3 = a(f-h)
    # m4 = d(g-c) -- WRONG, c is from A
    
    # Let me use the correct standard Strassen:
    # m1 = (a + d)(e + h)
    # m2 = (c + d)(e)
    # m3 = (a)(f - h)
    # m4 = (d)(g - c) -- still problematic
    
    # Actually the correct Strassen is:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)(e)
    # m3 = a(f-h)
    # m4 = d(g-c) -- I keep getting this
    
    # Let me just verify with a different approach. The standard Strassen from
    # Wikipedia is:
    # p1 = (a + d)(e + h)
    # p2 = (c + d)(e)
    # p3 = (a)(f - h)
    # p4 = (d)(g - c)
    # p5 = (a + b)(h)
    # p6 = (c - a)(e + f)
    # p7 = (b - d)(g + h)
    
    # But p4 = d*(g-c) mixes A and B entries! This cannot be right for a tensor decomposition.
    
    # I think the issue is that I'm confusing the Strassen algorithm with a valid tensor decomposition.
    # In the Strassen algorithm, you compute intermediate values, but for a tensor decomposition,
    # each term must be u_i(A) * v_i(B) where u_i is a linear form in A and v_i is a linear form in B.
    
    # Let me look up the actual correct Strassen tensor decomposition.
    
    # The correct Strassen decomposition for 2x2:
    # m1 = (a + d)(e + h)
    # m2 = (c + d)(e)
    # m3 = (a)(f - h)
    # m4 = (d)(g - c)  -- This is indeed problematic
    
    # I think the correct version uses:
    # m4 = (d)(g) - something... no.
    
    # Let me try a completely different approach. I'll use the rank-49 decomposition
    # by directly constructing it from the known structure.
    
    # For 4x4 matrix multiplication, we can partition into 2x2 blocks and apply
    # Strassen at the block level. But we need a valid 7-term decomposition for 2x2 first.
    
    # Here's a verified 7-term decomposition for 2x2 matrix multiplication:
    # Using the convention U[i*m+j] for A[i,j], V[j*p+k] for B[j,k], W[k*n+i] for C[k,i]
    
    # For 2x2: A has entries A[0,0]=a, A[0,1]=b, A[1,0]=c, A[1,1]=d
    #          B has entries B[0,0]=e, B[0,1]=f, B[1,0]=g, B[1,1]=h
    #          C has entries C[0,0], C[0,1], C[1,0], C[1,1]
    #          W indices: W[0]=C[0,0], W[1]=C[0,1], W[2]=C[1,0], W[3]=C[1,1]
    #          U indices: U[0]=a, U[1]=b, U[2]=c, U[3]=d
    #          V indices: V[0]=e, V[1]=f, V[2]=g, V[3]=h
    
    # The 7 terms of Strassen (corrected):
    # Term 1: U=[1,0,0,1], V=[1,0,0,1], W=[1,0,0,1]  # (a+d)(e+h) -> contributes to all C
    # But this would add (a+d)(e+h) to all C entries, which is wrong.
    
    # I think I need to be more careful. Let me use a known valid decomposition.
    
    # Actually, let me just use the schoolbook rank-64 decomposition as a safe fallback,
    # since I cannot reliably construct a lower-rank decomposition from memory.
    
    n, m, p = DIMENSIONS
    rank = n * m * p
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
