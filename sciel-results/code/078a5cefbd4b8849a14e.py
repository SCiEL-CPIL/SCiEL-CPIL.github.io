# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 7
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # Indices:
    # U[i*m + k, r]
    # V[k*p + j, r]
    # W[j*n + i, r]  <-- Note: prompt says W[k*n+i, r]. 
    # Let's stick to prompt: W[k*n+i, r]. 
    # In my previous analysis, I assumed W maps to C[i,k]. 
    # If W index is k*n+i, and it contributes to C[i,k], then:
    # Output C is n x p. 
    # Standard C[i,j]. 
    # If W[k*n+i] contributes to C[i,k], then the second index of C is k.
    # But C is A(n x m) * B(m x p) -> n x p.
    # So C[i,j]. 
    # If W[k*n+i] contributes to C[i,k], then j=k.
    # This implies the output is indexed by i (row) and k (col).
    # This is consistent if we view the result as n x p.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's 7 products:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let's map these to U, V, W.
    # U[i*m+k, r] extracts linear combo of A.
    # V[k*p+j, r] extracts linear combo of B.
    # W[j*n+i, r] (or k*n+i) weights the product into C[i,j].
    
    # r=0: M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    # M1 contributes to C00, C11.
    # C00 (i=0, j=0): W[0*n+0, 0] = W[0,0] = 1
    # C11 (i=1, j=1): W[1*n+1, 0] = W[3,0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # r=1: M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    # M2 contributes to C10, C11.
    # C10 (i=1, j=0): W[0*n+1, 1] = W[1,1] = 1
    # C11 (i=1, j=1): W[1*n+1, 1] = W[3,1] = -1
    W[1][1] = 1; W[3][1] = -1
    
    # r=2: M3 = A00(B10-B11)
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    # M3 contributes to C01, C11.
    # C01 (i=0, j=1): W[1*n+0, 2] = W[2,2] = 1
    # C11 (i=1, j=1): W[1*n+1, 2] = W[3,2] = 1
    W[2][2] = 1; W[3][2] = 1
    
    # r=3: M4 = A11(B01-B00)
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    # M4 contributes to C00, C10.
    # C00 (i=0, j=0): W[0*n+0, 3] = W[0,3] = 1
    # C10 (i=1, j=0): W[0*n+1, 3] = W[1,3] = 1
    W[0][3] = 1; W[1][3] = 1
    
    # r=4: M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    # M5 contributes to C00, C01.
    # C00 (i=0, j=0): W[0*n+0, 4] = W[0,4] = -1
    # C01 (i=0, j=1): W[1*n+0, 4] = W[2,4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # r=5: M6 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    # M6 contributes to C11.
    # C11 (i=1, j=1): W[1*n+1, 5] = W[3,5] = 1
    W[3][5] = 1
    
    # r=6: M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    # M7 contributes to C00.
    # C00 (i=0, j=0): W[0*n+0, 6] = W[0,6] = 1
    W[0][6] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
