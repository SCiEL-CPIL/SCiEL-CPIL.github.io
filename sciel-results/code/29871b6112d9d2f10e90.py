import time

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # We will use rank 49 by applying Strassen to the 2x2x2 block at i=0,1; j=0,1; k=0,1
    # Schoolbook rank is 50.
    # Strassen rank for 2x2x2 is 7.
    # We replace 8 terms with 7.
    
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen for 2x2x2 block (i in 0,1; j in 0,1; k in 0,1)
    # A is 2x2 (rows i, cols j)
    # B is 2x2 (rows j, cols k)
    # C is 2x2 (rows i, cols k)
    
    # Strassen 7 products:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - A00) -- Wait, M4 = A11(B01 - A00) is wrong. M4 = A11(B01 - A00) ??
    # Standard Strassen:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-c)
    # M5 = (a+b)(e+f)
    # M6 = (c+d)(g+h)
    # M7 = (a-b)(f+h)
    
    # Indices:
    # A: a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
    # B: e=B[0,0], f=B[0,1], g=B[1,0], h=B[1,1]
    
    # We need to add terms to U, V, W.
    # Each Strassen term is a product of a linear combo of A and a linear combo of B.
    # U corresponds to A, V to B, W to C.
    
    # Term 1: (A00 + A11)(B00 + B11)
    # U: A00 (1), A11 (1)
    # V: B00 (1), B11 (1)
    # W: C00 (1), C01 (1), C10 (-1), C11 (1) -- Wait, C = M1 + ...
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We will add 7 terms.
    
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        for i in range(n):
            for j in range(m):
                if (i, j) in u_coeffs:
                    U[i*m+j][col] += u_coeffs[(i,j)]
        for j in range(m):
            for k in range(p):
                if (j, k) in v_coeffs:
                    V[j*p+k][col] += v_coeffs[(j,k)]
        for k in range(p):
            for i in range(n):
                if (k, i) in w_coeffs:
                    W[k*n+i][col] += w_coeffs[(k,i)]
        col += 1

    # Strassen Terms
    # 1. (A00+A11)(B00+B11)
    add_term({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,0):-1, (0,1):1, (1,1):1})
    # Wait, W index is k*n+i.
    # C00 is i=0, k=0 -> W[0*2+0] = W[0]
    # C01 is i=0, k=1 -> W[1*2+0] = W[2]
    # C10 is i=1, k=0 -> W[0*2+1] = W[1]
    # C11 is i=1, k=1 -> W[1*2+1] = W[3]
    
    # Correct W coeffs for M1:
    # C00 += 1, C01 -= 1, C10 += 1, C11 += 1?
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # So for M1:
    # W[0] (C00) += 1
    # W[2] (C01) -= 1  (Wait, C01 coeff is 0 for M1? No. C01 = M3+M5. So 0)
    # W[1] (C10) += 1  (C10 = M2+M4. So 0)
    # W[3] (C11) += 1
    
    # Let's re-verify Strassen formulas.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-c)
    # M5 = (a+b)(e+f)
    # M6 = (c+d)(g+h)
    # M7 = (a-b)(f+h)
    
    # I will just implement the 7 terms correctly.
    
    # Reset col to 0 and use a helper to build the full 49 rank matrix.
    # This is getting complex to do inline. I'll write the full construct.

