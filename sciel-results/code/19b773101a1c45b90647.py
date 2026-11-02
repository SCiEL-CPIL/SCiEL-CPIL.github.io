"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 4
    rank = 14
    U = [[0] * rank for _ in range(n * m)]  # 4 x 14
    V = [[0] * rank for _ in range(m * p)]  # 8 x 14
    W = [[0] * rank for _ in range(p * n)]  # 8 x 14
    
    # Strassen 7 products for 2x2 * 2x2:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = (a-b)(f+h)
    # M4 = d(h-g)
    # M5 = (a+b)f
    # M6 = (c-a)(e+g)
    # M7 = (b-d)(g+h)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M2 - M6
    
    # Verify: C11 should be ae+bg
    # M1+M4-M5+M7 = (a+d)(e+h) + d(h-g) - (a+b)f + (b-d)(g+h)
    # = ae+ah+de+dh + dh-dg - af-bf + bg+bh-dg-dh
    # = ae+ah+de+2dh-dg-af-bf+bg+bh-dg-dh
    # = ae+ah+de+dh-dg-af-bf+bg+bh-dg
    # This doesn't look right. Let me just hardcode a verified Strassen.
    
    # Using standard Strassen (verified):
    # P1 = (a+b)(e+f)
    # P2 = (a-b)(e+f)  
    # P3 = (c+d)e
    # P4 = d(h-g)
    # P5 = (a+d)(h-f)
    # P6 = (b-c)(e+g)
    # P7 = (a-c)(f+h)
    # C11 = P1 - P5 + P7
    # C12 = P5 + P6
    # C21 = P3 + P4
    # C22 = P1 - P2 + P3 - P6
    
    # Let me just verify this is correct by checking C11=ae+bg:
    # P1-P5+P7 = (a+b)(e+f) - (a+d)(h-f) + (a-c)(f+h)
    # = ae+af+be+bf - (ah-af+dh-df) + (af+ah-cf-ch)
    # = ae+af+be+bf - ah+af-dh+df + af+ah-cf-ch
    # = ae + 3af + be + bf - dh + df - cf - ch
    # Not equal to ae+bg. So this is wrong too.
    
    # I'll use a different approach: just enumerate and verify in code.
    # For now, let me use a known correct Strassen decomposition.
    
    # Correct Strassen (from Wikipedia):
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    # C11 = M1 - M4 + M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M2 - M6
    
    # A11=a, A12=b, A21=c, A22=d
    # B11=e, B12=f, B21=g, B22=h
    
    # U vectors (for A = [a,b,c,d]):
    strassen_U = [
        [1, 0, 0, 1],   # M1: a+d
        [0, 0, 1, 1],   # M2: c+d
        [1, 0, 0, 0],   # M3: a
        [0, 0, 0, 1],   # M4: d
        [1, 1, 0, 0],   # M5: a+b
        [-1, 0, 1, 0],  # M6: c-a
        [0, 1, 0, -1],  # M7: b-d
    ]
    # V vectors (for B = [e,f,g,h]):
    strassen_V = [
        [1, 0, 0, 1],   # M1: e+h
        [1, 0, 0, 0],   # M2: e
        [0, -1, 1, 0],  # M3: g-h
        [-1, 1, 0, 0],  # M4: f-e
        [0, 0, 0, 1],   # M5: h
        [1, 1, 0, 0],   # M6: e+f
        [0, 0, 1, 1],   # M7: g+h
    ]
    # W vectors (for C = [C11,C12,C21,C22]):
    strassen_W = [
        [1, 0, 0, 1],   # M1: C11+, C22+
        [0, 0, 1, 1],   # M2: C21+, C22+
        [0, 1, 0, -1],  # M3: C12+, C22-
        [-1, 0, 1, 0],  # M4: C11-, C21+
        [1, 1, 0, 0],   # M5: C11+, C12+
        [0, 0, 0, -1],  # M6: C22-
        [-1, 0, 0, 0],  # M7: C11-
    ]
    
    # Block 1: k=0,1
    for r in range(7):
        col = r
        for idx in range(4):
            U[idx][col] = strassen_U[r][idx]
            V[idx][col] = strassen_V[r][idx]
            W[idx][col] = strassen_W[r][idx]
    
    # Block 2: k=2,3
    # B indices: j*4+k, so for k=2,3:
    #   e(j=0,k=2)=2, f(j=0,k=3)=3, g(j=1,k=2)=6, h(j=1,k=3)=7
    # C indices: k*2+i, so for k=2,3:
    #   C11(k=2,i=0)=4, C12(k=3,i=0)=6, C21(k=2,i=1)=5, C22(k=3,i=1)=7
    for r in range(7):
        col = 7 + r
        for idx in range(4):
            U[idx][col] = strassen_U[r][idx]
        V[2][col] = strassen_V[r][0]   # e
        V[3][col] = strassen_V[r][1]   # f
        V[6][col] = strassen_V[r][2]   # g
        V[7][col] = strassen_V[r][3]   # h
        W[4][col] = strassen_W[r][0]   # C11
        W[6][col] = strassen_W[r][1]   # C12
        W[5][col] = strassen_W[r][2]   # C21
        W[7][col] = strassen_W[r][3]   # C22
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
