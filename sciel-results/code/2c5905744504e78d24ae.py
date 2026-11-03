DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # We use Strassen for the top-left 2x2x2 block (indices 0,1 for i,j,k)
    # and schoolbook for the rest.
    # Rank = 7 (Strassen) + (3*3*5 - 2*2*2) schoolbook terms?
    # No, the Strassen decomposition replaces the 8 schoolbook terms for i,j,k in {0,1}.
    # The remaining terms are those where i=2 or j=2 or k=2.
    # Total schoolbook terms = 45.
    # Terms covered by Strassen: i in {0,1}, j in {0,1}, k in {0,1}. (8 terms)
    # Remaining terms: 45 - 8 = 37.
    # Total rank = 7 + 37 = 44.
    
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Strassen for i,j,k in {0,1}
    # M1 = (A00+A11)(B00+B11)
    # U: A00 (idx 0), A11 (idx 4)
    # V: B00 (idx 0), B11 (idx 6)
    # W: C00 (idx 0), C11 (idx 4) -> Wait, W is for C.
    # C00 is k=0, i=0 -> idx 0.
    # C11 is k=1, i=1 -> idx 1*3+1 = 4.
    
    # M1
    U[0][col] += 1; U[4][col] += 1
    V[0][col] += 1; V[6][col] += 1
    W[0][col] += 1; W[4][col] += 1
    col += 1
    
    # M2 = (A10+A11)B00
    U[3][col] += 1; U[4][col] += 1
    V[0][col] += 1
    W[3][col] += 1 # C10 (k=0, i=1) -> 0*3+1=1? No. W[k*n+i]. k=0, i=1 -> 1.
    # C10 is i=1, k=0. W[0*3+1] = W[1].
    W[1][col] += 1
    col += 1
    
    # M3 = A00(B10-B11)
    U[0][col] += 1
    V[5][col] += 1; V[6][col] -= 1 # B10 (j=1,k=0 -> 1*5+0=5), B11 (j=1,k=1 -> 1*5+1=6)
    W[1][col] += 1 # C10? No. C01?
    # M3 contributes to C01 (i=0, k=1) and C10 (i=1, k=0)?
    # Strassen:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # So M3 contributes to C01 (i=0, k=1 -> W[1*3+0]=W[3]) and C11? No.
    # M3 is in C01 and C11?
    # C01 = M3 + M5. So M3 -> C01.
    # C11 = M1 - M2 + M3 + M6. So M3 -> C11.
    W[3][col] += 1 # C01 (k=1, i=0) -> 1*3+0=3
    W[4][col] += 1 # C11 (k=1, i=1) -> 1*3+1=4
    col += 1
    
    # M4 = A11(B01-B00)
    U[4][col] += 1
    V[1][col] += 1; V[0][col] -= 1 # B01 (j=0,k=1 -> 1), B00 (j=0,k=0 -> 0)
    W[1][col] += 1 # C10 (k=0, i=1) -> 1
    W[4][col] += 1 # C11 (k=1, i=1) -> 4
    col += 1
    
    # M5 = (A00+A01)B11
    U[0][col] += 1; U[1][col] += 1
    V[6][col] += 1
    W[3][col] += 1 # C01 (k=1, i=0) -> 3
    W[0][col] -= 1 # C00 (k=0, i=0) -> 0
    col += 1
    
    # M6 = (A10-A00)(B00+B01)
    U[3][col] += 1; U[0][col] -= 1
    V[0][col] += 1; V[1][col] += 1
    W[4][col] += 1 # C11 (k=1, i=1) -> 4
    col += 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1][col] += 1; U[4][col] -= 1
    V[5][col] += 1; V[6][col] += 1
    W[0][col] += 1 # C00 (k=0, i=0) -> 0
    W[3][col] += 1 # C01? No.
    # C00 = M1 + M4 - M5 + M7. So M7 -> C00.
    # C01 = M3 + M5. No M7.
    # C10 = M2 + M4. No M7.
    # C11 = M1 - M2 + M3 + M6. No M7.
    # Wait, Strassen:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # So M7 only contributes to C00.
    col += 1
    
    # Schoolbook for the rest
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
