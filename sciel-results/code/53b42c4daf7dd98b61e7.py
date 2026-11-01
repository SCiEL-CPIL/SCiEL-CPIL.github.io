"""Matrix multiplication 2x2 x 2x8 with rank-28 Strassen-based decomposition."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 8
    
    # Standard Strassen for 2x2 matrices (0-indexed):
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B10-B11)
    # M4 = A11*(B01-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # U index: i*2+j (i=row of A, j=col of A)
    # V index: j*2+k (j=row of B, k=col of B)  
    # W index: k*2+i (k=col of C, i=row of C)
    
    strassen_terms = []
    # (U_local[4], V_local[4], W_local[4])
    
    # M1 = (A00+A11)*(B00+B11)
    # U: A00=1, A11=1 -> [1,0,0,1]
    # V: B00=1, B11=1 -> [1,0,0,1]
    # W: C00+=1, C11+=1 -> [1,0,0,1]
    strassen_terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))
    
    # M2 = (A10+A11)*B00
    # U: A10=1, A11=1 -> [0,0,1,1]
    # V: B00=1 -> [1,0,0,0]
    # W: C10+=1 -> [0,0,1,0]
    strassen_terms.append(([0,0,1,1], [1,0,0,0], [0,0,1,0]))
    
    # M3 = A00*(B10-B11)
    # U: A00=1 -> [1,0,0,0]
    # V: B10=1, B11=-1 -> [0,0,1,-1]
    # W: C01+=1, C11+=1 -> [0,1,0,1]
    strassen_terms.append(([1,0,0,0], [0,0,1,-1], [0,1,0,1]))
    
    # M4 = A11*(B01-B00)
    # U: A11=1 -> [0,0,0,1]
    # V: B01=1, B00=-1 -> [-1,1,0,0]
    # W: C00-=1, C10+=1 -> [-1,0,1,0]
    strassen_terms.append(([0,0,0,1], [-1,1,0,0], [-1,0,1,0]))
    
    # M5 = (A00+A01)*B11
    # U: A00=1, A01=1 -> [1,1,0,0]
    # V: B11=1 -> [0,0,0,1]
    # W: C00+=1, C01+=1 -> [1,1,0,0]
    strassen_terms.append(([1,1,0,0], [0,0,0,1], [1,1,0,0]))
    
    # M6 = (A10-A00)*(B00+B01)
    # U: A10=1, A00=-1 -> [-1,0,1,0]
    # V: B00=1, B01=1 -> [1,1,0,0]
    # W: C11+=1 -> [0,0,0,1]
    strassen_terms.append(([-1,0,1,0], [1,1,0,0], [0,0,0,1]))
    
    # M7 = (A01-A11)*(B10+B11)
    # U: A01=1, A11=-1 -> [0,1,0,-1]
    # V: B10=1, B11=1 -> [0,0,1,1]
    # W: C00-=1 -> [-1,0,0,0]
    strassen_terms.append(([0,1,0,-1], [0,0,1,1], [-1,0,0,0]))
    
    rank = 4 * 7  # 4 blocks * 7 Strassen terms
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
    col = 0
    for l in range(4):
        for (u_loc, v_loc, w_loc) in strassen_terms:
            # U: i*2+j (4 entries, same for all blocks)
            for idx in range(4):
                U[idx][col] = u_loc[idx]
            # V: j*2+k_local -> full index j*8 + (2*l + k_local)
            for j in range(2):
                for kl in range(2):
                    V[j*8 + (2*l + kl)][col] = v_loc[j*2 + kl]
            # W: k_local*2+i -> full index (2*l+k_local)*2 + i
            for kl in range(2):
                for i in range(2):
                    W[(2*l + kl)*2 + i][col] = w_loc[kl*2 + i]
            col += 1
    
    # Verify
    _verify(U, V, W, n, m, p)
    
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p):
    import random
    random.seed(12345)
    n, m, p = 2, 2, 8
    for trial in range(50):
        A = [[random.randint(-5,5) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(-5,5) for _ in range(p)] for _ in range(m)]
        C_true = [[sum(A[i][j]*B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
        C_tens = [[0]*p for _ in range(n)]
        for i in range(n):
            for k in range(p):
                s = 0
                for r in range(len(U[0])):
                    for j in range(m):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                C_tens[i][k] = s
        if C_true != C_tens:
            raise ValueError(f"Verification failed at trial {trial}: {C_true} vs {C_tens}")

# EVOLVE-BLOCK-END
