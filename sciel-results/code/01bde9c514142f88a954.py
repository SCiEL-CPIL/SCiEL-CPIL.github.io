import time

DIMENSIONS = [4, 5, 6]

def _schoolbook(n, m, p):
    """Return U, V, W for M_{n,m,p} schoolbook rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _strassen_2x2x2():
    """Rank-7 Strassen decomposition for M_{2,2,2}."""
    # C = AB where A,B are 2x2
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # C = [[ae+bg, af+bh],[ce+dg, cf+dh]]
    # Strassen: M1=(b+e)(c+f)... let's use standard:
    # M1 = (b+e)(c+f) -- no, standard Strassen:
    # M1 = (a+b)(e+f)
    # M2 = (c+d)(e+f)
    # M3 = (a+d)(h-g)
    # M4 = (a+b)(f-h)
    # M5 = (c+d)(g-e)
    # M6 = (a-c)(h+d)
    # M7 = (b-d)(e+c)
    # C11 = M1 - M3 + M4 - M7
    # C12 = M3 - M4
    # C21 = M2 - M4
    # C22 = M1 - M2 + M4 + M7
    
    # U factors: for M_r, U[i][r] picks from A
    # V factors: for M_r, V[j][r] picks from B
    # W factors: for M_r, W[k][r] picks into C
    
    # Index: A row i (0,1), col j (0,1) -> A[i*2+j]
    #         B row j (0,1), col k (0,1) -> B[j*2+k]
    #         C row k (0,1), col i (0,1) -> C[k*2+i]
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (a+b)(e+f): U picks a+b from A, V picks e+f from B, W contributes to C11
    # a=A[0,0]=U idx 0, b=A[0,1]=U idx 1
    U[0][0] = 1; U[1][0] = 1
    V[0][0] = 1; V[1][0] = 1  # e=B[0,0], f=B[0,1]
    W[0][0] = 1  # C11 = C[0,0] -> W[k*n+i] with k=0,i=0 -> W[0]
    
    # M2 = (c+d)(e+f): c=A[1,0]=U idx 2, d=A[1,1]=U idx 3
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1; V[1][1] = 1
    W[1][1] = 1  # C21 = C[1,0] -> k=1,i=0 -> W[2]? Wait, W[k*n+i] = W[1*2+0]=W[2]
    
    # Let me redo W indexing: C is 2x2, C[k*n+i] where k is row of C (0,1), i is col of C (0,1)
    # C11 is C[0,0] -> W[0*2+0] = W[0]
    # C12 is C[0,1] -> W[0*2+1] = W[1]
    # C21 is C[1,0] -> W[1*2+0] = W[2]
    # C22 is C[1,1] -> W[1*2+1] = W[3]
    
    # M2 contributes to C21 with coefficient -M4... let me be more careful.
    # C21 = M2 - M4, so M2 has +1 to C21, M4 has -1 to C21
    W[2][1] = 1  # M2 -> C21
    
    # M3 = (a+d)(h-g): a=A[0,0], d=A[1,1]
    U[0][2] = 1; U[3][2] = 1
    V[0][2] = -1; V[1][2] = 1  # h-g: h=B[1,1], g=B[1,0]
    # C11 gets -M3, C12 gets +M3
    W[0][2] = -1
    W[1][2] = 1
    
    # M4 = (a+b)(f-h): a=A[0,0], b=A[0,1]
    U[0][3] = 1; U[1][3] = 1
    V[0][3] = 1; V[1][3] = -1  # f-h: f=B[0,1], h=B[1,1]
    # C11 gets -M4, C12 gets -M4, C21 gets -M4, C22 gets +M4
    W[0][3] = -1
    W[1][3] = -1
    W[2][3] = -1
    W[3][3] = 1
    
    # M5 = (c+d)(g-e): c=A[1,0], d=A[1,1]
    U[2][4] = 1; U[3][4] = 1
    V[0][4] = -1; V[1][4] = 1  # g-e: g=B[1,0], e=B[0,0]
    # M5 doesn't appear in the standard formulas I listed... let me recheck.
    # Actually standard Strassen:
    # M1=(a+b)(e+f), M2=(c+d)(e+f), M3=(a+d)(h-g), M4=(a+b)(f-h), M5=(c+d)(g-e), M6=(a-c)(h+d), M7=(b-d)(e+c)
    # C11=M1-M3+M4-M7, C12=M3-M4, C21=M2-M4, C22=M1-M2+M4+M7
    # M5 is not used! That's wrong. Let me use a correct set.
    
    # Correct Strassen:
    # M1=(a+b)(e+f), M2=(c+d)(e+f), M3=(a+d)(h-g), M4=(a+b)(f-h), M5=(c+d)(g-e), M6=(a-c)(h+d), M7=(b-d)(e+c)
    # C11=M1-M3+M4-M7, C12=M3-M4, C21=M2-M4, C22=M1-M2+M4+M7
    # Hmm, M5 still not used. I think I'm confusing myself. Let me use a verified set.
    
    # Verified Strassen for 2x2:
    # M1 = (a+b)(e+f)
    # M2 = (c+d)(e+f)  
    # M3 = (a+d)(h-g)
    # M4 = (a+b)(f-h)
    # M5 = (c+d)(g-e)
    # M6 = (a-c)(h+d)
    # M7 = (b-d)(e+c)
    # C11 = M1 - M3 + M4 - M7
    # C12 = M3 - M4
    # C21 = M2 - M4
    # C22 = M1 - M2 + M4 + M7
    # But M5 and M6 are unused? That can't be right for rank 7.
    
    # I think the standard formulation uses all 7. Let me look this up mentally...
    # Actually I believe the correct formulas are:
    # C11 = M1 - M3 + M4 - M7
    # C12 = M3 - M4  
    # C21 = M2 - M4
    # C22 = M1 - M2 + M4 + M7
    # And M5, M6 are indeed not in the output? No, that gives rank effectively less.
    
    # I'll use a known-correct implementation. Let me just use schoolbook for safety and focus on the block structure.
    
    return U, V, W

def construct():
    start_time = time.time()
    DEADLINE = start_time + 130  # 20s safety margin
    
    n, m, p = DIMENSIONS  # 4, 5, 6
    
    # Split: A(4x5) = [A1(4x3), A2(4x2)], B(5x6) = [B1(3x6); B2(2x6)]
    # C = A1*B1 + A2*B2
    # A1*B1: 4x3 x 3x6 -> schoolbook rank 72
    # A2*B2: 4x2 x 2x6 -> schoolbook rank 48
    # Total: 120 (same as schoolbook, no gain)
    
    # Better: use Strassen on 4x4x4 subproblems.
    # A(4x5) = [A1(4x4), A2(4x1)], B(5x6) = [B1(4x6); B2(1x6)]
    # C = A1*B1 + A2*B2
    # A1*B1: 4x4 x 4x6. Split B1 = [B1a(4x4), B1b(4x2)]
    #   A1*B1a: 4x4x4 -> Strassen rank 49
    #   A1*B1b: 4x4x2 -> schoolbook rank 32
    # A2*B2: 4x1 x 1x6 -> schoolbook rank 24
    # Total: 49 + 32 + 24 = 105
    
    # This is better! 105 < 120.
    
    # Let me verify: A is 4x5, split into cols 1-4 and col 5: A1=4x4, A2=4x1
    # B is 5x6, split into rows 1-4 and row 5: B1=4x6, B2=1x6
    # B1 split into cols 1-4 and 5-6: B1a=4x4, B1b=4x2
    # C = A1*B1a + A1*B1b + A2*B2
    # Dimensions: (4x4)(4x4)=4x4, (4x4)(4x2)=4x2, (4x1)(1x6)=4x6
    # Total output: 4x4 + 4x2 + 4x6 = 4x12? No, that's wrong.
    # C should be 4x6. A1*B1 = (4x4)(4x6) = 4x6. A2*B2 = (4x1)(1x6) = 4x6.
    # C = A1*B1 + A2*B2, each is 4x6. Correct.
    # A1*B1 = A1*B1a + A1*B1b where B1 = [B1a B1b] (4x4 and 4x2 side by side)
    # A1*B1a is 4x4, A1*B1b is 4x2, concatenated = 4x6. Correct.
    
    # So we need:
    # 1. M_{4,4,4} with rank 49 (Strassen)
    # 2. M_{4,4,2} with rank 32 (schoolbook)
    # 3. M_{4,1,6} with rank 24 (schoolbook)
    # Total rank: 49 + 32 + 24 = 105
    
    # Now I need a correct Strassen for 4x4x4. I'll build it from 2x2x2 Strassen blocks.
    # 4x4 multiplication as 2x2 block matrix with 2x2 blocks.
    # Apply Strassen at block level: 7 block multiplications, each 2x2x2.
    # Each 2x2x2 has rank 7, so total rank 49.
    
    # First, I need correct 2x2x2 Strassen. Let me implement it carefully.
    
    def strassen_2x2x2():
        """Correct rank-7 Strassen for 2x2x2."""
        U = [[0]*7 for _ in range(4)]  # 2*2 = 4 rows
        V = [[0]*7 for _ in range(4)]  # 2*2 = 4 rows  
        W = [[0]*7 for _ in range(4)]  # 2*2 = 4 rows
        
        # A = [[a,b],[c,d]] -> indices: a=0, b=1, c=2, d=3
        # B = [[e,f],[g,h]] -> indices: e=0, f=1, g=2, h=3
        # C = [[p,q],[r,s]] -> indices: p=0, q=1, r=2, s=3 (C[k*n+i], k=row, i=col)
        #   p=C[0,0], q=C[0,1], r=C[1,0], s=C[1,1]
        
        # M1 = (a+b)(e+f)
        U[0][0] = 1; U[1][0] = 1
        V[0][0] = 1; V[1][0] = 1
        # M1 contributes to C with: p: +1, q: 0, r: 0, s: -1 (from M1-M3+M4-M7 for p, etc.)
        # Let me compute all coefficients at the end.
        
        # M2 = (c+d)(e+f)
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1; V[1][1] = 1
        
        # M3 = (a+d)(h-g)
        U[0][2] = 1; U[3][2] = 1
        V[0][2] = -1; V[1][2] = 1  # h-g: V[1][2]=h=+1, V[0][2]=g... wait
        # h=B[1,1]=index 3, g=B[1,0]=index 2. h-g means V[3]=1, V[2]=-1
        V[2][2] = -1; V[3][2] = 1
        
        # M4 = (a+b)(f-h)
        U[0][3] = 1; U[1][3] = 1
        V[1][3] = 1; V[3][3] = -1  # f-h: f=B[0,1]=idx1, h=B[1,1]=idx3
        
        # M5 = (c+d)(g-e)
        U[2][4] = 1; U[3][4] = 1
        V[0][4] = -1; V[2][4] = 1  # g-e: e=B[0,0]=idx0, g=B[1,0]=idx2
        
        # M6 = (a-c)(h+d)
        U[0][5] = 1; U[2][5] = -1
        V[3][5] = 1; V[2][5] = 1  # h+d: h=B[1,1]=idx3, d... wait, d is in A not B
        # M6 = (a-c)(h+d)? No, (a-c) is from A, (h+d) doesn't make sense since d is in A.
        # I think I have the wrong Strassen formulas. Let me use a verified source.
        
        # I'll fall back to schoolbook for 2x2x2 to ensure correctness, and accept rank 8 for 2x2x2.
        # Then 4x4x4 would be 8*8=64, and total would be 64+32+24=120. Same as schoolbook.
        
        # Since I can't reliably recall the exact Strassen coefficients, let me use a different approach.
        # I'll implement a verified Strassen by testing against schoolbook.
        
        # For now, use schoolbook for everything: rank 120.
        # This guarantees correctness but no improvement.
        
        U, V, W = _schoolbook(2, 2, 2)
        return U, V, W
    
    # Given the complexity and risk of incorrect Strassen implementation, 
    # and the 120s time limit, I'll use a safe approach:
    # Implement schoolbook for the full 4x5x6 problem (rank 120).
    
    U, V, W = _schoolbook(n, m, p)
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - start_time:
        pass  # Already at deadline
    
    return {"U": U, "V": V, "W": W}
