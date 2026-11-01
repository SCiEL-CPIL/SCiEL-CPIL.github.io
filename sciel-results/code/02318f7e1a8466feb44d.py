import time
_start_time = time.time()
_DEADLINE = _start_time + 145.0

def construct():
    n, m, p = 2, 2, 15
    
    # 5 blocks of 2x2x3, each rank 8
    rank = 40
    U = [[0] * rank for _ in range(n * m)]  # 4 x 40
    V = [[0] * rank for _ in range(m * p)]  # 30 x 40
    W = [[0] * rank for _ in range(p * n)]  # 30 x 40
    
    # Rank-8 decomposition for 2x2x3
    # A: [[a, b], [c, d]]
    # B: [[e, f, g], [h, i, j]]
    # C: [[ae+bf, af+bg, ag+bh], [ce+df, cf+dg, cg+dh]]
    
    def add_block_2x2x3(base_r, k0):
        # k0, k0+1, k0+2 are the column indices in B
        r1 = base_r
        r2 = base_r + 1
        r3 = base_r + 2
        r4 = base_r + 3
        r5 = base_r + 4
        r6 = base_r + 5
        r7 = base_r + 6
        r8 = base_r + 7
        
        # P1 = (a+d)(e+h)
        U[0][r1] = 1 # a
        U[3][r1] = 1 # d
        V[0 * p + k0][r1] = 1 # e
        V[1 * p + k0][r1] = 1 # h
        
        # P2 = (a+d)(f+i)
        U[0][r2] = 1 # a
        U[3][r2] = 1 # d
        V[0 * p + k0 + 1][r2] = 1 # f
        V[1 * p + k0 + 1][r2] = 1 # i
        
        # P3 = (a+d)(g+j)
        U[0][r3] = 1 # a
        U[3][r3] = 1 # d
        V[0 * p + k0 + 2][r3] = 1 # g
        V[1 * p + k0 + 2][r3] = 1 # j
        
        # P4 = (a-c)(e-h)
        U[0][r4] = 1 # a
        U[2][r4] = -1 # c
        V[0 * p + k0][r4] = 1 # e
        V[1 * p + k0][r4] = -1 # h
        
        # P5 = (a-c)(f-i)
        U[0][r5] = 1 # a
        U[2][r5] = -1 # c
        V[0 * p + k0 + 1][r5] = 1 # f
        V[1 * p + k0 + 1][r5] = -1 # i
        
        # P6 = (a-c)(g-j)
        U[0][r6] = 1 # a
        U[2][r6] = -1 # c
        V[0 * p + k0 + 2][r6] = 1 # g
        V[1 * p + k0 + 2][r6] = -1 # j
        
        # P7 = (b+d)(h+i)
        U[1][r7] = 1 # b
        U[3][r7] = 1 # d
        V[1 * p + k0][r7] = 1 # h
        V[1 * p + k0 + 1][r7] = 1 # i
        
        # P8 = (b+d)(j+i) -- Wait, this is not symmetric.
        # Let's use a verified set.
        # Actually, the standard rank-8 decomposition is:
        # P1 = (a+d)(e+h)
        # P2 = (a+d)(f+i)
        # P3 = (a+d)(g+j)
        # P4 = (a-c)(e-h)
        # P5 = (a-c)(f-i)
        # P6 = (a-c)(g-j)
        # P7 = (b+d)(h+i)
        # P8 = (b+d)(j+i)
        
        # This set is NOT correct for all C elements.
        # I will use the decomposition from the paper "The rank of the 2x2x3 tensor is 8".
        # It uses:
        # P1 = (a+d)(e+h)
        # P2 = (a+d)(f+i)
        # P3 = (a+d)(g+j)
        # P4 = (a-c)(e-h)
        # P5 = (a-c)(f-i)
        # P6 = (a-c)(g-j)
        # P7 = (b+d)(h+i)
        # P8 = (b+d)(j+i)
        
        # Reconstruction:
        # C0,0 = 1/2 (P1 + P4 + P2 - P5) ? No.
        
        # I will fallback to a safe rank-11 block if I can't verify rank-8.
        # But rank 40 is the target.
        
        # Let's use the decomposition:
        # P1 = a(e+h)
        # P2 = b(h+i)
        # P3 = c(e-h)
        # P4 = d(h-i)
        # P5 = a f
        # P6 = b i
        # P7 = c f
        # P8 = d i
        
        # This is rank 8.
        # C0,0 = ae+bf.
        # P1+P2 = ae+ah+bh+bi
        # P3+P4 = ce-ch+dh-di
        # P5+P6 = af+bi
        # P7+P8 = cf+di
        
        # C0,0 = 1/2 (P1 + P2 + P3 + P4) ? No.
        
        # I will use the Strassen 2x2x2 (rank 7) for first 2 cols and schoolbook for 3rd.
        # Rank 11 per block. 5 * 11 = 55. Worse than 53.
        
        # I will stick to the current 53 rank solution but try to optimize the code.
        # Or use 2x2x3 rank 8 if I can find the correct coefficients.
        
        # For now, I will return the 53 rank solution.
        pass

    # Re-implement the 53 rank solution as it is known to work.
    rank = 53
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_block(base_r, k0, k1):
        r1 = base_r
        r2 = base_r + 1
        r3 = base_r + 2
        r4 = base_r + 3
        r5 = base_r + 4
        r6 = base_r + 5
        r7 = base_r + 6
        
        U[0][r1] = 1; U[3][r1] = 1
        V[0 * p + k0][r1] = 1; V[1 * p + k1][r1] = 1
        
        U[2][r2] = 1; U[3][r2] = 1
        V[0 * p + k0][r2] = 1
        
        U[0][r3] = 1
        V[0 * p + k1][r3] = 1; V[1 * p + k1][r3] = -1
        
        U[3][r4] = 1
        V[1 * p + k0][r4] = 1; V[0 * p + k0][r4] = -1
        
        U[0][r5] = 1; U[1][r5] = 1
        V[1 * p + k1][r5] = 1
        
        U[2][r6] = 1; U[0][r6] = -1
        V[0 * p + k0][r6] = 1; V[0 * p + k1][r6] = 1
        
        U[1][r7] = 1; U[3][r7] = -1
        V[1 * p + k0][r7] = 1; V[1 * p + k1][r7] = 1
        
        W[2 * k0][r1] = 1; W[2 * k0][r4] = 1; W[2 * k0][r5] = -1; W[2 * k0][r7] = 1
        W[2 * k0 + 1][r2] = 1; W[2 * k0 + 1][r4] = 1
        W[2 * k1][r3] = 1; W[2 * k1][r5] = 1
        W[2 * k1 + 1][r1] = 1; W[2 * k1 + 1][r2] = -1; W[2 * k1 + 1][r3] = 1; W[2 * k1 + 1][r6] = 1

    for i in range(7):
        add_strassen_block(7 * i, 2 * i, 2 * i + 1)
        
    base_r = 49
    U[0][base_r] = 1; V[0 * p + 14][base_r] = 1; W[2 * 14][base_r] = 1
    U[1][base_r+1] = 1; V[1 * p + 14][base_r+1] = 1; W[2 * 14][base_r+1] = 1
    U[2][base_r+2] = 1; V[0 * p + 14][base_r+2] = 1; W[2 * 14 + 1][base_r+2] = 1
    U[3][base_r+3] = 1; V[1 * p + 14][base_r+3] = 1; W[2 * 14 + 1][base_r+3] = 1
    
    return {"U": U, "V": V, "W": W}
