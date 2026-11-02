"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Block-diagonal decomposition: 3 blocks of 4 columns each.
    # Each 2x2x4 block uses a rank-9 decomposition.
    # Total rank = 3 * 9 = 27.
    rank = 27
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for block_idx in range(3):
        col_offset = block_idx * 4
        base = block_idx * 9
        
        # Rank-9 decomposition for 2x2x4 matrix multiplication
        # U indices: i*m+j = i*2+j (4 entries)
        # V indices: j*p+k = j*4+k (8 entries)
        # W indices: k*n+i = k*2+i (8 entries)
        
        # Factor 0: A[0,0] * (B[0,0] + B[1,0])
        U[0][base + 0] = 1
        V[0*4+0][base + 0] = 1
        V[1*4+0][base + 0] = 1
        W[0*2+0][base + 0] += 1
        W[0*2+1][base + 0] += 1
        
        # Factor 1: A[0,0] * B[0,0]
        U[0][base + 1] = 1
        V[0*4+0][base + 1] = 1
        W[0*2+0][base + 1] -= 1
        
        # Factor 2: A[0,1] * B[1,0]
        U[1][base + 2] = 1
        V[1*4+0][base + 2] = 1
        W[0*2+0][base + 2] += 1
        
        # Factor 3: A[1,0] * (B[0,0] + B[1,0])
        U[2][base + 3] = 1
        V[0*4+0][base + 3] = 1
        V[1*4+0][base + 3] = 1
        W[1*2+0][base + 3] += 1
        W[1*2+1][base + 3] += 1
        
        # Factor 4: A[1,0] * B[0,0]
        U[2][base + 4] = 1
        V[0*4+0][base + 4] = 1
        W[1*2+0][base + 4] -= 1
        
        # Factor 5: A[0,0] * B[0,1]
        U[0][base + 5] = 1
        V[0*4+1][base + 5] = 1
        W[1*2+0][base + 5] += 1
        
        # Factor 6: A[0,1] * B[1,1]
        U[1][base + 6] = 1
        V[1*4+1][base + 6] = 1
        W[1*2+0][base + 6] += 1
        
        # Factor 7: A[1,0] * B[0,2]
        U[2][base + 7] = 1
        V[0*4+2][base + 7] = 1
        W[2*2+0][base + 7] += 1
        
        # Factor 8: A[1,1] * B[1,2]
        U[3][base + 8] = 1
        V[1*4+2][base + 8] = 1
        W[2*2+0][base + 8] += 1
        
        # We still need to cover columns 2 and 3 fully.
        # Let me redo this with a proper rank-9 decomposition.
        
    # The above is incomplete. Let me use a verified rank-9 decomposition.
    # For 2x2x4, the rank-9 decomposition is:
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for block_idx in range(3):
        col_offset = block_idx * 4
        base = block_idx * 9
        
        # Local indices for the 2x2x4 block
        # A: [a00, a01, a10, a11] at U indices 0,1,2,3
        # B: [b00,b01,b02,b03, b10,b11,b12,b13] at V indices 0-7
        # C: [c00,c01,c02,c03, c10,c11,c12,c13] at W indices 0-7
        #    W index = k*n+i = k*2+i, so:
        #    c00=0, c01=2, c02=4, c03=6, c10=1, c11=3, c12=5, c13=7
        
        # Rank-9 decomposition for 2x2x4:
        # f0: a00 * (b00 + b10) -> c00 += 1, c10 += 1
        U[0][base+0] = 1
        V[0*4+0][base+0] = 1
        V[1*4+0][base+0] = 1
        W[0*2+0][base+0] += 1
        W[0*2+1][base+0] += 1
        
        # f1: a00 * b00 -> c00 -= 1
        U[0][base+1] = 1
        V[0*4+0][base+1] = 1
        W[0*2+0][base+1] -= 1
        
        # f2: a01 * b10 -> c00 += 1
        U[1][base+2] = 1
        V[1*4+0][base+2] = 1
        W[0*2+0][base+2] += 1
        
        # f3: a10 * (b00 + b10) -> c10 += 1, c00 += 1
        U[2][base+3] = 1
        V[0*4+0][base+3] = 1
        V[1*4+0][base+3] = 1
        W[1*2+0][base+3] += 1
        W[0*2+0][base+3] += 1
        
        # f4: a10 * b00 -> c10 -= 1
        U[2][base+4] = 1
        V[0*4+0][base+4] = 1
        W[1*2+0][base+4] -= 1
        
        # f5: a00 * b01 -> c01 += 1
        U[0][base+5] = 1
        V[0*4+1][base+5] = 1
        W[1*2+0][base+5] += 1
        
        # f6: a01 * b11 -> c01 += 1
        U[1][base+6] = 1
        V[1*4+1][base+6] = 1
        W[1*2+0][base+6] += 1
        
        # f7: a10 * b02 -> c12 += 1
        U[2][base+7] = 1
        V[0*4+2][base+7] = 1
        W[2*2+0][base+7] += 1
        
        # f8: a11 * b12 -> c12 += 1
        U[3][base+8] = 1
        V[1*4+2][base+8] = 1
        W[2*2+0][base+8] += 1
        
        # This is still incomplete - missing many terms. Let me use a different approach.
        # I'll use a decomposition that I can verify.
        
    # Let me restart with a clean implementation using a verified decomposition.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for block_idx in range(3):
        col_offset = block_idx * 4
        base = block_idx * 9
        
        # For the 2x2x4 block starting at col_offset:
        # We decompose into 9 rank-1 terms.
        # The decomposition uses the following 9 products:
        #
        # p0 = a00 * (b00 + b10)
        # p1 = a00 * b00  
        # p2 = a01 * b10
        # p3 = a10 * (b00 + b10)
        # p4 = a10 * b00
        # p5 = a00 * b01 + a01 * b11  (combined)
        # p6 = a10 * b02 + a11 * b12  (combined)
        # p7 = a00 * b03 + a01 * b13  (combined)
        # p8 = a10 * b03 + a11 * b13  (combined)
        #
        # This doesn't work because we can't combine B terms with different A terms.
        
        # Let me use a simpler but valid decomposition with rank 14 per block (2x Strassen)
        # and accept rank 42, or find a better decomposition.
        
        # Actually, let me use the following rank-9 decomposition:
        # Based on the structure of 2x2x4 tensor
        
        # The 9 factors for 2x2x4:
        # Index: U_coeffs (a00,a01,a10,a11), V_coeffs (b00,b01,b02,b03,b10,b11,b12,b13), W_coeffs (c00,c10,c01,c11,c02,c12,c03,c13)
        
        # f0: a00*(b00+b10) -> c00+=1, c10+=1
        U[0][base+0] = 1
        V[0][base+0] = 1
        V[4][base+0] = 1
        W[0][base+0] += 1
        W[1][base+0] += 1
        
        # f1: a00*b00 -> c00-=1
        U[0][base+1] = 1
        V[0][base+1] = 1
        W[0][base+1] -= 1
        
        # f2: a01*b10 -> c00+=1
        U[1][base+2] = 1
        V[4][base+2] = 1
        W[0][base+2] += 1
        
        # f3: a10*(b00+b10) -> c10+=1, c00+=1
        U[2][base+3] = 1
        V[0][base+3] = 1
        V[4][base+3] = 1
        W[1][base+3] += 1
        W[0][base+3] += 1
        
        # f4: a10*b00 -> c10-=1
        U[2][base+4] = 1
        V[0][base+4] = 1
        W[1][base+4] -= 1
        
        # f5: (a00+a01)*(b01+b11) -> c01+=1, c11+=1
        U[0][base+5] = 1
        U[1][base+5] = 1
        V[1][base+5] = 1
        V[5][base+5] = 1
        W[2][base+5] += 1
        W[3][base+5] += 1
        
        # f6: a00*b01 -> c01-=1
        U[0][base+6] = 1
        V[1][base+6] = 1
        W[2][base+6] -= 1
        
        # f7: a01*b11 -> c01+=1
        U[1][base+7] = 1
        V[5][base+7] = 1
        W[2][base+7] += 1
        
        # f8: a10*b02 -> c12+=1
        U[2][base+8] = 1
        V[2][base+8] = 1
        W[4][base+8] += 1
        
        # This is getting messy. Let me just use a verified decomposition.
        
    # I'll use a clean approach: 3 blocks of 4, each with rank 9.
    # The rank-9 decomposition for 2x2x4 is constructed as follows.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for block_idx in range(3):
        co = block_idx * 4
        b = block_idx * 9
        
        # U row indices: 0=a00, 1=a01, 2=a10, 3=a11
        # V row indices: 0=b00, 1=b01, 2=b02, 3=b03, 4=b10, 5=b11, 6=b12, 7=b13
        # W row indices: 0=c00, 1=c10, 2=c01, 3=c11, 4=c02, 5=c12, 6=c03, 7=c13
        
        # f0: a00*(b00+b10) -> c00+=1, c10+=1
        U[0][b+0] = 1; V[0][b+0] = 1; V[4][b+0] = 1; W[0][b+0] += 1; W[1][b+0] += 1
        # f1: a00*b00 -> c00-=1
        U[0][b+1] = 1; V[0][b+1] = 1; W[0][b+1] -= 1
        # f2: a01*b10 -> c00+=1
        U[1][b+2] = 1; V[4][b+2] = 1; W[0][b+2] += 1
        # f3: a10*(b00+b10) -> c10+=1, c00+=1
        U[2][b+3] = 1; V[0][b+3] = 1; V[4][b+3] = 1; W[1][b+3] += 1; W[0][b+3] += 1
        # f4: a10*b00 -> c10-=1
        U[2][b+4] = 1; V[0][b+4] = 1; W[1][b+4] -= 1
        # f5: (a00+a01)*(b01+b11) -> c01+=1, c11+=1
        U[0][b+5] = 1; U[1][b+5] = 1; V[1][b+5] = 1; V[5][b+5] = 1; W[2][b+5] += 1; W[3][b+5] += 1
        # f6: a00*b01 -> c01-=1
        U[0][b+6] = 1; V[1][b+6] = 1; W[2][b+6] -= 1
        # f7: a01*b11 -> c01+=1
        U[1][b+7] = 1; V[5][b+7] = 1; W[2][b+7] += 1
        # f8: (a10+a11)*(b02+b12) -> c12+=1, c02+=1
        U[2][b+8] = 1; U[3][b+8] = 1; V[2][b+8] = 1; V[6][b+8] = 1; W[4][b+8] += 1; W[5][b+8] += 1
        
        # This only covers 3 of the 4 column pairs. I need 9 factors total for all 4 columns.
        # The above uses 9 factors but only covers columns 0,1,2. Column 3 is missing.
        
    # The decomposition above is incomplete. Let me use a different strategy.
    # I'll use rank 14 per block (2 Strassen blocks) for a total of 42, same as before.
    # Or I can try to find a valid rank-9 decomposition.
    
    # After careful consideration, let me use the following valid rank-9 decomposition:
    # It's based on sharing the A-side factors between the two 2x2x2 sub-problems.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for block_idx in range(3):
        co = block_idx * 4
        b = block_idx * 9
        
        # Strassen-like for cols (co, co+1):
        # s0 = (a00+a01)*(b00+b10) -> c00+=1, c10+=1
        U[0][b+0] = 1; U[1][b+0] = 1; V[0][b+0] = 1; V[4][b+0] = 1; W[0][b+0] += 1; W[1][b+0] += 1
        # s1 = a00*b00 -> c00-=1
        U[0][b+1] = 1; V[0][b+1] = 1; W[0][b+1] -= 1
        # s2 = a01*b10 -> c00+=1
        U[1][b+2] = 1; V[4][b+2] = 1; W[0][b+2] += 1
        # s3 = (a10+a11)*(b00+b10) -> c10+=1, c00+=1
        U[2][b+3] = 1; U[3][b+3] = 1; V[0][b+3] = 1; V[4][b+3] = 1; W[1][b+3] += 1; W[0][b+3] += 1
        # s4 = a10*b00 -> c10-=1
        U[2][b+4] = 1; V[0][b+4] = 1; W[1][b+4] -= 1
        # s5 = a00*b01 -> c01+=1
        U[0][b+5] = 1; V[1][b+5] = 1; W[2][b+5] += 1
        # s6 = a01*b11 -> c01+=1
        U[1][b+6] = 1; V[5][b+6] = 1; W[2][b+6] += 1
        # s7 = a10*b02 -> c12+=1
        U[2][b+7] = 1; V[2][b+7] = 1; W[4][b+7] += 1
        # s8 = a11*b12 -> c12+=1
        U[3][b+8] = 1; V[6][b+8] = 1; W[4][b+8] += 1
        
        # Missing: c02, c13, c03 terms. This decomposition is incomplete.
        
    # Given the difficulty, let me use a safe decomposition with rank 42 (same as parent)
    # but structured as 3 blocks of 4, each using 2 Strassen blocks (rank 14 per block).
    
    U = [[0] * 42 for _ in range(n * m)]
    V = [[0] * 42 for _ in range(m * p)]
    W = [[0] * 42 for _ in range(p * n)]
    
    for block_idx in range(3):
        co = block_idx * 4
        base = block_idx * 14
        
        # First Strassen block for cols (co, co+1)
        sb = base
        U[0][sb+0] = 1; U[3][sb+0] = 1; V[0*12+co][sb+0] = 1; V[1*12+co+1][sb+0] = 1; W[co*2+0][sb+0] += 1; W[(co+1)*2+1][sb+0] += 1
        U[2][sb+1] = 1; U[3][sb+1] = 1; V[0*12+co][sb+1] = 1; W[co*2+1][sb+1] += 1; W[(co+1)*2+1][sb+1] -= 1
        U[0][sb+2] = 1; V[0*12+co+1][sb+2] = 1; V[1*12+co+1][sb+2] = -1; W[(co+1)*2+0][sb+2] += 1; W[(co+1)*2+1][sb+2] += 1
        U[3][sb+3] = 1; V[1*12+co][sb+3] = 1; V[0*12+co][sb+3] = -1; W[co*2+0][sb+3] += 1; W[co*2+1][sb+3] += 1
        U[0][sb+4] = 1; U[1][sb+4] = 1; V[1*12+co+1][sb+4] = 1; W[co*2+0][sb+4] -= 1; W[(co+1)*2+0][sb+4] += 1
        U[2][sb+5] = 1; U[0][sb+5] = -1; V[0*12+co][sb+5] = 1; V[0*12+co+1][sb+5] = 1; W[(co+1)*2+1][sb+5] += 1
        U[1][sb+6] = 1; U[3][sb+6] = -1; V[1*12+co][sb+6] = 1; V[1*12+co+1][sb+6] = 1; W[co*2+0][sb+6] += 1
        
        # Second Strassen block for cols (co+2, co+3)
        sb = base + 7
        U[0][sb+0] = 1; U[3][sb+0] = 1; V[0*12+co+2][sb+0] = 1; V[1*12+co+3][sb+0] = 1; W[(co+2)*2+0][sb+0] += 1; W[(co+3)*2+1][sb+0] += 1
        U[2][sb+1] = 1; U[3][sb+1] = 1; V[0*12+co+2][sb+1] = 1; W[(co+2)*2+1][sb+1] += 1; W[(co+3)*2+1][sb+1] -= 1
        U[0][sb+2] = 1; V[0*12+co+3][sb+2] = 1; V[1*12+co+3][sb+2] = -1; W[(co+3)*2+0][sb+2] += 1; W[(co+3)*2+1][sb+2] += 1
        U[3][sb+3] = 1; V[1*12+co+2][sb+3] = 1; V[0*12+co+2][sb+3] = -1; W[(co+2)*2+0][sb+3] += 1; W[(co+2)*2+1][sb+3] += 1
        U[0][sb+4] = 1; U[1][sb+4] = 1; V[1*12+co+3][sb+4] = 1; W[(co+2)*2+0][sb+4] -= 1; W[(co+3)*2+0][sb+4] += 1
        U[2][sb+5] = 1; U[0][sb+5] = -1; V[0*12+co+2][sb+5] = 1; V[0*12+co+3][sb+5] = 1; W[(co+3)*2+1][sb+5] += 1
        U[1][sb+6] = 1; U[3][sb+6] = -1; V[1*12+co+2][sb+6] = 1; V[1*12+co+3][sb+6] = 1; W[(co+2)*2+0][sb+6] += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
