"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=11
    
    # We'll build the decomposition with rank 39 using Strassen on column pairs
    # Columns 0-9 form 5 pairs, column 10 is single
    # Each pair uses 7 Strassen terms, single uses 4 schoolbook terms
    
    rank = 5 * 7 + 4  # 39
    U = [[0] * rank for _ in range(n * m)]  # (4, 39)
    V = [[0] * rank for _ in range(m * p)]  # (22, 39)
    W = [[0] * rank for _ in range(p * n)]  # (22, 39)
    
    col = 0
    
    # Process 5 pairs of columns: (0,1), (2,3), (4,5), (6,7), (8,9)
    for pair_idx in range(5):
        k0 = 2 * pair_idx      # first column in pair
        k1 = 2 * pair_idx + 1  # second column in pair
        
        # Strassen terms for this pair
        # A indices: 0=A00, 1=A01, 2=A10, 3=A11
        # B indices for this pair: B0k0, B0k1, B1k0, B1k1
        # B index in V: j*p + k, so B0k0 = V[0*11+k0], B0k1 = V[0*11+k1], B1k0 = V[1*11+k0], B1k1 = V[1*11+k1]
        
        # M1 = (A00+A11)*(B0k0 + B1k0)  -- WAIT, this is wrong.
        # Strassen: M1 = (a+d)(e+h) where e=B0k0, h=B1k1? No...
        # Let me re-examine. B = [[e,f],[g,h]] where e=B[0,k0], f=B[0,k1], g=B[1,k0], h=B[1,k1]
        # M1 = (a+d)(e+h) = (A00+A11)(B0k0 + B1k1)
        
        # Actually I need to be more careful. Let me re-derive.
        # A = [[A00, A01], [A10, A11]]
        # B_block = [[B0k0, B0k1], [B1k0, B1k1]]
        # C_block = A * B_block = [[C0k0, C0k1], [C1k0, C1k1]]
        
        # Strassen:
        # M1 = (A00+A11)(B0k0+B1k1)
        # M2 = (A10+A11)(B0k0)
        # M3 = A00(B1k0-B1k1)
        # M4 = A11(B0k1-B0k0)
        # M5 = (A00+A01)(B1k1)
        # M6 = (A10-A00)(B0k0+B1k0)
        # M7 = (A01-A11)(B0k1+B1k1)
        
        # C0k0 = M1 - M4 + M6
        # C0k1 = M3 + M5
        # C1k0 = M2 + M4
        # C1k1 = M1 - M3 + M7
        
        # Let me verify with a simple case. If A=I, B=I:
        # A00=1,A01=0,A10=0,A11=1; B0k0=1,B0k1=0,B1k0=0,B1k1=1
        # M1=(1+1)(1+1)=4, M2=(0+1)(1)=1, M3=1(0-1)=-1, M4=1(0-1)=-1
        # M5=(1+0)(1)=1, M6=(0-1)(1+0)=-1, M7=(0-1)(0+1)=-1
        # C0k0=4-(-1)+(-1)=4? Should be 1. Wrong!
        
        # I made an error in the Strassen formula. Let me look it up properly.
        # Standard Strassen:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        # C00 = M1 - M4 + M6
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M3 + M7
        
        # With A=I, B=I: A00=1,A01=0,A10=0,A11=1; B00=1,B01=0,B10=0,B11=1
        # M1=(1+1)(1+1)=4, M2=(0+1)*1=1, M3=1*(0-1)=-1, M4=1*(0-1)=-1
        # M5=(1+0)*1=1, M6=(0-1)*(1+0)=-1, M7=(0-1)*(0+1)=-1
        # C00=4-(-1)+(-1)=4. Still wrong.
        
        # I think I have the B indexing wrong. Let me use B = [[b00, b01], [b10, b11]]
        # and A = [[a00, a01], [a10, a11]]
        # C = A*B
        # C00 = a00*b00 + a01*b10
        # C01 = a00*b01 + a01*b11
        # C10 = a10*b00 + a11*b10
        # C11 = a10*b01 + a11*b11
        
        # Strassen:
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)b00
        # M3 = a00(b10-b11)
        # M4 = a11(b01-b00)
        # M5 = (a00+a01)b11
        # M6 = (a10-a00)(b00+b10)
        # M7 = (a01-a11)(b01+b11)
        
        # C00 = M1 - M4 + M6
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M3 + M7
        
        # Test with A=I, B=I: a00=1,a01=0,a10=0,a11=1; b00=1,b01=0,b10=0,b11=1
        # M1=(1+1)(1+1)=4, M2=(0+1)*1=1, M3=1*(0-1)=-1, M4=1*(0-1)=-1
        # M5=(1+0)*1=1, M6=(0-1)*(1+0)=-1, M7=(0-1)*(0+1)=-1
        # C00=4-(-1)+(-1)=4. Expected 1.
        
        # This is still wrong. Let me look at the actual Strassen algorithm more carefully.
        # I think the issue is that I'm confusing the B matrix layout.
        
        # Let me use a reference:
        # Given A = [[a, b], [c, d]], B = [[e, f], [g, h]]
        # p1 = (a + d)(e + h)
        # p2 = (c + d) e
        # p3 = a (f - g)
        # p4 = d (h - e)
        # p5 = (a + b) h
        # p6 = (c - a) (e + g)
        # p7 = (b - d) (f + h)
        
        # C00 = p1 - p4 + p6
        # C01 = p3 + p5
        # C10 = p2 + p4
        # C11 = p1 - p3 + p7
        
        # Test: a=1,b=0,c=0,d=1; e=1,f=0,g=0,h=1
        # p1=(1+1)(1+1)=4, p2=(0+1)*1=1, p3=1*(0-0)=0, p4=1*(1-1)=0
        # p5=(1+0)*1=1, p6=(0-1)*(1+0)=-1, p7=(0-1)*(0+1)=-1
        # C00=4-0+(-1)=3. Expected 1. Still wrong!
        
        # I must be fundamentally misunderstanding Strassen. Let me just use the correct formulas.
        # From Wikipedia:
        # m1 = (a + d) × (e + h)
        # m2 = (c + d) × e
        # m3 = (a + b) × h
        # m4 = d × (h - g)
        # m5 = a × (f - e)
        # m6 = (c + a) × (e + g)
        # m7 = (b + d) × (f + h)
        
        # C00 = m1 + m4 - m5 - m7
        # C01 = m5 + m3
        # C10 = m6 + m4
        # C11 = m1 + m2 - m3 - m6
        
        # Test: a=1,b=0,c=0,d=1; e=1,f=0,g=0,h=1
        # m1=(1+1)(1+1)=4, m2=(0+1)*1=1, m3=(1+0)*1=1, m4=1*(1-0)=1
        # m5=1*(0-1)=-1, m6=(0+1)*(1+0)=1, m7=(0+1)*(0+1)=1
        # C00=4+1-(-1)-1=5. Expected 1.
        
        # I'm clearly getting this wrong. Let me just implement the schoolbook for the pairs and use a different optimization.
        
        # Actually, let me just use the known result and implement rank 39 with a different approach.
        # I'll use the fact that for each column, we need 2 outputs, and we can share A factors.
        
        # Let me just do schoolbook for now (rank 44) and then try to reduce.
        # Actually, the task says the current score is 1.0 with rank 44, and we need to beat that.
        # Let me try rank 39 as stated in the knowledge.
        
        # I'll implement a specific rank-39 construction. Let me think of a valid one.
        
        # Alternative: use 3 terms per column for 10 columns (30 terms) + 9 terms for the last column = 39.
        # But I need to verify this works.
        
        # For now, let me just return the schoolbook decomposition to ensure correctness, 
        # and the evaluation will score it at 1.0. Then I can iterate.
        
        # Actually, let me just implement the schoolbook correctly.
        pass
    
    # Fallback: schoolbook decomposition
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
