import time
import numpy as np

DIMENSIONS = [4, 4, 8]

def _strassen_2x2():
    """
    Returns U, V, W for 2x2x2 matrix multiplication with rank 7.
    U shape: (4, 7)  # indices i*2+j
    V shape: (4, 7)  # indices j*2+k
    W shape: (4, 7)  # indices k*2+i
    """
    # Strassen's 7 products for 2x2 matrices
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a-d)(e+f)
    # M2 = (c+d)e
    # M3 = a(h-f)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+g)
    # M7 = (b-d)(f+h)
    
    # Mapping to tensor indices:
    # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # C indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A00 - A11) * (B00 + B01)
    # A00=0, A11=3. B00=0, B01=1.
    # C00 += M1, C11 += M1, C01 -= M1, C10 -= M1 ? No.
    # C = [[a e + b g, a f + b h], [c e + d g, c f + d h]]
    # M1 = (a-d)(e+f) = ae + af - de - df
    # C00 = ae + bg. C11 = cf + dh. C01 = af + bh. C10 = ce + dg.
    # Let's use standard Strassen definitions carefully.
    
    # Standard Strassen:
    # p1 = (a - d) * (e + f)
    # p2 = (c + d) * e
    # p3 = a * (h - f)
    # p4 = d * (g - e)
    # p5 = (a + b) * h
    # p6 = (c - a) * (e + g)
    # p7 = (b - d) * (f + h)
    
    # C00 = p1 + p4 - p5 + p7
    # C01 = p3 + p5
    # C10 = p2 + p4
    # C11 = p1 - p2 + p3 + p6
    
    # Let's populate U, V, W based on this.
    # r=0: p1
    U[0][0] = 1; U[3][0] = -1
    V[0][0] = 1; V[1][0] = 1
    W[0][0] = 1; W[3][0] = -1; W[1][0] = 1; W[2][0] = -1 # C00=1, C11=-1, C01=1, C10=-1? 
    # Wait, W index is k*2+i.
    # C00 is k=0,i=0 -> idx 0.
    # C01 is k=0,i=1 -> idx 1.
    # C10 is k=1,i=0 -> idx 2.
    # C11 is k=1,i=1 -> idx 3.
    
    # Let's re-verify C coefficients for p1 = (a-d)(e+f) = ae + af - de - df.
    # C00 = ae + bg. Contribution to C00 is +1 (from ae).
    # C01 = af + bh. Contribution to C01 is +1 (from af).
    # C10 = ce + dg. Contribution to C10 is 0.
    # C11 = cf + dh. Contribution to C11 is -1 (from -df).
    # So W[0]=1, W[1]=1, W[2]=0, W[3]=-1.
    
    U[0][0] = 1; U[3][0] = -1
    V[0][0] = 1; V[1][0] = 1
    W[0][0] = 1; W[1][0] = 1; W[2][0] = 0; W[3][0] = -1
    
    # r=1: p2 = (c+d)e
    # c=A10(idx 2), d=A11(idx 3), e=B00(idx 0)
    # p2 = ce + de
    # C10 = ce + dg. Contrib +1.
    # C11 = cf + dh. Contrib 0.
    # C00, C01 contrib 0.
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    # r=2: p3 = a(h-f)
    # a=A00(0), h=B11(3), f=B01(1)
    # p3 = ah - af
    # C01 = af + bh. Contrib -1 (from -af).
    # C11 = cf + dh. Contrib +1 (from ah).
    U[0][2] = 1
    V[3][2] = 1; V[1][2] = -1
    W[1][2] = -1; W[3][2] = 1
    
    # r=3: p4 = d(g-e)
    # d=A11(3), g=B10(2), e=B00(0)
    # p4 = dg - de
    # C10 = ce + dg. Contrib +1.
    # C11 = cf + dh. Contrib -1.
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[2][3] = 1; W[3][3] = -1
    
    # r=4: p5 = (a+b)h
    # a=A00(0), b=A01(1), h=B11(3)
    # p5 = ah + bh
    # C01 = af + bh. Contrib +1.
    # C11 = cf + dh. Contrib +1.
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[1][4] = 1; W[3][4] = 1
    
    # r=5: p6 = (c-a)(e+g)
    # c=A10(2), a=A00(0), e=B00(0), g=B10(2)
    # p6 = ce + cg - ae - ag
    # C10 = ce + dg. Contrib +1 (from ce).
    # C00 = ae + bg. Contrib -1 (from -ae).
    # C01 = af + bh. Contrib 0.
    # C11 = cf + dh. Contrib -1 (from -ag? No, ag is not in C11. C11=cf+dh. g is in B10. B10 multiplies A01 and A11? No.
    # Matrix mult: C = AB.
    # C00 = A00 B00 + A01 B10
    # C01 = A00 B01 + A01 B11
    # C10 = A10 B00 + A11 B10
    # C11 = A10 B01 + A11 B11
    # So ag (A00 B10) is in C00.
    # cg (A10 B10) is in C10.
    # So:
    # C00: -1 (from -ae, -ag? No, p6 = (c-a)(e+g) = ce + cg - ae - ag.
    # ae is in C00. ag is in C00.
    # So C00 gets -1 -1 = -2?
    # Let's check C00 = ae + bg.
    # p6 contributes -ae - ag to C00?
    # Wait, ag is A00 B10. Yes, that is in C00.
    # So C00 contribution is -2?
    # Let's check the sum for C00: p1 + p4 - p5 + p7.
    # p1: +ae
    # p4: 0 (dg, de)
    # p5: 0
    # p7: (b-d)(f+h) = bf + bh - df - dh.
    # bf is in C00.
    # So p7 contributes +bf to C00.
    # Sum for C00: ae (from p1) + bf (from p7). This matches C00 = ae + bg? No, C00 = ae + bg.
    # My p7 has bf. That's wrong.
    # Standard Strassen p7 is (b-d)(f+h).
    # b=A01, d=A11. f=B01, h=B11.
    # p7 = bf + bh - df - dh.
    # bf: A01 B01. This is in C01? No. C01 = A00 B01 + A01 B11.
    # A01 B01 is not in C01.
    # Let's re-verify Strassen formulas.
    
    # Correct Strassen:
    # p1 = (a - d)(e + f)
    # p2 = (c + d)e
    # p3 = a(h - f)
    # p4 = d(g - e)
    # p5 = (a + b)h
    # p6 = (c - a)(e + g)
    # p7 = (b - d)(f + h)
    
    # C00 = p1 + p4 - p5 + p7
    # C01 = p3 + p5
    # C10 = p2 + p4
    # C11 = p1 - p2 + p3 + p6
    
    # Let's check C00 = p1 + p4 - p5 + p7
    # p1 = ae + af - de - df
    # p4 = dg - de
    # p5 = ah + bh
    # p7 = bf + bh - df - dh
    
    # Sum for C00 terms (ae, bg):
    # ae: from p1 (+1)
    # bg: from p7? p7 has bf. No bg.
    # Wait, C00 = a e + b g.
    # Is bg in p7? p7 = (b-d)(f+h) = bf + bh - df - dh. No g.
    # This implies my memory of Strassen is slightly off or the indices are mixed.
    
    # Let's use a known correct set.
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # p1 = (a - d)(e + f)
    # p2 = (c + d)e
    # p3 = a(h - f)
    # p4 = d(g - e)
    # p5 = (a + b)h
    # p6 = (c - a)(e + g)
    # p7 = (b - d)(f + h)
    
    # C = [[p1 + p4 - p5 + p7, p3 + p5], [p2 + p4, p1 - p2 + p3 + p6]]
    
    # Let's expand C00 = p1 + p4 - p5 + p7
    # p1 = ae + af - de - df
    # p4 = dg - de
    # p5 = ah + bh
    # p7 = bf + bh - df - dh
    
    # C00 = (ae + af - de - df) + (dg - de) - (ah + bh) + (bf + bh - df - dh)
    # = ae + af - 2de - 2df + dg - ah - bf + dh
    # This does not look like ae + bg.
    
    # I will use the standard algorithm from Wikipedia or a reliable source in code.
    # Actually, I can just use the numpy based verification to fix it if I get it wrong, 
    # but it's better to get it right.
    
    # Let's try this set:
    # m1 = (a - d) * (e + f)
    # m2 = (c + d) * e
    # m3 = a * (h - f)
    # m4 = d * (g - e)
    # m5 = (a + b) * h
    # m6 = (c - a) * (e + g)
    # m7 = (b - d) * (f + h)
    
    # C00 = m1 + m4 - m5 + m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m6
    
    # Let's re-evaluate C00 with this.
    # m1 = ae + af - de - df
    # m4 = dg - de
    # m5 = ah + bh
    # m7 = bf + bh - df - dh
    
    # C00 = ae + af - de - df + dg - de - ah - bh + bf + bh - df - dh
    # = ae + af - 2de - 2df + dg - ah + bf - dh
    # Still not ae + bg.
    
    # There is a mistake in my expansion or the formula.
    # Let's look at C00 = ae + bg.
    # We need terms ae and bg.
    # m1 has ae.
    # m7 has bf. We need bg.
    # Maybe m7 is (b - d) * (g + h)?
    # If m7 = (b - d)(g + h) = bg + bh - dg - dh.
    # Then C00 = m1 + m4 - m5 + m7
    # = (ae + af - de - df) + (dg - de) - (ah + bh) + (bg + bh - dg - dh)
    # = ae + af - 2de - df + dg - ah + bg - dh
    # Still messy.
    
    # Let's use a different known set.
    # p1 = (a - d)(e + f)
    # p2 = (c + d)e
    # p3 = a(h - f)
    # p4 = d(g - e)
    # p5 = (a + b)h
    # p6 = (c - a)(e + g)
    # p7 = (b - d)(f + h)
    
    # Wait, C00 = p1 + p4 - p5 + p7 is correct for the above?
    # Let's check a reference.
    # Yes, that is the standard formula.
    # Let's re-expand carefully.
    # p1 = ae + af - de - df
    # p4 = dg - de
    # p5 = ah + bh
    # p7 = bf + bh - df - dh
    
    # Sum:
    # ae: +1
    # af: +1
    # -de: -1 (from p1) - 1 (from p4) = -2
    # -df: -1 (from p1) - 1 (from p7) = -2
    # dg: +1 (from p4)
    # -ah: -1 (from p5)
    # -bh: -1 (from p5) + 1 (from p7) = 0
    # bf: +1 (from p7)
    # -dh: -1 (from p7)
    
    # Result: ae + af - 2de - 2df + dg - ah + bf - dh.
    # This is definitely not ae + bg.
    
    # I suspect the formula for C00 is different.
    # C00 = p1 + p4 - p5 + p7
    # Let's check C01 = p3 + p5
    # p3 = ah - af
    # p5 = ah + bh
    # Sum = 2ah - af + bh.
    # C01 should be af + bh.
    # So p3 must be -ah + af? Or p5 is different?
    
    # Okay, I will hardcode a verified Strassen decomposition.
    # I will use the one from the current program's _strassen_2x2 and verify it mentally or by code.
    # The current program's _strassen_2x2 is:
    # M1=(A11-A10)(B00+B01) -> This is (d-c)(e+f).
    # M2=(A10+A11)(B10+B11) -> (c+d)(g+h).
    # M3=(A00+A10)(B00) -> (a+c)e.
    # M4=(A11+A01)(B11) -> (d+b)h.
    # M5=(A00-A10)(B01+B11) -> (a-c)(f+h).
    # M6=(A01+A11)(B10-B00) -> (b+d)(g-e).
    # M7=(A00+A01)(B00-B10) -> (a+b)(e-g).
    
    # This is also a valid set of 7 products, just permuted/signed differently.
    # I will reuse the logic from the current program's _strassen_2x2 as it was verified to work (score 1.0).
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1=(A11-A10)(B00+B01)
    # A11=3, A10=2. B00=0, B01=1.
    # U[3][0]=1, U[2][0]=-1. V[0][0]=1, V[1][0]=1.
    # C = ?
    # (d-c)(e+f) = de + df - ce - cf.
    # C11 = cf + dh. Contrib -cf.
    # C10 = ce + dg. Contrib -ce.
    # C01 = af + bh. Contrib 0.
    # C00 = ae + bg. Contrib 0.
    # Wait, de is in C11? No, C11 = A10 B01 + A11 B11 = cf + dh.
    # de = A11 B00. Not in C11.
    # de is in C10? C10 = A10 B00 + A11 B10 = ce + dg.
    # So de is not in C10 either.
    # de = d e. A11 B00.
    # C10 = c e + d g.
    # C11 = c f + d h.
    # So de is not in C10 or C11.
    # It is in C00? C00 = a e + b g. No.
    # C01 = a f + b h. No.
    # This product (d-c)(e+f) involves terms that don't map directly to C?
    # Ah, the decomposition sums them up.
    
    # I will trust the current program's _strassen_2x2 implementation as it is known to be correct for the 2x2 case.
    
    # M1=(A11-A10)(B00+B01)
    U[3][0]=1; U[2][0]=-1; V[0][0]=1; V[1][0]=1; W[0][0]=1; W[2][0]=-1; W[3][0]=1
    # M2=(A10+A11)(B10+B11)
    U[2][1]=1; U[3][1]=1; V[2][1]=1; V[3][1]=1; W[2][1]=1; W[3][1]=1
    # M3=(A00+A10)(B00)
    U[0][2]=1; U[2][2]=1; V[0][2]=1; W[2][2]=1; W[3][2]=1
    # M4=(A11+A01)(B11)
    U[3][3]=1; U[1][3]=1; V[3][3]=1; W[0][3]=1; W[2][3]=-1
    # M5=(A00-A10)(B01+B11)
    U[0][4]=1; U[2][4]=-1; V[1][4]=1; V[3][4]=1; W[0][4]=-1; W[2][4]=1
    # M6=(A01+A11)(B10-B00)
    U[1][5]=1; U[3][5]=1; V[2][5]=1; V[0][5]=-1; W[2][5]=-1; W[3][5]=-1
    # M7=(A00+A01)(B00-B10)
    U[0][6]=1; U[1][6]=1; V[0][6]=1; V[2][6]=-1; W[0][6]=1; W[2][6]=-1
    
    return U, V, W

def _strassen_2x2x4():
    """
    Construct 2x2x4 decomposition.
    A is 2x2, B is 2x4, C is 2x4.
    B = [[B00, B01], [B10, B11]] where B00,B01,B10,B11 are 2x2? No.
    B is 2x4. We can view it as two 2x2 blocks: B_left, B_right.
    C = A * B = [A * B_left, A * B_right].
    So we can use two 2x2x2 Strassen multiplications.
    Rank = 7 + 7 = 14.
    
    U: (4, 14)
    V: (8, 14)  # 2x4 matrix has 8 elements. Index j*4+k.
    W: (8, 14)  # 2x4 matrix has 8 elements. Index k*2+i.
    
    For the left block (k=0,1):
    U_left = U_strassen
    V_left: B_left indices are 0,1,2,3 (j*2+k for j=0,1 and k=0,1).
             In 2x4, B_left is cols 0,1. Indices 0,1,2,3.
    W_left: C_left indices are 0,1,2,3 (k*2+i for k=0,1 and i=0,1).
             In 2x4, C_left is cols 0,1. Indices 0,1,2,3.
             
    For the right block (k=2,3):
    U_right = U_strassen
    V_right: B_right indices are 4,5,6,7 (j*2+k for j=0,1 and k=2,3 -> 0*4+2=2? No.
             B is 2x4. Index j*4+k.
             B00=0, B01=1, B02=2, B03=3.
             B10=4, B11=5, B12=6, B13=7.
             B_left (cols 0,1): 0,1,4,5.
             B_right (cols 2,3): 2,3,6,7.
             
    W_right: C_right indices.
             C is 2x4. Index k*2+i.
             C00=0, C01=1, C02=2, C03=3.
             C10=4, C11=5, C12=6, C13=7.
             C_left (cols 0,1): 0,1,4,5.
             C_right (cols 2,3): 2,3,6,7.
    """
    U2, V2, W2 = _strassen_2x2()
    
    U = [[0]*14 for _ in range(4)]
    V = [[0]*14 for _ in range(8)]
    W = [[0]*14 for _ in range(8)]
    
    # Left block (r=0..6)
    for r in range(7):
        for i in range(4):
            U[i][r] = U2[i][r]
        # V2 indices 0..3 correspond to B_left 0,1,4,5?
        # V2 is for 2x2 B. Indices 0,1,2,3.
        # In 2x4 B, B_left is at indices 0,1,4,5.
        # So V[0][r] = V2[0][r], V[1][r] = V2[1][r], V[4][r] = V2[2][r], V[5][r] = V2[3][r].
        V[0][r] = V2[0][r]
        V[1][r] = V2[1][r]
        V[4][r] = V2[2][r]
        V[5][r] = V2[3][r]
        
        # W2 indices 0..3 correspond to C_left 0,1,4,5?
        # W2 is for 2x2 C. Indices 0,1,2,3.
        # In 2x4 C, C_left is at indices 0,1,4,5.
        W[0][r] = W2[0][r]
        W[1][r] = W2[1][r]
        W[4][r] = W2[2][r]
        W[5][r] = W2[3][r]
        
    # Right block (r=7..13)
    for r in range(7):
        rr = r + 7
        for i in range(4):
            U[i][rr] = U2[i][r]
        # V_right indices 2,3,6,7 in 2x4 B.
        V[2][rr] = V2[0][r]
        V[3][rr] = V2[1][r]
        V[6][rr] = V2[2][r]
        V[7][rr] = V2[3][r]
        
        # W_right indices 2,3,6,7 in 2x4 C.
        W[2][rr] = W2[0][r]
        W[3][rr] = W2[1][r]
        W[6][rr] = W2[2][r]
        W[7][rr] = W2[3][r]
        
    return U, V, W

def _kronecker_product_4x4x8():
    """
    Construct 4x4x8 using Kronecker product of 2x2x2 (Strassen) and 2x2x4 (Strassen-based).
    
    A is 4x4. View as 2x2 blocks of 2x2.
    A = [[A00, A01], [A10, A11]]
    B is 4x8. View as 2x2 blocks of 2x4.
    B = [[B00, B01], [B10, B11]]
    C is 4x8. View as 2x2 blocks of 2x4.
    C = [[C00, C01], [C10, C11]]
    
    C00 = A00 B00 + A01 B10
    C01 = A00 B01 + A01 B11
    C10 = A10 B00 + A11 B10
    C11 = A10 B01 + A11 B11
    
    This is a 2x2 block matrix multiplication where:
    "A" blocks are 2x2.
    "B" blocks are 2x4.
    "C" blocks are 2x4.
    
    So we use the 2x2x4 algorithm (rank 14) for the block multiplication.
    But wait, the 2x2x4 algorithm multiplies a 2x2 matrix by a 2x4 matrix.
    Here, the "A" is a 2x2 matrix of 2x2 blocks.
    The "B" is a 2x2 matrix of 2x4 blocks.
    
    The Kronecker product of the 2x2x2 tensor (for block structure) and the 2x2x4 tensor (for internal structure)
    gives the 4x4x8 tensor.
    
    Rank = 7 * 14 = 98.
    
    U_4x4x8: Shape (16, 98).
    V_4x4x8: Shape (32, 98).
    W_4x4x8: Shape (32, 98).
    
    Indexing:
    U index a: i*4+j. i,j in 0..3.
    Block indices: ib = i//2, i0 = i%2.
    Internal index: i0*2 + j? No, A is 4x4.
    A[i,j]. Block (ib, jb). Internal (i0, j0).
    i = ib*2 + i0, j = jb*2 + j0.
    a = i*4 + j = (ib*2+i0)*4 + jb*2+j0 = ib*8 + i0*4 + jb*2 + j0.
    
    V index b: j*8+k. j in 0..3, k in 0..7.
    Block indices: jb = j//2, j0 = j%2.
    Internal B block is 2x4.
    B[j,k]. Block (jb, kb). Internal (j0, k0).
    j = jb*2 + j0.
    k = kb*4 + k0? No, B is 4x8.
    B[j,k].
    Block (jb, kb).
    j = jb*2 + j0.
    k = kb*4 + k0?
    B is 2x2 blocks of 2x4.
    B00 is rows 0,1 cols 0,1,2,3.
    B01 is rows 0,1 cols 4,5,6,7.
    B10 is rows 2,3 cols 0,1,2,3.
    B11 is rows 2,3 cols 4,5,6,7.
    
    So j = jb*2 + j0.
    k = kb*4 + k0.
    b = j*8 + k = (jb*2+j0)*8 + kb*4 + k0 = jb*16 + j0*8 + kb*4 + k0.
    
    W index c: k*4+i. k in 0..7, i in 0..3.
    C is 4x8.
    Block (kb, ib).
    k = kb*4 + k0.
    i = ib*2 + i0.
    c = k*4 + i = (kb*4+k0)*4 + ib*2 + i0 = kb*16 + k0*4 + ib*2 + i0.
    
    U_4x4x8[a, rb*14 + rs] = U_block[ab, rb] * U_internal[as, rs]
    where ab = ib*2 + jb (index in 2x2 block A)
    as = i0*2 + j0 (index in 2x2 internal A)
    
    V_4x4x8[b, rb*14 + rs] = V_block[bb, rb] * V_internal[bs, rs]
    where bb = jb*2 + kb (index in 2x2 block B)
    bs = j0*4 + k0 (index in 2x4 internal B)
    
    W_4x4x8[c, rb*14 + rs] = W_block[cb, rb] * W_internal[cs, rs]
    where cb = kb*2 + ib (index in 2x2 block C)
    cs = k0*2 + i0 (index in 2x4 internal C)
    """
    U_block, V_block, W_block = _strassen_2x2()
    U_int, V_int, W_int = _strassen_2x2x4()
    
    rank_block = 7
    rank_int = 14
    total_rank = rank_block * rank_int # 98
    
    U = [[0]*total_rank for _ in range(16)]
    V = [[0]*total_rank for _ in range(32)]
    W = [[0]*total_rank for _ in range(32)]
    
    for rb in range(rank_block):
        for rs in range(rank_int):
            col = rb * rank_int + rs
            
            # U
            for ib in range(2):
                for i0 in range(2):
                    for jb in range(2):
                        for j0 in range(2):
                            a = ib * 8 + i0 * 4 + jb * 2 + j0
                            ab = ib * 2 + jb
                            as_idx = i0 * 2 + j0
                            U[a][col] = U_block[ab][rb] * U_int[as_idx][rs]
                            
            # V
            for jb in range(2):
                for j0 in range(2):
                    for kb in range(2):
                        for k0 in range(4):
                            b = jb * 16 + j0 * 8 + kb * 4 + k0
                            bb = jb * 2 + kb
                            bs = j0 * 4 + k0
                            V[b][col] = V_block[bb][rb] * V_int[bs][rs]
                            
            # W
            for kb in range(2):
                for k0 in range(4):
                    for ib in range(2):
                        for i0 in range(2):
                            c = kb * 16 + k0 * 4 + ib * 2 + i0
                            cb = kb * 2 + ib
                            cs = k0 * 2 + i0
                            W[c][col] = W_block[cb][rb] * W_int[cs][rs]
                            
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, rank):
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check diagonal
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                actual = np.sum(U_arr[a, :] * V_arr[b, :] * W_arr[c, :])
                if abs(actual - 1.0) > 1e-9:
                    return False
                    
    # Check off-diagonal with random sampling
    np.random.seed(42)
    num_checks = 5000
    for _ in range(num_checks):
        a = np.random.randint(n * m)
        b = np.random.randint(m * p)
        c = np.random.randint(p * n)
        actual = np.sum(U_arr[a, :] * V_arr[b, :] * W_arr[c, :])
        
        i = a // m
        j = a % m
        j2 = b // p
        k = b % p
        k2 = c // n
        i2 = c % n
        expected = 1.0 if (j == j2 and i == i2 and k == k2) else 0.0
        
        if abs(actual - expected) > 1e-9:
            return False
            
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    # Build 4x4x8 using Kronecker product
    try:
        U, V, W = _kronecker_product_4x4x8()
        rank = len(U[0])
        
        if time.time() < deadline:
            if _verify_decomposition(U, V, W, n, m, p, rank):
                return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fallback to schoolbook
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

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
