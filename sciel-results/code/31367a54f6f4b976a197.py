import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = 4, 4, 9
    nm = n * m       # 16
    mp = m * p       # 36
    pn = p * n       # 36
    
    # Strassen's 2x2x2 decomposition (rank 7)
    # For C = AB where A,B are 2x2:
    # m1 = (A00+A11)(B00+B11)
    # m2 = (A10+A11)B00
    # m3 = A00(B10-B11)
    # m4 = A11(B01-B00)
    # m5 = (A00+A10)B11
    # m6 = (A11-A00)(B00+B10)
    # m7 = (A01-A11)(B01+B11)
    # C00 = m1+m4-m5-m7
    # C01 = m3+m5
    # C10 = m2+m4
    # C11 = m1-m2+m3+m7
    
    # Build 4x4x4 using Kronecker product of Strassen (rank 49)
    # Then extend to 4x4x9
    
    # Actually, let's use a direct approach with known decompositions
    # and verify with exact arithmetic
    
    def verify(U, V, W):
        """Verify that U,V,W give exact tensor decomposition using Fraction arithmetic."""
        rank = len(U[0]) if U else 0
        # Check dimensions
        if len(U) != nm or len(V) != mp or len(W) != pn:
            return False
        if any(len(row) != rank for row in U):
            return False
        if any(len(row) != rank for row in V):
            return False
        if any(len(row) != rank for row in W):
            return False
        # Convert to Fraction for exact arithmetic
        U_frac = [[Fraction(x).limit_denominator(10**9) for x in row] for row in U]
        V_frac = [[Fraction(x).limit_denominator(10**9) for x in row] for row in V]
        W_frac = [[Fraction(x).limit_denominator(10**9) for x in row] for row in W]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = Fraction(0)
                    for r in range(rank):
                        total += U_frac[i*m+j][r] * V_frac[j*p+k][r] * W_frac[k*n+i][r]
                    if total != 1:
                        return False
        return True
    
    # Build Strassen 2x2x2
    def strassen_222():
        """Return U,V,W for 2x2x2 matrix multiplication, rank 7."""
        # U[i*2+j, r] for i,j in 0..1
        # V[j*2+k, r] for j,k in 0..1
        # W[k*2+i, r] for k,i in 0..1
        
        U = [[0,0,0,0,0,0,0],
             [0,0,0,0,0,0,0],
             [0,0,0,0,0,0,0],
             [0,0,0,0,0,0,0]]
        V = [[0,0,0,0,0,0,0],
             [0,0,0,0,0,0,0],
             [0,0,0,0,0,0,0],
             [0,0,0,0,0,0,0]]
        W = [[0,0,0,0,0,0,0],
             [0,0,0,0,0,0,0],
             [0,0,0,0,0,0,0],
             [0,0,0,0,0,0,0]]
        
        # m1 = (A00+A11)(B00+B11)
        U[0,0] = 1; U[3,0] = 1  # A00, A11
        V[0,0] = 1; V[3,0] = 1  # B00, B11
        W[0,0] = 1; W[1,0] = 1; W[2,0] = 1; W[3,0] = 1  # C00, C01, C10, C11 contributions... 
        # Wait, I need to be more careful.
        # W[k*2+i, r] where the output C[i,k] gets contribution
        # C00 = m1+m4-m5-m7, so W[0*2+0,0]=1, W[1*2+0,0]=1, W[2*2+0,0]=1, W[3*2+0,0]=1
        # But that's not right either. Let me reconsider.
        # The tensor is T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
        # For 2x2x2, i,j,k in {0,1}
        
        # Let me just use the standard Strassen coefficients
        # m1: A00*A00, A00*A11, A11*A00, A11*A11 -> U[0,0]=1,U[3,0]=1; V[0,0]=1,V[3,0]=1; W[0,0]=1,W[1,0]=1,W[2,0]=1,W[3,0]=1
        # No wait, that's not how it works. Each m_r is a single product.
        
        # m1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11
        # This contributes to C00, C01, C10, C11
        # U[0,0]=1, U[3,0]=1 (A00, A11)
        # V[0,0]=1, V[3,0]=1 (B00, B11) -- wait, V[j,k,r]
        # V[0,0]=1 is B00, V[0,1]=1 is B01, V[1,0]=1 is B10, V[1,1]=1 is B11
        # For m1: B00+B11, so V[0,0,0]=1, V[1,1,0]=1
        # W[k,i,r]: W[0,0]=C00, W[0,1]=C10, W[1,0]=C01, W[1,1]=C11
        # m1 contributes to all: W[0,0,0]=1, W[0,1,0]=1, W[1,0,0]=1, W[1,1,0]=1
        
        # This is getting complex. Let me use a known correct Strassen decomposition.
        
        # Standard Strassen for 2x2:
        # m1 = (a00+a11)(b00+b11)
        # m2 = (a10+a11)b00
        # m3 = a00(b10-b11)
        # m4 = a11(b01-b00)
        # m5 = (a00+a10)b11
        # m6 = (a11-a00)(b00+b10)
        # m7 = (a01-a11)(b01+b11)
        
        # c00 = m1+m4-m5-m7
        # c01 = m3+m5
        # c10 = m2+m4
        # c11 = m1-m2+m3+m7
        
        # U[i,j,r] where i=row of A, j=col of A
        # V[j,k,r] where j=col of A = row of B, k=col of B
        # W[k,i,r] where k=col of B, i=row of C (i=row of A)
        
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # m1: (a00+a11)(b00+b11)
        U[0,0]=1; U[3,0]=1  # a00, a11
        V[0,0]=1; V[3,0]=1  # b00, b11
        W[0,0]=1; W[1,0]=1; W[2,0]=1; W[3,0]=1  # c00,c01,c10,c11 all get +1
        
        # m2: (a10+a11)b00
        U[2,1]=1; U[3,1]=1  # a10, a11
        V[0,1]=1  # b00
        W[2,1]=1  # c10 gets +1
        
        # m3: a00(b10-b11)
        U[0,2]=1  # a00
        V[2,2]=1; V[3,2]=-1  # b10, -b11
        W[1,2]=1; W[3,2]=1  # c01, c11 get +1
        
        # m4: a11(b01-b00)
        U[3,3]=1  # a11
        V[1,3]=1; V[0,3]=-1  # b01, -b00
        W[0,3]=1; W[2,3]=1  # c00, c10 get +1
        
        # m5: (a00+a10)b11
        U[0,4]=1; U[2,4]=1  # a00, a10
        V[3,4]=1  # b11
        W[0,4]=-1; W[1,4]=1  # c00 gets -1, c01 gets +1
        
        # m6: (a11-a00)(b00+b10)
        U[3,5]=1; U[0,5]=-1  # a11, -a00
        V[0,5]=1; V[2,5]=1  # b00, b10
        W[0,5]=1; W[2,5]=1  # c00, c10 get +1
        
        # m7: (a01-a11)(b01+b11)
        U[1,6]=1; U[3,6]=-1  # a01, -a11
        V[1,6]=1; V[3,6]=1  # b01, b11
        W[0,6]=-1; W[3,6]=1  # c00 gets -1, c11 gets +1
        
        return U, V, W
    
    # Verify Strassen 2x2x2
    U2, V2, W2 = strassen_222()
    # Quick check with numpy
    U2n = np.array(U2, dtype=float)
    V2n = np.array(V2, dtype=float)
    W2n = np.array(W2, dtype=float)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                total = sum(U2n[i*2+j,r]*V2n[j*2+k,r]*W2n[k*2+i,r] for r in range(7))
                if abs(total - 1) > 1e-10:
                    print(f"Strassen check failed at i={i},j={j},k={k}: {total}")
                    return {"U": [[1]+[0]*143 for _ in range(16)], 
                            "V": [[1]+[0]*143 for _ in range(36)], 
                            "W": [[1]+[0]*143 for _ in range(36)]}
    
    # Build 4x4x4 from 2x2x2 using Kronecker product
    # 4x4 matrix = 2x2 block matrix of 2x2 blocks
    # A = [[A00, A01], [A10, A11]] where each Ai is 2x2
    # B = [[B00, B01], [B10, B11]]
    # C = AB = [[A00B00+A01B10, A00B01+A01B11], [A10B00+A11B10, A10B01+A11B11]]
    # Each 2x2 product can be done with Strassen rank 7
    # Total: 4 block products * 7 = 28? No, that's not right.
    # Actually, the Kronecker product of two 2x2x2 tensors gives 4x4x4 with rank 7*7=49
    
    def kronecker_tensor(U1, V1, W1, U2, V2, W2):
        """Kronecker product of two tensor decompositions."""
        r1 = len(U1[0]) if U1 else 0
        r2 = len(U2[0]) if U2 else 0
        r = r1 * r2
        
        n1 = int(len(U1)**0.5)  # should be 2
        m1 = int(len(V1)**0.5)  # should be 2
        p1 = int(len(W1)**0.5)  # should be 2
        
        n2 = int(len(U2)**0.5)
        m2 = int(len(V2)**0.5)
        p2 = int(len(W2)**0.5)
        
        n = n1 * n2
        m = m1 * m2
        p = p1 * p2
        
        U = [[0]*r for _ in range(n*m)]
        V = [[0]*r for _ in range(m*p)]
        W = [[0]*r for _ in range(p*n)]
        
        for i1 in range(n1):
            for j1 in range(m1):
                for i2 in range(n2):
                    for j2 in range(m2):
                        i = i1*n2 + i2
                        j = j1*m2 + j2
                        for r1_idx in range(r1):
                            for r2_idx in range(r2):
                                r = r1_idx * r2 + r2_idx
                                U[i*m+j, r] = U1[i1*m1+j1, r1_idx] * U2[i2*m2+j2, r2_idx]
        
        for j1 in range(m1):
            for k1 in range(p1):
                for j2 in range(m2):
                    for k2 in range(p2):
                        j = j1*m2 + j2
                        k = k1*p2 + k2
                        for r1_idx in range(r1):
                            for r2_idx in range(r2):
                                r = r1_idx * r2 + r2_idx
                                V[j*p+k, r] = V1[j1*p1+k1, r1_idx] * V2[j2*p2+k2, r2_idx]
        
        for k1 in range(p1):
            for i1 in range(n1):
                for k2 in range(p2):
                    for i2 in range(n2):
                        k = k1*p2 + k2
                        i = i1*n2 + i2
                        for r1_idx in range(r1):
                            for r2_idx in range(r2):
                                r = r1_idx * r2 + r2_idx
                                W[k*n+i, r] = W1[k1*n1+i1, r1_idx] * W2[k2*n2+i2, r2_idx]
        
        return U, V, W
    
    # Build 4x4x4 with rank 49
    U4, V4, W4 = kronecker_tensor(U2, V2, W2, U2, V2, W2)
    
    # Verify 4x4x4
    U4n = np.array(U4, dtype=float)
    V4n = np.array(V4, dtype=float)
    W4n = np.array(W4, dtype=float)
    rank49 = 49
    valid_444 = True
    for i in range(4):
        for j in range(4):
            for k in range(4):
                total = sum(U4n[i*4+j,r]*V4n[j*4+k,r]*W4n[k*4+i,r] for r in range(rank49))
                if abs(total - 1) > 1e-10:
                    valid_444 = False
                    break
            if not valid_444:
                break
        if not valid_444:
            break
    
    if not valid_444:
        # Fallback to schoolbook
        rank_sb = 144
        U_sb = np.zeros((16, rank_sb))
        V_sb = np.zeros((36, rank_sb))
        W_sb = np.zeros((36, rank_sb))
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(9):
                    U_sb[i*4+j, col] = 1
                    V_sb[j*9+k, col] = 1
                    W_sb[k*4+i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    # Now build 4x4x9 from 4x4x4 (rank 49) + 4x4x4 (rank 49) + 4x4x1 (rank 16)
    # Total: 114
    # 4x4x1: C = AB where B is 4x1, rank 16 (schoolbook)
    
    def build_441():
        """4x4x1 matrix multiplication, rank 16."""
        U = [[0]*16 for _ in range(16)]
        V = [[0]*16 for _ in range(4)]  # m*p = 4*1 = 4
        W = [[0]*16 for _ in range(4)]  # p*n = 1*4 = 4
        
        for i in range(4):
            for j in range(4):
                r = i*4+j
                U[i*4+j, r] = 1
                V[j*1+0, r] = 1
                W[0*4+i, r] = 1
        
        return U, V, W
    
    # Combine 4x4x4 (cols 0-3), 4x4x4 (cols 4-7), 4x4x1 (col 8)
    total_rank = 49 + 49 + 16
    U_final = [[0]*total_rank for _ in range(16)]
    V_final = [[0]*total_rank for _ in range(36)]
    W_final = [[0]*total_rank for _ in range(36)]
    
    # First 4x4x4: columns 0-3 of B, columns 0-3 of C
    for i in range(4):
        for j in range(4):
            for r in range(49):
                U_final[i*4+j, r] = U4[i*4+j, r]
    
    for j in range(4):
        for k in range(4):
            for r in range(49):
                V_final[j*9+k, r] = V4[j*4+k, r]
    
    for k in range(4):
        for i in range(4):
            for r in range(49):
                W_final[k*4+i, r] = W4[k*4+i, r]
    
    # Second 4x4x4: columns 4-7 of B, columns 4-7 of C
    offset = 49
    for i in range(4):
        for j in range(4):
            for r in range(49):
                U_final[i*4+j, offset+r] = U4[i*4+j, r]
    
    for j in range(4):
        for k in range(4):
            for r in range(49):
                V_final[j*9+(4+k), offset+r] = V4[j*4+k, r]
    
    for k in range(4):
        for i in range(4):
            for r in range(49):
                W_final[(4+k)*4+i, offset+r] = W4[k*4+i, r]
    
    # 4x4x1: column 8 of B, column 8 of C
    offset2 = 98
    U1, V1, W1 = build_441()
    for i in range(4):
        for j in range(4):
            for r in range(16):
                U_final[i*4+j, offset2+r] = U1[i*4+j, r]
    
    for j in range(4):
        for r in range(16):
            V_final[j*9+8, offset2+r] = V1[j, r]
    
    for i in range(4):
        for r in range(16):
            W_final[8*4+i, offset2+r] = W1[i, r]
    
    # Verify with numpy
    Uf = np.array(U_final, dtype=float)
    Vf = np.array(V_final, dtype=float)
    Wf = np.array(W_final, dtype=float)
    valid = True
    for i in range(4):
        for j in range(4):
            for k in range(9):
                total = sum(Uf[i*4+j,r]*Vf[j*9+k,r]*Wf[k*4+i,r] for r in range(total_rank))
                if abs(total - 1) > 1e-10:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if valid:
        return {"U": U_final, "V": V_final, "W": W_final}
    
    # Fallback
    rank_sb = 144
    U_sb = np.zeros((16, rank_sb))
    V_sb = np.zeros((36, rank_sb))
    W_sb = np.zeros((36, rank_sb))
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(9):
                U_sb[i*4+j, col] = 1
                V_sb[j*9+k, col] = 1
                W_sb[k*4+i, col] = 1
                col += 1
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
