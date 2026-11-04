import time
import numpy as np

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # We split m=5 into m1=2 and m2=3.
    # We split p=6 into p1=2, p2=2, p3=2.
    # For each pk, we compute C_pk = A_m1 B_m1_pk + A_m2 B_m2_pk.
    # A_m1 is 2x2, B_m1_pk is 2x2. Use Strassen (rank 7).
    # A_m2 is 2x3, B_m2_pk is 3x2. Use Schoolbook (rank 12).
    # Total rank per pk = 7 + 12 = 19.
    # Total rank = 3 * 19 = 57.
    
    rank = 57
    U = [[0] * rank for _ in range(n * m)] # 10 x 57
    V = [[0] * rank for _ in range(m * p)] # 30 x 57
    W = [[0] * rank for _ in range(p * n)] # 12 x 57
    
    col = 0
    for k in range(3):
        # p-block k (columns 2k to 2k+1 of B, rows 2k to 2k+1 of C)
        p_start = 2 * k
        
        # --- Part 1: m1=2 (rows 0,1 of A, rows 0,1 of B) ---
        # Strassen decomposition for 2x2 * 2x2
        # A_m1 indices in U: i*5 + j. i in 0..1, j in 0..1.
        # B_m1 indices in V: j*6 + (2k+l). j in 0..1, l in 0..1.
        # C_m1 indices in W: (2k+l)*2 + i.
        
        # Strassen factors for 2x2x2
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-c)
        # M5 = (a+b)h
        # M6 = (c-d)(e+f)
        # M7 = (b-d)(g+h)
        
        # U part (A)
        # a=U[0, col], d=U[1, col] -> M1: U[0]+=1, U[1]+=1
        # c=U[1, col], d=U[1, col] -> M2: U[1]+=1 (wait, c is row 1 col 0? No, A is 2x2. a=A00, b=A01, c=A10, d=A11)
        # Indices in U for A_m1:
        # A00 -> U[0*5+0] = U[0]
        # A01 -> U[0*5+1] = U[1]
        # A10 -> U[1*5+0] = U[5]
        # A11 -> U[1*5+1] = U[6]
        
        # V part (B)
        # B is 2x2 block at rows 0,1 and cols 2k, 2k+1.
        # B00 -> V[0*6 + 2k] = V[2k]
        # B01 -> V[0*6 + 2k+1] = V[2k+1]
        # B10 -> V[1*6 + 2k] = V[6+2k]
        # B11 -> V[1*6 + 2k+1] = V[6+2k+1]
        
        # W part (C)
        # C is 2x2 block at rows 2k, 2k+1 and cols 0,1.
        # C00 -> W[(2k)*2 + 0] = W[4k]
        # C01 -> W[(2k)*2 + 1] = W[4k+1]
        # C10 -> W[(2k+1)*2 + 0] = W[4k+2]
        # C11 -> W[(2k+1)*2 + 1] = W[4k+3]
        
        # Strassen M1: (a+d)(e+h)
        # U: a(0), d(6) -> 1, 1
        # V: e(2k), h(6+2k+1) -> 1, 1
        # W: C00(4k) +1, C11(4k+3) -1? 
        # C00 = M1 + M4 - M5 + M7
        # C11 = M1 - M2 + M3 + M6
        # Let's map coefficients.
        
        # M1
        U[0][col] += 1; U[6][col] += 1
        V[2*k][col] += 1; V[6+2*k+1][col] += 1
        W[4*k][col] += 1; W[4*k+3][col] += 1
        col += 1
        
        # M2: (c+d)e
        # c=U[5], d=U[6]
        # e=V[2k]
        # C00 += M2? No. C21 = M2 + M4.
        # C indices: C21 is row 1, col 1? No.
        # C = [[C00, C01], [C10, C11]]
        # C10 = M2 + M4.
        # C10 index: W[4k+2].
        U[5][col] += 1; U[6][col] += 1
        V[2*k][col] += 1
        W[4*k+2][col] += 1
        col += 1
        
        # M3: a(f-h)
        # a=U[0]
        # f=V[2k+1], h=V[6+2k+1]
        # C01 = M3 + M5.
        # C01 index: W[4k+1].
        U[0][col] += 1
        V[2*k+1][col] += 1; V[6+2*k+1][col] -= 1
        W[4*k+1][col] += 1
        col += 1
        
        # M4: d(g-c)
        # d=U[6]
        # g=V[6+2k], c=V[2k] (Wait, c is from A? No, g,c are from B? No.)
        # Strassen: M4 = d(g-c). d is A11. g is B10, c is B00? No.
        # Standard Strassen:
        # M4 = d * (g - c) is NOT standard.
        # M4 = d * (g - c) ?
        # Let's re-verify Strassen.
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-c)
        # M5 = (a+b)h
        # M6 = (c-d)(e+f)
        # M7 = (b-d)(g+h)
        
        # A: a=A00, b=A01, c=A10, d=A11.
        # B: e=B00, f=B01, g=B10, h=B11.
        
        # M4: d(g-c). d=A11=U[6]. g=B10=V[6+2k]. c=A10=U[5]? No, c is A10.
        # Wait, M4 = d * (g - c). c is A10.
        # So U: d(6)=1, c(5)=-1.
        # V: g(6+2k)=1.
        # C: C00 += M4, C10 += M4.
        U[6][col] += 1; U[5][col] -= 1
        V[6+2*k][col] += 1
        W[4*k][col] += 1; W[4*k+2][col] += 1
        col += 1
        
        # M5: (a+b)h
        # a=U[0], b=U[1]. h=V[6+2k+1].
        # C: C00 -= M5, C01 += M5.
        U[0][col] += 1; U[1][col] += 1
        V[6+2*k+1][col] += 1
        W[4*k][col] -= 1; W[4*k+1][col] += 1
        col += 1
        
        # M6: (c-d)(e+f)
        # c=U[5], d=U[6]. e=V[2k], f=V[2k+1].
        # C: C11 += M6.
        U[5][col] += 1; U[6][col] -= 1
        V[2*k][col] += 1; V[2*k+1][col] += 1
        W[4*k+3][col] += 1
        col += 1
        
        # M7: (b-d)(g+h)
        # b=U[1], d=U[6]. g=V[6+2k], h=V[6+2k+1].
        # C: C00 += M7.
        U[1][col] += 1; U[6][col] -= 1
        V[6+2*k][col] += 1; V[6+2*k+1][col] += 1
        W[4*k][col] += 1
        col += 1

        # --- Part 2: m2=3 (rows 2,3,4 of A, rows 2,3,4 of B) ---
        # Schoolbook for 2x3 * 3x2.
        # A_m2:
        # A20=U[2*5+0]=U[10], A21=U[11], A22=U[12]
        # A30=U[3*5+0]=U[15], A31=U[16], A32=U[17]
        # B_m2 (rows 2,3,4 of B, cols 2k, 2k+1):
        # B20=V[2*6+2k]=V[12+2k], B21=V[13+2k]
        # B30=V[3*6+2k]=V[18+2k], B31=V[19+2k]
        # B40=V[4*6+2k]=V[24+2k], B41=V[25+2k]
        # C_m2 (rows 2k, 2k+1 of C):
        # C00=W[4k], C01=W[4k+1], C10=W[4k+2], C11=W[4k+3]
        
        # 6 terms for C00: A20B20, A21B30, A22B40, A30B20... No.
        # C00 = A20 B20 + A21 B30 + A22 B40 + A30 B20? No.
        # C00 = A[0,0]B[0,0] + A[0,1]B[1,0] + A[0,2]B[2,0]
        # A_m2 row 0: A20, A21, A22.
        # B_m2 col 0: B20, B30, B40.
        
        # Term 1: A20 B20
        U[10][col] += 1
        V[12+2*k][col] += 1
        W[4*k][col] += 1
        col += 1
        
        # Term 2: A21 B30
        U[11][col] += 1
        V[18+2*k][col] += 1
        W[4*k][col] += 1
        col += 1
        
        # Term 3: A22 B40
        U[12][col] += 1
        V[24+2*k][col] += 1
        W[4*k][col] += 1
        col += 1
        
        # Term 4: A20 B21 (for C01)
        U[10][col] += 1
        V[13+2*k][col] += 1
        W[4*k+1][col] += 1
        col += 1
        
        # Term 5: A21 B31
        U[11][col] += 1
        V[19+2*k][col] += 1
        W[4*k+1][col] += 1
        col += 1
        
        # Term 6: A22 B41
        U[12][col] += 1
        V[25+2*k][col] += 1
        W[4*k+1][col] += 1
        col += 1
        
        # Term 7: A30 B20 (for C10)
        U[15][col] += 1
        V[12+2*k][col] += 1
        W[4*k+2][col] += 1
        col += 1
        
        # Term 8: A31 B30
        U[16][col] += 1
        V[18+2*k][col] += 1
        W[4*k+2][col] += 1
        col += 1
        
        # Term 9: A32 B40
        U[17][col] += 1
        V[24+2*k][col] += 1
        W[4*k+2][col] += 1
        col += 1
        
        # Term 10: A30 B21 (for C11)
        U[15][col] += 1
        V[13+2*k][col] += 1
        W[4*k+3][col] += 1
        col += 1
        
        # Term 11: A31 B31
        U[16][col] += 1
        V[19+2*k][col] += 1
        W[4*k+3][col] += 1
        col += 1
        
        # Term 12: A32 B41
        U[17][col] += 1
        V[25+2*k][col] += 1
        W[4*k+3][col] += 1
        col += 1

    return {"U": U, "V": V, "W": W}
