"""
Matrix multiplication tensor decomposition: 4x4 * 4x9 -> 4x9.

Strategy:
  Use Strassen's algorithm for the 4x4 * 4x4 blocks (rank 49 each)
  and schoolbook for the 4x4 * 4x1 block (rank 16).
  Split 9 columns as 4 + 4 + 1.
  Total rank: 49 + 49 + 16 = 114.

The rank-7 decomposition for 2x2 * 2x2 -> 2x2 (Strassen):
  p1 = (a+c)(e+g)
  p2 = (a-d)(h-f)
  p3 = (a+b)(f-h)
  p4 = (c+d)(e+f)
  p5 = a*e
  p6 = b*f
  p7 = d*h

  C11 = p1 + p4 - p5 - p7
  C12 = p3 + p5
  C21 = p2 + p7
  C22 = p1 - p2 + p3 + p6

  Wait, let me verify this. For A=[[a,b],[c,d]], B=[[e,f],[g,h]]:
  C = AB = [[ae+cg, af+dh], [ce+dg, cf+gh]]

  p1 = (a+c)(e+g) = ae + ag + ce + cg
  p2 = (a-d)(h-f) = ah - af - dh + df
  p3 = (a+b)(f-h) = af - ah + bf - bh
  p4 = (c+d)(e+f) = ce + cf + de + df
  p5 = ae
  p6 = bf
  p7 = dh

  C11 = ae + cg
  C12 = af + dh
  C21 = ce + dg
  C22 = cf + gh

  Let me check: p1 + p4 - p5 - p7 = (ae+ag+ce+cg) + (ce+cf+de+df) - ae - dh
  = ag + 2ce + cg + cf + de + df - dh
  That's not ae + cg. So this decomposition is wrong.

Let me use the correct Strassen decomposition for 2x2:
  p1 = (a+c)(e+g)
  p2 = (a-d)(h-f)
  p3 = (a+b)(f-h)
  p4 = (c+d)(e+f)
  p5 = a*e
  p6 = b*f
  p7 = d*h

  C11 = p1 + p4 - p5 - p7
  Let me recompute:
  p1 = ae + ag + ce + cg
  p4 = ce + cf + de + df
  p5 = ae
  p7 = dh
  p1 + p4 - p5 - p7 = ae + ag + ce + cg + ce + cf + de + df - ae - dh
  = ag + 2ce + cg + cf + de + df - dh
  Still wrong.

The correct Strassen 2x2 decomposition:
  p1 = (a+c)(e+g)
  p2 = (a-d)(h-f)
  p3 = (a+b)(f-h)
  p4 = (c+d)(e+f)
  p5 = a*e
  p6 = b*f
  p7 = d*h

  C11 = p1 - p4 + p5 + p7
  C12 = p3 + p5
  C21 = p2 + p7
  C22 = p1 - p2 + p3 - p4 + p6

Let me verify:
  C11 = p1 - p4 + p5 + p7 = (ae+ag+ce+cg) - (ce+cf+de+df) + ae + dh
  = ae + ag + ce + cg - ce - cf - de - df + ae + dh
  = 2ae + ag + cg - cf - de - df + dh
  Still wrong.

I need to find the correct rank-7 decomposition. Let me just use the known one:
  p1 = (a+c)(e+g)
  p2 = (a-d)(h-f)
  p3 = (a+b)(f-h)
  p4 = (c+d)(e+f)
  p5 = a*e
  p6 = b*f
  p7 = d*h

  C11 = p1 + p4 - p5 - p7  ... this doesn't work as shown.

Let me try a different known decomposition. The standard Strassen for 2x2:
  M1 = (A11 + A22)(B11 + B22)
  M2 = (A21 + A22)B11
  M3 = A11(B12 - B22)
  M4 = A22(B21 - B11)
  M5 = (A11 + A12)B22
  M6 = (A21 - A11)(B11 + B12)
  M7 = (A12 - A22)(B21 + B22)

  C11 = M1 + M4 - M5 + M7
  C12 = M3 + M5
  C21 = M2 + M4
  C22 = M1 - M2 + M3 + M6

For 2x2, A11=a, A12=b, A21=c, A22=d, B11=e, B12=f, B21=g, B22=h:
  M1 = (a+d)(e+h)
  M2 = (c+d)e
  M3 = a(f-h)
  M4 = d(g-e)
  M5 = (a+b)h
  M6 = (c-a)(e+f)
  M7 = (b-d)(g+h)

  C11 = M1 + M4 - M5 + M7
  = (a+d)(e+h) + d(g-e) - (a+b)h + (b-d)(g+h)
  = ae + ah + de + dh + dg - de - ah - bh + bg + bh - dg - dh
  = ae + bg
  But C11 should be ae + cg. So this is also wrong!

I'm clearly making errors. Let me just use schoolbook for everything and accept rank 144, but try to be smarter.

Actually, for 4x4 * 4x9, schoolbook is 4*4*9 = 144.

Let me just implement a clean schoolbook solution.
"""

import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS  # n=4, m=4, p=9
    U_rows = n * m  # 16
    V_rows = m * p  # 36
    W_rows = p * n  # 36
    
    rank = n * m * p  # 144
    U = [[0] * rank for _ in range(U_rows)]
    V = [[0] * rank for _ in range(V_rows)]
    W = [[0] * rank for _ in range(W_rows)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
