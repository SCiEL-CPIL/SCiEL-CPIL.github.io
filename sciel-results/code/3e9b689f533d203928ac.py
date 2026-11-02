import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    # 1. Generate Strassen 2x2 decomposition
    # Rank 7
    # u, v, w are lists of 7 vectors of length 4
    # Index 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
    
    # Strassen terms
    # M1: (a+d)(e+h)
    # M2: (c+d)e
    # M3: a(f-h)
    # M4: d(g-e)
    # M5: (a+b)h
    # M6: (c-a)(e+f)
    # M7: (b-d)(g+h)
    
    # We need to map these to u, v, w vectors
    # u[i,j], v[j,k], w[k,i]
    
    U2 = [[0]*4 for _ in range(7)]
    V2 = [[0]*4 for _ in range(7)]
    W2 = [[0]*4 for _ in range(7)]
    
    # Helper to set u[i,j]
    def set_u(r, i, j, val): U2[r][i*2+j] = val
    def set_v(r, j, k, val): V2[r][j*2+k] = val
    def set_w(r, k, i, val): W2[r][k*2+i] = val
    
    # M1: (A00+A11)(B00+B11)
    # u: (0,0)=1, (1,1)=1
    # v: (0,0)=1, (1,1)=1
    # w: (0,0)=1, (1,1)=1
    set_u(0, 0, 0, 1); set_u(0, 1, 1, 1)
    set_v(0, 0, 0, 1); set_v(0, 1, 1, 1)
    set_w(0, 0, 0, 1); set_w(0, 1, 1, 1)
    
    # M2: (A10+A11)B00
    # u: (1,0)=1, (1,1)=1
    # v: (0,0)=1
    # w: (0,1)=1  (C10)
    set_u(1, 1, 0, 1); set_u(1, 1, 1, 1)
    set_v(1, 0, 0, 1)
    set_w(1, 0, 1, 1)
    
    # M3: A00(B01-B11)
    # u: (0,0)=1
    # v: (0,1)=1, (1,1)=-1
    # w: (1,0)=1 (C10? No, C01)
    # C01 = M3 + M5.
    # C_{0,1}. k=0, i=1.
    set_u(2, 0, 0, 1)
    set_v(2, 0, 1, 1); set_v(2, 1, 1, -1)
    set_w(2, 0, 1, 1)
    
    # M4: A11(B10-B00)
    # u: (1,1)=1
    # v: (1,0)=1, (0,0)=-1
    # w: (0,0)=1 (C00), (1,1)=1 (C11)?
    # C00 = M1 + M4 - M5 + M7.
    # C11 = M1 - M2 + M3 + M6.
    # M4 contributes to C00 and C11?
    # M4 = d(g-e). d=A11, g=B10, e=B00.
    # d*g -> A11 B10 -> C10.
    # d*-e -> -A11 B00 -> -C10?
    # Wait. C = AB.
    # C10 = A10 B00 + A11 B10.
    # M4 = A11(B10-B00).
    # This contributes to C10.
    # But Strassen formula: C10 = M2 + M4.
    # M2 = (A10+A11)B00 = A10 B00 + A11 B00.
    # M4 = A11 B10 - A11 B00.
    # Sum = A10 B00 + A11 B10. Correct.
    # So M4 contributes to C10.
    # k=1, i=0.
    set_u(3, 1, 1, 1)
    set_v(3, 1, 0, 1); set_v(3, 0, 0, -1)
    set_w(3, 1, 0, 1)
    
    # M5: (A00+A01)B11
    # u: (0,0)=1, (0,1)=1
    # v: (1,1)=1
    # w: (1,0)=-1 (C00), (1,1)=1 (C01)?
    # C00 = ... - M5 ...
    # C01 = M3 + M5.
    # M5 = A00 B11 + A01 B11.
    # A00 B11 -> C01.
    # A01 B11 -> C01.
    # So M5 contributes to C01.
    # k=1, i=1? No.
    # A00 is row 0. B11 is col 1. C01.
    # k=1, i=1? No, C_{k,i}.
    # C_{1,1}?
    # A_{0,0} B_{1,1} -> C_{0,1}.
    # A_{0,1} B_{1,1} -> C_{0,1}.
    # So k=1, i=1?
    # Wait. C_{i,k} in standard?
    # Task: W[k*n+i].
    # C_{k,i}.
    # A_{i,j} B_{j,k}.
    # A_{0,0} B_{1,1} -> i=0, j=0, k=1.
    # So C_{1,0}.
    # My previous M3 analysis: A00(B01-B11).
    # A00 B01 -> i=0, j=0, k=1 -> C_{1,0}.
    # A00 -B11 -> i=0, j=1, k=1 -> C_{1,0}.
    # So M3 contributes to C_{1,0}.
    # M5: A00 B11 -> C_{1,0}. A01 B11 -> C_{1,0}.
    # So M5 contributes to C_{1,0}.
    # Strassen: C_{1,0} = M2 + M4.
    # C_{0,1} = M3 + M5.
    # My indices:
    # C_{k,i}.
    # C_{1,0} is k=1, i=0.
    # C_{0,1} is k=0, i=1.
    
    # Let's re-verify M3.
    # M3 = A00(B01-B11).
    # A00 B01 -> i=0, j=0, k=1. C_{1,0}.
    # A00 -B11 -> i=0, j=1, k=1. C_{1,0}.
    # So M3 contributes to C_{1,0}.
    # But Strassen says C_{0,1} = M3 + M5.
    # There is a mismatch in my index convention vs Strassen standard.
    # Standard Strassen:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # In my W[k*n+i], C_{k,i} corresponds to C_{row k, col i}.
    # Standard C_{row, col}.
    # So C_{1,1} (row 1, col 1) is k=1, i=1.
    # C_{1,2} (row 1, col 2) is k=1, i=2? No, n=2.
    # C_{1,2} doesn't exist.
    # C_{0,1} (row 0, col 1) is k=0, i=1.
    
    # So:
    # M3 contributes to C_{0,1}.
    # My calculation: A00 B01 -> C_{1,0}.
    # Why?
    # A_{i,j} B_{j,k}.
    # A_{0,0} B_{0,1}. i=0, j=0, k=1.
    # Result C_{0,1}.
    # Task W index: k*n+i = 1*2+0 = 2.
    # So W[2] is C_{1,0}.
    # W[1] is C_{0,1} (0*2+1).
    
    # So M3 should set W[1] (k=0, i=1).
    
    # Let's fix the W assignments.
    
    # M1: C00, C11. W[0], W[3].
    # M2: C10. W[2].
    # M3: C01. W[1].
    # M4: C10, C11?
    # M4 = A11(B10-B00).
    # A11 B10 -> i=1, j=1, k=0. C_{1,0}. W[2].
    # A11 -B00 -> i=1, j=0, k=0. C_{1,0}. W[2].
    # So M4 contributes to C10.
    # Strassen: C21 = M2 + M4. (C10).
    # C11 = M1 + M4 - M5 + M7.
    # M4 contributes to C11?
    # A11 B10 -> C10.
    # A11 -B00 -> C10.
    # Where does M4 contribute to C11?
    # It doesn't.
    # Strassen formula: C11 = M1 + M4 - M5 + M7.
    # M1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11.
    # M4 = A11B10 - A11B00.
    # M5 = A00B11 + A01B11.
    # M7 = A01B10 + A01B11 - A11B10 - A11B11.
    # Sum:
    # A00B00 (M1)
    # A00B11 (M1 - M5) = 0?
    # A11B00 (M1 - M4) = 0?
    # A11B11 (M1 + M7) = A11B11 + A01B10 + A01B11 - A11B10 - A11B11 = A01B10 + A01B11.
    # This is getting messy.
    
    # I will use a known correct Strassen tensor decomposition.
    # Or simply use the schoolbook for 2x2 as well, to ensure correctness, then optimize?
    # No, rank 8 is worse than 7.
    
    # I will hardcode the correct Strassen vectors.
    
    pass
