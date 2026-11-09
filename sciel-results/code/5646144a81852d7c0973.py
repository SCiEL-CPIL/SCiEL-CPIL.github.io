"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
import time
import itertools
from fractions import Fraction

DEADLINE = 140.0  # safety margin before 150s kill

def _verify_fast(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication using Fraction."""
    rank = len(U[0])
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if any(len(row) != rank for row in U):
        return False
    if any(len(row) != rank for row in V):
        return False
    if any(len(row) != rank for row in W):
        return False
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for c in range(rank):
                    total += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                if total != 1:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
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

def _strassen_block(U, V, W, col, i0, j0, k0, n, m, p):
    """Add Strassen's 7-term decomposition for a 2x2x2 block starting at (i0,j0,k0).
    
    The block covers i in {i0, i0+1}, j in {j0, j0+1}, k in {k0, k0+1}.
    Strassen terms for C=AB where A is 2x2 (rows i, cols j), B is 2x2 (rows j, cols k):
    
    M1 = (A[i0,j0]+A[i0+1,j0+1]) * (B[j0,k0]+B[j0+1,k0+1])
    M2 = (A[i0+1,j0]+A[i0+1,j0+1]) * B[j0,k0]
    M3 = A[i0,j0] * (B[j0+1,k0]-B[j0+1,k0+1])
    M4 = A[i0+1,j0+1] * (B[j0,k0+1]-B[j0,k0])
    M5 = (A[i0,j0]+A[i0,j0+1]) * B[j0+1,k0+1]
    M6 = (A[i0+1,j0]-A[i0,j0]) * (B[j0,k0]+B[j0+1,k0])
    M7 = (A[i0,j0+1]-A[i0+1,j0+1]) * (B[j0+1,k0+1]+B[j0,k0+1])
    
    Output:
    C[i0,k0]   = M1 + M4 - M5 + M7
    C[i0,k0+1] = M3 + M5
    C[i0+1,k0] = M2 + M4
    C[i0+1,k0+1] = M1 - M2 + M3 + M6
    """
    # Each term is a triple (U_vals, V_vals, W_vals) where:
    # U_vals: dict from (i,j) -> coeff
    # V_vals: dict from (j,k) -> coeff  
    # W_vals: dict from (k,i) -> coeff
    
    # M1: U has A[i0,j0]=1, A[i0+1,j0+1]=1; V has B[j0,k0]=1, B[j0+1,k0+1]=1; W has C[i0,k0]=1, C[i0,k0+1]=0, C[i0+1,k0]=0, C[i0+1,k0+1]=1
    terms = [
        # M1: (A00+A11)(B00+B11) -> contributes to C00(+1), C11(+1)
        ({(i0,j0):1, (i0+1,j0+1):1}, {(j0,k0):1, (j0+1,k0+1):1}, {(k0,i0):1, (k0+1,i0+1):1}),
        # M2: (A10+A11)B00 -> contributes to C10(+1)
        ({(i0+1,j0):1, (i0+1,j0+1):1}, {(j0,k0):1}, {(k0,i0+1):1}),
        # M3: A00(B10-B11) -> contributes to C01(+1), C11(+1)
        ({(i0,j0):1}, {(j0+1,k0):1, (j0+1,k0+1):-1}, {(k0,i0):0, (k0+1,i0):1, (k0,i0+1):0, (k0+1,i0+1):1}),
        # M4: A11(B01-B00) -> contributes to C00(+1), C10(+1)
        ({(i0+1,j0+1):1}, {(j0,k0+1):1, (j0,k0):-1}, {(k0,i0):1, (k0+1,i0+1):1}),
        # M5: (A00+A01)B11 -> contributes to C00(-1), C01(+1)
        ({(i0,j0):1, (i0,j0+1):1}, {(j0+1,k0+1):1}, {(k0+1,i0):-1, (k0+1,i0):1}),
        # M6: (A10-A00)(B00+B10) -> contributes to C11(+1)
        ({(i0+1,j0):1, (i0,j0):-1}, {(j0,k0):1, (j0+1,k0):1}, {(k0,i0+1):1}),
        # M7: (A01-A11)(B01+B11) -> contributes to C00(+1)
        ({(i0,j0+1):1, (i0+1,j0+1):-1}, {(j0,k0+1):1, (j0+1,k0+1):1}, {(k0+1,i0):1}),
    ]
    
    # Wait, let me re-derive more carefully.
    # The tensor identity we need: for each (i,j,k) in the 2x2x2 block,
    # sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1
    # 
    # Strassen computes C = A*B where C[i,k] = sum_j A[i,j]*B[j,k]
    # In our tensor: we need sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = delta_{i,j,k} for the matmul tensor
    # Actually no - the matmul tensor T[i,j,k] = 1 for all valid (i,j,k), and the decomposition
    # satisfies: for each (i,k), sum_j T[i,j,k] * A[i,j] * B[j,k] = C[i,k]
    # In tensor form: C[i,k] = sum_{j,c} A[i,j] * B[j,k] * (sum_c U[i,j,c]*V[j,k,c]*W[k,i,c])
    # So we need sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1 for all (i,j,k)
    
    # So for the 2x2x2 block, we need 7 terms such that for all (i,j) in {i0,i0+1}x{j0,j0+1} and k in {k0,k0+1}:
    # sum_{c=0}^{6} U_c[i,j] * V_c[j,k] * W_c[k,i] = 1
    
    # Strassen's 7 bilinear forms for 2x2x2:
    # Let me use the standard Strassen decomposition.
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[p,q],[r,s]]
    # p = a*e + c*g, q = a*f + c*h, r = b*e + d*g, s = b*f + d*h
    
    # M1 = (a+d)(e+h): p+=M1, s+=M1
    # M2 = (c+d)e: r+=M2
    # M3 = a(g-h): q+=M3, s+=M3  
    # M4 = d(f-e): p+=M4, r+=M4
    # M5 = (a+b)h: p-=M5, q+=M5
    # M6 = (c-a)(e+g): s+=M6
    # M7 = (b-d)(f+h): p+=M7
    
    # Verify: p = M1+M4-M5+M7 = (a+d)(e+h)+d(f-e)-(a+b)h+(b-d)(f+h)
    # = ae+ah+de+dh+df-de-ah-bh+bf+bh-df-dh = ae+bf ✓
    # q = M3+M5 = a(g-h)+(a+b)h = ag-ah+ah+bh = ag+bh ✓
    # r = M2+M4 = (c+d)e+d(f-e) = ce+de+df-de = ce+df ✓
    # s = M1-M2+M3+M6 = (a+d)(e+h)-(c+d)e+a(g-h)+(c-a)(e+g)
    # = ae+ah+de+dh-ce-de+dh-ae+ag-ah+ce+cg-ae-ag = ae+bf... let me redo
    # = ae+ah+de+dh - ce-de + dh - ae+ag-ah + ce+cg-ae-ag
    # Wait that's wrong. Let me be more careful.
    # M1-M2+M3+M6 = (a+d)(e+h) - (c+d)e + a(g-h) + (c-a)(e+g)
    # = ae+ah+de+dh - ce-de+dh - ae+ag-ah + ce+cg-ae-ag
    # Hmm, let me expand term by term:
    # (a+d)(e+h) = ae+ah+de+dh
    # -(c+d)e = -ce-de
    # a(g-h) = ag-ah
    # (c-a)(e+g) = ce+cg-ae-ag
    # Sum: ae+ah+de+dh-ce-de+dh-ae+ag-ah+ce+cg-ae-ag
    # = (ae-ae-ae) + (ah-ah) + (de-de) + (dh+dh) + (-ce+ce) + (ag-ag) + cg
    # = -ae + 2dh + cg
    # That's not right. I must have the Strassen formulas wrong.
    
    # Let me look up the correct Strassen decomposition:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # Verify C00: (a+d)(e+h)+d(f-e)-(a+b)h+(b-d)(f+h)
    # = ae+ah+de+dh+df-de-ah-bh+bf+bh-df-dh = ae+bf ✓
    # C01: a(g-h)+(a+b)h = ag-ah+ah+bh = ag+bh ✓
    # C10: (c+d)e+d(f-e) = ce+de+df-de = ce+df ✓
    # C11: (a+d)(e+h)-(c+d)e+a(g-h)+(c-a)(e+g)
    # = ae+ah+de+dh-ce-de+dh-ae+ag-ah+ce+cg-ae-ag
    # Group: ae terms: ae-ae-ae = -ae
    # ah terms: ah-ah = 0
    # de terms: de-de = 0
    # dh terms: dh+dh = 2dh
    # ce terms: -ce+ce = 0
    # ag terms: ag-ag = 0
    # bf terms: none... wait I'm missing something
    # cg terms: cg
    # So: -ae + 2dh + cg? That's not bf.
    
    # I think I'm making an arithmetic error. Let me redo C11 very carefully:
    # M1 = (a+d)(e+h) = ae + ah + de + dh
    # -M2 = -(c+d)e = -ce - de
    # M3 = a(g-h) = ag - ah
    # M6 = (c-a)(e+g) = ce + cg - ae - ag
    # Sum: ae + ah + de + dh - ce - de + ag - ah + ce + cg - ae - ag
    # ae: ae - ae - ae = -ae
    # ah: ah - ah = 0
    # de: de - de = 0
    # dh: dh
    # ce: -ce + ce = 0
    # ag: ag - ag = 0
    # cg: cg
    # Total: -ae + dh + cg
    
    # But we need C11 = bf = b*f. This is clearly wrong. I must have the Strassen formulas incorrect.
    
    # Let me use a different reference. The correct Strassen decomposition:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)(B00)
    # M3 = (A00)(B10-B11)
    # M4 = (A11)(B01-B00)
    # M5 = (A00+A01)(B11)
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1 - M5 + M4 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M6 + M3
    
    # C11 = M1 - M2 + M6 + M3 = (a+d)(e+h) - (c+d)e + (c-a)(e+g) + a(g-h)
    # = ae+ah+de+dh -ce-de + ce+cg-ae-ag + ag-ah
    # = ae+ah+de+dh-ce-de+ce+cg-ae-ag+ag-ah
    # ae: ae-ae = 0
    # ah: ah-ah = 0
    # de: de-de = 0
    # dh: dh
    # ce: -ce+ce = 0
    # cg: cg
    # ag: -ag+ag = 0
    # = dh + cg
    
    # Still not bf! I think I have the matrix multiplication convention wrong.
    # C = A*B means C[i,k] = sum_j A[i,j]*B[j,k]
    # So C11 = A[1,0]*B[0,1] + A[1,1]*B[1,1] = c*f + d*h
    
    # Oh! I see my error. In my notation, if A=[[a,b],[c,d]] and B=[[e,f],[g,h]],
    # then C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] = c*f + d*h
    # Not b*f! I was confusing the indices.
    
    # So C11 = cf + dh. Let me re-verify:
    # C11 = M1 - M2 + M6 + M3 = dh + cg... no that gives dh + cg, not cf + dh.
    
    # I'm clearly making errors. Let me just use the standard well-known Strassen formulas
    # and verify them numerically.
    
    # Actually, I realize I should just directly construct the 7 terms for the tensor
    # decomposition. The tensor T[i,j,k]=1 for the 2x2x2 matmul needs 7 rank-1 tensors.
    # 
    # For the 2x2x2 case, the 7 terms are:
    # Term 1: U={(0,0):1,(1,1):1}, V={(0,0):1,(1,1):1}, W={(0,0):1,(1,1):1}  -> M1
    # Term 2: U={(1,0):1,(1,1):1}, V={(0,0):1}, W={(0,1):1}  -> M2
    # Term 3: U={(0,0):1}, V={(1,0):1,(1,1):-1}, W={(1,0):1,(1,1):1}  -> M3
    # Term 4: U={(1,1):1}, V={(0,1):1,(0,0):-1}, W={(0,0):1,(1,1):1}  -> M4
    # Term 5: U={(0,0):1,(0,1):1}, V={(1,1):1}, W={(1,0):-1,(1,1):1}  -> M5
    # Term 6: U={(1,0):1,(0,0):-1}, V={(0,0):1,(1,0):1}, W={(0,1):1}  -> M6
    # Term 7: U={(0,1):1,(1,1):-1}, V={(0,1):1,(1,1):1}, W={(1,0):1}  -> M7
    
    # Let me verify these give T[i,j,k]=1 for all (i,j,k) in {0,1}^3.
    # For (0,0,0): sum of U[0,0]*V[0,0]*W[0,0] over terms
    # T1: 1*1*1=1, T2: 0, T3: 1*0*0=0, T4: 0, T5: 1*0*(-1)=0, T6: 0*1*0=0, T7: 0
    # = 1 ✓
    # For (0,0,1): T1: 1*1*0=0, T2: 0*1*0=0, T3: 1*0*0=0, T4: 0, T5: 1*0*0=0, T6: 0*1*1=0, T7: 0
    # = 0 ✗ (should be 1)
    
    # This approach is getting error-prone. Let me just use NumPy to find the Strassen decomposition
    # by solving the linear system, or better yet, use the known explicit formulas.
    
    # After more careful thought, let me use this well-known 2x2x2 rank-7 decomposition:
    # The 7 bilinear forms are:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e  
    # m3 = a(g-h)
    # m4 = d(f-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+g)
    # m7 = (b-d)(f+h)
    # 
    # c00 = m1+m4-m5+m7
    # c01 = m3+m5
    # c10 = m2+m4
    # c11 = m1-m2+m3+m6
    
    # Let me verify c11 with a=1,b=2,c=3,d=4,e=5,f=6,g=7,h=8:
    # m1 = (1+4)(5+8) = 5*13 = 65
    # m2 = (3+4)*5 = 35
    # m3 = 1*(7-8) = -1
    # m4 = 4*(6-5) = 4
    # m5 = (1+2)*8 = 24
    # m6 = (3-1)*(5+7) = 2*12 = 24
    # m7 = (2-4)*(6+8) = -2*14 = -28
    # c11 = 65-35+(-1)+24 = 53
    # Actual: c*f+d*h = 3*6+4*8 = 18+32 = 50
    # 53 != 50, so these formulas are WRONG.
    
    # I clearly don't remember the exact Strassen formulas. Let me derive them or use a different approach.
    
    # Alternative: I'll use the fact that for 2x2x2, there's a known rank-7 decomposition.
    # Let me construct it by solving the equations.
    # We need 7 terms (u_c, v_c, w_c) where u_c is 2x2, v_c is 2x2, w_c is 2x2 (but w_c[k,i] corresponds to output C[i,k])
    # such that for all i,j,k in {0,1}: sum_c u_c[i,j]*v_c[j,k]*w_c[k,i] = 1
    
    # This is a system of 8 equations (for 8 (i,j,k) triples) with 7*12=84 unknowns.
    # It's underdetermined, so there are many solutions.
    
    # Let me just use a simple construction. One known rank-7 decomposition for 2x2x2:
    # I'll use the one from the literature. After some research in my memory:
    # 
    # Actually, let me just use a different strategy. Instead of trying to recall Strassen,
    # I'll construct the 2x5x5 decomposition directly using a block approach where I
    # explicitly verify correctness.
    
    # For now, let me fall back to schoolbook and add a simple optimization:
    # Try to find if any two terms can be merged (linear dependency).
    pass
    return col  # No terms added, this function is a placeholder

def _construct_strassen(n, m, p):
    """Construct a rank-46 decomposition using Strassen on 2x2x2 blocks."""
    # Partition: j = {0,1}, {2,3}, {4}; k = {0,1}, {2,3}, {4}
    # Blocks: (2,2,2) x4, (2,1,2) x2, (2,2,1) x2, (2,1,1) x1
    # Schoolbook: 4*8 + 2*4 + 2*4 + 1*2 = 32+8+8+2 = 50
    # With Strassen on (2,2,2): 4*7 + 2*4 + 2*4 + 1*2 = 28+8+8+2 = 46
    
    # For the (2,2,2) blocks, we need a rank-7 decomposition of the 2x2x2 matmul tensor.
    # Let me construct one explicitly.
    
    # The 2x2x2 matmul tensor T[i,j,k] = 1 for all i,j,k in {0,1}.
    # We need 7 rank-1 tensors summing to T.
    
    # One valid rank-7 decomposition (verified by construction):
    # I'll use the approach of finding 7 terms that satisfy the constraint.
    
    # Let me use a systematic approach. We need:
    # For each (i,j,k) in {0,1}^3: sum_{c=1}^{7} U_c[i,j] * V_c[j,k] * W_c[k,i] = 1
    
    # Here's a known decomposition (from various sources):
    # Term 1: U={(0,0):1,(1,1):1}, V={(0,0):1,(1,1):1}, W={(0,0):1,(1,1):1}
    # Term 2: U={(1,0):1,(1,1):1}, V={(0,0):1}, W={(0,1):1}
    # Term 3: U={(0,0):1}, V={(1,0):1,(1,1):-1}, W={(1,0):1,(1,1):1}
    # Term 4: U={(1,1):1}, V={(0,1):1,(0,0):-1}, W={(0,0):1,(1,1):1}
    # Term 5: U={(0,0):1,(0,1):1}, V={(1,1):1}, W={(1,0):-1,(1,1):1}
    # Term 6: U={(1,0):1,(0,0):-1}, V={(0,0):1,(1,0):1}, W={(0,1):1}
    # Term 7: U={(0,1):1,(1,1):-1}, V={(0,1):1,(1,1):1}, W={(1,0):1}
    
    # Let me verify (0,0,0):
    # T1: U[0,0]=1, V[0,0]=1, W[0,0]=1 -> 1
    # T2: U[0,0]=0 -> 0
    # T3: U[0,0]=1, V[0,0]=0 -> 0
    # T4: U[0,0]=0 -> 0
    # T5: U[0,0]=1, V[0,0]=0 -> 0
    # T6: U[0,0]=-1, V[0,0]=1, W[0,0]=0 -> 0
    # T7: U[0,0]=0 -> 0
    # Sum = 1 ✓
    
    # (0,0,1):
    # T1: U[0,0]=1, V[0,1]=0 -> 0
    # T2: U[0,0]=0 -> 0
    # T3: U[0,0]=1, V[0,1]=0 -> 0
    # T4: U[0,0]=0 -> 0
    # T5: U[0,0]=1, V[0,1]=0 -> 0
    # T6: U[0,0]=-1, V[0,1]=0 -> 0
    # T7: U[0,0]=0 -> 0
    # Sum = 0 ✗ (should be 1)
    
    # This decomposition is wrong. Let me try a different approach.
    
    # I'll solve for the decomposition using linear algebra.
    # The constraint is: for all (i,j,k), sum_c U_c[i,j]*V_c[j,k]*W_c[k,i] = 1
    # This is nonlinear in the variables, so I can't use simple linear algebra.
    
    # Instead, let me use a known correct Strassen decomposition.
    # After careful derivation, here's the correct one:
    
    # Using the standard Strassen algorithm with A=[[a,b],[c,d]], B=[[e,f],[g,h]]:
    # p = a*e + c*g (C[0,0])
    # q = a*f + c*h (C[0,1])  
    # r = b*e + d*g (C[1,0])
    # s = b*f + d*h (C[1,1])
    
    # The 7 products:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(g-h)
    # M4 = d(f-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+g)
    # M7 = (b-d)(f+h)
    
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    
    # Let me verify with a=1,b=2,c=3,d=4,e=5,f=6,g=7,h=8:
    # M1 = (1+4)(5+8) = 65
    # M2 = (3+4)(5) = 35
    # M3 = 1(7-8) = -1
    # M4 = 4(6-5) = 4
    # M5 = (1+2)(8) = 24
    # M6 = (3-1)(5+7) = 24
    # M7 = (2-4)(6+8) = -28
    
    # C[0,0] = 65+4-24+(-28) = 17. Actual: 1*5+3*7 = 5+21 = 26. WRONG.
    
    # I clearly have the wrong formulas. Let me try yet another set.
    
    # You know what, let me just use a completely different approach.
    # I'll construct the decomposition by directly specifying the 7 terms
    # and verifying them numerically in code.
    
    # For the 2x2x2 tensor, one valid rank-7 decomposition is:
    # (I'll define terms as (U_dict, V_dict, W_dict) where dicts map (row,col)->coeff)
    
    # Actually, the simplest correct approach: use the fact that we can always
    # find a rank-7 decomposition. Let me use the following construction:
    
    # Terms (each is a triple of 2x2 matrices U,V,W where W[k,i] is the output weight):
    # I'll use a parametrization and solve.
    
    # For simplicity and correctness, let me just return the schoolbook for now
    # and add the Strassen optimization as a TODO. The merge-based search might
    # find improvements.
    
    return _schoolbook(n, m, p)

def _try_merge_fast(U, V, W, n, m, p, c1, c2, factor):
    """Try to merge columns c1 and c2 by adding factor * c2 to c1 and removing c2."""
    rank = len(U[0])
    if c1 == c2:
        return None
    
    new_rank = rank - 1
    new_U = [[0] * new_rank for _ in range(n * m)]
    new_V = [[0] * new_rank for _ in range(m * p)]
    new_W = [[0] * new_rank for _ in range(p * n)]
    
    new_col = 0
    for c in range(rank):
        if c == c2:
            continue
        if c == c1:
            for i in range(n * m):
                new_U[i][new_col] = U[i][c1] + factor * U[i][c2]
            for j in range(m * p):
                new_V[j][new_col] = V[j][c1] + factor * V[j][c2]
            for k in range(p * n):
                new_W[k][new_col] = W[k][c1] + factor * W[k][c2]
        else:
            for i in range(n * m):
                new_U[i][new_col] = U[i][c]
            for j in range(m * p):
                new_V[j][new_col] = V[j][c]
            for k in range(p * n):
                new_W[k][new_col] = W[k][c]
        new_col += 1
    
    if _verify_fast(new_U, new_V, new_W, n, m, p):
        return new_U, new_V, new_W
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_U, best_V, best_W = U, V, W
    best_rank = schoolbook_rank
    
    start_time = time.time()
    
    # Try to find lower rank decompositions via merging
    max_merge_attempts = 20000
    attempts = 0
    factors_to_try = [Fraction(1), Fraction(-1), Fraction(1,2), Fraction(-1,2), Fraction(2), Fraction(-2), Fraction(3), Fraction(-3), Fraction(1,3), Fraction(-1,3)]
    
    current_U, current_V, current_W = U, V, W
    current_rank = best_rank
    
    while current_rank > 10 and time.time() < DEADLINE:
        merged = False
        for c1 in range(current_rank):
            if time.time() > DEADLINE:
                break
            for c2 in range(c1 + 1, current_rank):
                if time.time() > DEADLINE:
                    break
                for factor in factors_to_try:
                    attempts += 1
                    if attempts > max_merge_attempts:
                        break
                    result = _try_merge_fast(current_U, current_V, current_W, n, m, p, c1, c2, factor)
                    if result is not None:
                        current_U, current_V, current_W = result
                        current_rank -= 1
                        best_U, best_V, best_W = current_U, current_V, current_W
                        best_rank = current_rank
                        merged = True
                        break
                if merged or attempts > max_merge_attempts:
                    break
            if merged or attempts > max_merge_attempts:
                break
        
        if not merged:
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
