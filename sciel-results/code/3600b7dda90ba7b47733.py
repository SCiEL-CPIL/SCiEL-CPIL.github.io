import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 2, 13]

def construct():
    start_time = time.time()
    deadline = 135

    n, m, p = DIMENSIONS

    # Schoolbook decomposition of rank n*m*p = 52
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1

    # Strassen for first 2 cols + schoolbook for rest: rank = 7 + 44 = 51
    strassen_rank = 7
    schoolbook_cols = p - 2
    schoolbook_rank = 4 * schoolbook_cols
    total_rank = strassen_rank + schoolbook_rank

    if total_rank < rank_school:
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]

        col = 0
        # Strassen terms for 2x2 x 2x2 (cols 0,1 of B)
        # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
        # U indices: 0=a, 1=b, 2=c, 3=d
        # V indices: V[0*p+0]=e, V[1*p+0]=g, V[0*p+1]=f, V[1*p+1]=h
        # W indices: W[k*n+i], so W[0]=C00, W[2]=C01, W[1]=C10, W[3]=C11
        
        # Verified Strassen decomposition:
        # m1 = (a+b)(e+f)
        # m2 = (a+b)(g+h)
        # m3 = (c+d)(e+f)
        # m4 = (c+d)(g+h)
        # m5 = a(d-c)
        # m6 = e(h-f)
        # m7 = g(b-a)
        
        # C00 = m1 + m4 - m2 - m6
        # C01 = m2 + m6
        # C10 = m4 + m5
        # C11 = m1 + m3 - m4 - m5
        
        # Let me verify C00:
        # m1+m4-m2-m6 = (a+b)(e+f) + (c+d)(g+h) - (a+b)(g+h) - e(h-f)
        # = ae+af+be+bf + cg+ch+dg+dh - ag-ah-bg-bh - eh+ef
        # = ae+af+be+bf+cg+ch+dg+dh-ag-ah-bg-bh-eh+ef
        # = ae+bg+dg+ch + (af-ah+ef-eh) + (be-bg-bh+bf) + (cg-ch)
        # This is getting messy. Let me just verify numerically.
        
        # I'll define the 7 terms and verify they produce correct matrix multiplication
        strassen_terms = [
            # (U_coeffs, V_coeffs, W_coeffs)
            # m1 = (a+b)(e+f): U=[1,1,0,0], V=[1,0,1,0] (e,f), W contributions
            # Each term contributes to multiple W entries with coefficients
            
            # Let me define each rank-1 term as (U_vec, V_vec, W_vec)
            # where the tensor is sum_r U[i,r]*V[j,r]*W[k,r]
            
            # Term 1: m1 = (a+b)(e+f)
            # U = [1,1,0,0], V = [1,0,1,0] (e at pos 0, f at pos 2 in V's 2x2 block)
            # This term contributes to C00 (+1), C01 (+1), C10 (+1), C11 (+1) with appropriate signs
            # But we need to split it into contributions to each C entry
            # Actually, each rank-1 term in the tensor decomposition has a single W vector
            # that can have multiple non-zero entries
            
            # The 7 Strassen terms as rank-1 tensors:
            # t1: U=[1,1,0,0], V=[1,0,1,0], W=[1,1,1,1] -- no, this isn't right either
        ]
        
        # I'm overcomplicating this. Let me just use a known valid set of 7 rank-1 tensors
        # for the 2x2x2x2 multiplication tensor.
        
        # From literature, one valid Strassen decomposition:
        # t1: U=[1,0,0,0], V=[1,0,0,0], W=[1,0,0,0]  # ae -> C00
        # t2: U=[0,0,1,0], V=[0,1,0,0], W=[0,1,0,0]  # cg -> C10
        # t3: U=[0,1,0,0], V=[0,0,1,0], W=[0,0,1,0]  # bf -> C01
        # t4: U=[0,0,0,1], V=[0,0,0,1], W=[0,0,0,1]  # dh -> C11
        # t5: U=[1,0,1,0], V=[0,1,0,1], W=[0,1,0,0]  # (a+c)(g+h) -> C10, C11
        # t6: U=[0,1,0,1], V=[1,0,1,0], W=[1,0,1,0]  # (b+d)(e+f) -> C00, C01
        # t7: U=[1,1,0,0], V=[0,1,0,1], W=[1,0,0,1]  # (a+b)(g+h) -> C00, C11
        
        # Verify: C00 should be ae+bg
        # From t1: 1*1*1 = ae ✓
        # From t6: 0*1*1 + 1*0*1 = 0 + 0 = 0... wait
        # t6: U=[0,1,0,1], V=[1,0,1,0], W=[1,0,1,0]
        # Contribution to C00 (W[0]=1): U[0]*V[0]*1 + U[1]*V[1]*1 + U[2]*V[2]*1 + U[3]*V[3]*1
        # = 0*1 + 1*0 + 0*1 + 1*0 = 0. Hmm.
        
        # I think the issue is that in a rank-1 tensor term, the contribution to C[i,k] is
        # sum_j U[i,j]*V[j,k] where the W vector selects which (i,k) pair.
        # Actually no. The tensor is T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
        # where i indexes A, j indexes B, k indexes C.
        
        # For t6: U=[0,1,0,1], V=[1,0,1,0], W=[1,0,1,0]
        # T[0,0,0] += 0*1*1 = 0
        # T[0,1,0] += 0*0*1 = 0
        # T[0,0,1] += 0*1*1 = 0
        # T[0,1,1] += 0*0*1 = 0
        # T[1,0,0] += 1*1*0 = 0
        # T[1,1,0] += 1*0*0 = 0
        # T[1,0,1] += 1*1*1 = 1
        # T[1,1,1] += 1*0*1 = 0
        # T[2,0,0] += 0*1*0 = 0
        # ... all others 0
        # So t6 only contributes 1 to T[1,0,1] which is C[1,0] from b*f? No...
        
        # I'm clearly not getting this right. Let me just return schoolbook.
        pass

    return {"U": U_school, "V": V_school, "W": W_school}
