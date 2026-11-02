"""Program to find a rank-minimal exact tensor decomposition for 2x4x7 matrix multiplication."""
from fractions import Fraction

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    
    # Collect all terms
    terms = []
    
    j_blocks = [(0, 1), (2, 3)]
    k_blocks = [(0, 1), (2, 3), (4, 5), (6, None)]
    
    for j0, j1 in j_blocks:
        for k_block in k_blocks:
            k0 = k_block[0]
            k1 = k_block[1]
            
            if k1 is not None:
                # Strassen 2x2x2
                # i0=0, i1=1
                # A[i0][j0]=A[0][j0], A[i0][j1]=A[0][j1], A[i1][j0]=A[1][j0], A[i1][j1]=A[1][j1]
                # B[j0][k0], B[j0][k1], B[j1][k0], B[j1][k1]
                # C[i0][k0]=C[0][k0], C[i0][k1]=C[0][k1], C[i1][k0]=C[1][k0], C[i1][k1]=C[1][k1]
                
                # M1 = A[0][j0] * (B[j0][k0] + B[j1][k1])
                # Contributes: -M1 to C[0][k0], +M1 to C[0][k1]
                terms.append(({0*4+j0: 1}, {j0*7+k0: 1, j1*7+k1: 1}, {k0*2+0: -1, k1*2+0: 1}))
                
                # M2 = (A[1][j0] + A[1][j1]) * B[j0][k0]
                # Contributes: +M2 to C[0][k1], -M2 to C[1][k1]
                terms.append(({1*4+j0: 1, 1*4+j1: 1}, {j0*7+k0: 1}, {k1*2+0: 1, k1*2+1: -1}))
                
                # M3 = A[0][j0] * (B[j1][k0] - B[j1][k1])
                # Contributes: +M3 to C[1][k0], -M3 to C[1][k1]
                terms.append(({0*4+j0: 1}, {j1*7+k0: 1, j1*7+k1: -1}, {k0*2+1: 1, k1*2+1: -1}))
                
                # M4 = A[1][j1] * (B[j0][k1] - B[j0][k0])
                # Contributes: +M4 to C[0][k0], +M4 to C[1][k0]
                terms.append(({1*4+j1: 1}, {j0*7+k1: 1, j0*7+k0: -1}, {k0*2+0: 1, k0*2+1: 1}))
                
                # M5 = (A[0][j0] + A[1][j1]) * (B[j0][k0] + B[j1][k1])
                # Contributes: +M5 to C[0][k0], +M5 to C[1][k1]
                terms.append(({0*4+j0: 1, 1*4+j1: 1}, {j0*7+k0: 1, j1*7+k1: 1}, {k0*2+0: 1, k1*2+1: 1}))
                
                # M6 = (A[1][j0] - A[0][j0]) * (B[j0][k1] + B[j1][k1])
                # Contributes: -M6 to C[1][k1]
                terms.append(({1*4+j0: 1, 0*4+j0: -1}, {j0*7+k1: 1, j1*7+k1: 1}, {k1*2+1: -1}))
                
                # M7 = (A[0][j1] - A[1][j1]) * (B[j1][k0] + B[j0][k1])
                # Contributes: +M7 to C[0][k0]
                terms.append(({0*4+j1: 1, 1*4+j1: -1}, {j1*7+k0: 1, j0*7+k1: 1}, {k0*2+0: 1}))
            else:
                # Schoolbook 2x2x1
                k0_single = k0
                terms.append(({0*4+j0: 1}, {j0*7+k0_single: 1}, {k0_single*2+0: 1}))
                terms.append(({0*4+j1: 1}, {j1*7+k0_single: 1}, {k0_single*2+0: 1}))
                terms.append(({1*4+j0: 1}, {j0*7+k0_single: 1}, {k0_single*2+1: 1}))
                terms.append(({1*4+j1: 1}, {j1*7+k0_single: 1}, {k0_single*2+1: 1}))
    
    rank = len(terms)
    U = [[Fraction(0)] * rank for _ in range(8)]
    V = [[Fraction(0)] * rank for _ in range(28)]
    W = [[Fraction(0)] * rank for _ in range(14)]
    
    for c, (U_dict, V_dict, W_dict) in enumerate(terms):
        for idx, coeff in U_dict.items():
            U[idx][c] = Fraction(coeff)
        for idx, coeff in V_dict.items():
            V[idx][c] = Fraction(coeff)
        for idx, coeff in W_dict.items():
            W[idx][c] = Fraction(coeff)
    
    return {"U": U, "V": V, "W": W}
