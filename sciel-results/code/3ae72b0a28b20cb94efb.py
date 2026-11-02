"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-49 decomposition using Strassen's algorithm on 7 independent 2x2 blocks.
    We treat B as a 2x(2*7) matrix of 7 independent 2x2 blocks, and C as a (2*7)x2
    matrix of 7 independent 2x2 blocks. The tensor is the direct sum of 7 copies of
    the 2x2x2 multiplication tensor, each of rank 7.
    
    Strassen's algorithm for a 2x2 matmul:
    Let A = [[a, b], [c, d]] (2x2)
    Let B = [[e, f], [g, h]] (2x2)
    
    M1 = (a+d)(e+h)
    M2 = (c+d)(e)
    M3 = (a-b)(g+h)
    M4 = (a+b)(h)
    M5 = (a+c)(e) - d(g+h)
    M6 = (b+d)(f) - a(g+h)
    M7 = (a-c)(h) - b(e+f)
    
    Output C = [[C11, C12], [C21, C22]] where:
    C11 = M1 + M4 - M5 - M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    In the tensor decomposition:
    U[i*m + j, r] is the coefficient for A_ij
    V[j*p + k, r] is the coefficient for B_jk
    W[k*n + i, r] is the coefficient for C_ki  (W[k*n + i] where k is col, i is row of C)
    
    For block l (0..6):
    B block columns are 2l, 2l+1.
    B_{0, 2l} has index 0*14 + 2l = 2l
    B_{0, 2l+1} has index 0*14 + (2l+1) = 2l+1
    B_{1, 2l} has index 1*14 + 2l = 14 + 2l
    B_{1, 2l+1} has index 1*14 + (2l+1) = 14 + 2l + 1
    
    C block: column in full C is 2l or 2l+1.
    C_{0, 2l} index: (2l)*2 + 0 = 4l
    C_{0, 2l+1} index: (2l+1)*2 + 0 = 4l + 2
    C_{1, 2l} index: (2l)*2 + 1 = 4l + 1
    C_{1, 2l+1} index: (2l+1)*2 + 1 = 4l + 3
    """
    n, m, p = DIMENSIONS
    rank = 7 * p  # 7 * 14 = 98? Wait, p=14, but we have 7 blocks of size 2.
    # Actually, 7 blocks, each rank 7, so total rank = 7*7 = 49.
    
    U = [[0.0] * rank for _ in range(n * m)]  # 4 x 49
    V = [[0.0] * rank for _ in range(m * p)]  # 28 x 49
    W = [[0.0] * rank for _ in range(p * n)]  # 28 x 49
    
    # Strassen intermediate coefficients
    # Each M_s contributes to A, B, and C
    # s = 0..6 for the 7 Strassen products
    
    # A coefficients for each M_s: [a, b, c, d]
    A_coeff = [
        [1, 0, 0, 1],   # M1: (a+d)
        [0, 0, 1, 1],   # M2: (c+d)
        [1, -1, 0, 0],  # M3: (a-b)
        [1, 1, 0, 0],   # M4: (a+b)
        [1, 0, 1, 0],   # M5: (a+c) - d(...) -> the (a+c) part
        [0, 1, 0, 1],   # M6: (b+d) - a(...) -> the (b+d) part
        [1, 0, -1, 0],  # M7: (a-c) - b(...) -> the (a-c) part
    ]
    
    # Also, M5 has -d*(g+h), M6 has -a*(g+h), M7 has -b*(e+f)
    # These are additional A coefficients:
    # M5: -d -> [0, 0, 0, -1]
    # M6: -a -> [-1, 0, 0, 0]
    # M7: -b -> [0, -1, 0, 0]
    # But wait, these are part of the same M5, M6, M7.
    # Actually, M5 = (a+c) - d*(g+h), so A coeff for M5 is [1, 0, 1, -1]
    # M6 = (b+d) - a*(g+h), so A coeff for M6 is [-1, 1, 0, 1]
    # M7 = (a-c) - b*(e+f), so A coeff for M7 is [1, -1, -1, 0]
    
    A_coeff = [
        [1, 0, 0, 1],   # M1: (a+d)
        [0, 0, 1, 1],   # M2: (c+d)
        [1, -1, 0, 0],  # M3: (a-b)
        [1, 1, 0, 0],   # M4: (a+b)
        [1, 0, 1, -1],  # M5: (a+c) - d
        [-1, 1, 0, 1],  # M6: (b+d) - a
        [1, -1, -1, 0], # M7: (a-c) - b
    ]
    
    # B coefficients for each M_s: [e, f, g, h]
    B_coeff = [
        [1, 0, 0, 1],   # M1: (e+h)
        [1, 0, 0, 0],   # M2: e
        [0, 0, 1, 1],   # M3: (g+h)
        [0, 0, 0, 1],   # M4: h
        [0, 0, 1, 1],   # M5: (g+h)
        [0, 0, 1, 1],   # M6: (g+h)
        [1, 1, 0, 0],   # M7: (e+f)
    ]
    
    # C coefficients for each M_s: [C11, C12, C21, C22]
    C_coeff = [
        [1, 0, 0, 1],   # M1
        [0, 0, -1, -1], # M2
        [0, 1, 0, 1],   # M3
        [1, 0, 1, 0],   # M4
        [-1, 1, 0, 0],  # M5
        [0, 0, 0, 1],   # M6
        [-1, 0, 0, 0],  # M7
    ]
    
    for l in range(7):
        for s in range(7):
            r = l * 7 + s  # column in U, V, W
            
            # U entries: A_i_j at index i*m + j = i*2 + j
            # A_coeff[s] = [a, b, c, d] for i=0,1 and j=0,1
            U[0 * 2 + 0][r] += A_coeff[s][0]  # a
            U[0 * 2 + 1][r] += A_coeff[s][1]  # b
            U[1 * 2 + 0][r] += A_coeff[s][2]  # c
            U[1 * 2 + 1][r] += A_coeff[s][3]  # d
            
            # V entries: B_j_k at index j*p + k
            # B_coeff[s] = [e, f, g, h] for j=0,1 and k=2l, 2l+1
            e, f, g, h = B_coeff[s]
            V[0 * 14 + (2 * l)][r] += e
            V[0 * 14 + (2 * l + 1)][r] += f
            V[1 * 14 + (2 * l)][r] += g
            V[1 * 14 + (2 * l + 1)][r] += h
            
            # W entries: C_k_i at index k*n + i
            # C_coeff[s] = [C11, C12, C21, C22]
            # C11 -> k=2l, i=0 -> index (2l)*2 + 0 = 4l
            # C12 -> k=2l+1, i=0 -> index (2l+1)*2 + 0 = 4l+2
            # C21 -> k=2l, i=1 -> index (2l)*2 + 1 = 4l+1
            # C22 -> k=2l+1, i=1 -> index (2l+1)*2 + 1 = 4l+3
            c11, c12, c21, c22 = C_coeff[s]
            W[4 * l + 0][r] += c11   # C_{0, 2l}
            W[4 * l + 2][r] += c12   # C_{0, 2l+1}
            W[4 * l + 1][r] += c21   # C_{1, 2l}
            W[4 * l + 3][r] += c22   # C_{1, 2l+1}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
