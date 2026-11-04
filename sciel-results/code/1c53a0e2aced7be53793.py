import numpy as np
import time
from fractions import Fraction

START_TIME = time.time()
TIME_LIMIT = 110  # Safety margin for 120s limit

def construct():
    n, m, p = 3, 5, 7
    
    # Split m=5 into m1=2, m2=3
    # C = A[:,:2] @ B[:2,:] + A[:,2:] @ B[2:,:]
    # Term 1: 3x2 x 2x7 -> rank 42 (trivial)
    # Term 2: 3x3 x 3x7 -> rank 49 (using Strassen on 3x3x3 per "slice")
    
    m1, m2 = 2, 3
    
    # Term 1: 3x2x7 trivial
    r1 = n * m1 * p  # 42
    U1 = [[0.0 for _ in range(r1)] for _ in range(n * m1)]
    V1 = [[0.0 for _ in range(r1)] for _ in range(m1 * p)]
    W1 = [[0.0 for _ in range(r1)] for _ in range(p * n)]
    
    idx = 0
    for i in range(n):
        for j in range(m1):
            for k in range(p):
                U1[i*m1 + j][idx] = 1.0
                V1[j*p + k][idx] = 1.0
                W1[k*n + i][idx] = 1.0
                idx += 1
    
    # Term 2: 3x3x7 using Strassen-like decomposition
    # For 3x3x3, Strassen rank is 7. For 3x3x7, we can use rank 7*7=49? 
    # Actually, let's use trivial for now: rank = 3*3*7 = 63
    # But let's try to get 49.
    
    # The 3x3x7 tensor: we can decompose it as the "tensor product" of 3x3x3 and 1x1x7?
    # No. Let's just use trivial 63 for now, total = 42+63 = 105. No gain.
    
    # Better: use the fact that 3x3x7 can be done with rank 49.
    # Idea: 3x3x7 = (3x3x3) * (1x1x7) in some sense.
    # Actually, the correct way: the 3x3x7 multiplication tensor has the same
    # structure as 3x3x3 but with 7 output columns. We can use the Strassen
    # decomposition of 3x3x3 and apply it to each "column block".
    
    # For 3x3x3, Strassen uses 7 terms. Each term involves products of linear
    # combinations. For 3x3x7, we can use the same 7 linear combinations on
    # the 3x3 part, but the 7 output dimensions interact.
    
    # Actually, I think the rank of 3x3x7 is at most 7 * 7 = 49, where one 7
    # comes from Strassen's 7 multiplications and the other 7 from the output
    # dimension. Let me construct this.
    
    # Strassen for 3x3x3:
    # A = [[a11,a12,a13],[a21,a22,a23],[a31,a32,a33]]
    # B = [[b11,b12,b13],[b21,b22,b23],[b31,b32,b33]]
    # 7 products:
    # m1 = (a11+a33)*(b11+b33)
    # m2 = (a21+a23)*b13
    # m3 = a11*(b23-b33)
    # m4 = a33*(b11-b21)
    # m5 = (a13+a22)*(b21+b22)
    # m6 = (a11-a22)*(b12+b32)
    # m7 = (a23-a33)*(b21+b31)
    
    # For 3x3x7, B is 3x7. Let B = [B1 B2 B3] where each Bi is 3x... no,
    # B is 3 rows, 7 columns. Let's denote B[j][k] for j in 0..2, k in 0..6.
    
    # The 7 Strassen products for 3x3x3 generalize to 3x3x7 by having each
    # "b" term be a row of B (length 7).
    
    # m1 = (a11+a33) * (b1 + b3)  where b1, b3 are rows 0,2 of B (each length 7)
    # m2 = (a21+a23) * b1
    # m3 = a11 * (b2 - b3)
    # m4 = a33 * (b1 - b2)
    # m5 = (a13+a22) * (b2)
    # m6 = (a11-a22) * (b2 + b3)  # wait, this doesn't look right
    
    # Let me look up the exact Strassen formula for 3x3.
    
    # Actually, for 3x3x7, the trivial rank is 63. To get 49, I need a 
    # non-trivial decomposition. Let me just use 63 for now.
    
    r2 = n * m2 * p  # 63
    U2 = [[0.0 for _ in range(r2)] for _ in range(n * m2)]
    V2 = [[0.0 for _ in range(r2)] for _ in range(m2 * p)]
    W2 = [[0.0 for _ in range(r2)] for _ in range(p * n)]
    
    idx = 0
    for i in range(n):
        for j in range(m2):
            for k in range(p):
                U2[i*m2 + j][idx] = 1.0
                V2[j*p + k][idx] = 1.0
                W2[k*n + i][idx] = 1.0
                idx += 1
    
    # Combine: total rank = r1 + r2 = 42 + 63 = 105. Same as before!
    # This splitting doesn't help with trivial decomposition.
    
    # Let me try a different approach: use Strassen on the whole thing.
    # Pad 3x5 to 4x8, 5x7 to 8x8, use Strassen on 8x8x8.
    # 8x8x8 Strassen rank = 7^3 = 343. Worse.
    
    # Alternative: use the "lattice" method or known small-case decompositions.
    
    # For now, let's just return the trivial 105-rank decomposition.
    rank = n * m * p
    U = [[0.0 for _ in range(rank)] for _ in range(n * m)]
    V = [[0.0 for _ in range(rank)] for _ in range(m * p)]
    W = [[0.0 for _ in range(rank)] for _ in range(p * n)]
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][idx] = 1.0
                V[j*p + k][idx] = 1.0
                W[k*n + i][idx] = 1.0
                idx += 1
    
    return {'U': U, 'V': V, 'W': W}
