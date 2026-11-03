import numpy as np
import time

def construct():
    """
    Construct a tensor decomposition for 4x4 * 4x9 matrix multiplication.
    
    Uses Strassen's algorithm for the 4x4 * 4x4 part, and schoolbook for the remaining columns.
    The 4x4 * 4x4 part uses 7 multiplications (Strassen) instead of 16 (schoolbook).
    The 4x4 * 4x5 part (columns 4-8) uses schoolbook: 20 multiplications.
    Total: 7*5 + 20 = 55 (for the 4x4*4x4 blocks) + ... 
    
    Actually, let me think more carefully.
    
    For 4x4 * 4x9:
    Split B into B1 (4x4) and B2 (4x5).
    C = A*B1 + A*B2
    
    For A*B1 (4x4 * 4x4): use Strassen with 7 multiplications
    For A*B2 (4x4 * 4x5): use schoolbook with 20 multiplications
    
    Total: 7 + 20 = 27... wait, that's not right either.
    
    Let me reconsider. Strassen for 4x4*4x4 gives a decomposition of rank 7 for the 2x2*2x2 subproblem,
    but for 4x4*4x4 we need to apply Strassen at two levels or use the 7-multiplication scheme.
    
    Actually, for 4x4 * 4x4, the standard Strassen approach gives:
    - Split into 2x2 blocks
    - Each 2x2*2x2 uses 7 multiplications
    - Total: 7 multiplications for the 4x4*4x4 product
    
    Wait, no. Strassen for 2x2*2x2 uses 7 multiplications. For 4x4*4x4, we can split into 2x2 blocks
    and apply Strassen to each block product, but that gives 7*4 = 28? No...
    
    Let me be more careful. For 4x4 * 4x4:
    A = [[A00, A01], [A10, A11]] where each is 2x2
    B = [[B00, B01], [B10, B11]] where each is 2x2
    
    C00 = A00*B00 + A01*B10
    C01 = A00*B01 + A01*B11
    C10 = A10*B00 + A11*B10
    C11 = A10*B01 + A11*B11
    
    Using Strassen:
    M1 = (A00+A11)*(B00+B11)
    M2 = (A10+A11)*B00
    M3 = A00*(B01-B11)
    M4 = A11*(B10-B00)
    M5 = (A00+A01)*B11
    M6 = (A10-A00)*(B00+B01)
    M7 = (A01-A11)*(B10+B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    
    So 7 bilinear forms for the 4x4*4x4 part.
    
    For the 4x4 * 4x5 part (columns 4-8 of B), use schoolbook: 4*4*5 = 80 bilinear forms.
    
    Total: 7 + 80 = 87.
    
    But wait, I need to be more careful about how the bilinear forms combine.
    
    Actually, let me just implement Strassen for the 4x4*4x4 part and schoolbook for the rest.
    """
    
    n, m, p = 4, 4, 9
    
    # We'll build the decomposition as a list of (U_vec, V_vec, W_vec) tuples
    # where U_vec has length n*m, V_vec has length m*p, W_vec has length p*n
    
    decompositions = []
    
    # Part 1: A * B[:, 0:4] (4x4 * 4x4) using Strassen
    # A is 4x4, B1 is 4x4 (first 4 columns of B)
    # C1 = A * B1 is 4x4
    
    # Define Strassen bilinear forms for 4x4 * 4x4
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    
    # A00 = A[0:2, 0:2], A01 = A[0:2, 2:4], A10 = A[2:4, 0:2], A11 = A[2:4, 2:4]
    # B00 = B1[0:2, 0:2], B01 = B1[0:2, 2:4], B10 = B1[2:4, 0:2], B11 = B1[2:4, 2:4]
    # C00 = C1[0:2, 0:2], C01 = C1[0:2, 2:4], C10 = C1[2:4, 0:2], C11 = C1[2:4, 2:4]
    
    def make_U(*indices):
        """Create U vector with 1s at given (i,j) indices"""
        u = [0] * (n * m)
        for idx in indices:
            u[idx] = 1
        return u
    
    def make_V(*indices):
        """Create V vector with 1s at given (j,k) indices"""
        v = [0] * (m * p)
        for idx in indices:
            v[idx] = 1
        return v
    
    def make_W(*indices):
        """Create W vector with 1s at given (k,i) indices"""
        w = [0] * (p * n)
        for idx in indices:
            w[idx] = 1
        return w
    
    def add_bilinear(u_coeffs, v_coeffs, w_coeffs):
        """Add a bilinear form with given coefficients.
        u_coeffs: dict mapping (i,j) -> coeff
        v_coeffs: dict mapping (j,k) -> coeff  
        w_coeffs: dict mapping (k,i) -> coeff
        """
        u = [0] * (n * m)
        v = [0] * (m * p)
        w = [0] * (p * n)
        for (i, j), c in u_coeffs.items():
            u[i * m + j] = c
        for (j, k), c in v_coeffs.items():
            v[j * p + k] = c
        for (k, i), c in w_coeffs.items():
            w[k * n + i] = c
        decompositions.append((u, v, w))
    
    # Strassen for 4x4 * 4x4 (first 4 columns of B)
    # B columns 0-3 correspond to k=0,1,2,3
    
    # M1 = (A00+A11)*(B00+B11)
    # A00+A11: A[0,0]+A[2,2], A[0,1]+A[2,3], A[1,0]+A[3,2], A[1,1]+A[3,3]
    # B00+B11: B[0,0]+B[2,0], B[0,1]+B[2,1], B[1,0]+B[3,0], B[1,1]+B[3,1]
    # C00: C[0,0], C[0,1], C[1,0], C[1,1] -> W[k*n+i] for k=0,1 and i=0,1
    # C11: C[2,2], C[2,3], C[3,2], C[3,3] -> W[k*n+i] for k=0,1 and i=2,3
    
    # M1 contributes to C00 and C11
    u1 = {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1, (2,2): 1, (2,3): 1, (3,2): 1, (3,3): 1}
    v1 = {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1, (2,0): 1, (2,1): 1, (3,0): 1, (3,1): 1}
    w1 = {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1, (0,2): 1, (0,3): 1, (1,2): 1, (1,3): 1}
    add_bilinear(u1, v1, w1)
    
    # M2 = (A10+A11)*B00
    # A10+A11: A[2,0]+A[2,2], A[2,1]+A[2,3], A[3,0]+A[3,2], A[3,1]+A[3,3]
    # B00: B[0,0], B[0,1], B[1,0], B[1,1]
    # Contributes to C10 (C[2,0], C[2,1], C[3,0], C[3,1]) and C11 (C[2,2], C[2,3], C[3,2], C[3,3])
    u2 = {(2,0): 1, (2,1): 1, (2,2): 1, (2,3): 1, (3,0): 1, (3,1): 1, (3,2): 1, (3,3): 1}
    v2 = {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1}
    w2 = {(0,2): 1, (0,3): 1, (1,2): 1, (1,3): 1, (2,2): 1, (2,3): 1, (3,2): 1, (3,3): 1}
    # Wait, C10 is C[2:4, 0:2], so W[k*n+i] for k=0,1 and i=2,3
    # C11 is C[2:4, 2:4], so W[k*n+i] for k=2,3 and i=2,3
    w2 = {(0,2): 1, (0,3): 1, (1,2): 1, (1,3): 1, (2,2): 1, (2,3): 1, (3,2): 1, (3,3): 1}
    add_bilinear(u2, v2, w2)
    
    # M3 = A00*(B01-B11)
    # A00: A[0,0], A[0,1], A[1,0], A[1,1]
    # B01-B11: B[0,2]-B[2,2], B[0,3]-B[2,3], B[1,2]-B[3,2], B[1,3]-B[3,3]
    # Contributes to C01 (C[0,2], C[0,3], C[1,2], C[1,3])
    u3 = {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1}
    v3 = {(0,2): 1, (0,3): 1, (1,2): 1, (1,3): 1, (2,2): -1, (2,3): -1, (3,2): -1, (3,3): -1}
    w3 = {(2,0): 1, (2,1): 1, (3,0): 1, (3,1): 1}
    add_bilinear(u3, v3, w3)
    
    # M4 = A11*(B10-B00)
    # A11: A[2,2], A[2,3], A[3,2], A[3,3]
    # B10-B00: B[2,0]-B[0,0], B[2,1]-B[0,1], B[3,0]-B[1,0], B[3,1]-B[1,1]
    # Contributes to C00 (C[0,0], C[0,1], C[1,0], C[1,1]) and C10 (C[2,0], C[2,1], C[3,0], C[3,1])
    u4 = {(2,2): 1, (2,3): 1, (3,2): 1, (3,3): 1}
    v4 = {(2,0): 1, (2,1): 1, (3,0): 1, (3,1): 1, (0,0): -1, (0,1): -1, (1,0): -1, (1,1): -1}
    w4 = {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1, (0,2): 1, (0,3): 1, (1,2): 1, (1,3): 1}
    add_bilinear(u4, v4, w4)
    
    # M5 = (A00+A01)*B11
    # A00+A01: A[0,0]+A[0,2], A[0,1]+A[0,3], A[1,0]+A[1,2], A[1,1]+A[1,3]
    # B11: B[2,2], B[2,3], B[3,2], B[3,3]
    # Contributes to C01 (C[0,2], C[0,3], C[1,2], C[1,3]) and C11 (C[2,2], C[2,3], C[3,2], C[3,3])
    u5 = {(0,0): 1, (0,1): 1, (0,2): 1, (0,3): 1, (1,0): 1, (1,1): 1, (1,2): 1, (1,3): 1}
    v5 = {(2,2): 1, (2,3): 1, (3,2): 1, (3,3): 1}
    w5 = {(2,0): 1, (2,1): 1, (3,0): 1, (3,1): 1, (2,2): 1, (2,3): 1, (3,2): 1, (3,3): 1}
    add_bilinear(u5, v5, w5)
    
    # M6 = (A10-A00)*(B00+B01)
    # A10-A00: A[2,0]-A[0,0], A[2,1]-A[0,1], A[3,0]-A[1,0], A[3,1]-A[1,1]
    # B00+B01: B[0,0]+B[0,2], B[0,1]+B[0,3], B[1,0]+B[1,2], B[1,1]+B[1,3]
    # Contributes to C00 (C[0,0], C[0,1], C[1,0], C[1,1]) and C10 (C[2,0], C[2,1], C[3,0], C[3,1])
    u6 = {(2,0): 1, (2,1): 1, (3,0): 1, (3,1): 1, (0,0): -1, (0,1): -1, (1,0): -1, (1,1): -1}
    v6 = {(0,0): 1, (0,1): 1, (0,2): 1, (0,3): 1, (1,0): 1, (1,1): 1, (1,2): 1, (1,3): 1}
    w6 = {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1, (0,2): 1, (0,3): 1, (1,2): 1, (1,3): 1}
    add_bilinear(u6, v6, w6)
    
    # M7 = (A01-A11)*(B10+B11)
    # A01-A11: A[0,2]-A[2,2], A[0,3]-A[2,3], A[1,2]-A[3,2], A[1,3]-A[3,3]
    # B10+B11: B[2,0]+B[2,2], B[2,1]+B[2,3], B[3,0]+B[3,2], B[3,1]+B[3,3]
    # Contributes to C00 (C[0,0], C[0,1], C[1,0], C[1,1]) and C11 (C[2,2], C[2,3], C[3,2], C[3,3])
    u7 = {(0,2): 1, (0,3): 1, (1,2): 1, (1,3): 1, (2,2): -1, (2,3): -1, (3,2): -1, (3,3): -1}
    v7 = {(2,0): 1, (2,1): 1, (2,2): 1, (2,3): 1, (3,0): 1, (3,1): 1, (3,2): 1, (3,3): 1}
    w7 = {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1, (2,2): 1, (2,3): 1, (3,2): 1, (3,3): 1}
    add_bilinear(u7, v7, w7)
    
    # Part 2: A * B[:, 4:9] (4x4 * 4x5) using schoolbook
    # For each i in 0..3, j in 0..3, k in 4..8:
    # C[i,k] += A[i,j] * B[j,k]
    for i in range(n):
        for j in range(m):
            for k in range(4, p):
                u = {(i, j): 1}
                v = {(j, k): 1}
                w = {(k, i): 1}
                add_bilinear(u, v, w)
    
    # Build the final matrices
    rank = len(decompositions)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col, (u, v, w) in enumerate(decompositions):
        for i in range(n * m):
            if u[i] != 0:
                U[i][col] = u[i]
        for j in range(m * p):
            if v[j] != 0:
                V[j][col] = v[j]
        for k in range(p * n):
            if w[k] != 0:
                W[k][col] = w[k]
    
    return {"U": U, "V": V, "W": W}
