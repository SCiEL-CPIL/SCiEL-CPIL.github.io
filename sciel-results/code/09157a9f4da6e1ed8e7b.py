import numpy as np

def construct():
    n, m, p = 4, 4, 7
    
    # Use Strassen's algorithm for 2x2 block matrix multiplication
    # A is 4x4, B is 4x7
    # Split into 2x2 blocks:
    # A = [[A11, A12], [A21, A22]] where each is 2x2
    # B = [[B11, B12], [B21, B22]] where B11,B12 are 2x4, B21,B22 are 2x3
    # Wait, B is 4x7, so split as:
    # B = [[B11, B12], [B21, B22]] where B11 is 2x4, B12 is 2x3, B21 is 2x4, B22 is 2x3
    
    # Actually, let's be more careful. B is 4x7.
    # If we split B into two blocks of size 2x7 each:
    # B = [B_top; B_bottom] where B_top is 2x7, B_bottom is 2x7
    # A = [A_top_left A_top_right; A_bottom_left A_bottom_right] where each is 2x2
    
    # C = A*B is 4x7
    # C = [C_top; C_bottom] where C_top is 2x7, C_bottom is 2x7
    
    # Using Strassen-like approach:
    # M1 = A_top_left * B_top
    # M2 = A_bottom_left * B_top  
    # M3 = A_top_left * B_bottom
    # M4 = A_bottom_left * B_bottom
    # M5 = (A_top_left + A_bottom_left) * B_top
    # M6 = (A_top_left + A_bottom_left) * B_bottom
    # M7 = A_top_right * (B_top + B_bottom)
    # M8 = A_bottom_right * (B_top + B_bottom)
    
    # This doesn't really save anything. Let me think differently.
    
    # For 4x4 * 4x7, the best known approach might be to use the fact that
    # we can decompose this using block Strassen.
    
    # Let's try a different approach: use the known result that rank(4,4,7) 
    # might be achievable with fewer than 112 multiplications.
    
    # Actually, for now, let's implement a proper Strassen decomposition for blocks.
    
    # Split A (4x4) into 2x2 blocks of size 2x2
    # Split B (4x7) into 2x7 blocks (top and bottom)
    
    # A = [[A11, A12], [A21, A22]] each 2x2
    # B = [[B1], [B2]] where B1 is 2x7, B2 is 2x7
    
    # C = A*B = [[A11*B1 + A12*B2], [A21*B1 + A22*B2]]
    
    # We need to compute:
    # C11 = A11*B1 + A12*B2 (2x7)
    # C21 = A21*B1 + A22*B2 (2x7)
    
    # Each of these requires 2 matrix multiplications of 2x2 * 2x7
    # A 2x2 * 2x7 multiplication has rank 4 (schoolbook)
    # So total rank = 2 * 2 * 4 = 16? No, that's not right.
    
    # Actually, a 2x2 * 2x7 multiplication:
    # Result is 2x7, so 14 entries
    # Schoolbook: for each of the 2 rows of result, and each of 7 columns,
    # we do 2 multiplications. Total = 2*7*2 = 28 multiplications.
    
    # But we can use Strassen for the 2x2 part:
    # M1 = A11 * B1_col1 (scalar * scalar)
    # Actually, let's just use schoolbook for simplicity but structure it properly.
    
    # Let me just use the schoolbook approach but verify it's correct.
    
    terms = []
    
    def make_u(i, j, val):
        u = [0] * (n * m)
        u[i * m + j] = val
        return u
    
    def make_v(j, k, val):
        v = [0] * (m * p)
        v[j * p + k] = val
        return v
    
    def make_w(k, i, val):
        w = [0] * (p * n)
        w[k * n + i] = val
        return w
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = make_u(i, j, 1)
                v = make_v(j, k, 1)
                w = make_w(k, i, 1)
                terms.append((u, v, w))
    
    rank = len(terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, (u, v, w) in enumerate(terms):
        for r in range(n * m):
            U[r][c] = u[r]
        for r in range(m * p):
            V[r][c] = v[r]
        for r in range(p * n):
            W[r][c] = w[r]
    
    return {"U": U, "V": V, "W": W}
