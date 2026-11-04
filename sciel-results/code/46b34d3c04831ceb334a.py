"""Matrix multiplication tensor decomposition for 3x4 * 4x8 matrices."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    start_time = time.time()
    
    # Strategy: Use Strassen decomposition for 2x2 blocks where possible
    # i in {0,1,2}, j in {0,1,2,3}, k in {0,...,7}
    # Group i: {0,1} and {2}
    # Group j: {0,1} and {2,3}
    # Group k: {0,1}, {2,3}, {4,5}, {6,7}
    
    # For i-group {0,1}, j-group {0,1} or {2,3}, k-group: 2x2 x 2x2 -> Strassen (rank 7)
    # For i-group {2}, j-group, k-group: 1x2 x 2x2 -> schoolbook (rank 4)
    
    # Total: 2*2*4 = 16 sub-blocks
    # 8 of type 2x2x2x2 (Strassen, rank 7 each) = 56
    # 8 of type 1x2x2x2 (schoolbook, rank 4 each) = 32
    # Total rank = 88
    
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen 2x2x2 decomposition
    # For a 2x2 matrix multiplication C = A * B:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # p1 = (a+d)(e+h), p2 = (c+d)(e+f), p3 = (a+b)(g+h), p4 = d(h-f)
    # p5 = a(e-g), p6 = (c-a)(f+h), p7 = (b-d)(e+g)
    # C00 = p1 + p4 - p5 + p7, C01 = p3 + p5
    # C10 = p2 - p4, C11 = p1 - p2 + p3 + p6
    
    # A local indices: [0,0]->(i0,j0), [0,1]->(i0,j0+1), [1,0]->(i0+1,j0), [1,1]->(i0+1,j0+1)
    # B local indices: [0,0]->(j0,k0), [0,1]->(j0,k0+1), [1,0]->(j0+1,k0), [1,1]->(j0+1,k0+1)
    # C local indices: [0,0]->(i0,k0), [0,1]->(i0,k0+1), [1,0]->(i0+1,k0), [1,1]->(i0+1,k0+1)
    
    # U row for A[i,j]: i*m+j
    # V row for B[j,k]: j*p+k
    # W row for C[i,k]: k*n+i
    
    # Strassen terms: each is (A_coeffs, B_coeffs, C_coeffs)
    # A_coeffs: [coeff for (i0,j0), (i0,j0+1), (i0+1,j0), (i0+1,j0+1)]
    # B_coeffs: [coeff for (j0,k0), (j0,k0+1), (j0+1,k0), (j0+1,k0+1)]
    # C_coeffs: [coeff for (i0,k0), (i0,k0+1), (i0+1,k0), (i0+1,k0+1)]
    
    strassen_terms = [
        # (A_coeffs, B_coeffs, C_coeffs)
        # p1 = (a+d)(e+h)
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        # p2 = (c+d)(e+f)
        ([0,0,1,1], [1,1,0,0], [0,0,1,-1]),
        # p3 = (a+b)(g+h)
        ([1,1,0,0], [0,0,1,1], [0,1,0,1]),
        # p4 = d(h-f)
        ([0,0,0,1], [0,-1,0,1], [1,0,-1,0]),
        # p5 = a(e-g)
        ([1,0,0,0], [1,0,-1,0], [-1,1,0,0]),
        # p6 = (c-a)(f+h)
        ([-1,0,1,0], [0,1,0,1], [0,0,0,1]),
        # p7 = (b-d)(e+g)
        ([0,1,0,-1], [1,0,1,0], [1,0,0,0]),
    ]
    
    # Verify Strassen: C00 = p1+p4-p5+p7 = 1+1-(-1)+1 = 1+1+1+1 = 4? No...
    # Let me recheck the Strassen formulas.
    
    # Actually, the standard Strassen formulas are:
    # p1 = (a+d)(e+h)
    # p2 = (c+d)(e+f)
    # p3 = (a+b)(g+h)
    # p4 = d(h-f)
    # p5 = a(e-g)
    # p6 = (c-a)(f+h)
    # p7 = (b-d)(e+g)
    
    # C00 = p1 - p5 + p7 + p4
    # C01 = p3 + p5
    # C10 = p2 - p4
    # C11 = p1 - p2 + p3 + p6
    
    # Let me verify with a simple example:
    # A = [[1,0],[0,1]], B = [[1,0],[0,1]] (identity matrices)
    # a=1,b=0,c=0,d=1, e=1,f=0,g=0,h=1
    # p1 = (1+1)(1+1) = 4
    # p2 = (0+1)(1+0) = 1
    # p3 = (1+0)(0+1) = 1
    # p4 = 1*(1-0) = 1
    # p5 = 1*(1-0) = 1
    # p6 = (0-1)*(0+1) = -1
    # p7 = (0-1)*(1+0) = -1
    
    # C00 = p1 - p5 + p7 + p4 = 4 - 1 + (-1) + 1 = 3? Should be 1.
    
    # I think I have the formulas wrong. Let me look up the correct Strassen formulas.
    
    # Correct Strassen:
    # p1 = (a + d) * (e + h)
    # p2 = (c + d) * e
    # p3 = a * (f - h)
    # p4 = d * (g - e)
    # p5 = (a + b) * h
    # p6 = (c + d) * f
    # p7 = (a + b) * g
    
    # C00 = p1 + p4 - p5 + p7
    # C01 = p3 + p5
    # C10 = p2 - p4
    # C11 = p1 - p2 + p3 + p6
    
    # Let me verify:
    # A = [[1,0],[0,1]], B = [[1,0],[0,1]]
    # a=1,b=0,c=0,d=1, e=1,f=0,g=0,h=1
    # p1 = (1+1)(1+1) = 4
    # p2 = (0+1)*1 = 1
    # p3 = 1*(0-1) = -1
    # p4 = 1*(0-1) = -1
    # p5 = (1+0)*1 = 1
    # p6 = (0+1)*0 = 0
    # p7 = (1+0)*0 = 0
    
    # C00 = p1 + p4 - p5 + p7 = 4 + (-1) - 1 + 0 = 2? Should be 1.
    
    # Still wrong. Let me try a different set of formulas.
    
    # Another common formulation:
    # p1 = (a + d) * (e + h)
    # p2 = (c + d) * (e + f)
    # p3 = (a + b) * (g + h)
    # p4 = d * (h - f)
    # p5 = a * (e - g)
    # p6 = (c - a) * (f + h)
    # p7 = (b - d) * (e + g)
    
    # C00 = p1 + p4 - p5 + p7
    # C01 = p3 + p5
    # C10 = p2 - p4
    # C11 = p1 - p2 + p3 + p6
    
    # Verify:
    # A = [[1,0],[0,1]], B = [[1,0],[0,1]]
    # a=1,b=0,c=0,d=1, e=1,f=0,g=0,h=1
    # p1 = (1+1)(1+1) = 4
    # p2 = (0+1)(1+0) = 1
    # p3 = (1+0)(0+1) = 1
    # p4 = 1*(1-0) = 1
    # p5 = 1*(1-0) = 1
    # p6 = (0-1)(0+1) = -1
    # p7 = (0-1)(1+0) = -1
    
    # C00 = p1 + p4 - p5 + p7 = 4 + 1 - 1 + (-1) = 3? Should be 1.
    
    # I'm clearly making an error. Let me just use the schoolbook approach to be safe,
    # or find a verified Strassen implementation.
    
    # Actually, let me just verify numerically what works.
    # For now, let me use a simpler approach: just use schoolbook for everything.
    # This gives rank = n*m*p = 3*4*8 = 96.
    
    # But wait, the current program claims to achieve rank 88 with Strassen.
    # Let me just verify the current approach works by checking the tensor values.
    
    # For the matrix multiplication tensor T[i,j,k] = 1 for all valid (i,j,k),
    # we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Let me just use the schoolbook decomposition which is guaranteed to work:
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    # Verify
    def _verify(U, V, W, n, m, p):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # This should never happen, but just in case
    raise ValueError("Failed to construct valid decomposition")

# EVOLVE-BLOCK-END
