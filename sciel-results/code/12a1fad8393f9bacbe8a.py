"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import itertools

DIMENSIONS = [4, 4, 8]

def _build_tensor_equations(n, m, p, r, coeff_set):
    """
    Build the linear system for rank-r tensor decomposition.
    
    We want to find U (n*m x r), V (m*p x r), W (p*n x r) with entries from coeff_set
    such that for all (i,j,k):
        sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    This is a bilinear system. We linearize by fixing U and V, then solving for W,
    or use a more sophisticated approach.
    
    For small coefficient sets, we can enumerate possible values.
    """
    # The tensor T has entries T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Number of unknowns: (n*m + m*p + p*n) * r
    # Number of equations: n * m * p
    
    n_eq = n * m * p
    n_unk = (n * m + m * p + p * n) * r
    
    # We'll use a heuristic: try small coefficient sets and use MILP
    return n_eq, n_unk

def _try_rank(n, m, p, r, max_time=60):
    """
    Try to find a rank-r decomposition using MILP with small integer coefficients.
    """
    # For the bilinear identity sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1,
    # we can fix U and V to have entries in {-1, 0, 1} and solve for W.
    # But this is still nonlinear.
    
    # Alternative: use the fact that for matrix multiplication, we can use
    # structured decompositions. Let's try a direct approach with small coefficients.
    
    # For 4x4*4x8, let's try to exploit the structure.
    # The schoolbook gives rank 128. Can we do better?
    
    # One approach: use the 2x2 Winograd decomposition (rank 7) for pairs of rows/cols.
    # For 4x4*4x4, we can potentially achieve rank around 49 using Strassen-like methods.
    # For 4x4*4x8, we might do better than 128.
    
    # Let's try a constructive approach:
    # Split A (4x4) into 2x2 blocks: A = [[A11, A12], [A21, A22]]
    # Split B (4x8) into 2x4 blocks: B = [[B11, B12], [B21, B22]] where each is 2x4
    # C = A*B = [[A11*B11 + A12*B21, A11*B12 + A12*B22],
    #            [A21*B11 + A22*B21, A21*B12 + A22*B22]]
    
    # Each 2x2 * 2x4 multiplication can be done with rank 7*2 = 14? 
    # Actually, 2x2 * 2x4: schoolbook is 2*2*4 = 16.
    # With Winograd for 2x2*2x2 (rank 7), we can do 2x2*2x4 as two 2x2*2x2 multiplications: rank 14.
    
    # So for 4x4*4x8:
    # We have 4 blocks of 2x2*2x4, each with rank 14, total rank = 4*14 = 56.
    # But we need to be careful about how the blocks interact.
    
    # Let's construct this explicitly.
    # A = [[a11, a12, a13, a14],
    #      [a21, a22, a23, a24],
    #      [a31, a32, a33, a34],
    #      [a41, a42, a43, a44]]
    
    # B = [[b11, b12, b13, b14, b15, b16, b17, b18],
    #      [b21, b22, b23, b24, b25, b26, b27, b28],
    #      [b31, b32, b33, b34, b35, b36, b37, b38],
    #      [b41, b42, b43, b44, b45, b46, b47, b48]]
    
    # C = A*B is 4x8.
    
    # Using 2x2 Winograd for each 2x2 block of A with each 2x4 block of B:
    # Block (0,0): A11 (2x2) * B11 (2x4) -> contributes to C[0:2, 0:4]
    # Block (0,1): A12 (2x2) * B21 (2x4) -> contributes to C[0:2, 0:4]
    # Block (0,2): A11 (2x2) * B12 (2x4) -> contributes to C[0:2, 4:8]
    # Block (0,3): A12 (2x2) * B22 (2x4) -> contributes to C[0:2, 4:8]
    # Block (1,0): A21 (2x2) * B11 (2x4) -> contributes to C[2:4, 0:4]
    # Block (1,1): A22 (2x2) * B21 (2x4) -> contributes to C[2:4, 0:4]
    # Block (1,2): A21 (2x2) * B12 (2x4) -> contributes to C[2:4, 4:8]
    # Block (1,3): A22 (2x2) * B22 (2x4) -> contributes to C[2:4, 4:8]
    
    # Each 2x2*2x4 uses rank 14 (two Winograd 2x2*2x2).
    # Total rank = 8 * 14 = 112. That's better than 128!
    
    # Let's implement this.
    
    # Winograd 2x2*2x2 with rank 7:
    # For X (2x2) and Y (2x2):
    # X = [[x11, x12], [x21, x22]]
    # Y = [[y11, y12], [y21, y22]]
    # Z = X*Y
    #
    # Winograd decomposition:
    # u0 = x11
    # u1 = x12
    # u2 = x21
    # u3 = x22
    # u4 = x11 + x12
    # u5 = x21 + x22
    # u6 = x11 + x21
    #
    # v0 = y11
    # v1 = y12
    # v2 = y21
    # v3 = y22
    # v4 = y21 + y22
    # v5 = y11 + y12
    # v6 = y11 - y12
    #
    # Z[0,0] = u0*v0 + u2*v2 + u5*v5 + u6*v6
    # Z[0,1] = u1*v1 + u0*v3 + u4*v4 - u6*v6
    # Z[1,0] = u0*v2 + u2*v0 + u4*v4 - u6*v6
    # Z[1,1] = u3*v3 + u1*v2 + u5*v5 - u6*v6
    
    # Let me verify this is correct by checking Z[0,0]:
    # = x11*y11 + x21*y21 + (x21+x22)*(y11+y12) + (x11+x21)*(y11-y12)
    # = x11*y11 + x21*y21 + x21*y11 + x21*y12 + x22*y11 + x22*y12 + x11*y11 - x11*y12 + x21*y11 - x21*y12
    # = 2*x11*y11 + 2*x21*y11 + x21*y21 + x21*y12 - x21*y12 + x22*y11 + x22*y12 - x11*y12
    # = 2*x11*y11 + 2*x21*y11 + x21*y21 + x22*y11 + x22*y12 - x11*y12
    # This doesn't look right. The correct Winograd formula is different.
    
    # Let me use a verified Winograd 2x2 formula.
    # From standard references, the rank-7 decomposition for 2x2*2x2 is:
    
    # For A = [[a, b], [c, d]] and B = [[e, f], [g, h]]:
    # m1 = (a+b)(e+h)
    # m2 = (c+d)(e+f)
    # m3 = a(f-h)
    # m4 = d(g-f)
    # m5 = (a-c)(e+g)
    # m6 = (b+d)(g+h)
    # m7 = (b+c)(e+f)  # This might not be right either
    
    # Actually, let me just use a simple approach: I'll construct the decomposition
    # using the known fact that 2x2*2x2 has rank 7, and build up from there.
    
    # For now, let's return the schoolbook decomposition as a fallback, but try
    # to find a better one using the block structure.
    
    # Let's try rank 112 with the block approach.
    if r >= 112:
        # Construct the decomposition using 2x2 Winograd blocks
        # We'll use a simple verification-based approach
        
        # For 2x2*2x2, use this verified rank-7 decomposition:
        # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
        # C = A*B
        #
        # t1 = a
        # t2 = b
        # t3 = c
        # t4 = d
        # t5 = a + b
        # t6 = c + d
        # t7 = a + c
        #
        # s1 = e
        # s2 = f
        # s3 = g
        # s4 = h
        # s5 = g + h
        # s6 = e + f
        # s7 = e - h
        #
        # C[0,0] = t1*s1 + t3*s3 + t6*s6 + t7*s7
        # C[0,1] = t2*s2 + t1*s4 + t5*s5 - t7*s7
        # C[1,0] = t1*s3 + t3*s1 + t5*s5 - t7*s7
        # C[1,1] = t4*s4 + t2*s3 + t6*s6 - t7*s7
        
        # Let me verify C[0,0]:
        # = a*e + c*g + (c+d)*(e+f) + (a+c)*(e-h)
        # = ae + cg + ce + cf + de + df + ae - ah + ce - ch
        # = 2ae + 2ce + cg + cf + de + df - ah - ch
        # This is not equal to ae + cg. So this formula is wrong.
        
        # I'll use a different approach: just return schoolbook for now.
        pass
    
    # Return schoolbook decomposition as fallback
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
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-START
def construct():
    """
    Tensor decomposition for multiplying a 4x4 matrix by a 4x8 matrix.
    """
    n, m, p = DIMENSIONS  # n=4, m=4, p=8
    
    # Try to find a better decomposition than schoolbook (rank 128)
    # For now, use schoolbook as a safe fallback
    rank = n * m * p  # 128
    
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
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
