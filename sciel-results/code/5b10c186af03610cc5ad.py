"""FTW-inspired decomposition for 2x4x6 matrix multiplication."""
DIMENSIONS = [2, 4, 6]

def _frac(x):
    """Represent half-integer as float."""
    return float(x)

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 4, 6
    
    # FTW approach: decompose 2x4x6 as Kronecker product of 2x2x2 (Strassen, rank 7)
    # and 1x2x3 (rank 6), giving total rank 42.
    # 
    # The key insight: m=4 = 2*2, p=6 = 2*3, n=2 = 2*1
    # So T_{2,4,6} = T_{2,2,2} ⊗ T_{1,2,3}
    # rank(T_{2,2,2}) = 7 (Strassen)
    # rank(T_{1,2,3}) = 6 (three independent dot products of length 2)
    # Total rank = 7 * 6 = 42
    
    # Constructing the decomposition:
    # 
    # For the 2x2x2 Strassen core, we have 7 products M1..M7.
    # For the 1x2x3 core, we have 6 products (3 dot products of length 2).
    # 
    # The combined decomposition has 42 terms.
    
    rank = 42
    U = [[0.0] * rank for _ in range(n*m)]   # shape (8, 42)
    V = [[0.0] * rank for _ in range(m*p)]   # shape (24, 42)
    W = [[0.0] * rank for _ in range(p*n)]   # shape (12, 42)
    
    # Strassen 2x2x2 coefficients:
    # For X*Y = Z where X,Y,Z are 2x2:
    # X[i,j], Y[j,k], Z[i,k] with i,j,k in {0,1}
    # 
    # The 7 Strassen products and their contributions to U, V, W:
    
    # Strassen product definitions:
    # Each product s has:
    #   u_coeffs: list of (i, j, coeff) for X[i,j]
    #   v_coeffs: list of (j, k, coeff) for Y[j,k]
    #   w_coeffs: list of (i, k, coeff) for Z[i,k]
    
    strassen = [
        # M1 = (X00 + X11)(Y00 + Y11)
        {'u': [(0,0,1), (1,1,1)], 'v': [(0,0,1), (1,1,1)], 
         'w': [(0,0,1), (0,1,0), (1,0,0), (1,1,-1)]},  # Z00=M1, Z11=-M1 (partial)
        # M2 = (X10 + X11)Y00
        {'u': [(1,0,1), (1,1,1)], 'v': [(0,0,1)], 
         'w': [(1,0,1), (1,1,-1)]},  # Z10=M2, Z11=-M2
        # M3 = X00(Y01 - Y11)
        {'u': [(0,0,1)], 'v': [(0,1,1), (1,1,-1)], 
         'w': [(0,1,1), (1,1,1)]},  # Z01=M3, Z11=M3
        # M4 = X11(Y10 - Y00)
        {'u': [(1,1,1)], 'v': [(1,0,1), (0,0,-1)], 
         'w': [(0,0,1), (1,0,1)]},  # Z00=M4, Z10=M4
        # M5 = (X00 + X01)Y11
        {'u': [(0,0,1), (0,1,1)], 'v': [(1,1,1)], 
         'w': [(0,0,-1), (0,1,1)]},  # Z00=-M5, Z01=M5
        # M6 = (X10 - X00)(Y00 + Y01)
        {'u': [(1,0,1), (0,0,-1)], 'v': [(0,0,1), (0,1,1)], 
         'w': [(1,1,1)]},  # Z11=M6
        # M7 = (X01 - X11)(Y10 + Y11)
        {'u': [(0,1,1), (1,1,-1)], 'v': [(1,0,1), (1,1,1)], 
         'w': [(0,0,1)]},  # Z00=M7
    ]
    
    # 1x2x3 core: computing 3 dot products of length 2
    # For a row vector x of length 2 and matrix Y of shape 2x3,
    # compute z = x * Y (shape 1x3)
    # z[k] = x[0]*Y[0,k] + x[1]*Y[1,k] for k=0,1,2
    # 
    # Each dot product has rank 2 (cannot be done with 1 multiplication)
    # Total rank = 3 * 2 = 6
    # 
    # The 6 products:
    # For k=0: P1 = x[0]*Y[0,0], P2 = x[1]*Y[1,0], z[0] = P1 + P2
    # For k=1: P3 = x[0]*Y[0,1], P4 = x[1]*Y[1,1], z[1] = P3 + P4
    # For k=2: P5 = x[0]*Y[0,2], P6 = x[1]*Y[1,2], z[2] = P5 + P6
    
    core_123 = [
        # P1 = x[0] * Y[0,0]
        {'u': [(0,0,1)], 'v': [(0,0,1)], 'w': [(0,1)]},  # contributes to z[0]
        # P2 = x[1] * Y[1,0]
        {'u': [(0,1,1)], 'v': [(1,0,1)], 'w': [(0,1)]},  # contributes to z[0]
        # P3 = x[0] * Y[0,1]
        {'u': [(0,0,1)], 'v': [(0,1,1)], 'w': [(1,1)]},  # contributes to z[1]
        # P4 = x[1] * Y[1,1]
        {'u': [(0,1,1)], 'v': [(1,1,1)], 'w': [(1,1)]},  # contributes to z[1]
        # P5 = x[0] * Y[0,2]
        {'u': [(0,0,1)], 'v': [(0,2,1)], 'w': [(2,1)]},  # contributes to z[2]
        # P6 = x[1] * Y[1,2]
        {'u': [(0,1,1)], 'v': [(1,2,1)], 'w': [(2,1)]},  # contributes to z[2]
    ]
    
    # Now combine: for each Strassen product s and each core product t,
    # we get one term in the rank-42 decomposition.
    # 
    # The Kronecker product structure:
    # T_{2,4,6}[a, b, c] = sum_{s,t} U_s[a_s] * V_t[b_t] * W_s[c_s] * W_t[c_t]
    # 
    # Where a = a_s * 2 + a_t (since m=4 = 2*2)
    #       b = b_s * 3 + b_t (since p=6 = 2*3)
    #       c = c_s * 3 + c_t (since n*p = 2*6 = 12, and c = b*n + i)
    # 
    # Wait, let me re-derive the indexing.
    # 
    # A is 2x4: A[i, a] where i in {0,1}, a in {0,1,2,3}
    # U row index = i*m + a = i*4 + a
    # 
    # B is 4x6: B[a, b] where a in {0,1,2,3}, b in {0,1,2,3,4,5}
    # V row index = a*p + b = a*6 + b
    # 
    # C is 2x6: C[i, c] where i in {0,1}, c in {0,1,2,3,4,5}
    # W row index = c*n + i = c*2 + i (trace convention)
    # 
    # Decomposition:
    # a = a_s * 2 + a_t where a_s in {0,1} (from 2x2x2), a_t in {0,1} (from 1x2x3)
    # b = b_s * 3 + b_t where b_s in {0,1} (from 2x2x2), b_t in {0,1,2} (from 1x2x3)
    # c = c_s * 3 + c_t where c_s in {0,1} (from 2x2x2, output row), c_t in {0,1,2} (from 1x2x3, output col)
    # i = c_s (output row of C)
    # 
    # Actually, let me think about this more carefully.
    # 
    # The tensor T_{2,4,6} has indices (i, a, b) -> C[i, b] = sum_a A[i,a] B[a,b]
    # 
    # Using Kronecker: i = i_s * 1 + i_t where i_s in {0,1} (from 2x2x2), i_t = 0 (from 1x2x3, since n=1)
    # a = a_s * 2 + a_t where a_s in {0,1}, a_t in {0,1}
    # b = b_s * 3 + b_t where b_s in {0,1}, b_t in {0,1,2}
    # 
    # So:
    # U row = i*4 + a = (i_s)*4 + a_s*2 + a_t
    # V row = a*6 + b = (a_s*2 + a_t)*6 + b_s*3 + b_t
    # W row = b*2 + i = (b_s*3 + b_t)*2 + i_s
    
    col = 0
    for s in range(7):
        for t in range(6):
            # U contributions from Strassen product s and core product t
            for (i_s, a_s, coeff_s) in strassen[s]['u']:
                for (i_t, a_t, coeff_t) in core_123[t]['u']:
                    i = i_s  # i_t is always 0 for 1x2x3
                    a = a_s * 2 + a_t
                    U[i*4 + a][col] += float(coeff_s * coeff_t)
            
            # V contributions
            for (a_s, b_s, coeff_s) in strassen[s]['v']:
                for (a_t, b_t, coeff_t) in core_123[t]['v']:
                    a = a_s * 2 + a_t
                    b = b_s * 3 + b_t
                    V[a*6 + b][col] += float(coeff_s * coeff_t)
            
            # W contributions
            for (i_s, c_s, coeff_s) in strassen[s]['w']:
                for (c_t, coeff_t) in core_123[t]['w']:
                    i = i_s
                    b = c_s * 3 + c_t
                    W[b*2 + i][col] += float(coeff_s * coeff_t)
            
            col += 1
    
    # Verify dimensions
    assert len(U) == 8 and len(U[0]) == 42, f"U shape: {len(U)}x{len(U[0])}"
    assert len(V) == 24 and len(V[0]) == 42, f"V shape: {len(V)}x{len(V[0])}"
    assert len(W) == 12 and len(W[0]) == 42, f"W shape: {len(W)}x{len(W[0])}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
