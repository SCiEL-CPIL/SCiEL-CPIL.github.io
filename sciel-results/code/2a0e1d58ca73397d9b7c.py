"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Exploit block structure: C = A * B where A is 2x2, B is 2x8.
    # C is 2x8. C[:, 2*b:2*b+2] = A @ B[:, 2*b:2*b+2].
    # Each block multiplication is 2x2 @ 2x2, rank 7 via Strassen.
    # Total rank = 4 * 7 = 28.
    
    # Strassen's algorithm for 2x2 matrix multiplication.
    # Given A = [[a, b], [c, d]] and E = [[e, f], [g, h]] (using E to avoid confusion with B)
    # Result F = [[p1*p5 - p3*(p4+p6), p1*p4 + p7], [p2*p3 + p5, p3*(p1+p2) - p5*(p6+p7)]]
    # where:
    # p1 = (a + d) * (e + h)
    # p2 = (c + d) * e
    # p3 = a * (f - h)
    # p4 = d * (g - e)
    # p5 = (a + b) * h
    # p6 = (c + a) * (e + f)
    # p7 = (d + b) * (g + h)
    
    # The multiplication tensor for 2x2 has rank 7.
    # We represent it as U_sub, V_sub, W_sub of shape (4, 7), (4, 7), (4, 7).
    # U_sub[i*2+j, s] = alpha_{s, i, j}
    # V_sub[k*2+l, s] = beta_{s, k, l}  (Wait, V is for the second matrix, indices correspond to E)
    # W_sub[(k*2+l)*2 + i]? No.
    
    # Let's stick to the tensor definition.
    # T_{i,j,k,l} = \delta_{j,k}.
    # Decomposition: T = \sum_s U_s \otimes V_s \otimes W_s.
    # U_s is a 2x2 matrix, V_s is a 2x2 matrix, W_s is a 2x2 matrix.
    # The product is (U_s * V_s)_{i,l}? No.
    # The formula is C_{i,l} = \sum_j A_{i,j} B_{j,l}.
    # So the tensor is T_{i,j,l} = \delta_{j,j}? No.
    # T_{i,j,l} contributes to C_{i,l} via A_{i,j} B_{j,l}.
    # The factorization is T_{i,j,l} = \sum_s U_{i,j}^{(s)} V_{j,l}^{(s)} W_{i,l}^{(s)}?
    # No, the standard decomposition is:
    # C_{i,l} = \sum_s ( \sum_j U_{i,j}^{(s)} A_{i,j} ) ( \sum_l V_{j,l}^{(s)} B_{j,l} ) W_{i,l}^{(s)} ?
    # No, the rank-r decomposition of the multiplication tensor T_{ijk} (where i, j, k correspond to rows of A, cols of A/rows of B, cols of B) is:
    # T_{ijk} = \sum_{s=1}^r U_{i,k}^{(s)} V_{j,k}^{(s)} W_{j,l}^{(s)} ... wait.
    
    # Let's use the explicit entries.
    # For a 2x2 block:
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = A B
    # C00 = A00 B00 + A01 B10
    # C01 = A00 B01 + A01 B11
    # C10 = A10 B00 + A11 B10
    # C11 = A10 B01 + A11 B11
    
    # Strassen steps:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 + A00)(B00 + B01)
    # M7 = (A11 + A01)(B10 + B11)
    
    # C00 = M1 - M4 + M6 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M2 - M5
    
    # Let's map this to U, V, W matrices.
    # U is 4x7. Indices: i*2+j for A.
    # V is 4x7. Indices: j*2+k for B. (j is row of B, k is col of B).
    # W is 4x7. Indices: k*2+i for C?
    # Task says: W[k*n+i, r]. Here n=2 (for the block). So index is 2*k + i.
    # Wait, the task W is for the whole matrix.
    # For the block b, the C indices are i and k (where k is local col).
    # Global C row is i, global C col is 2*b + k.
    # The task W matrix has size p*n = 8*2 = 16.
    # Index is k_global * n + i.
    # k_global = 2*b + k_local.
    # So W index = (2*b + k_local)*2 + i = 4*b + 2*k_local + i.
    
    # Let's build U, V, W for the whole problem directly.
    rank = 28
    U = [[0.0] * rank for _ in range(n * m)]  # 4x28
    V = [[0.0] * rank for _ in range(m * p)]  # 16x28
    W = [[0.0] * rank for _ in range(p * n)]  # 16x28
    
    # Strassen coefficients for the 7 products.
    # Each product s has:
    # alpha_s: 4 coefficients (for A00, A01, A10, A11)
    # beta_s: 4 coefficients (for B00, B01, B10, B11)
    # gamma_s: 4 coefficients (for C00, C01, C10, C11)
    
    # s=0 (M1): (A00+A11)(B00+B11)
    # alpha: [1, 0, 0, 1]
    # beta: [1, 0, 0, 1]
    # gamma: C00=1, C11=1 -> [1, 0, 0, 1] (order C00, C01, C10, C11)
    
    # s=1 (M2): (A10+A11)B00
    # alpha: [0, 0, 1, 1]
    # beta: [1, 0, 0, 0]
    # gamma: C10=1, C11=-1 -> [0, 0, 1, -1]
    
    # s=2 (M3): A00(B01-B11)
    # alpha: [1, 0, 0, 0]
    # beta: [0, 1, 0, -1]
    # gamma: C01=1, C11=-1 -> [0, 1, 0, -1]
    
    # s=3 (M4): A11(B10-B00)
    # alpha: [0, 0, 0, 1]
    # beta: [-1, 0, 1, 0]
    # gamma: C00=-1, C10=1 -> [-1, 0, 1, 0]
    
    # s=4 (M5): (A00+A01)B11
    # alpha: [1, 1, 0, 0]
    # beta: [0, 0, 0, 1]
    # gamma: C01=1, C11=-1 -> [0, 1, 0, -1]
    
    # s=5 (M6): (A10+A00)(B00+B01)
    # alpha: [1, 0, 1, 0]
    # beta: [1, 1, 0, 0]
    # gamma: C00=1 -> [1, 0, 0, 0]
    
    # s=6 (M7): (A11+A01)(B10+B11)
    # alpha: [0, 1, 0, 1]
    # beta: [0, 0, 1, 1]
    # gamma: C00=-1 -> [-1, 0, 0, 0]
    
    # Store these as lists of lists.
    # alpha[s][idx_A]
    # beta[s][idx_B]
    # gamma[s][idx_C]
    
    alphas = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ]
    
    betas = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    
    gammas = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 1, 0, -1],
        [1, 0, 0, 0],
        [-1, 0, 0, 0]
    ]
    
    # Now fill U, V, W.
    # U has 4 rows (A indices 0,1,2,3 corresponding to 00,01,10,11).
    # V has 16 rows. For block b, B indices are (0,0), (0,1), (1,0), (1,1).
    # V index for B_{j,k} is j*2+k.
    # Global B index for block b:
    # j is 0 or 1. k_local is 0 or 1.
    # Global col = 2*b + k_local.
    # So V index = j*2 + (2*b + k_local)? No.
    # V index is j*p + k_global = 2*j + (2*b + k_local).
    
    # W has 16 rows.
    # W index is k_global*n + i = (2*b + k_local)*2 + i.
    
    for b in range(4):  # 4 blocks
        for s in range(7):
            r = 7 * b + s  # rank index
            
            # Fill U
            for idx_A in range(4):
                U[idx_A][r] = alphas[s][idx_A]
            
            # Fill V
            # idx_B corresponds to (j, k_local)
            # idx_B: 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
            for idx_B in range(4):
                j = idx_B // 2
                k_local = idx_B % 2
                k_global = 2 * b + k_local
                v_idx = 2 * j + k_global
                V[v_idx][r] = betas[s][idx_B]
            
            # Fill W
            # idx_C corresponds to (i, k_local)
            # idx_C: 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
            for idx_C in range(4):
                i = idx_C // 2
                k_local = idx_C % 2
                k_global = 2 * b + k_local
                w_idx = 2 * k_global + i
                W[w_idx][r] = gammas[s][idx_C]
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
