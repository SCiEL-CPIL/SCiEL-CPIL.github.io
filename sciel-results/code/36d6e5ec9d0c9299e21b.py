import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS  # n=4, m=4, p=6
    
    # We use the tensor identity M_{4,4,6} ≅ M_{2,2,2} ⊗ M_{2,2,3}
    # Rank of M_{2,2,2} is 7 (Strassen)
    # For M_{2,2,3}, we use a rank-4 decomposition (schoolbook-like with sharing)
    # Total rank = 7 * 4 = 28
    
    # Strassen decomposition for 2x2 * 2x2
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # C = A*B
    # 7 products:
    # p0 = (a00+a11)*(b00+b11)
    # p1 = (a10+a11)*b00
    # p2 = a00*(b01-b11)
    # p3 = a11*(b10-b00)
    # p4 = (a00+a10)*b11
    # p5 = (a01-a10)*(b10+b11)
    # p6 = (a01-a11)*(b00+b01)
    # C00 = p0 + p3 - p4 + p6
    # C01 = p2 + p4
    # C10 = p1 + p5
    # C11 = p0 - p1 + p2 + p5
    
    # Strassen coefficients
    # alpha[s][a] where a = i*2+k (row i, col k of A)
    # beta[s][b] where b = k*2+j (row k, col j of B)  
    # recon[s] = list of (i, j, coeff) for C[i,j]
    
    alpha_s = [
        [1, 0, 0, 1],  # p0: a00, a11
        [0, 0, 1, 1],  # p1: a10, a11
        [1, 0, 0, 0],  # p2: a00
        [0, 0, 0, 1],  # p3: a11
        [1, 0, 1, 0],  # p4: a00, a10
        [0, 1, -1, 0], # p5: a01, -a10
        [0, 1, 0, -1]  # p6: a01, -a11
    ]
    beta_s = [
        [1, 0, 0, 1],  # p0: b00, b11
        [1, 0, 0, 0],  # p1: b00
        [0, 1, 0, -1], # p2: b01, -b11
        [0, 0, 1, -1], # p3: b10, -b00
        [0, 0, 0, 1],  # p4: b11
        [0, 0, 1, 1],  # p5: b10, b11
        [1, 1, 0, 0]   # p6: b00, b01
    ]
    recon_s = [
        [(0,0,1), (1,1,1)],      # p0 -> C00+1, C11+1
        [(1,0,1), (1,1,-1)],     # p1 -> C10+1, C11-1
        [(0,1,1), (1,1,1)],      # p2 -> C01+1, C11+1
        [(0,0,1), (1,0,1)],      # p3 -> C00+1, C10+1
        [(0,0,-1), (0,1,1)],     # p4 -> C00-1, C01+1
        [(1,0,1), (1,1,1)],      # p5 -> C10+1, C11+1
        [(0,0,1), (0,1,-1)]      # p6 -> C00+1, C01-1
    ]
    
    # For M_{2,2,3}: A is 2x2, B is 2x3, C is 2x3
    # We can view this as: for the first 2 columns, use Strassen (7 terms)
    # For the 3rd column, use schoolbook (4 terms)
    # But wait, that gives 7+4=11, not 4.
    # 
    # Actually, let me reconsider. The Kronecker product approach:
    # M_{4,4,6} = M_{2,2,2} (outer) ⊗ M_{2,2,3} (inner)
    # But I need a rank-4 decomposition of M_{2,2,3} to get total rank 28.
    # 
    # Hmm, rank-4 for 2x2*2x3 seems too optimistic. Let me think differently.
    # 
    # Alternative: M_{4,4,6} can be seen as:
    # A (4x4), B (4x6), C (4x6)
    # Split into blocks: A = [[A11, A12], [A21, A22]] where each is 2x2
    # B = [[B11, B12, B13], [B21, B22, B23]] where each is 2x2 (first 2 cols) and 2x1 (last col)
    # Actually B is 4x6, so split as:
    # B = [[B1, B2], [B3, B4]] where B1,B2 are 2x3 and B3,B4 are 2x3
    # C = [[C1, C2], [C3, C4]] where each is 2x3
    #
    # C1 = A11*B1 + A12*B3
    # C2 = A11*B2 + A12*B4
    # C3 = A21*B1 + A22*B3
    # C4 = A21*B2 + A22*B4
    #
    # This is M_{2,2,2} (for the 2x2 block structure) ⊗ M_{2,2,3} (for inner)
    #
    # For the inner M_{2,2,3}, I'll use schoolbook which has rank 12 (2*2*3).
    # Total: 7 * 12 = 84. That's worse.
    #
    # Let me try: use Strassen for outer (7) and for inner, note that 2x2*2x3 can be done
    # by treating it as 2x2*2x2 (7 terms via Strassen) + 2x2*2x1 (4 terms schoolbook) = 11
    # Total: 7 * 11 = 77. Same as current.
    #
    # To beat 77, I need a better inner decomposition.
    # 
    # What if I use a different splitting?
    # M_{4,4,6}: split A as 4x4, B as 4x6.
    # Use Strassen on a 2x2 substructure and handle the rest differently.
    #
    # Actually, let me just implement the rank-77 solution cleanly and verify it works.
    # The key insight from the current program is:
    # Outer: Strassen 2x2 (7 terms)
    # Inner: Strassen 2x2*2x2 (7 terms) + schoolbook 2x2*2x1 (4 terms) = 11 terms
    # Total: 7 * 11 = 77
    
    R = 77
    U = np.zeros((16, R))  # 4*4 = 16
    V = np.zeros((24, R))  # 4*6 = 24
    W = np.zeros((24, R))  # 4*6 = 24 (output C is 4x6)
    
    # Indexing:
    # A is 4x4: A[i,k] where i in 0..3, k in 0..3
    #   Split: i = i1*2+i2, k = k1*2+k2 where i1,k1 in {0,1} (outer), i2,k2 in {0,1} (inner)
    # B is 4x6: B[k,j] where k in 0..3, j in 0..5
    #   Split: k = k1*2+k2, j = j1*3+j2 where k1,j1 in {0,1} (outer), k2 in {0,1}, j2 in {0,1,2} (inner)
    # C is 4x6: C[i,j] where i in 0..3, j in 0..5
    #   Split: i = i1*2+i2, j = j1*3+j2 where i1,j1 in {0,1} (outer), i2 in {0,1}, j2 in {0,1,2} (inner)
    #   W index: k*n+i = j*4+i where j in 0..5, i in 0..3
    
    col = 0
    for s in range(7):  # outer Strassen term
        # Inner Strassen for 2x2 * 2x2 (handles columns 0,1 of inner)
        for si in range(7):
            # U: A part
            for a in range(4):  # inner A index: a = i2*2+k2
                if alpha_s[si][a] != 0:
                    i2, k2 = a // 2, a % 2
                    for r in range(4):  # outer A index: r = i1*2+k1
                        if alpha_s[s][r] != 0:
                            i1, k1 = r // 2, r % 2
                            u_idx = (i1*2 + i2) * 4 + (k1*2 + k2)
                            U[u_idx, col] += alpha_s[si][a] * alpha_s[s][r]
            
            # V: B part
            for b in range(4):  # inner B index: b = k2*2+j2 where j2 in {0,1}
                if beta_s[si][b] != 0:
                    k2, j2 = b // 2, b % 2
                    for c in range(4):  # outer B index: c = k1*2+j1
                        if beta_s[s][c] != 0:
                            k1, j1 = c // 2, c % 2
                            v_idx = (k1*2 + k2) * 6 + (j1*3 + j2)
                            V[v_idx, col] += beta_s[si][b] * beta_s[s][c]
            
            # W: C part
            for (i_out, j_out, g_out) in recon_s[s]:
                for (i_in, j_in, g_in) in recon_s[si]:
                    i = i_out*2 + i_in
                    j = j_out*3 + j_in
                    w_idx = j * 4 + i
                    W[w_idx, col] += g_out * g_in
            
            col += 1
        
        # Schoolbook for 2x2 * 2x1 (handles column 2 of inner)
        for u_in in range(2):  # i2
            for v_in in range(2):  # k2
                # U: A part
                for r in range(4):  # outer A index
                    if alpha_s[s][r] != 0:
                        i1, k1 = r // 2, r % 2
                        u_idx = (i1*2 + u_in) * 4 + (k1*2 + v_in)
                        U[u_idx, col] += alpha_s[s][r]
                
                # V: B part
                for c in range(4):  # outer B index
                    if beta_s[s][c] != 0:
                        k1, j1 = c // 2, c % 2
                        v_idx = (k1*2 + v_in) * 6 + (j1*3 + 2)
                        V[v_idx, col] += beta_s[s][c]
                
                # W: C part
                for (i_out, j_out, g_out) in recon_s[s]:
                    i = i_out*2 + u_in
                    j = j_out*3 + 2
                    w_idx = j * 4 + i
                    W[w_idx, col] += g_out
                
                col += 1
    
    def to_list(mat):
        result = []
        for row in mat:
            result_row = []
            for x in row:
                if abs(x - round(x)) < 1e-9:
                    result_row.append(int(round(x)))
                else:
                    result_row.append(float(x))
            result.append(result_row)
        return result
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
