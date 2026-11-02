"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110  # safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # Build rank-49 decomposition for 4x4 x 4x4 using recursive Strassen
    # Strassen 2x2: 7 rank-1 terms
    # 4x4: split into 2x2 blocks, 7 Strassen terms, each is 2x2x2 (rank 7)
    # Total: 49 rank-1 terms for 4x4x4
    
    # For 4x4 x 4x8: two independent 4x4x4 multiplications
    # Total rank: 98
    
    # First, define Strassen 2x2x2 rank-1 terms
    # U_2x2[i,j,term], V_2x2[i,j,term], W_2x2[i,j,term]
    # Indices: i,j in 0..1 for A (2x2), i,j in 0..1 for B, i,j in 0..1 for C
    
    # Strassen 2x2 terms (7 terms):
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # For each term t, U_2x2[i,j,t], V_2x2[i,j,t], W_2x2[i,j,t]
    # A index: (i,j) where i=row, j=col of A (2x2)
    # B index: (j,k) where j=row of B, k=col of B (2x2)  
    # C index: (i,k) where i=row, k=col of C (2x2)
    
    # Let me define the 7 Strassen terms for 2x2:
    # Each term has coefficients for A elements, B elements, C elements
    
    # Term 1: (A00+A11)(B00+B11) -> C00 += 1, C01 += 1, C10 += 1, C11 += 1
    # Actually, let me be more careful.
    
    # In Strassen, we compute:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # where:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B10-B11)
    # M4 = A11*(B01-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    
    # For tensor decomposition, each Mi corresponds to a rank-1 term.
    # Mi = sum_{i,j,k} u_{i,j} * v_{j,k} * w_{i,k}
    
    # M1: A part: A00+A11, B part: B00+B11
    # u: A00=1, A11=1 (others 0)
    # v: B00=1, B11=1 (others 0)
    # w: C00=1, C01=1, C10=1, C11=1
    
    # Let me define for 2x2:
    # A indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # B indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # C indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    
    # But in the problem, for 4x4x4:
    # A is 4x4, indexed by (i,j) with i=0..3, j=0..3
    # B is 4x4, indexed by (j,k) with j=0..3, k=0..3
    # C is 4x4, indexed by (i,k) with i=0..3, k=0..3
    
    # For the 2x2 sub-blocks within 4x4:
    # A block (bi,bj) where bi in {0,1}, bj in {0,1}
    # A[2*bi+di, 2*bj+dj] for di,dj in {0,1}
    
    # Let me build the 4x4x4 Strassen decomposition directly.
    
    # 4x4 split into 2x2 blocks:
    # A = [[A00, A01], [A10, A11]] where each is 2x2
    # A00 = A[0:2, 0:2], A01 = A[0:2, 2:4]
    # A10 = A[2:4, 0:2], A11 = A[2:4, 2:4]
    
    # Similarly for B and C.
    
    # Strassen on the 2x2 block structure gives 7 "super-terms", each being a 2x2x2 product.
    # Each 2x2x2 product has rank 7 (Strassen).
    # Total rank: 7 * 7 = 49.
    
    # Let me define the 49 rank-1 terms.
    
    # For each of the 7 Strassen block-level terms (t=0..6),
    # and for each of the 7 Strassen 2x2x2 terms (s=0..6),
    # we get a rank-1 term in the 4x4x4 decomposition.
    
    # Block-level Strassen:
    # T0: (A00+A11) * (B00+B11) -> contributes to C00, C01, C10, C11 with coeff 1
    # T1: (A10+A11) * B00 -> contributes to C10, C11
    # T2: A00 * (B10-B11) -> contributes to C00, C01
    # T3: A11 * (B01-B00) -> contributes to C00, C01
    # T4: (A00+A01) * B11 -> contributes to C00, C01
    # T5: (A10-A00) * (B00+B01) -> contributes to C10, C11
    # T6: (A01-A11) * (B10+B11) -> contributes to C00, C01
    
    # Wait, I need to be more careful with the Strassen formulas.
    
    # Standard Strassen for 2x2:
    # C = A * B
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # So for the block-level (2x2 blocks of 2x2 matrices):
    # C00_block = M1 + M4 - M5 + M7  (each Mi is a 2x2 matrix product)
    # C01_block = M3 + M5
    # C10_block = M2 + M4
    # C11_block = M1 - M2 + M3 + M6
    
    # Each Mi = (linear combo of A blocks) * (linear combo of B blocks)
    
    # For tensor decomposition, each Mi is itself a 2x2x2 tensor with rank 7.
    
    # So the total 4x4x4 tensor is a sum of 7 * 7 = 49 rank-1 terms.
    
    # Let me define the coefficients.
    
    # For block-level term Mi (i=0..6), the A-block coefficients and B-block coefficients:
    # M0 (M1): A: [1,0,0,1] (A00=1, A11=1), B: [1,0,0,1] (B00=1, B11=1)
    # M1 (M2): A: [0,0,1,1] (A10=1, A11=1), B: [1,0,0,0] (B00=1)
    # M2 (M3): A: [1,0,0,0] (A00=1), B: [0,0,1,-1] (B10=1, B11=-1)
    # M3 (M4): A: [0,0,0,1] (A11=1), B: [0,1,-1,0] (B01=1, B00=-1)
    # M4 (M5): A: [1,1,0,0] (A00=1, A01=1), B: [0,0,0,1] (B11=1)
    # M5 (M6): A: [0,0,1,-1] (A10=1, A00=-1), B: [1,1,0,0] (B00=1, B01=1)
    # M6 (M7): A: [0,1,0,-1] (A01=1, A11=-1), B: [0,0,1,1] (B10=1, B11=1)
    
    # A-block indices: 0=A00, 1=A01, 2=A10, 3=A11
    # B-block indices: 0=B00, 1=B01, 2=B10, 3=B11
    
    # For each block-level term mi (i=0..6) with A-block coeff a_i[0..3] and B-block coeff b_i[0..3],
    # and for each 2x2x2 Strassen term sj (j=0..6) with 2x2 A-coeff u_j[0..3], B-coeff v_j[0..3], C-coeff w_j[0..3],
    # the combined rank-1 term has:
    # A-coeff: for block (bi,bj) and 2x2 element (di,dj): a_i[bi*2+bj] * u_j[di*2+dj]
    # B-coeff: for block (bi,bj) and 2x2 element (di,dj): b_i[bi*2+bj] * v_j[di*2+dj]
    # C-coeff: for block (bi,bj) and 2x2 element (di,dj): w_j[di*2+dj] * (C-block coeff from Strassen)
    
    # Wait, I also need the C-block coefficients from the block-level Strassen.
    
    # C-block coefficients for each Mi:
    # M0 (M1): C00=1, C01=1, C10=1, C11=1 -> [1,1,1,1]
    # M1 (M2): C10=1, C11=1 -> [0,0,1,1]
    # M2 (M3): C00=1, C01=1 -> [1,1,0,0]
    # M3 (M4): C00=1, C01=1 -> [1,1,0,0]
    # M4 (M5): C00=-1, C01=-1 -> [-1,-1,0,0]  (from -M5 in C00,C01)
    # Wait, C00 = M1+M4-M5+M7, C01 = M3+M5
    # So M5 appears in C00 with coeff -1 and in C01 with coeff +1
    # M4 appears in C00 with coeff +1 and in C10 with coeff +1
    # etc.
    
    # Let me define C-block coefficients c_i[0..3] for each Mi, where 0=C00,1=C01,2=C10,3=C11:
    # M0 (M1): c=[1,0,0,1] (C00+=1, C11+=1)
    # M1 (M2): c=[0,0,1,-1] (C10+=1, C11-=1)
    # M2 (M3): c=[1,1,0,0] (C00+=1, C01+=1)
    # M3 (M4): c=[1,0,1,0] (C00+=1, C10+=1)
    # M4 (M5): c=[-1,1,0,0] (C00-=1, C01+=1)
    # M5 (M6): c=[0,0,0,1] (C11+=1)
    # M6 (M7): c=[1,0,0,1] (C00+=1, C11+=1)
    
    # Hmm, let me re-derive:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # So:
    # M1: C00+=1, C11+=1 -> c=[1,0,0,1]
    # M2: C10+=1, C11-=1 -> c=[0,0,1,-1]
    # M3: C01+=1, C11+=1 -> c=[0,1,0,1]
    # M4: C00+=1, C10+=1 -> c=[1,0,1,0]
    # M5: C00-=1, C01+=1 -> c=[-1,1,0,0]
    # M6: C11+=1 -> c=[0,0,0,1]
    # M7: C00+=1, C11+=1 -> c=[1,0,0,1]
    
    # Now for the 2x2x2 Strassen (7 terms), let me define u_j, v_j, w_j for j=0..6:
    # Each is for a 2x2 A, 2x2 B, 2x2 C.
    # 2x2 indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    # Same Strassen structure:
    # M0 (M1): u=[1,0,0,1], v=[1,0,0,1], w=[1,0,0,1]
    # M1 (M2): u=[0,0,1,1], v=[1,0,0,0], w=[0,0,1,-1]
    # M2 (M3): u=[1,0,0,0], v=[0,0,1,-1], w=[0,1,0,1]
    # M3 (M4): u=[0,0,0,1], v=[0,1,-1,0], w=[1,0,1,0]
    # M4 (M5): u=[1,1,0,0], v=[0,0,0,1], w=[-1,1,0,0]
    # M5 (M6): u=[0,0,1,-1], v=[1,1,0,0], w=[0,0,0,1]
    # M6 (M7): u=[0,1,0,-1], v=[0,0,1,1], w=[1,0,0,1]
    
    # Now for the 4x4x4 decomposition:
    # Total 49 terms. For term (mi, sj) where mi=0..6, sj=0..6:
    # A-coeff for element (i,j) in 4x4:
    #   i = 2*bi + di, j = 2*bj + dj
    #   coeff = a_mi[bi*2+bj] * u_sj[di*2+dj]
    # B-coeff for element (j,k) in 4x4:
    #   j = 2*bi + di, k = 2*bj + dj
    #   coeff = b_mi[bi*2+bj] * v_sj[di*2+dj]
    # C-coeff for element (i,k) in 4x4:
    #   i = 2*bi + di, k = 2*bj + dj
    #   coeff = c_mi[bi*2+bj] * w_sj[di*2+dj]
    
    # Let me define the coefficient arrays.
    
    a_coeffs = [
        [1, 0, 0, 1],  # M0
        [0, 0, 1, 1],  # M1
        [1, 0, 0, 0],  # M2
        [0, 0, 0, 1],  # M3
        [1, 1, 0, 0],  # M4
        [0, 0, 1, -1], # M5
        [0, 1, 0, -1], # M6
    ]
    
    b_coeffs = [
        [1, 0, 0, 1],  # M0
        [1, 0, 0, 0],  # M1
        [0, 0, 1, -1], # M2
        [0, 1, -1, 0], # M3
        [0, 0, 0, 1],  # M4
        [1, 1, 0, 0],  # M5
        [0, 0, 1, 1],  # M6
    ]
    
    c_coeffs = [
        [1, 0, 0, 1],   # M0
        [0, 0, 1, -1],  # M1
        [0, 1, 0, 1],   # M2
        [1, 0, 1, 0],   # M3
        [-1, 1, 0, 0],  # M4
        [0, 0, 0, 1],   # M5
        [1, 0, 0, 1],   # M6
    ]
    
    u_2x2 = [
        [1, 0, 0, 1],  # M0
        [0, 0, 1, 1],  # M1
        [1, 0, 0, 0],  # M2
        [0, 0, 0, 1],  # M3
        [1, 1, 0, 0],  # M4
        [0, 0, 1, -1], # M5
        [0, 1, 0, -1], # M6
    ]
    
    v_2x2 = [
        [1, 0, 0, 1],  # M0
        [1, 0, 0, 0],  # M1
        [0, 0, 1, -1], # M2
        [0, 1, -1, 0], # M3
        [0, 0, 0, 1],  # M4
        [1, 1, 0, 0],  # M5
        [0, 0, 1, 1],  # M6
    ]
    
    w_2x2 = [
        [1, 0, 0, 1],   # M0
        [0, 0, 1, -1],  # M1
        [0, 1, 0, 1],   # M2
        [1, 0, 1, 0],   # M3
        [-1, 1, 0, 0],  # M4
        [0, 0, 0, 1],   # M5
        [1, 0, 0, 1],   # M6
    ]
    
    # Build 4x4x4 decomposition (rank 49)
    rank_4x4 = 49
    U_4x4 = [[0]*rank_4x4 for _ in range(16)]  # 4*4
    V_4x4 = [[0]*rank_4x4 for _ in range(16)]  # 4*4
    W_4x4 = [[0]*rank_4x4 for _ in range(16)]  # 4*4
    
    for mi in range(7):
        for sj in range(7):
            col = mi * 7 + sj
            a = a_coeffs[mi]
            b = b_coeffs[mi]
            c = c_coeffs[mi]
            u = u_2x2[sj]
            v = v_2x2[sj]
            w = w_2x2[sj]
            
            for bi in range(2):
                for bj in range(2):
                    for di in range(2):
                        for dj in range(2):
                            i = 2*bi + di
                            j = 2*bj + dj
                            k = 2*bj + dj  # for B: (j,k) where j=2*bi+di, k=2*bj+dj
                            # A[i,j], B[j,k], C[i,k]
                            a_coeff = a[bi*2+bj] * u[di*2+dj]
                            b_coeff = b[bi*2+bj] * v[di*2+dj]
                            c_coeff = c[bi*2+bj] * w[di*2+dj]
                            
                            U_4x4[i*4+j][col] += a_coeff
                            V_4x4[j*4+k][col] += b_coeff
                            W_4x4[i*4+k][col] += c_coeff
    
    # Now for 4x4 x 4x8: two 4x4x4 multiplications
    # C[:, 0:4] = A * B[:, 0:4]
    # C[:, 4:8] = A * B[:, 4:8]
    
    # U is same for both (A is 4x4)
    # V is split: V[:, 0:4] and V[:, 4:8]
    # W is split: W[:, 0:4] and W[:, 4:8]
    
    # Total rank: 49 + 49 = 98
    total_rank = 98
    U = [[0]*total_rank for _ in range(16)]  # n*m = 4*4
    V = [[0]*total_rank for _ in range(32)]  # m*p = 4*8
    W = [[0]*total_rank for _ in range(32)]  # p*n = 8*4
    
    # First half: k=0..3
    for i in range(16):
        for col in range(49):
            U[i][col] = U_4x4[i][col]
            V[i][col] = V_4x4[i][col]  # B[:, 0:4] -> V index = j*4+k where k=0..3
            W[i][col] = W_4x4[i][col]  # C[:, 0:4] -> W index = k*4+i where k=0..3
    
    # Second half: k=4..7
    for i in range(16):
        for col in range(49):
            col2 = 49 + col
            U[i][col2] = U_4x4[i][col]
            # B[:, 4:8]: V index = j*4+(k+4) where k=0..3
            V[i][col2] = V_4x4[i][col]  # Wait, V_4x4[i] is for B element (j,k) where i=j*4+k
            # Actually V_4x4 is indexed by (j,k) for 4x4 B. For B[:, 4:8], we need B[j, 4+k]
            # So V element (j, 4+k) should get the same coeff as V_4x4 element (j, k)
            # V_4x4[j*4+k][col] -> V[j*4+(k+4)][col2]
            pass
    
    # Let me redo this more carefully.
    # V_4x4 is a 16x49 matrix where V_4x4[j*4+k][col] corresponds to B[j,k]
    # For the first 4x4x4 (B[:, 0:4]): V[j*4+k][col] for k=0..3
    # For the second 4x4x4 (B[:, 4:8]): V[j*4+(4+k)][col2] for k=0..3
    
    # Similarly for W:
    # W_4x4[i*4+k][col] corresponds to C[i,k]
    # For first: W[k*4+i][col] for k=0..3 (note: W index is k*n+i = k*4+i)
    # Wait, the problem says W[k*n+i, r] so W index = k*4+i
    # W_4x4 is indexed by (i,k) where i=0..3, k=0..3, so W_4x4[i*4+k]
    # But we need W[k*4+i], so we need to transpose the index.
    
    # Let me re-examine. In the 4x4x4 decomposition I built:
    # U_4x4[i*4+j][col] for A[i,j]
    # V_4x4[j*4+k][col] for B[j,k]
    # W_4x4[i*4+k][col] for C[i,k]
    
    # But the problem wants W[k*n+i] = W[k*4+i]
    # So W_4x4[i*4+k] should map to W[k*4+i]
    
    # Let me fix the W indexing.
    
    # Rebuild with correct W indexing.
    U_4x4 = [[0]*rank_4x4 for _ in range(16)]
    V_4x4 = [[0]*rank_4x4 for _ in range(16)]
    W_4x4 = [[0]*rank_4x4 for _ in range(16)]  # Will store as W[k*4+i]
    
    for mi in range(7):
        for sj in range(7):
            col = mi * 7 + sj
            a = a_coeffs[mi]
            b = b_coeffs[mi]
            c = c_coeffs[mi]
            u = u_2x2[sj]
            v = v_2x2[sj]
            w = w_2x2[sj]
            
            for bi in range(2):
                for bj in range(2):
                    for di in range(2):
                        for dj in range(2):
                            i = 2*bi + di
                            j = 2*bj + dj
                            k = 2*bj + dj
                            
                            a_coeff = a[bi*2+bj] * u[di*2+dj]
                            b_coeff = b[bi*2+bj] * v[di*2+dj]
                            c_coeff = c[bi*2+bj] * w[di*2+dj]
                            
                            U_4x4[i*4+j][col] += a_coeff
                            V_4x4[j*4+k][col] += b_coeff
                            W_4x4[k*4+i][col] += c_coeff  # W index = k*4+i
    
    # Now assemble for 4x4 x 4x8
    U = [[0]*total_rank for _ in range(16)]
    V = [[0]*total_rank for _ in range(32)]
    W = [[0]*total_rank for _ in range(32)]
    
    # First half: B[:, 0:4], C[:, 0:4]
    for i in range(16):
        for col in range(49):
            U[i][col] = U_4x4[i][col]
            V[i][col] = V_4x4[i][col]  # V[j*4+k] for k=0..3
            W[i][col] = W_4x4[i][col]  # W[k*4+i] for k=0..3
    
    # Second half: B[:, 4:8], C[:, 4:8]
    for i in range(16):
        for col in range(49):
            col2 = 49 + col
            U[i][col2] = U_4x4[i][col]
            # V_4x4[j*4+k][col] -> V[j*4+(4+k)][col2]
            V[i][col2] = V_4x4[i][col]
            # W_4x4[k*4+i][col] -> W[(4+k)*4+i][col2]
            W[i][col2] = W_4x4[i][col]
    
    # Wait, I need to be careful. V_4x4 is indexed by j*4+k.
    # For the second half, B[j, 4+k] should get the same coeff as B[j, k] in the first half.
    # So V[j*4+(4+k)][col2] = V_4x4[j*4+k][col]
    # That means V[i][col2] where i = j*4+(4+k) should equal V_4x4[i-16][col]
    
    # Let me redo:
    U = [[0]*total_rank for _ in range(16)]
    V = [[0]*total_rank for _ in range(32)]
    W = [[0]*total_rank for _ in range(32)]
    
    for i in range(16):
        for col in range(49):
            U[i][col] = U_4x4[i][col]
            V[i][col] = V_4x4[i][col]
            W[i][col] = W_4x4[i][col]
            
            col2 = 49 + col
            U[i][col2] = U_4x4[i][col]
            # V: i = j*4+k, need V[j*4+(4+k)] = V_4x4[j*4+k]
            # So V[i+16][col2] = V_4x4[i][col]
            V[i+16][col2] = V_4x4[i][col]
            # W: i = k*4+di, need W[(4+k)*4+di] = W_4x4[k*4+di]
            # So W[i+16][col2] = W_4x4[i][col]
            W[i+16][col2] = W_4x4[i][col]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

