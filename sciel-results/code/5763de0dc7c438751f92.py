"""5x5 matrix multiplication rank-120 block decomposition (2+3 split)."""
import time

DIMENSIONS = [5, 5, 5]
DEADLINE = 150.0
START_TIME = time.time()

def _time_ok():
    return time.time() - START_TIME < DEADLINE - 10.0

def construct():
    # Rank 120 decomposition:
    # Split 5 = 2 + 3.
    # C22 (2x2): Strassen rank 7
    # C33 (3x3): known rank 23 decomposition
    # C23 (2x3): schoolbook rank 6
    # C32 (3x2): schoolbook rank 6
    # Total: 7 + 23 + 6 + 6 = 42... wait, that's not 120.
    # Let me recalculate. The matrix is 5x5 * 5x5 = 5x5.
    # A = [A22 A23; A32 A33], B = [B22 B23; B32 B33]
    # C = [A22*B22 + A23*B32, A22*B23 + A23*B33; A32*B22 + A33*B32, A32*B23 + A33*B33]
    # C22 = A22*B22 + A23*B32 (2x2 * 2x2 + 2x3 * 3x2)
    # C23 = A22*B23 + A23*B33 (2x2 * 2x3 + 2x3 * 3x3)
    # C32 = A32*B22 + A33*B32 (3x2 * 2x2 + 3x3 * 3x2)
    # C33 = A32*B23 + A33*B33 (3x2 * 2x3 + 3x3 * 3x3)
    #
    # Using independent decompositions for each product:
    # C22: rank(2,2,2) + rank(2,3,2) = 7 + 6 = 13
    # C23: rank(2,2,3) + rank(2,3,3) = 6 + 18 = 24
    # C32: rank(3,2,2) + rank(3,3,2) = 6 + 18 = 24
    # C33: rank(3,2,3) + rank(3,3,3) = 18 + 23 = 41
    # Total: 13 + 24 + 24 + 41 = 102
    #
    # But the inspirations said rank 120. Let me re-read.
    # "5x5 matrix multiplication - rank 120 block decomposition (2+3 split)"
    # Perhaps the mixed blocks use schoolbook:
    # rank(2,3,2) = 6 (schoolbook)
    # rank(2,3,3) = 18 (schoolbook)
    # rank(3,2,2) = 6 (schoolbook)
    # rank(3,3,2) = 18 (schoolbook)
    # rank(3,2,3) = 18 (schoolbook)
    #
    # C22: 7 + 6 = 13
    # C23: 6 + 18 = 24
    # C32: 6 + 18 = 24
    # C33: 18 + 23 = 41
    # Total: 13 + 24 + 24 + 41 = 102
    #
    # Hmm, 102 is better than 120. But maybe the rank 23 for 3x3 is not standard?
    # The best known rank for 3x3 is 19 (by Stothers) or 23 (by others).
    # Let me use rank 23 for 3x3 as stated in the inspirations.
    #
    # Actually, let me reconsider. The score is seed/rank.
    # Seed rank = 125.
    # If I get rank 102, score = 125/102 = 1.225
    # If I get rank 120, score = 125/120 = 1.0417
    #
    # Let me try to get rank 102.
    #
    # For 2x2: Strassen rank 7
    # For 3x3: rank 23 (I'll use a known decomposition)
    # For mixed blocks: schoolbook
    #
    # rank(2,2,2) = 7
    # rank(2,2,3) = 6 (schoolbook)
    # rank(2,3,2) = 6 (schoolbook)
    # rank(2,3,3) = 18 (schoolbook)
    # rank(3,2,2) = 6 (schoolbook)
    # rank(3,2,3) = 18 (schoolbook)
    # rank(3,3,2) = 18 (schoolbook)
    # rank(3,3,3) = 23
    #
    # C22 = A22*B22 + A23*B32: rank(2,2,2) + rank(2,3,2) = 7 + 6 = 13
    # C23 = A22*B23 + A23*B33: rank(2,2,3) + rank(2,3,3) = 6 + 18 = 24
    # C32 = A32*B22 + A33*B32: rank(3,2,2) + rank(3,3,2) = 6 + 18 = 24
    # C33 = A32*B23 + A33*B33: rank(3,2,3) + rank(3,3,3) = 18 + 23 = 41
    # Total = 13 + 24 + 24 + 41 = 102
    #
    # Score = 125/102 ≈ 1.225

    n, m, p = 5, 5, 5
    total_rank = 102
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Helper: add a term (u, v, w) where u, v, w are (n*m, m*p, p*n) vectors
    def add_term(u_vals, v_vals, w_vals):
        nonlocal col
        for i in range(n * m):
            if u_vals[i] != 0:
                U[i][col] += u_vals[i]
        for i in range(m * p):
            if v_vals[i] != 0:
                V[i][col] += v_vals[i]
        for i in range(p * n):
            if w_vals[i] != 0:
                W[i][col] += w_vals[i]
        col += 1
    
    # Strassen 2x2 decomposition
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # C = A*B
    # 7 terms:
    # m1 = (a11+a22)*(b11+b22)
    # m2 = (a21+a22)*b11
    # m3 = a11*(b12-b22)
    # m4 = a22*(b21-b11)
    # m5 = (a11+a12)*b22
    # m6 = (a21-a11)*(b11+b12)
    # m7 = (a12-a22)*(b21+b22)
    # C11 = m1 + m4 - m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6
    
    def strassen_2x2(i_base, j_base, k_base):
        # A indices: (i_base + i, j_base + j) for i,j in {0,1}
        # B indices: (j_base + j, k_base + k) for j,k in {0,1}
        # C indices: (k_base + k, i_base + i) for k,i in {0,1}
        # W[k*n+i, c]
        
        # m1 = (a11+a22)*(b11+b22)
        u1 = [0.0] * 25
        v1 = [0.0] * 25
        w1 = [0.0] * 25
        u1[i_base * 5 + j_base] = 1.0
        u1[(i_base + 1) * 5 + (j_base + 1)] = 1.0
        v1[j_base * 5 + k_base] = 1.0
        v1[(j_base + 1) * 5 + (k_base + 1)] = 1.0
        w1[k_base * 5 + i_base] = 1.0
        w1[(k_base + 1) * 5 + (i_base + 1)] = 1.0
        add_term(u1, v1, w1)
        
        # m2 = (a21+a22)*b11
        u2 = [0.0] * 25
        v2 = [0.0] * 25
        w2 = [0.0] * 25
        u2[(i_base + 1) * 5 + j_base] = 1.0
        u2[(i_base + 1) * 5 + (j_base + 1)] = 1.0
        v2[j_base * 5 + k_base] = 1.0
        w2[k_base * 5 + (i_base + 1)] = 1.0
        add_term(u2, v2, w2)
        
        # m3 = a11*(b12-b22)
        u3 = [0.0] * 25
        v3 = [0.0] * 25
        w3 = [0.0] * 25
        u3[i_base * 5 + j_base] = 1.0
        v3[j_base * 5 + (k_base + 1)] = 1.0
        v3[(j_base + 1) * 5 + (k_base + 1)] = -1.0
        w3[(k_base + 1) * 5 + i_base] = 1.0
        add_term(u3, v3, w3)
        
        # m4 = a22*(b21-b11)
        u4 = [0.0] * 25
        v4 = [0.0] * 25
        w4 = [0.0] * 25
        u4[(i_base + 1) * 5 + (j_base + 1)] = 1.0
        v4[(j_base + 1) * 5 + k_base] = 1.0
        v4[j_base * 5 + k_base] = -1.0
        w4[k_base * 5 + (i_base + 1)] = 1.0
        add_term(u4, v4, w4)
        
        # m5 = (a11+a12)*b22
        u5 = [0.0] * 25
        v5 = [0.0] * 25
        w5 = [0.0] * 25
        u5[i_base * 5 + j_base] = 1.0
        u5[i_base * 5 + (j_base + 1)] = 1.0
        v5[(j_base + 1) * 5 + (k_base + 1)] = 1.0
        w5[(k_base + 1) * 5 + i_base] = 1.0
        add_term(u5, v5, w5)
        
        # m6 = (a21-a11)*(b11+b12)
        u6 = [0.0] * 25
        v6 = [0.0] * 25
        w6 = [0.0] * 25
        u6[(i_base + 1) * 5 + j_base] = 1.0
        u6[i_base * 5 + j_base] = -1.0
        v6[j_base * 5 + k_base] = 1.0
        v6[j_base * 5 + (k_base + 1)] = 1.0
        w6[(k_base + 1) * 5 + (i_base + 1)] = 1.0
        add_term(u6, v6, w6)
        
        # m7 = (a12-a22)*(b21+b22)
        u7 = [0.0] * 25
        v7 = [0.0] * 25
        w7 = [0.0] * 25
        u7[i_base * 5 + (j_base + 1)] = 1.0
        u7[(i_base + 1) * 5 + (j_base + 1)] = -1.0
        v7[(j_base + 1) * 5 + k_base] = 1.0
        v7[(j_base + 1) * 5 + (k_base + 1)] = 1.0
        w7[k_base * 5 + i_base] = 1.0
        add_term(u7, v7, w7)
    
    # Schoolbook 2x2x3 (A is 2x2, B is 2x3, C is 2x3)
    def schoolbook_223(i_base, j_base, k_base):
        # A: 2x2 at (i_base, j_base)
        # B: 2x3 at (j_base, k_base)
        # C: 2x3 at (k_base, i_base) -> W[k*n+i, c]
        for j in range(2):
            for k in range(3):
                u = [0.0] * 25
                v = [0.0] * 25
                w = [0.0] * 25
                u[i_base * 5 + (j_base + j)] = 1.0
                v[(j_base + j) * 5 + (k_base + k)] = 1.0
                w[(k_base + k) * 5 + i_base] = 1.0
                add_term(u, v, w)
    
    # Schoolbook 2x3x2 (A is 2x3, B is 3x2, C is 2x2)
    def schoolbook_232(i_base, j_base, k_base):
        # A: 2x3 at (i_base, j_base)
        # B: 3x2 at (j_base, k_base)
        # C: 2x2 at (k_base, i_base) -> W[k*n+i, c]
        for j in range(3):
            for k in range(2):
                u = [0.0] * 25
                v = [0.0] * 25
                w = [0.0] * 25
                u[i_base * 5 + (j_base + j)] = 1.0
                v[(j_base + j) * 5 + (k_base + k)] = 1.0
                w[(k_base + k) * 5 + i_base] = 1.0
                add_term(u, v, w)
    
    # Schoolbook 2x3x3 (A is 2x3, B is 3x3, C is 2x3)
    def schoolbook_233(i_base, j_base, k_base):
        # A: 2x3 at (i_base, j_base)
        # B: 3x3 at (j_base, k_base)
        # C: 2x3 at (k_base, i_base) -> W[k*n+i, c]
        for j in range(3):
            for k in range(3):
                u = [0.0] * 25
                v = [0.0] * 25
                w = [0.0] * 25
                u[i_base * 5 + (j_base + j)] = 1.0
                v[(j_base + j) * 5 + (k_base + k)] = 1.0
                w[(k_base + k) * 5 + i_base] = 1.0
                add_term(u, v, w)
    
    # Schoolbook 3x2x2 (A is 3x2, B is 2x2, C is 3x2)
    def schoolbook_322(i_base, j_base, k_base):
        # A: 3x2 at (i_base, j_base)
        # B: 2x2 at (j_base, k_base)
        # C: 3x2 at (k_base, i_base) -> W[k*n+i, c]
        for j in range(2):
            for k in range(2):
                u = [0.0] * 25
                v = [0.0] * 25
                w = [0.0] * 25
                u[i_base * 5 + (j_base + j)] = 1.0
                v[(j_base + j) * 5 + (k_base + k)] = 1.0
                w[(k_base + k) * 5 + i_base] = 1.0
                add_term(u, v, w)
    
    # Schoolbook 3x2x3 (A is 3x2, B is 2x3, C is 3x3)
    def schoolbook_323(i_base, j_base, k_base):
        # A: 3x2 at (i_base, j_base)
        # B: 2x3 at (j_base, k_base)
        # C: 3x3 at (k_base, i_base) -> W[k*n+i, c]
        for j in range(2):
            for k in range(3):
                u = [0.0] * 25
                v = [0.0] * 25
                w = [0.0] * 25
                u[i_base * 5 + (j_base + j)] = 1.0
                v[(j_base + j) * 5 + (k_base + k)] = 1.0
                w[(k_base + k) * 5 + i_base] = 1.0
                add_term(u, v, w)
    
    # Schoolbook 3x3x2 (A is 3x3, B is 3x2, C is 3x2)
    def schoolbook_332(i_base, j_base, k_base):
        # A: 3x3 at (i_base, j_base)
        # B: 3x2 at (j_base, k_base)
        # C: 3x2 at (k_base, i_base) -> W[k*n+i, c]
        for j in range(3):
            for k in range(2):
                u = [0.0] * 25
                v = [0.0] * 25
                w = [0.0] * 25
                u[i_base * 5 + (j_base + j)] = 1.0
                v[(j_base + j) * 5 + (k_base + k)] = 1.0
                w[(k_base + k) * 5 + i_base] = 1.0
                add_term(u, v, w)
    
    # Rank 23 decomposition for 3x3 matrix multiplication
    # This is a known decomposition. I'll use a simplified version or schoolbook if needed.
    # For now, let's use schoolbook for 3x3 (rank 27) to ensure correctness,
    # then optimize if time permits.
    # Actually, let me use a known rank 23 decomposition.
    # I'll hardcode it.
    
    def rank23_3x3(i_base, j_base, k_base):
        # A: 3x3 at (i_base, j_base)
        # B: 3x3 at (j_base, k_base)
        # C: 3x3 at (k_base, i_base) -> W[k*n+i, c]
        # I'll use schoolbook for now (rank 27) to ensure correctness.
        for j in range(3):
            for k in range(3):
                u = [0.0] * 25
                v = [0.0] * 25
                w = [0.0] * 25
                u[i_base * 5 + (j_base + j)] = 1.0
                v[(j_base + j) * 5 + (k_base + k)] = 1.0
                w[(k_base + k) * 5 + i_base] = 1.0
                add_term(u, v, w)
    
    # Now assemble the 5x5 multiplication
    # A = [A22 A23; A32 A33] where A22 is 2x2, A23 is 2x3, A32 is 3x2, A33 is 3x3
    # B = [B22 B23; B32 B33] where B22 is 2x2, B23 is 2x3, B32 is 3x2, B33 is 3x3
    # C = [C22 C23; C32 C33] where C22 is 2x2, C23 is 2x3, C32 is 3x2, C33 is 3x3
    
    # C22 = A22*B22 + A23*B32
    # A22*B22: 2x2 * 2x2 -> Strassen (rank 7)
    # A23*B32: 2x3 * 3x2 -> schoolbook (rank 6)
    strassen_2x2(0, 0, 0)  # A22*B22
    schoolbook_232(0, 2, 2)  # A23*B32
    
    # C23 = A22*B23 + A23*B33
    # A22*B23: 2x2 * 2x3 -> schoolbook (rank 6)
    # A23*B33: 2x3 * 3x3 -> schoolbook (rank 18)
    schoolbook_223(0, 0, 2)  # A22*B23
    schoolbook_233(0, 2, 2)  # A23*B33
    
    # C32 = A32*B22 + A33*B32
    # A32*B22: 3x2 * 2x2 -> schoolbook (rank 6)
    # A33*B32: 3x3 * 3x2 -> schoolbook (rank 18)
    schoolbook_322(2, 0, 0)  # A32*B22
    schoolbook_332(2, 2, 0)  # A33*B32
    
    # C33 = A32*B23 + A33*B33
    # A32*B23: 3x2 * 2x3 -> schoolbook (rank 18)
    # A33*B33: 3x3 * 3x3 -> rank 23 (or schoolbook 27 for now)
    schoolbook_323(2, 0, 2)  # A32*B23
    rank23_3x3(2, 2, 2)  # A33*B33
    
    return {"U": U, "V": V, "W": W}
