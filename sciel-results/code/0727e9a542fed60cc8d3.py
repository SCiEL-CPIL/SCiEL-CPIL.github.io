import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    
    # We will construct the decomposition in parts.
    # Part 1: B cols 0,1 (3x3 * 3x2). Rank 17.
    # Part 2: B cols 2,3,4 (3x3 * 3x3). Rank 23 (verified) or 18.
    # Part 3: B cols 5,6 (3x3 * 3x2). Rank 17.
    
    total_rank = 17 + 23 + 17
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    r_offset = 0
    
    # Helper to add a rank-1 component
    def add_component(u_vec, v_vec, w_vec):
        nonlocal r_offset
        for i in range(n * m):
            U[i][r_offset] = u_vec[i]
        for i in range(m * p):
            V[i][r_offset] = v_vec[i]
        for i in range(p * n):
            W[i][r_offset] = w_vec[i]
        r_offset += 1
        
    # Helper to add a schoolbook product A[i,j] * B[k,l] -> C[i,l]
    def add_schoolbook(i, j, k, l, coeff=1):
        nonlocal r_offset
        u_vec = [0] * (n * m)
        v_vec = [0] * (m * p)
        w_vec = [0] * (p * n)
        
        u_vec[i * m + j] = coeff
        v_vec[k * p + l] = coeff
        w_vec[l * n + i] = coeff
        
        add_component(u_vec, v_vec, w_vec)

    # --- Part 1: Cols 0, 1 ---
    # Strassen for A[0:2, 0:2] * B[0:2, 0:2]
    # A indices: 00, 01, 10, 11 -> 0, 1, 3, 4
    # B indices: 00, 01, 10, 11 -> 0, 1, 3, 4 (l is 0,1)
    
    # p1 = (a00+a11)(b00+b11)
    u = [0]*9; v = [0]*21; w = [0]*21
    u[0]=1; u[4]=1
    v[0]=1; v[4]=1
    # c00 += p1, c11 += p1
    w[0] = 1 # c00 (l=0, i=0) -> 0*3+0=0
    w[1] = 1 # c11 (l=1, i=1) -> 1*3+1=4? No.
    # W index: l*n + i.
    # c00: l=0, i=0 -> 0.
    # c11: l=1, i=1 -> 1*3+1 = 4.
    # c01: l=1, i=0 -> 1*3+0 = 3.
    # c10: l=0, i=1 -> 0*3+1 = 1.
    
    # Let's use a helper for Strassen terms to avoid index errors.
    
    def add_strassen_term(a_coeff, b_coeff, c_coeff):
        # a_coeff: dict {idx: val}
        # b_coeff: dict {idx: val}
        # c_coeff: dict {idx: val}
        nonlocal r_offset
        u_vec = [0]*9
        v_vec = [0]*21
        w_vec = [0]*21
        for idx, val in a_coeff.items():
            u_vec[idx] = val
        for idx, val in b_coeff.items():
            v_vec[idx] = val
        for idx, val in c_coeff.items():
            w_vec[idx] = val
        add_component(u_vec, v_vec, w_vec)

    # Strassen 7 products for top-left 2x2
    # A: 00(0), 01(1), 10(3), 11(4)
    # B: 00(0), 01(1), 10(3), 11(4)
    # C: 00(0), 01(3), 10(1), 11(4)
    
    add_strassen_term({0:1, 4:1}, {0:1, 4:1}, {0:1, 4:1}) # p1
    add_strassen_term({3:1, 4:1}, {0:1}, {1:1}) # p2 -> c10
    add_strassen_term({0:1}, {3:-1, 4:1}, {3:1}) # p3 -> c01 (b10-b11)
    add_strassen_term({4:1}, {1:-1, 0:1}, {3:1}) # p4 -> c01 (b01-b00)
    add_strassen_term({0:1, 1:1}, {4:1}, {0:1}) # p5 -> c00 (a00+a01)b11
    add_strassen_term({3:-1, 0:1}, {0:1, 1:1}, {1:-1}) # p6 -> c10 (a10-a00)(b00+b01)
    add_strassen_term({1:-1, 4:1}, {3:1, 4:1}, {4:-1, 3:1}) # p7 -> c11, c01 (a01-a11)(b10+b11)
    
    # Cross terms for C_top (2x2)
    # c00 += a02 b20
    add_schoolbook(0, 2, 2, 0)
    # c01 += a02 b21
    add_schoolbook(0, 2, 2, 1)
    # c10 += a12 b20
    add_schoolbook(1, 2, 2, 0)
    # c11 += a12 b21
    add_schoolbook(1, 2, 2, 1)
    
    # C_bottom (Row 2)
    # c20 = a20 b00 + a21 b10 + a22 b20
    add_schoolbook(2, 0, 0, 0)
    add_schoolbook(2, 1, 1, 0)
    add_schoolbook(2, 2, 2, 0)
    # c21 = a20 b01 + a21 b11 + a22 b21
    add_schoolbook(2, 0, 0, 1)
    add_schoolbook(2, 1, 1, 1)
    add_schoolbook(2, 2, 2, 1)
    
    # Part 1 done. r_offset should be 17.
    
    # --- Part 2: Cols 2, 3, 4 ---
    # Use Rank 23 if available, else 18.
    # I will attempt Rank 23.
    # If I don't have a verified set, I use Schoolbook 18.
    # For now, I will use Schoolbook 18 to ensure correctness.
    # This makes total rank 17 + 18 + 17 = 52.
    
    for l in range(2, 5):
        for i in range(3):
            for j in range(3):
                add_schoolbook(i, j, j, l) # A[i,j] * B[j,l] -> C[i,l]
                
    # Part 2 done. r_offset should be 17 + 18 = 35.
    
    # --- Part 3: Cols 5, 6 ---
    # Same as Part 1, but for columns 5, 6.
    # B indices for col 5: 0*7+5=5, 1*7+5=12, 2*7+5=19
    # B indices for col 6: 0*7+6=6, 1*7+6=13, 2*7+6=20
    
    # Strassen for A[0:2, 0:2] * B[0:2, 5:7]
    # A: 00(0), 01(1), 10(3), 11(4)
    # B: 05(5), 06(6), 15(12), 16(13)
    # C: 05(15), 06(16), 15(18), 16(19)
    
    add_strassen_term({0:1, 4:1}, {5:1, 13:1}, {15:1, 19:1}) # p1
    add_strassen_term({3:1, 4:1}, {5:1}, {18:1}) # p2
    add_strassen_term({0:1}, {12:-1, 13:1}, {16:1}) # p3
    add_strassen_term({4:1}, {6:-1, 5:1}, {16:1}) # p4
    add_strassen_term({0:1, 1:1}, {13:1}, {15:1}) # p5
    add_strassen_term({3:-1, 0:1}, {5:1, 6:1}, {18:-1}) # p6
    add_strassen_term({1:-1, 4:1}, {12:1, 13:1}, {19:-1, 16:1}) # p7
    
    # Cross terms
    add_schoolbook(0, 2, 2, 5)
    add_schoolbook(0, 2, 2, 6)
    add_schoolbook(1, 2, 2, 5)
    add_schoolbook(1, 2, 2, 6)
    
    # C_bottom
    add_schoolbook(2, 0, 0, 5)
    add_schoolbook(2, 1, 1, 5)
    add_schoolbook(2, 2, 2, 5)
    add_schoolbook(2, 0, 0, 6)
    add_schoolbook(2, 1, 1, 6)
    add_schoolbook(2, 2, 2, 6)
    
    return {"U": U, "V": V, "W": W}
