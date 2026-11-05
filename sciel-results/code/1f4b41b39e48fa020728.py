import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # We will construct U, V, W matrices of shape (n*m, rank), (m*p, rank), (p*n, rank)
    # rank = 64
    
    rank = 64
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: Laderman for 4x4x4 (columns 0-3 of B)
    # Laderman decomposition for 4x4x4
    # A is 4x4, B1 is 4x4 (first 4 cols of B), C1 is 4x4 (first 4 cols of C)
    
    # Laderman's algorithm uses 20 multiplications.
    # We define the 20 bilinear forms.
    # Let A = [[a, b, c, d], [e, f, g, h], [i, j, k, l], [m, n, o, p]]
    # Let B1 = [[q, r, s, t], [u, v, w, x], [y, z, A1, B1], [C1, D1, E1, F1]]
    
    # Laderman 20 mults:
    # M1 = (a+c) * (q+y)
    # M2 = (a-c) * (q-y)
    # M3 = (b+d) * (r+z)
    # M4 = (b-d) * (r-z)
    # M5 = (e+g) * (u+w)
    # M6 = (e-g) * (u-w)
    # M7 = (f+h) * (v+x)
    # M8 = (f-h) * (v-x)
    # M9 = (i+k) * (y+A1)  # Note: indices in A row 2 are i,j,k,l
    # M10 = (i-k) * (y-A1)
    # M11 = (j+l) * (z+B1)
    # M12 = (j-l) * (z-B1)
    # M13 = (m+o) * (C1+E1)
    # M14 = (m-o) * (C1-E1)
    # M15 = (n+p) * (D1+F1)
    # M16 = (n-p) * (D1-F1)
    # M17 = (a+e) * (q+u)
    # M18 = (b+f) * (r+v)
    # M19 = (c+g) * (s+w)
    # M20 = (d+h) * (t+x)
    
    # Wait, standard Laderman is slightly different. 
    # Let's use a known explicit Laderman decomposition.
    
    # Actually, implementing Laderman correctly from scratch in the time is risky.
    # Alternative: Use Strassen for 4x4x4?
    # Strassen for 4x4x4: Split into 2x2 blocks.
    # 7 mults of 2x2x2. Each 2x2x2 is 7 mults. Total 49.
    # 49 > 20. So Laderman is better.
    
    # Let's try to find a simpler improvement.
    # What if we just use Strassen for the whole 4x4x7?
    # Split A into 2x2 blocks (4 blocks).
    # Split B (4x7) into 2x7 blocks (2 blocks).
    # C (4x7) into 2x7 blocks (2 blocks).
    # C_top = A11 B_top + A12 B_bot
    # C_bot = A21 B_top + A22 B_bot
    # This requires 4 multiplications of type 2x2x7.
    # Rank of 2x2x7?
    # 2x2x2 is 7.
    # 2x2x7 can be done by 7 * 7 = 49? No.
    # 2x2x7 is equivalent to 7 multiplications of 2x2x1? No.
    # 2x2x7: A is 2x2, B is 2x7.
    # We can compute A * B using Strassen-like methods?
    # Actually, 2x2x7 can be done with 14 multiplications?
    # If we split B into B1 (2x2), B2 (2x2), B3 (2x2), B4 (2x1).
    # 2x2x2 (7) + 2x2x2 (7) + 2x2x2 (7) + 2x2x1 (4) = 25.
    # So 4 * 25 = 100.
    # 100 < 112.
    # Score 112/100 = 1.12.
    
    # This is a safe, easy-to-implement improvement!
    # Let's implement the 4x4x7 via 2x2 blocks.
    
    # Rank calculation:
    # 4 blocks of 2x2x7.
    # Each 2x2x7:
    #   Split B (2x7) into B_a (2x2), B_b (2x2), B_c (2x2), B_d (2x1).
    #   Mult 1: 2x2x2 (Strassen, rank 7)
    #   Mult 2: 2x2x2 (Strassen, rank 7)
    #   Mult 3: 2x2x2 (Strassen, rank 7)
    #   Mult 4: 2x2x1 (Schoolbook, rank 4)
    #   Total per 2x2x7: 25.
    # Total Rank: 4 * 25 = 100.
    
    # Let's refine the 2x2x1 rank.
    # A (2x2) * B (2x1) = C (2x1).
    # C1 = a11*b1 + a12*b2
    # C2 = a21*b1 + a22*b2
    # Schoolbook: 4 mults.
    # Can we do it in 3? No, rank is 4.
    
    # So Rank 100 is achievable.
    
    # Let's check if we can do better for 2x2x7.
    # Maybe 2x2x7 rank is 14?
    # If we use Strassen for 2x2x2 (7 mults) for the first 4 columns (two 2x2x2s = 14 mults).
    # And schoolbook for last 3 columns (2x2x3).
    # 2x2x3 rank:
    #   2x2x2 (7) + 2x2x1 (4) = 11.
    #   So 14 + 11 = 25.
    #   Same result.
    
    # What if we use Laderman for 2x2x7? No, Laderman is for 4x4x4.
    
    # Is there a better 2x2x7?
    # The rank of 2x2x7 is 14?
    # If rank(2x2x7) = 14, then total rank = 4 * 14 = 56.
    # Score 112/56 = 2.0.
    
    # Let's assume rank(2x2x7) = 14.
    # How to get 14?
    # 2x2x7 = 2x2x2 + 2x2x2 + 2x2x2 + 2x2x1?
    # 7+7+7+4 = 25.
    # Maybe we can share?
    # No, they are independent columns.
    
    # Let's stick to Rank 100 for safety, as implementing Strassen correctly is easier than finding a complex 14-rank 2x2x7 algorithm.
    
    rank = 100
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen 2x2x2 decomposition
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)(e)
    # M3 = (a-b)(f+h)
    # M4 = (d)(g-h)
    # M5 = (a)(e)
    # M6 = (b)(e+f)
    # M7 = (c)(g+h)
    # C11 = M1 - M4 + M5 + M6
    # C12 = M3 + M4
    # C21 = M2 + M5
    # C22 = M1 - M2 + M3 + M7
    
    def add_strassen_2x2x2(U, V, W, col, a_idx, b_idx, c_idx):
        # a_idx: indices of A (4 elements)
        # b_idx: indices of B (4 elements)
        # c_idx: indices of C (4 elements)
        # U rows: a_idx
        # V rows: b_idx
        # W rows: c_idx
        
        # M1 = (a+d)(e+h)
        U[a_idx[0]][col] += 1
        U[a_idx[3]][col] += 1
        V[b_idx[0]][col] += 1
        V[b_idx[3]][col] += 1
        W[c_idx[0]][col] += 1
        W[c_idx[2]][col] += 1
        W[c_idx[1]][col] -= 1
        W[c_idx[3]][col] -= 1
        col += 1
        
        # M2 = (c+d)(e)
        U[a_idx[2]][col] += 1
        U[a_idx[3]][col] += 1
        V[b_idx[0]][col] += 1
        W[c_idx[2]][col] += 1
        W[c_idx[3]][col] -= 1
        col += 1
        
        # M3 = (a-b)(f+h)
        U[a_idx[0]][col] += 1
        U[a_idx[1]][col] -= 1
        V[b_idx[1]][col] += 1
        V[b_idx[3]][col] += 1
        W[c_idx[1]][col] += 1
        W[c_idx[3]][col] += 1
        col += 1
        
        # M4 = (d)(g-h)
        U[a_idx[3]][col] += 1
        V[b_idx[2]][col] += 1
        V[b_idx[3]][col] -= 1
        W[c_idx[0]][col] -= 1
        W[c_idx[1]][col] += 1
        W[c_idx[3]][col] -= 1
        col += 1
        
        # M5 = (a)(e)
        U[a_idx[0]][col] += 1
        V[b_idx[0]][col] += 1
        W[c_idx[0]][col] += 1
        W[c_idx[2]][col] += 1
        col += 1
        
        # M6 = (b)(e+f)
        U[a_idx[1]][col] += 1
        V[b_idx[0]][col] += 1
        V[b_idx[1]][col] += 1
        W[c_idx[0]][col] += 1
        col += 1
        
        # M7 = (c)(g+h)
        U[a_idx[2]][col] += 1
        V[b_idx[2]][col] += 1
        V[b_idx[3]][col] += 1
        W[c_idx[3]][col] += 1
        col += 1
        
        return col

    # Schoolbook 2x2x1
    # A = [[a, b], [c, d]], B = [[e], [g]]
    # C1 = ae + bg
    # C2 = ce + dg
    
    def add_schoolbook_2x2x1(U, V, W, col, a_idx, b_idx, c_idx):
        # M1 = a*e
        U[a_idx[0]][col] += 1
        V[b_idx[0]][col] += 1
        W[c_idx[0]][col] += 1
        col += 1
        
        # M2 = b*g
        U[a_idx[1]][col] += 1
        V[b_idx[1]][col] += 1
        W[c_idx[0]][col] += 1
        col += 1
        
        # M3 = c*e
        U[a_idx[2]][col] += 1
        V[b_idx[0]][col] += 1
        W[c_idx[1]][col] += 1
        col += 1
        
        # M4 = d*g
        U[a_idx[3]][col] += 1
        V[b_idx[1]][col] += 1
        W[c_idx[1]][col] += 1
        col += 1
        
        return col

    # Main loop for 4 blocks of 2x2x7
    # A is 4x4. Indices:
    # A11: 0,1,4,5
    # A12: 2,3,6,7
    # A21: 8,9,12,13
    # A22: 10,11,14,15
    
    # B is 4x7.
    # Top 2 rows: 0..6, 7..13
    # Bot 2 rows: 14..20, 21..27
    
    # C is 4x7.
    # Top 2 rows: 0..6, 7..13 (k*n+i -> k*4+i)
    # Bot 2 rows: 16..22, 23..29
    
    # Block 1: A11 * B_top
    a_idx = [0, 1, 4, 5]
    # B_top is 2x7.
    # Split B_top into B_t1 (2x2), B_t2 (2x2), B_t3 (2x2), B_t4 (2x1)
    # B_t1: cols 0,1 -> indices 0,1, 7,8
    # B_t2: cols 2,3 -> indices 2,3, 9,10
    # B_t3: cols 4,5 -> indices 4,5, 11,12
    # B_t4: col 6 -> indices 6, 13
    
    # C_top is 2x7.
    # C_t1: cols 0,1 -> indices 0,1, 4,5
    # C_t2: cols 2,3 -> indices 2,3, 6,7
    # C_t3: cols 4,5 -> indices 4,5, 8,9
    # C_t4: col 6 -> indices 6, 10
    
    col = add_strassen_2x2x2(U, V, W, col, a_idx, [0,1,7,8], [0,1,4,5])
    col = add_strassen_2x2x2(U, V, W, col, a_idx, [2,3,9,10], [2,3,6,7])
    col = add_strassen_2x2x2(U, V, W, col, a_idx, [4,5,11,12], [4,5,8,9])
    col = add_schoolbook_2x2x1(U, V, W, col, a_idx, [6,13], [6,10])
    
    # Block 2: A12 * B_bot
    a_idx = [2, 3, 6, 7]
    # B_bot is 2x7.
    # B_b1: cols 0,1 -> indices 14,15, 21,22
    # B_b2: cols 2,3 -> indices 16,17, 23,24
    # B_b3: cols 4,5 -> indices 18,19, 25,26
    # B_b4: col 6 -> indices 20, 27
    
    # C_top is 2x7.
    col = add_strassen_2x2x2(U, V, W, col, a_idx, [14,15,21,22], [0,1,4,5])
    col = add_strassen_2x2x2(U, V, W, col, a_idx, [16,17,23,24], [2,3,6,7])
    col = add_strassen_2x2x2(U, V, W, col, a_idx, [18,19,25,26], [4,5,8,9])
    col = add_schoolbook_2x2x1(U, V, W, col, a_idx, [20,27], [6,10])
    
    # Block 3: A21 * B_top
    a_idx = [8, 9, 12, 13]
    # C_bot is 2x7.
    # C_b1: cols 0,1 -> indices 16,17, 20,21
    # C_b2: cols 2,3 -> indices 18,19, 22,23
    # C_b3: cols 4,5 -> indices 20,21, 24,25
    # C_b4: col 6 -> indices 22, 26
    
    col = add_strassen_2x2x2(U, V, W, col, a_idx, [0,1,7,8], [16,17,20,21])
    col = add_strassen_2x2x2(U, V, W, col, a_idx, [2,3,9,10], [18,19,22,23])
    col = add_strassen_2x2x2(U, V, W, col, a_idx, [4,5,11,12], [20,21,24,25])
    col = add_schoolbook_2x2x1(U, V, W, col, a_idx, [6,13], [22,26])
    
    # Block 4: A22 * B_bot
    a_idx = [10, 11, 14, 15]
    
    col = add_strassen_2x2x2(U, V, W, col, a_idx, [14,15,21,22], [16,17,20,21])
    col = add_strassen_2x2x2(U, V, W, col, a_idx, [16,17,23,24], [18,19,22,23])
    col = add_strassen_2x2x2(U, V, W, col, a_idx, [18,19,25,26], [20,21,24,25])
    col = add_schoolbook_2x2x1(U, V, W, col, a_idx, [20,27], [22,26])
    
    # Total rank used should be 100
    # 4 blocks * (3*7 + 4) = 4 * 25 = 100.
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
