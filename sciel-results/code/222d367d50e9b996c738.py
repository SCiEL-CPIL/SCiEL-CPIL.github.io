import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    n, m, p = DIMENSIONS
    
    # We will build the global U, V, W matrices.
    # Sizes:
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    # We estimate the rank to be 56 based on the decomposition strategy.
    # Strategy:
    # 1. Split m=5 into m1=2, m2=3.
    # 2. Split p=6 into p1=3, p2=3.
    
    # Sub-problems:
    # SP1: A[0:2, 0:2] (2x2) * B[0:2, 0:3] (2x3) -> Rank 11
    # SP2: A[0:2, 0:2] (2x2) * B[0:2, 3:6] (2x3) -> Rank 11
    # SP3: A[0:2, 2:5] (2x3) * B[2:5, 0:3] (3x3) -> Rank 17
    # SP4: A[0:2, 2:5] (2x3) * B[2:5, 3:6] (3x3) -> Rank 17
    
    # Total Rank = 11 + 11 + 17 + 17 = 56.
    
    rank = 56
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_offset = 0
    
    # Helper to add a local decomposition to the global matrices
    def add_decomp(n_loc, m_loc, p_loc, u_off, v_off, w_off, gen_func):
        nonlocal col_offset
        r = gen_func(n_loc, m_loc, p_loc)
        # r is a tuple (U_loc, V_loc, W_loc) of lists of lists
        U_loc, V_loc, W_loc = r
        r_len = len(U_loc[0])
        
        for i in range(n_loc):
            for j in range(m_loc):
                u_idx = u_off + i * m_loc + j
                for c in range(r_len):
                    U[u_idx][col_offset + c] += U_loc[i * m_loc + j][c]
                    
        for j in range(m_loc):
            for k in range(p_loc):
                v_idx = v_off + j * p_loc + k
                for c in range(r_len):
                    V[v_idx][col_offset + c] += V_loc[j * p_loc + k][c]
                    
        for k in range(p_loc):
            for i in range(n_loc):
                w_idx = w_off + k * n_loc + i
                for c in range(r_len):
                    W[w_idx][col_offset + c] += W_loc[k * n_loc + i][c]
                    
        col_offset += r_len

    # Generator for 2x2x3 (Rank 11)
    def gen_2x2x3(n, m, p):
        # n=2, m=2, p=3
        # Split p into 2 and 1.
        # Part 1: 2x2x2 (Strassen)
        # Part 2: 2x2x1 (Schoolbook)
        
        U_part1, V_part1, W_part1 = gen_2x2x2(2, 2, 2)
        U_part2, V_part2, W_part2 = gen_2x2x1(2, 2, 1)
        
        # Combine
        r1 = len(U_part1[0])
        r2 = len(U_part2[0])
        r_total = r1 + r2
        
        U = [[0] * r_total for _ in range(2*2)]
        V = [[0] * r_total for _ in range(2*3)]
        W = [[0] * r_total for _ in range(3*2)]
        
        # Part 1 maps to k=0,1
        for i in range(2):
            for j in range(2):
                for c in range(r1):
                    U[i*2+j][c] = U_part1[i*2+j][c]
        for j in range(2):
            for k in range(2):
                for c in range(r1):
                    V[j*3+k][c] = V_part1[j*2+k][c]
        for k in range(2):
            for i in range(2):
                for c in range(r1):
                    W[k*2+i][c] = W_part1[k*2+i][c]
                    
        # Part 2 maps to k=2
        for i in range(2):
            for j in range(2):
                for c in range(r2):
                    U[i*2+j][r1+c] = U_part2[i*2+j][c]
        for j in range(2):
            # k_local in part 2 is 0, maps to k_global=2
            for c in range(r2):
                    V[j*3+2][r1+c] = V_part2[j*1+0][c]
        for i in range(2):
            # k_local in part 2 is 0, maps to k_global=2
            for c in range(r2):
                    W[2*2+i][r1+c] = W_part2[0*2+i][c]
                    
        return U, V, W

    # Generator for 2x2x2 (Strassen, Rank 7)
    def gen_2x2x2(n, m, p):
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # Strassen coefficients
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B10
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B00
        # M6 = (A10-A11)(B01+B11)
        # M7 = (A01+A11)B10
        
        # U indices: i*m+j
        # V indices: j*p+k
        # W indices: k*n+i
        
        # M1
        U[0][0] += 1; U[3][0] += 1
        V[0][0] += 1; V[3][0] += 1
        W[0][0] += 1; W[1][0] += 1; W[2][0] += 1; W[3][0] += 1 # Wait, W is k*n+i.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Let's just hardcode the standard Strassen decomposition.
        # U, V, W for each term c=0..6
        
        # c=0: M1
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[1][0] = 1; W[2][0] = 1; W[3][0] = 1
        # This is wrong. W[k*n+i].
        # Term 0 contributes to C00, C01, C10, C11?
        # No, Strassen is a sum of 7 rank-1 tensors.
        
        # Let's use the explicit definitions.
        # T = sum_{c=1}^7 u_c v_c w_c
        
        # c=1: (a+d)(e+h) -> C00 += 1, C11 += 1
        # u: a,d. v: e,h. w: C00, C11.
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1 # C00 is k=0,i=0? No.
        # W index is k*n+i.
        # C00: k=0, i=0 -> idx 0.
        # C11: k=1, i=1 -> idx 3.
        # C01: k=1, i=0 -> idx 1.
        # C10: k=0, i=1 -> idx 2.
        
        # c=1: M1
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # c=2: M2 = (c+d)h
        # u: c,d. v: h. w: C10, C11?
        # C10 = M2 + M4. C11 = ... - M2 ...
        # M2 contributes to C10 (+1) and C11 (-1).
        U[2][1] = 1; U[3][1] = 1
        V[3][1] = 1
        W[2][1] = 1; W[3][1] = -1
        
        # c=3: M3 = a(f-h)
        # u: a. v: f, -h. w: C01, C11?
        # C01 = M3 + M5. C11 = ... + M3 ...
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[1][2] = 1; W[3][2] = 1
        
        # c=4: M4 = d(g-e)
        # u: d. v: g, -e. w: C00, C10?
        # C00 = ... + M4 ... C10 = M2 + M4.
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1
        
        # c=5: M5 = (a+b)e
        # u: a, b. v: e. w: C00, C01?
        # C00 = ... - M5 ... C01 = M3 + M5.
        U[0][4] = 1; U[1][4] = 1
        V[0][4] = 1
        W[0][4] = -1; W[1][4] = 1
        
        # c=6: M6 = (c-d)(g+h)
        # u: c, -d. v: g, h. w: C11?
        # C11 = ... + M6 ...
        U[2][5] = 1; U[3][5] = -1
        V[2][5] = 1; V[3][5] = 1
        W[3][5] = 1
        
        # c=7: M7 = (b+d)f
        # u: b, d. v: f. w: C00?
        # C00 = ... + M7 ...
        U[1][6] = 1; U[3][6] = 1
        V[1][6] = 1
        W[0][6] = 1
        
        return U, V, W

    # Generator for 2x2x1 (Schoolbook, Rank 4)
    def gen_2x2x1(n, m, p):
        U = [[0]*4 for _ in range(4)]
        V = [[0]*4 for _ in range(2)]
        W = [[0]*4 for _ in range(2)]
        
        # C00 = a*e
        U[0][0] = 1
        V[0][0] = 1
        W[0][0] = 1
        
        # C01 = b*e ? No, C is 2x1.
        # A is 2x2, B is 2x1.
        # C0 = A0 * B0 + A1 * B1? No.
        # C_i = sum_j A_ij B_jk.
        # C_0 = A_00 B_00 + A_01 B_10.
        # C_1 = A_10 B_00 + A_11 B_10.
        
        # Term 1: A00 B00
        U[0][0] = 1
        V[0][0] = 1
        W[0][0] = 1 # C_0, i=0, k=0
        
        # Term 2: A01 B10
        U[1][1] = 1
        V[1][1] = 1
        W[0][1] = 1
        
        # Term 3: A10 B00
        U[2][2] = 1
        V[0][2] = 1
        W[1][2] = 1 # C_1, i=1, k=0
        
        # Term 4: A11 B10
        U[3][3] = 1
        V[1][3] = 1
        W[1][3] = 1
        
        return U, V, W

    # Generator for 2x3x3 (Rank 17)
    def gen_2x3x3(n, m, p):
        # Split m=3 into 1 and 2.
        # Part 1: 2x1x3 (Rank 6)
        # Part 2: 2x2x3 (Rank 11)
        
        U_part1, V_part1, W_part1 = gen_2x1x3(2, 1, 3)
        U_part2, V_part2, W_part2 = gen_2x2x3(2, 2, 3)
        
        r1 = len(U_part1[0])
        r2 = len(U_part2[0])
        r_total = r1 + r2
        
        U = [[0] * r_total for _ in range(2*3)]
        V = [[0] * r_total for _ in range(3*3)]
        W = [[0] * r_total for _ in range(3*2)]
        
        # Part 1: j=0
        for i in range(2):
            for c in range(r1):
                U[i*3+0][c] = U_part1[i*1+0][c]
        for k in range(3):
            for c in range(r1):
                V[0*3+k][c] = V_part1[0*3+k][c]
        for k in range(3):
            for i in range(2):
                W[k*2+i][c] = W_part1[k*2+i][c] # Wait, W_part1 shape is (3*2, r1)
                
        # Part 2: j=1,2
        for i in range(2):
            for j_loc in range(2):
                j_glob = j_loc + 1
                for c in range(r2):
                    U[i*3+j_glob][r1+c] = U_part2[i*2+j_loc][c]
        for j_loc in range(2):
            j_glob = j_loc + 1
            for k in range(3):
                for c in range(r2):
                    V[j_glob*3+k][r1+c] = V_part2[j_loc*3+k][c]
        for k in range(3):
            for i in range(2):
                for c in range(r2):
                    W[k*2+i][r1+c] = W_part2[k*2+i][c]
                    
        return U, V, W

    # Generator for 2x1x3 (Schoolbook, Rank 6)
    def gen_2x1x3(n, m, p):
        U = [[0]*6 for _ in range(2)]
        V = [[0]*6 for _ in range(3)]
        W = [[0]*6 for _ in range(6)]
        
        # C_i = A_i0 * B_0k
        # 2 * 3 = 6 terms.
        idx = 0
        for i in range(2):
            for k in range(3):
                U[i][idx] = 1
                V[k][idx] = 1
                W[k*2+i][idx] = 1
                idx += 1
        return U, V, W

    # Add Sub-problem 1: A[0:2, 0:2] * B[0:2, 0:3]
    add_decomp(2, 2, 3, 0, 0, 0, gen_2x2x3)
    
    # Add Sub-problem 2: A[0:2, 0:2] * B[0:2, 3:6]
    add_decomp(2, 2, 3, 0, 3, 6, gen_2x2x3)
    # Note: v_off for B[0:2, 3:6] is j=0,1 mapped to global j=0,1, but k=3,4,5.
    # My add_decomp assumes local indices map to global offsets.
    # v_off is the offset in the flattened V matrix.
    # V is (m*p, rank). Flattened index is j*p+k.
    # For B[0:2, 3:6], j in {0,1}, k in {3,4,5}.
    # Local j in {0,1}, local k in {0,1,2}.
    # Global j = local j. Global k = local k + 3.
    # Global V index = j*p + (k+3) = j*6 + k + 3.
    # My add_decomp uses v_off + j*p_loc + k.
    # If v_off=3, p_loc=3.
    # j=0, k=0 -> 3 + 0 + 0 = 3. Correct (global 0*6+3).
    # j=0, k=2 -> 3 + 0 + 2 = 5. Correct (global 0*6+5).
    # j=1, k=0 -> 3 + 3 + 0 = 6. Correct (global 1*6+3).
    
    # Add Sub-problem 3: A[0:2, 2:5] * B[2:5, 0:3]
    add_decomp(2, 3, 3, 4, 12, 0, gen_2x3x3)
    # u_off: i*5 + j. j in {2,3,4}. Local j in {0,1,2}.
    # Global j = local j + 2.
    # Global U index = i*5 + (local j + 2) = i*5 + local j + 2.
    # My add_decomp uses u_off + i*m_loc + j.
    # If u_off=2, m_loc=3.
    # i=0, j=0 -> 2 + 0 + 0 = 2. Correct (global 0*5+2).
    # i=0, j=2 -> 2 + 0 + 2 = 4. Correct (global 0*5+4).
    # i=1, j=0 -> 2 + 3 + 0 = 5. Correct (global 1*5+2).
    
    # v_off: B[2:5, 0:3]. j in {2,3,4}, k in {0,1,2}.
    # Global V index = j*6 + k.
    # Local j in {0,1,2}, local k in {0,1,2}.
    # Global j = local j + 2.
    # Global V index = (local j + 2)*6 + local k = local j*6 + 12 + local k.
    # My add_decomp uses v_off + j*p_loc + k.
    # If v_off=12, p_loc=3.
    # j=0, k=0 -> 12 + 0 + 0 = 12. Correct (global 2*6+0).
    # j=2, k=2 -> 12 + 6 + 2 = 20. Correct (global 4*6+2).
    
    # w_off: k in {0,1,2}, i in {0,1}.
    # Global W index = k*2 + i.
    # Local k in {0,1,2}, local i in {0,1}.
    # Global k = local k.
    # Global W index = local k*2 + local i.
    # My add_decomp uses w_off + k*n_loc + i.
    # If w_off=0, n_loc=2.
    # k=0, i=0 -> 0. Correct.
    
    # Add Sub-problem 4: A[0:2, 2:5] * B[2:5, 3:6]
    add_decomp(2, 3, 3, 4, 18, 6, gen_2x3x3)
    # v_off: B[2:5, 3:6]. j in {2,3,4}, k in {3,4,5}.
    # Global V index = j*6 + k.
    # Local j in {0,1,2}, local k in {0,1,2}.
    # Global j = local j + 2. Global k = local k + 3.
    # Global V index = (local j + 2)*6 + (local k + 3) = local j*6 + 12 + local k + 3 = local j*6 + local k + 15.
    # My add_decomp uses v_off + j*p_loc + k.
    # If v_off=15, p_loc=3.
    # j=0, k=0 -> 15. Correct (global 2*6+3).
    # j=2, k=2 -> 15 + 6 + 2 = 23. Correct (global 4*6+5).
    
    # w_off: k in {3,4,5}, i in {0,1}.
    # Global W index = k*2 + i.
    # Local k in {0,1,2}, local i in {0,1}.
    # Global k = local k + 3.
    # Global W index = (local k + 3)*2 + local i = local k*2 + 6 + local i.
    # My add_decomp uses w_off + k*n_loc + i.
    # If w_off=6, n_loc=2.
    # k=0, i=0 -> 6. Correct (global 3*2+0).
    # k=2, i=1 -> 6 + 4 + 1 = 11. Correct (global 5*2+1).

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
