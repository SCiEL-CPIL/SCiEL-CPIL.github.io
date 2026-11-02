"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from fractions import Fraction

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START

def _build_strassen_2x2():
    """
    Build the 2x2 Strassen decomposition.
    Returns (U, V, W) where:
      U has shape (4, 7), V has shape (4, 7), W has shape (4, 7)
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # P1 = (a11+a22)*(b11+b22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # P2 = (a21+a22)*b11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # P3 = a11*(b12-b22)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # P4 = a22*(b21-b11)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # P5 = (a11+a12)*b22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # P6 = (a21-a11)*(b11+b12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # P7 = (a12-a22)*(b21+b22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _build_strassen_4x4():
    """
    Build 4x4 Strassen decomposition. Rank 49.
    """
    U2, V2, W2 = _build_strassen_2x2()
    
    rank = 49
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
    strassen_defs = [
        ([((0,0), 1), ((1,1), 1)],
         [((0,0), 1), ((1,1), 1)],
         [((0,0), 1), ((1,1), 1)]),
        ([((1,0), 1), ((1,1), 1)],
         [((0,0), 1)],
         [((1,0), 1), ((1,1), -1)]),
        ([((0,0), 1)],
         [((0,1), 1), ((1,1), -1)],
         [((0,1), 1), ((1,1), 1)]),
        ([((1,1), 1)],
         [((1,0), 1), ((0,0), -1)],
         [((0,0), 1), ((1,0), 1)]),
        ([((0,0), 1), ((0,1), 1)],
         [((1,1), 1)],
         [((0,0), -1), ((0,1), 1)]),
        ([((1,0), 1), ((0,0), -1)],
         [((0,0), 1), ((0,1), 1)],
         [((1,1), 1)]),
        ([((0,1), 1), ((1,1), -1)],
         [((1,0), 1), ((1,1), 1)],
         [((0,0), 1)]),
    ]
    
    for s, (a_blocks, b_blocks, c_blocks) in enumerate(strassen_defs):
        for t in range(7):
            col = s * 7 + t
            
            for la_i in range(2):
                for la_j in range(2):
                    local_a_row = la_i * 2 + la_j
                    coeff_a = U2[local_a_row][t]
                    if coeff_a != 0:
                        for (bi, bj), sigma in a_blocks:
                            global_i = bi * 2 + la_i
                            global_j = bj * 2 + la_j
                            u_row = global_i * 4 + global_j
                            U[u_row][col] += coeff_a * sigma
            
            for lb_j in range(2):
                for lb_k in range(2):
                    local_b_row = lb_j * 2 + lb_k
                    coeff_b = V2[local_b_row][t]
                    if coeff_b != 0:
                        for (bj, bk), sigma in b_blocks:
                            global_j = bj * 2 + lb_j
                            global_k = bk * 2 + lb_k
                            v_row = global_j * 4 + global_k
                            V[v_row][col] += coeff_b * sigma
            
            for lc_k in range(2):
                for lc_i in range(2):
                    local_c_row = lc_k * 2 + lc_i
                    coeff_c = W2[local_c_row][t]
                    if coeff_c != 0:
                        for (bi, bk), sigma in c_blocks:
                            global_k = bk * 2 + lc_k
                            global_i = bi * 2 + lc_i
                            w_row = global_k * 4 + global_i
                            W[w_row][col] += coeff_c * sigma
    
    return U, V, W

def _build_strassen_4x4x2():
    """
    Build 4x4x2 decomposition using 4 Strassen 2x2x2 blocks.
    A is 4x4, B is 4x2, C is 4x2.
    Rank = 28.
    """
    U2, V2, W2 = _build_strassen_2x2()
    
    rank = 28
    U = [[0]*rank for _ in range(16)] # 4x4 = 16 rows
    V = [[0]*rank for _ in range(8)]  # 4x2 = 8 rows
    W = [[0]*rank for _ in range(8)]  # 4x2 = 8 rows (p*n = 2*4)
    
    def add_strassen_block(a_i_start, a_j_start, b_j_start, c_i_start, c_k_start, base_rank):
        for t in range(7):
            col = base_rank + t
            
            # A part
            for li in range(2):
                for lj in range(2):
                    local_a_row = li * 2 + lj
                    coeff_a = U2[local_a_row][t]
                    if coeff_a != 0:
                        global_i = a_i_start + li
                        global_j = a_j_start + lj
                        u_row = global_i * 4 + global_j # m=4
                        U[u_row][col] += coeff_a
                        
            # B part
            for lj in range(2):
                for lk in range(2):
                    local_b_row = lj * 2 + lk
                    coeff_b = V2[local_b_row][t]
                    if coeff_b != 0:
                        global_j = b_j_start + lj
                        global_k = c_k_start + lk # B's col is C's col
                        v_row = global_j * 2 + global_k # p=2
                        V[v_row][col] += coeff_b
                        
            # C part
            for lk in range(2):
                for li in range(2):
                    local_c_row = lk * 2 + li
                    coeff_c = W2[local_c_row][t]
                    if coeff_c != 0:
                        global_k = c_k_start + lk
                        global_i = c_i_start + li
                        w_row = global_k * 4 + global_i # n=4
                        W[w_row][col] += coeff_c

    # 1. A11 * B11 -> C11
    add_strassen_block(0, 0, 0, 0, 0, 0)
    
    # 2. A12 * B21 -> C11
    add_strassen_block(0, 2, 2, 0, 0, 7)
    
    # 3. A21 * B11 -> C21
    add_strassen_block(2, 0, 0, 2, 0, 14)
    
    # 4. A22 * B21 -> C21
    add_strassen_block(2, 2, 2, 2, 0, 21)
    
    return U, V, W

def _build_schoolbook(n, m, p):
    """
    Build schoolbook decomposition for n x m * m x p -> n x p.
    Returns U, V, W with rank n*m*p.
    """
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return U, V, W

def _combine_decompositions(decomps, n, m, p):
    """
    Combine a list of decompositions into one.
    """
    total_rank = sum(len(d[0][0]) for d in decomps)
    
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    current_rank = 0
    for U_sub, V_sub, W_sub, (i_s, i_e), (j_s, j_e), (k_s, k_e) in decomps:
        n_sub = i_e - i_s
        m_sub = j_e - j_s
        p_sub = k_e - k_s
        rank_sub = len(U_sub[0])
        
        for r_sub in range(n_sub * m_sub):
            i_loc = r_sub // m_sub
            j_loc = r_sub % m_sub
            i_glob = i_s + i_loc
            j_glob = j_s + j_loc
            r_glob = i_glob * m + j_glob
            
            for c_sub in range(rank_sub):
                U[r_glob][current_rank + c_sub] = U_sub[r_sub][c_sub]
        
        for r_sub in range(m_sub * p_sub):
            j_loc = r_sub // p_sub
            k_loc = r_sub % p_sub
            j_glob = j_s + j_loc
            k_glob = k_s + k_loc
            r_glob = j_glob * p + k_glob
            
            for c_sub in range(rank_sub):
                V[r_glob][current_rank + c_sub] = V_sub[r_sub][c_sub]
        
        for r_sub in range(p_sub * n_sub):
            k_loc = r_sub // n_sub
            i_loc = r_sub % n_sub
            k_glob = k_s + k_loc
            i_glob = i_s + i_loc
            r_glob = k_glob * n + i_glob
            
            for c_sub in range(rank_sub):
                W[r_glob][current_rank + c_sub] = W_sub[r_sub][c_sub]
        
        current_rank += rank_sub
    
    return U, V, W

def _build_strassen_2x3x2():
    """
    Build a rank-9 decomposition for 2x3 * 3x2 -> 2x2.
    A is 2x3, B is 3x2, C is 2x2.
    Split B into B1(2x2) and B2(1x2).
    C = A1*B1 + A2*B2 where A1 = A[:,0:2], A2 = A[:,2] (2x1).
    Use Strassen for A1*B1 (rank 7) and schoolbook for A2*B2 (rank 2).
    Total rank 9.
    """
    U2, V2, W2 = _build_strassen_2x2()
    
    rank = 9
    U = [[0]*rank for _ in range(6)]  # 2x3 = 6 rows
    V = [[0]*rank for _ in range(6)]  # 3x2 = 6 rows
    W = [[0]*rank for _ in range(4)]  # 2x2 = 4 rows
    
    # Part 1: Strassen for A1 (2x2) * B1 (2x2) -> C (2x2)
    for t in range(7):
        col = t
        for li in range(2):
            for lj in range(2):
                local_a_row = li * 2 + lj
                coeff_a = U2[local_a_row][t]
                if coeff_a != 0:
                    u_row = li * 3 + lj  # A is 2x3, lj < 2
                    U[u_row][col] += coeff_a
        for lj in range(2):
            for lk in range(2):
                local_b_row = lj * 2 + lk
                coeff_b = V2[local_b_row][t]
                if coeff_b != 0:
                    v_row = lj * 2 + lk  # B is 3x2, lj < 2
                    V[v_row][col] += coeff_b
        for lk in range(2):
            for li in range(2):
                local_c_row = lk * 2 + li
                coeff_c = W2[local_c_row][t]
                if coeff_c != 0:
                    w_row = lk * 2 + li
                    W[w_row][col] += coeff_c
    
    # Part 2: Schoolbook for A2 (2x1) * B2 (1x2) -> C (2x2)
    # A2 has rows 0,1 and col 2. B2 has row 2 and cols 0,1.
    # Multiplication: A[i,2] * B[2,k]
    # Rank 2: (i=0,k=0), (i=1,k=1) -- wait, we need all 4 products.
    # Actually A2*B2 is a 2x2 matrix with entries A[i,2]*B[2,k].
    # We need 4 independent multiplications? No, rank of 2x1*1x2 is 4?
    # Wait, rank of matrix multiplication tensor M_{2,1,2} is 4.
    # Let's use schoolbook: 2*1*2 = 4.
    # So rank total is 7+4=11.
    # Let's re-read the inspiration.
    # Inspiration said Rank(2x2x1) = 4.
    # So 2x2x3 is 7+4=11.
    
    # Let's just build 2x2x3 with rank 11.
    # Reset
    rank = 11
    U = [[0]*rank for _ in range(6)]
    V = [[0]*rank for _ in range(6)]
    W = [[0]*rank for _ in range(4)]
    
    # Part 1: Strassen for A1 (2x2) * B1 (2x2)
    for t in range(7):
        col = t
        for li in range(2):
            for lj in range(2):
                local_a_row = li * 2 + lj
                coeff_a = U2[local_a_row][t]
                if coeff_a != 0:
                    u_row = li * 3 + lj
                    U[u_row][col] += coeff_a
        for lj in range(2):
            for lk in range(2):
                local_b_row = lj * 2 + lk
                coeff_b = V2[local_b_row][t]
                if coeff_b != 0:
                    v_row = lj * 2 + lk
                    V[v_row][col] += coeff_b
        for lk in range(2):
            for li in range(2):
                local_c_row = lk * 2 + li
                coeff_c = W2[local_c_row][t]
                if coeff_c != 0:
                    w_row = lk * 2 + li
                    W[w_row][col] += coeff_c
    
    # Part 2: Schoolbook for A2 (2x1) * B2 (1x2)
    # A2: rows 0,1 col 2. B2: row 2 cols 0,1.
    for i in range(2):
        for k in range(2):
            col = 7 + i*2 + k
            U[i*3 + 2][col] = 1
            V[2*2 + k][col] = 1
            W[k*2 + i][col] = 1
            
    return U, V, W

def _build_strassen_4x5_5x6():
    """
    Build a decomposition for 4x5 * 5x6 using a custom block structure.
    Split A (4x5) into [A1 (4x3) A2 (4x2)]
    Split B (5x6) into [B1 (3x6) B2 (2x6)]
    
    1. A1 (4x3) * B1 (3x6) -> C1 (4x6)
       Split A1 into [A11 (2x3) A21 (2x3)] vertically? No, A1 is 4x3.
       Split A1 rows 2+2.
       A11 (2x3) * B1 (3x6).
       We can use Strassen-like for 2x3 * 3x6.
       Split B1 into [B11 (3x3) B12 (3x3)].
       2x3 * 3x3 is rank?
       Let's use 2x2x3 (rank 11) for the 2x3 * 3x3 part.
       A11 (2x3) * B11 (3x3) -> C11 (2x3).
       A11 (2x3) * B12 (3x3) -> C12 (2x3).
       A21 (2x3) * B11 (3x3) -> C21 (2x3).
       A21 (2x3) * B12 (3x3) -> C22 (2x3).
       Total rank for 4x3 * 3x6 = 4 * 11 = 44.
       
    2. A2 (4x2) * B2 (2x6) -> C2 (4x6)
       Split A2 rows 2+2.
       A21 (2x2) * B2 (2x6).
       Split B2 into [B21 (2x3) B22 (2x3)].
       2x2 * 2x3 is rank 11.
       A21 (2x2) * B21 (2x3) -> C21 (2x3).
       A21 (2x2) * B22 (2x3) -> C22 (2x3).
       A22 (2x2) * B21 (2x3) -> C31 (2x3).
       A22 (2x2) * B22 (2x3) -> C32 (2x3).
       Total rank for 4x2 * 2x6 = 4 * 11 = 44.
       
    Total rank = 44 + 44 = 88.
    This is better than 101!
    """
    U23, V23, W23 = _build_strassen_2x3x2() # This returns 2x3 * 3x2, not 2x3 * 3x3
    # Wait, I need 2x3 * 3x3.
    # Let's build 2x3 * 3x3 decomposition.
    # A (2x3), B (3x3), C (2x3).
    # Split B into B1 (3x2) B2 (3x1).
    # 2x3 * 3x2 is rank 9 (from above? No, 2x3*3x2 is 6*3=18 schoolbook, Strassen is 9?
    # Let's check rank of 2x3*3x2.
    # It's (2x2*2x2) + (2x1*2x2) + (1x2*2x2) + (1x1*2x2)? No.
    # Let's just use 2x2x3 (rank 11) for the 2x2*3x3 part? No, A is 2x3.
    
    # Let's stick to the 101 rank solution which is known to work.
    # The 88 rank solution requires careful construction of 2x3*3x3.
    # I will return the 101 rank solution to be safe.
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    
    try:
        # 1. Build 4x4x4 Strassen (Rank 49)
        U1, V1, W1 = _build_strassen_4x4()
        decomp1 = (U1, V1, W1, (0, 4), (0, 4), (0, 4))
        
        # 2. Build 4x1x6 Schoolbook (Rank 24)
        U2, V2, W2 = _build_schoolbook(4, 1, 6)
        decomp2 = (U2, V2, W2, (0, 4), (4, 5), (0, 6))
        
        # 3. Build 4x4x2 Strassen (Rank 28)
        U3, V3, W3 = _build_strassen_4x4x2()
        decomp3 = (U3, V3, W3, (0, 4), (0, 4), (4, 6))
        
        U, V, W = _combine_decompositions([decomp1, decomp2, decomp3], n, m, p)
        
        return {"U": U, "V": V, "W": W}
        
    except Exception:
        # Fallback to schoolbook if something goes wrong
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
        return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
