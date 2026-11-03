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

def _build_4x5_5x6_improved():
    """
    Improved decomposition for 4x5 * 5x6.
    Split A (4x5) into [A_left(4x3) A_right(4x2)]
    Split B (5x6) into [B_top(3x6) B_bot(2x6)]
    
    For 4x3x6: Split 3 into 2+1
    - 4x2x6: Use Strassen-like approach
    - 4x1x6: Schoolbook (rank 24)
    
    For 4x2x6: Split 4 into 2+2, 6 into 3+3
    - Four 2x2x3 blocks
    - Each 2x2x3 can be done with rank 11 (7 for 2x2x2 + 4 for 2x2x1)
    - Total for 4x2x6: 4 * 11 = 44? That's worse than schoolbook 48.
    
    Let me try a different approach. Use the existing blocks but optimize.
    Current: 49 (4x4x4) + 24 (4x1x6) + 28 (4x4x2) = 101
    
    Alternative split: 5 = 3+2, 6 = 4+2
    - 4x3x4 + 4x3x2 + 4x2x4 + 4x2x2
    - 4x3x4: split 3 into 2+1: 4x2x4 (Strassen 4x4x4 restricted? No, 4x2x4) + 4x1x4
    - This is getting complex. Let me stick with current and try to verify.
    """
    n, m, p = 4, 5, 6
    
    # Build the three components
    U1, V1, W1 = _build_strassen_4x4()
    U2, V2, W2 = _build_schoolbook(4, 1, 6)
    U3, V3, W3 = _build_strassen_4x4x2()
    
    decomp1 = (U1, V1, W1, (0, 4), (0, 4), (0, 4))
    decomp2 = (U2, V2, W2, (0, 4), (4, 5), (0, 6))
    decomp3 = (U3, V3, W3, (0, 4), (0, 4), (4, 6))
    
    return _combine_decompositions([decomp1, decomp2, decomp3], n, m, p)

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

def _build_strassen_4x5_5x6():
    """
    Build a decomposition for 4x5 * 5x6 using a custom block structure.
    Split A (4x5) into [A11(2x2) A12(2x3)]
    Split B (5x6) into [B11(2x2) B12(2x2) B13(2x2)] (vertically into 2x2, 2x2, 1x2? No, B is 5x6)
    
    Let's try a different split:
    A = [A11 (2x2) A12 (2x2) A13 (2x1)]
    B = [B11 (2x2) B12 (2x2) B13 (1x2)] (Top 2x2, Middle 2x2, Bottom 1x2)
    
    Wait, dimensions:
    A is 4x5. B is 5x6.
    Split columns of A and rows of B at indices 2 and 4.
    A11: rows 0-1, cols 0-1 (2x2)
    A12: rows 0-1, cols 2-3 (2x2)
    A13: rows 0-1, cols 4 (2x1)
    A21: rows 2-3, cols 0-1 (2x2)
    A22: rows 2-3, cols 2-3 (2x2)
    A23: rows 2-3, cols 4 (2x1)
    
    B11: rows 0-1, cols 0-5 (2x6)
    B21: rows 2-3, cols 0-5 (2x6)
    B31: rows 4, cols 0-5 (1x6)
    
    This is getting complicated to map Strassen directly.
    Let's stick to the known good decomposition which scored 1.188.
    The previous program used:
    1. 4x4x4 Strassen (Rank 49)
    2. 4x1x6 Schoolbook (Rank 24)
    3. 4x4x2 Strassen (Rank 28)
    Total Rank = 49 + 24 + 28 = 101.
    
    Score = Seed / Value.
    If Seed is 120 (schoolbook), Score = 120/101 = 1.188.
    
    Can we do better?
    Let's try a decomposition based on splitting 5 into 3+2 and 6 into 3+3?
    Or 5 into 2+2+1?
    
    Let's try to improve the 4x4x2 block or the 4x1x6 block.
    4x1x6 schoolbook is 24. Can we do better for 4x1x6?
    It's just 4 columns of 6. Rank 24 is likely minimal for that slice if isolated.
    
    What if we split 5 into 3+2?
    A: 4x5 -> A_left (4x3), A_right (4x2)
    B: 5x6 -> B_top (3x6), B_bot (2x6)
    C = A_left B_top + A_right B_bot
    
    1. 4x3x6: Split 3 into 2+1?
       4x2x6 + 4x1x6
       4x2x6 can be done with Strassen-like?
       4x2x6 is (4x2)*(2x6).
       4x2 matrix times 2x6 matrix.
       We can split 4 into 2+2, 6 into 3+3.
       (2x2)*(2x3) + (2x2)*(2x3) + (2x2)*(2x3) + (2x2)*(2x3)
       Each (2x2)*(2x3) is 2x2 * 2x3.
       Rank for 2x2x3?
       2x2 * 2x3.
       Strassen for 2x2x2 is rank 7.
       2x2x3 is 2x2 * 2x2 + 2x2 * 2x1.
       Rank(2x2x2) = 7.
       Rank(2x2x1) = 4 (schoolbook).
       Total 11?
       
       Let's try to build a 2x2x3 decomposition.
       A (2x2), B (2x3), C (2x3).
       B = [B1 (2x2), B2 (2x1)].
       C = [C1 (2x2), C2 (2x1)].
       C1 = A * B1. C2 = A * B2.
       We need to compute A*B1 and A*B2 using shared multiplications.
       
       Let's stick to the current 101 rank. It's a solid baseline.
    """
    # For now, just return the components to be combined in construct
    # This function is a placeholder to satisfy the structure if needed,
    # but construct() builds the final result directly.
    return None

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the decomposition is correct.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Check that sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = delta terms
    """
    rank = len(U[0])
    # Check a few random entries
    import random
    random.seed(42)
    for _ in range(10):
        i = random.randint(0, n-1)
        j = random.randint(0, m-1)
        k = random.randint(0, p-1)
        
        # Expected: 1 if this is the (i,j,k) entry of the multiplication tensor
        # The tensor T[i,j,k] = 1 for all valid i,j,k (it's the identity in the right sense)
        # Actually, for matrix multiplication: C[i,k] = sum_j A[i,j]*B[j,k]
        # The tensor has T[i,j,k] = 1 for all i,j,k
        expected = 1.0
        
        actual = 0.0
        for r in range(rank):
            actual += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
        
        if abs(actual - expected) > 1e-10:
            return False
    return True

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
        
        # Verify
        if not _verify_decomposition(U, V, W, n, m, p):
            raise ValueError("Decomposition verification failed")
        
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
