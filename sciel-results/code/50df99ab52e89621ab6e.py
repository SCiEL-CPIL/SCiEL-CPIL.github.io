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

def _build_recursive(n, m, p):
    """
    Build a decomposition for n x m * m x p using recursive Strassen.
    Returns (U, V, W) with rank as low as possible via Strassen where dimensions allow.
    """
    # Base cases
    if n == 1 and m == 1 and p == 1:
        return [[1]], [[1]], [[1]]
    
    # If any dimension is 1, use schoolbook for that sub-problem
    if n == 1:
        return _build_schoolbook(1, m, p)
    if p == 1:
        return _build_schoolbook(n, m, 1)
    if m == 1:
        return _build_schoolbook(n, 1, p)
    
    # Try to split into halves and use Strassen
    # Strassen works best when we can split into equal halves
    n_half = n // 2
    m_half = m // 2
    p_half = p // 2
    
    # Check if we can use Strassen (need even dimensions for clean split)
    if n % 2 == 0 and m % 2 == 0 and p % 2 == 0:
        # Use 2x2 Strassen: split into 4 sub-problems of size (n/2, m/2, p/2)
        # But we need to be careful - Strassen gives rank 7*(n/2)*(m/2)*(p/2) vs schoolbook n*m*p
        # Strassen is better when 7*(n/2)*(m/2)*(p/2) < n*m*p, i.e., 7/8 < 1, always true
        
        # Build the 7 Strassen sub-problems
        U2, V2, W2 = _build_strassen_2x2()
        rank_sub = (n // 2) * (m // 2) * (p // 2)
        
        # We need to build each of the 7 sub-problems recursively
        # For efficiency, let's build the full Strassen decomposition
        
        total_rank = 7 * rank_sub
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        # Strassen definitions: (a_blocks, b_blocks, c_blocks)
        # Each block is a list of (offset, sign) pairs
        strassen_defs = [
            # P1 = (A11+A22)(B11+B22)
            ([((0,0), 1), ((1,1), 1)],
             [((0,0), 1), ((1,1), 1)],
             [((0,0), 1), ((1,1), 1)]),
            # P2 = (A21+A22)B11
            ([((1,0), 1), ((1,1), 1)],
             [((0,0), 1)],
             [((1,0), 1), ((1,1), -1)]),
            # P3 = A11(B12-B22)
            ([((0,0), 1)],
             [((0,1), 1), ((1,1), -1)],
             [((0,1), 1), ((1,1), 1)]),
            # P4 = A22(B21-B11)
            ([((1,1), 1)],
             [((1,0), 1), ((0,0), -1)],
             [((0,0), 1), ((1,0), 1)]),
            # P5 = (A11+A12)B22
            ([((0,0), 1), ((0,1), 1)],
             [((1,1), 1)],
             [((0,0), -1), ((0,1), 1)]),
            # P6 = (A21-A11)(B11+B12)
            ([((1,0), 1), ((0,0), -1)],
             [((0,0), 1), ((0,1), 1)],
             [((1,1), 1)]),
            # P7 = (A12-A22)(B21+B22)
            ([((0,1), 1), ((1,1), -1)],
             [((1,0), 1), ((1,1), 1)],
             [((0,0), 1)]),
        ]
        
        for s, (a_blocks, b_blocks, c_blocks) in enumerate(strassen_defs):
            for t in range(7):
                col = s * 7 * rank_sub + t * rank_sub
                
                # Get the sub-decomposition for this Strassen term
                # We need to figure out which sub-problem this corresponds to
                # For now, let's use schoolbook for sub-problems and see if we can improve
                
                # Actually, let's just use schoolbook for sub-problems for simplicity
                # and check if the total rank is better than pure schoolbook
                
                # This approach is getting complex. Let me use a simpler strategy:
                # Just use schoolbook for all sub-problems in the Strassen framework
                pass
        
        # Fallback: just use schoolbook if Strassen is complex
        return _build_schoolbook(n, m, p)
    
    # If dimensions are odd, try splitting into even + 1
    # For now, just use schoolbook
    return _build_schoolbook(n, m, p)

def _build_strassen_4x5x3():
    """
    Build decomposition for 4x5 * 5x3 using block structure.
    Split B and C into 3x3 + 3x0 or use direct methods.
    Actually, let's split as: A is 4x5, B is 5x3, C is 4x3
    We can split A into 2x3 + 2x2, B into 3x3 + 2x3, C into 2x3 + 2x3
    Or simpler: use schoolbook for 4x5x3 = 60, or find better.
    """
    # For 4x5x3, schoolbook gives rank 60
    # Let's try: split into (2x5x3) + (2x5x3) = 2 * rank(2x5x3)
    # 2x5x3 schoolbook = 30, so total = 60. No improvement.
    # Try: (4x3x3) + (4x2x3) = 4*3*3 + 4*2*3 = 36 + 24 = 60. Same.
    # For 4x3x3: can we do better than 36? 4x3x3 schoolbook = 36.
    # Strassen doesn't apply cleanly since 3 is odd.
    # Let's just use schoolbook for now.
    return _build_schoolbook(4, 5, 3)

def _build_strassen_4x3x6():
    """
    Build decomposition for 4x3 * 3x6.
    Schoolbook: 4*3*6 = 72
    Try: split C into 6 columns as 3+3: (4x3x3) + (4x3x3) = 2 * 36 = 72. Same.
    Try: split A into 2+2: (2x3x6) + (2x3x6) = 2 * 36 = 72. Same.
    For 2x3x6: schoolbook = 36. Can we do better?
    2x3x6: split B into 3+3: (2x3x3) + (2x3x3) = 2*18 = 36. Same.
    So schoolbook seems optimal for these small cases.
    """
    return _build_schoolbook(4, 3, 6)

def _build_strassen_4x2x6():
    """
    Build decomposition for 4x2 * 2x6.
    Schoolbook: 4*2*6 = 48
    Try: split into (4x2x3) + (4x2x3) = 2 * 24 = 48. Same.
    For 4x2x3: schoolbook = 24. 
    Can we do better? 4x2x3: split A into 2+2: (2x2x3) + (2x2x3) = 2*6 = 12. 
    Wait, 2x2x3 schoolbook = 12. So 2*12 = 24. Same.
    But 2x2x3: can we do better? Strassen for 2x2x2 gives rank 7 for 2x2*2x2->2x2.
    For 2x2x3: split C into 2+1: (2x2x2) + (2x2x1) = 7 + 4 = 11!
    So 4x2x3 = (2x2x3) + (2x2x3) = 2 * 11 = 22.
    And 4x2x6 = (4x2x3) + (4x2x3) = 2 * 22 = 44.
    """
    # Build 2x2x3 decomposition: (2x2x2 Strassen rank 7) + (2x2x1 schoolbook rank 4) = 11
    U222, V222, W222 = _build_strassen_2x2()  # rank 7
    U221, V221, W221 = _build_schoolbook(2, 2, 1)  # rank 4
    
    # Combine for 2x2x3
    decomp_222 = (U222, V222, W222, (0, 2), (0, 2), (0, 2))
    decomp_221 = (U221, V221, W221, (0, 2), (0, 2), (2, 3))
    U223, V223, W223 = _combine_decompositions([decomp_222, decomp_221], 2, 2, 3)
    
    # Now build 4x2x3: split A into 2+2
    decomp_top = (U223, V223, W223, (0, 2), (0, 2), (0, 3))
    decomp_bot = (U223, V223, W223, (2, 4), (0, 2), (0, 3))
    U423, V423, W423 = _combine_decompositions([decomp_top, decomp_bot], 4, 2, 3)
    
    # Now build 4x2x6: split C into 3+3
    decomp_left = (U423, V423, W423, (0, 4), (0, 2), (0, 3))
    decomp_right = (U423, V423, W423, (0, 4), (0, 2), (3, 6))
    U426, V426, W426 = _combine_decompositions([decomp_left, decomp_right], 4, 2, 6)
    
    return U426, V426, W426

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    
    try:
        # Option 1: Current approach - 4x4x4 + 4x1x6 + 4x4x2 = 49 + 24 + 28 = 101
        U1, V1, W1 = _build_strassen_4x4()
        decomp1 = (U1, V1, W1, (0, 4), (0, 4), (0, 4))
        
        U2, V2, W2 = _build_schoolbook(4, 1, 6)
        decomp2 = (U2, V2, W2, (0, 4), (4, 5), (0, 6))
        
        U3, V3, W3 = _build_strassen_4x4x2()
        decomp3 = (U3, V3, W3, (0, 4), (0, 4), (4, 6))
        
        U_opt1, V_opt1, W_opt1 = _combine_decompositions([decomp1, decomp2, decomp3], n, m, p)
        rank1 = len(U_opt1[0])
        
        # Option 2: 4x3x6 + 4x2x6 = 72 + 44 = 116 (worse)
        # Option 3: 4x5x3 + 4x5x3 = 60 + 60 = 120 (worse)
        
        # Option 4: Try 4x4x3 + 4x1x6 + 4x4x3
        # 4x4x3: schoolbook = 48, or can we do better?
        # 4x4x3: split C into 2+1: (4x4x2) + (4x4x1) = 28 + 16 = 44
        U442, V442, W442 = _build_strassen_4x4x2()  # rank 28
        U441, V441, W441 = _build_schoolbook(4, 4, 1)  # rank 16
        decomp_442 = (U442, V442, W442, (0, 4), (0, 4), (0, 2))
        decomp_441 = (U441, V441, W441, (0, 4), (0, 4), (2, 3))
        U443, V443, W443 = _combine_decompositions([decomp_442, decomp_441], 4, 4, 3)
        # rank 44
        
        # Now 4x4x6 = (4x4x3) + (4x4x3) = 44 + 44 = 88
        decomp_443a = (U443, V443, W443, (0, 4), (0, 4), (0, 3))
        decomp_443b = (U443, V443, W443, (0, 4), (0, 4), (3, 6))
        U446, V446, W446 = _combine_decompositions([decomp_443a, decomp_443b], 4, 4, 6)
        # rank 88
        
        # Total: 4x4x6 + 4x1x6 = 88 + 24 = 112 (worse than 101)
        
        # Option 5: Try 4x5x4 + 4x5x2
        # 4x5x4: split A into 4x4 + 4x1: (4x4x4) + (4x1x4) = 49 + 16 = 65
        # 4x5x2: split A into 4x4 + 4x1: (4x4x2) + (4x1x2) = 28 + 8 = 36
        # Total: 65 + 36 = 101 (same as current)
        
        # Option 6: Try 4x5x5 + 4x5x1
        # 4x5x5: split A into 4x4 + 4x1: (4x4x5) + (4x1x5)
        # 4x4x5: split C into 4+1: (4x4x4) + (4x4x1) = 49 + 16 = 65
        # 4x1x5: schoolbook = 20
        # So 4x5x5 = 65 + 20 = 85
        # 4x5x1: schoolbook = 20
        # Total: 85 + 20 = 105 (worse)
        
        # Option 7: Try 4x3x5 + 4x2x5 + 4x0x5 (not useful)
        # Or 4x5x5 + 4x5x1 = 105 as above
        
        # Option 8: Try to improve 4x4x2 from 28 to lower?
        # 4x4x2: currently 28. Can we do better?
        # 4x4x2: split A into 2+2: (2x4x2) + (2x4x2) = 2 * rank(2x4x2)
        # 2x4x2: split B into 2+2: (2x2x2) + (2x2x2) = 2 * 7 = 14
        # So 4x4x2 = 2 * 14 = 28. Same as current.
        
        # It seems 101 is hard to beat with these simple splits.
        # Let's stick with Option 1.
        
        return {"U": U_opt1, "V": V_opt1, "W": W_opt1}
        
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
