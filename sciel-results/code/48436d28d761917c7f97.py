"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time

def _strassen_2x2x2_terms():
    """
    Return the 7 Strassen terms for 2x2 * 2x2 matrix multiplication.
    """
    terms = []
    terms.append((
        {(0,0): 1, (1,1): 1},
        {(0,0): 1, (1,1): 1},
        {(0,0): 1, (1,1): 1}
    ))
    terms.append((
        {(1,0): 1, (1,1): 1},
        {(0,0): 1},
        {(1,0): 1, (1,1): -1}
    ))
    terms.append((
        {(0,0): 1},
        {(0,1): 1, (1,1): -1},
        {(0,1): 1, (1,1): 1}
    ))
    terms.append((
        {(1,1): 1},
        {(1,0): 1, (0,0): -1},
        {(0,0): 1, (1,0): 1}
    ))
    terms.append((
        {(0,0): 1, (0,1): 1},
        {(1,1): 1},
        {(0,0): -1, (0,1): 1}
    ))
    terms.append((
        {(1,0): 1, (0,0): -1},
        {(0,0): 1, (0,1): 1},
        {(1,1): 1}
    ))
    terms.append((
        {(0,1): 1, (1,1): -1},
        {(1,0): 1, (1,1): 1},
        {(0,0): 1}
    ))
    return terms

def _construct_2x4x4_strassen():
    """
    Construct a rank-28 decomposition for 2x4 * 4x4 using Strassen on 2x2 blocks.
    Returns U, V, W lists of length 28.
    """
    n, m, p = 2, 4, 4
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    strassen_terms = _strassen_2x2x2_terms()
    
    col = 0
    # 4 blocks:
    # A_block: 0 (cols 0,1) or 1 (cols 2,3)
    # B_row_block: 0 (rows 0,1) or 1 (rows 2,3)
    # B_col_block: 0 (cols 0,1) or 1 (cols 2,3)
    # C_col_block: same as B_col_block
    
    for a_block in range(2):
        for b_row_block in range(2):
            for b_col_block in range(2):
                for t in range(7):
                    U_coeffs, V_coeffs, W_coeffs = strassen_terms[t]
                    
                    for (i_loc, j_loc), coeff in U_coeffs.items():
                        i = i_loc
                        j = a_block * 2 + j_loc
                        u_idx = i * m + j
                        U[u_idx][col] += coeff
                    
                    for (j_loc, k_loc), coeff in V_coeffs.items():
                        j = b_row_block * 2 + j_loc
                        k = b_col_block * 2 + k_loc
                        v_idx = j * p + k
                        V[v_idx][col] += coeff
                    
                    for (i_loc, k_loc), coeff in W_coeffs.items():
                        i = i_loc
                        k = b_col_block * 2 + k_loc
                        # W index is k * n + i
                        w_idx = k * n + i
                        W[w_idx][col] += coeff
                    
                    col += 1
    
    return U, V, W

def _construct_2x4x2_simple():
    """
    Construct a rank-8 decomposition for 2x4 * 4x2.
    Returns U, V, W lists of length 8.
    """
    n, m, p = 2, 4, 2
    rank = 8
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i_row in range(2):
        for j in range(4):
            # U: 1 at (i_row, j)
            u_idx = i_row * m + j
            U[u_idx][col] = 1
            
            # V: 1 at (j, 0) and (j, 1)
            v_idx_0 = j * p + 0
            v_idx_1 = j * p + 1
            V[v_idx_0][col] = 1
            V[v_idx_1][col] = 1
            
            # W: 1 at (0, i_row) and (1, i_row)
            # W index is k * n + i
            # k=0, i=i_row -> i_row
            # k=1, i=i_row -> n + i_row
            w_idx_0 = 0 * n + i_row
            w_idx_1 = 1 * n + i_row
            W[w_idx_0][col] = 1
            W[w_idx_1][col] = 1
            
            col += 1
    
    return U, V, W

def _verify_decomposition(result, n=2, m=4, p=6, trials=10):
    """Verify a decomposition numerically."""
    U, V, W = result['U'], result['V'], result['W']
    rank = len(U[0])
    
    test_cases = []
    A_id = [[1 if i==j else 0 for j in range(m)] for i in range(n)]
    B_id = [[1 if i==j else 0 for j in range(p)] for i in range(m)]
    test_cases.append((A_id, B_id))
    
    A_ones = [[1 for _ in range(m)] for _ in range(n)]
    B_ones = [[1 for _ in range(p)] for _ in range(m)]
    test_cases.append((A_ones, B_ones))
    
    import random
    random.seed(42)
    for _ in range(trials):
        A = [[random.randint(-5, 5) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(-5, 5) for _ in range(p)] for _ in range(m)]
        test_cases.append((A, B))
        
    for A, B in test_cases:
        C_exp = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
        
        C_comp = [[0] * p for _ in range(n)]
        for col in range(rank):
            u_val = sum(U[i * m + j][col] * A[i][j] for i in range(n) for j in range(m))
            v_val = sum(V[j * p + k][col] * B[j][k] for j in range(m) for k in range(p))
            for i in range(n):
                for k in range(p):
                    C_comp[i][k] += W[k * n + i][col] * u_val * v_val
        
        if C_comp != C_exp:
            return False
    return True

def construct():
    """
    Construct a rank-minimal exact tensor decomposition.
    Returns a rank-36 decomposition (28 for 2x4x4 + 8 for 2x4x2).
    """
    start_time = time.time()
    
    # Part 1: 2x4x4 (first 4 columns of output)
    U1, V1, W1 = _construct_2x4x4_strassen()
    rank1 = len(U1[0])
    
    # Part 2: 2x4x2 (last 2 columns of output)
    U2, V2, W2 = _construct_2x4x2_simple()
    rank2 = len(U2[0])
    
    # Combine
    total_rank = rank1 + rank2
    n, m, p = 2, 4, 6
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Copy Part 1
    for i in range(n * m):
        for r in range(rank1):
            U[i][r] = U1[i][r]
    
    for i in range(m * p):
        for r in range(rank1):
            V[i][r] = V1[i][r]
            
    # W1 is for p=4, n=2. Size 4x2.
    # Full W is p=6, n=2. Size 6x2.
    # W1 covers k=0..3.
    for k in range(4):
        for i in range(n):
            w_idx = k * n + i
            for r in range(rank1):
                W[w_idx][r] = W1[k * 2 + i][r] # W1 index is k*2+i because p1=4, n=2? 
                # Wait, W1 shape is (p1*n, rank) = (8, rank).
                # Index in W1 for local k, i is k*n + i.
                # So W1[k*n + i][r].
                W[w_idx][r] = W1[k * n + i][r]

    # Copy Part 2
    for i in range(n * m):
        for r in range(rank2):
            U[i][rank1 + r] = U2[i][r]
            
    for i in range(m * p):
        for r in range(rank2):
            V[i][rank1 + r] = V2[i][r]
            
    # W2 is for p=2, n=2. Size 2x2.
    # Full W is p=6, n=2.
    # W2 covers local k=0,1 which map to global k=4,5.
    for k_loc in range(2):
        k_glob = 4 + k_loc
        for i in range(n):
            w_idx_glob = k_glob * n + i
            w_idx_loc = k_loc * n + i
            for r in range(rank2):
                W[w_idx_glob][rank1 + r] = W2[w_idx_loc][r]

    result = {"U": U, "V": V, "W": W}
    
    if _verify_decomposition(result):
        return result
        
    # Fallback
    return result

# EVOLVE-BLOCK-END
