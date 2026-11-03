"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def _schoolbook_factors(n, m, p):
    """Return (U, V, W) as lists of lists for schoolbook multiplication."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _strassen_222():
    """Return (U, V, W) for 2x2x2 Strassen multiplication."""
    # Standard Strassen decomposition for 2x2 matrices.
    # Indices:
    # U rows: i*m + j -> 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # V rows: j*p + k -> 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # W rows: k*n + i -> 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A00 + A11)(B00 + B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1  # C00
    W[3][0] = 1  # C11
    
    # M2 = (A10 + A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1  # C10
    W[3][1] = -1 # C11
    
    # M3 = A00(B01 - B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1  # C01
    W[3][2] = 1  # C11
    
    # M4 = A11(B10 - B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1  # C00
    W[2][3] = 1  # C10
    
    # M5 = (A00 + A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1 # C00
    W[1][4] = 1  # C01
    
    # M6 = (A10 - A00)(B00 + B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1  # C11
    
    # M7 = (A01 - A11)(B10 + B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1  # C00
    
    return U, V, W

def _embed_into_global(global_U, global_V, global_W, 
                       local_U, local_V, local_W,
                       i_off, j_off, k_off,
                       n, m, p):
    """
    Embed local factors into global factors.
    
    local_U: (n_loc*m_loc, rank)
    local_V: (m_loc*p_loc, rank)
    local_W: (p_loc*n_loc, rank)
    
    Global dimensions: n, m, p.
    Offsets: i_off, j_off, k_off.
    """
    n_loc = len(local_U) // m_loc if m_loc else 0 # This logic is flawed if we don't pass n_loc, m_loc, p_loc
    # Better to pass n_loc, m_loc, p_loc explicitly or infer from shapes if unique
    # Since shapes are (n*m, R), (m*p, R), (p*n, R), we can infer if we know one dimension.
    # We will pass n_loc, m_loc, p_loc.
    
    # Actually, let's just iterate based on the provided local shapes and offsets.
    # We need to know n_loc, m_loc, p_loc to map indices.
    # Let's modify the call to include these.
    pass

def _embed(global_U, global_V, global_W, 
           local_U, local_V, local_W,
           i_off, j_off, k_off,
           n_loc, m_loc, p_loc,
           n, m, p):
    """
    Embed local factors into global factors.
    """
    rank_loc = len(local_U[0]) if local_U else 0
    
    # Pre-compute global row indices for local rows
    # Local U row (i_l, j_l) -> Global U row (i_off + i_l, j_off + j_l)
    # Global U index: (i_off + i_l) * m + (j_off + j_l)
    
    # Local V row (j_l, k_l) -> Global V row (j_off + j_l, k_off + k_l)
    # Global V index: (j_off + j_l) * p + (k_off + k_l)
    
    # Local W row (k_l, i_l) -> Global W row (k_off + k_l, i_off + i_l)
    # Global W index: (k_off + k_l) * n + (i_off + i_l)
    
    for i_l in range(n_loc):
        for j_l in range(m_loc):
            u_row_local = i_l * m_loc + j_l
            u_row_global = (i_off + i_l) * m + (j_off + j_l)
            for r in range(rank_loc):
                val = local_U[u_row_local][r]
                if val != 0:
                    global_U[u_row_global][r] += val

    for j_l in range(m_loc):
        for k_l in range(p_loc):
            v_row_local = j_l * p_loc + k_l
            v_row_global = (j_off + j_l) * p + (k_off + k_l)
            for r in range(rank_loc):
                val = local_V[v_row_local][r]
                if val != 0:
                    global_V[v_row_global][r] += val

    for k_l in range(p_loc):
        for i_l in range(n_loc):
            w_row_local = k_l * n_loc + i_l
            w_row_global = (k_off + k_l) * n + (i_off + i_l)
            for r in range(rank_loc):
                val = local_W[w_row_local][r]
                if val != 0:
                    global_W[w_row_global][r] += val

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    # Initialize global factors with 0. We don't know the final rank yet, 
    # so we can use a list of dicts or just accumulate and trim.
    # However, the interface requires lists of lists.
    # We can estimate max rank (schoolbook 75) and trim later, 
    # or just build it dynamically.
    # Given the small size, we can build a list of columns.
    
    # Let's use a dictionary for columns to handle dynamic rank, then convert.
    # Or simpler: since we know the blocks, we can sum their ranks.
    # Block 1: 3x1x5 -> 15
    # Block 2a: 3x4x1 -> 12
    # Block 2b-i: 1x4x4 -> 16
    # Block 2b-ii: 4x2x2 Strassen blocks -> 4 * 7 = 28
    # Total rank = 71.
    
    total_rank = 15 + 12 + 16 + 28
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    
    # Helper to add local factors to global ones at a specific column offset
    def add_local(local_U, local_V, local_W, i_off, j_off, k_off, n_loc, m_loc, p_loc):
        nonlocal col_offset
        rank_loc = len(local_U[0])
        
        for r in range(rank_loc):
            global_col = col_offset + r
            
            for i_l in range(n_loc):
                for j_l in range(m_loc):
                    u_row_local = i_l * m_loc + j_l
                    u_row_global = (i_off + i_l) * m + (j_off + j_l)
                    U[u_row_global][global_col] = local_U[u_row_local][r]
                    
            for j_l in range(m_loc):
                for k_l in range(p_loc):
                    v_row_local = j_l * p_loc + k_l
                    v_row_global = (j_off + j_l) * p + (k_off + k_l)
                    V[v_row_global][global_col] = local_V[v_row_local][r]
                    
            for k_l in range(p_loc):
                for i_l in range(n_loc):
                    w_row_local = k_l * n_loc + i_l
                    w_row_global = (k_off + k_l) * n + (i_off + i_l)
                    W[w_row_global][global_col] = local_W[w_row_local][r]
        
        col_offset += rank_loc

    # Block 1: A[:, 0] * B[0, :]
    # n_loc=3, m_loc=1, p_loc=5
    # i_off=0, j_off=0, k_off=0
    U1, V1, W1 = _schoolbook_factors(3, 1, 5)
    add_local(U1, V1, W1, 0, 0, 0, 3, 1, 5)
    
    # Block 2a: A[:, 1:5] * B[1:5, 0]
    # n_loc=3, m_loc=4, p_loc=1
    # i_off=0, j_off=1, k_off=0
    U2a, V2a, W2a = _schoolbook_factors(3, 4, 1)
    add_local(U2a, V2a, W2a, 0, 1, 0, 3, 4, 1)
    
    # Block 2b-i: A[0, 1:5] * B[1:5, 1:5]
    # n_loc=1, m_loc=4, p_loc=4
    # i_off=0, j_off=1, k_off=1
    U2bi, V2bi, W2bi = _schoolbook_factors(1, 4, 4)
    add_local(U2bi, V2bi, W2bi, 0, 1, 1, 1, 4, 4)
    
    # Block 2b-ii: A[1:3, 1:5] * B[1:5, 1:5]
    # n_loc=2, m_loc=4, p_loc=4
    # i_off=1, j_off=1, k_off=1
    # Split into 4 blocks of 2x2x2
    
    # Sub-block A: n_loc=2, m_loc=2, p_loc=2
    # i_off=1, j_off=1, k_off=1
    U_A, V_A, W_A = _strassen_222()
    add_local(U_A, V_A, W_A, 1, 1, 1, 2, 2, 2)
    
    # Sub-block B: n_loc=2, m_loc=2, p_loc=2
    # i_off=1, j_off=3, k_off=1
    U_B, V_B, W_B = _strassen_222()
    add_local(U_B, V_B, W_B, 1, 3, 1, 2, 2, 2)
    
    # Sub-block C: n_loc=2, m_loc=2, p_loc=2
    # i_off=1, j_off=1, k_off=3
    U_C, V_C, W_C = _strassen_222()
    add_local(U_C, V_C, W_C, 1, 1, 3, 2, 2, 2)
    
    # Sub-block D: n_loc=2, m_loc=2, p_loc=2
    # i_off=1, j_off=3, k_off=3
    U_D, V_D, W_D = _strassen_222()
    add_local(U_D, V_D, W_D, 1, 3, 3, 2, 2, 2)
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
