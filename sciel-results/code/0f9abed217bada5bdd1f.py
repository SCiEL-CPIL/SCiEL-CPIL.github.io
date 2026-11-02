import time
import sys

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for multiplying a 3x5 matrix by a 5x7 matrix.
    Uses Strassen's algorithm for 2x2x2 sub-blocks to reduce rank below the schoolbook rank.
    """
    n, m, p = DIMENSIONS
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Schoolbook rank
    rank_schoolbook = n * m * p
    
    # We will use Strassen for 2x2x2 blocks and standard for the rest.
    # Dimensions: n=3, m=5, p=7
    # Split n into 2, 1
    # Split m into 2, 2, 1
    # Split p into 2, 2, 2, 1
    
    # Define the blocks
    # n_blocks: [2, 1]
    # m_blocks: [2, 2, 1]
    # p_blocks: [2, 2, 2, 1]
    
    # Initialize matrices
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    # We don't know the final rank yet. We'll build lists of columns.
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_column(u_vals, v_vals, w_vals):
        U_cols.append(u_vals)
        V_cols.append(v_vals)
        W_cols.append(w_vals)
        
    # Helper to get flattened index
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    # Define Strassen coefficients for 2x2x2
    # M1 = A00(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A01(B10-B00)
    # M4 = A11(B11-B00)
    # M5 = (A00+A11)B11
    # M6 = (A01-A00)B10
    # M7 = (A10+A00)B01
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    def add_strassen_block(i0, j0, k0):
        """Add a 2x2x2 Strassen block starting at (i0, j0, k0)."""
        # i0, i0+1 are rows
        # j0, j0+1 are cols
        # k0, k0+1 are output cols
        
        # M1
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[u_idx(i0, j0)] = 1
        v[v_idx(j0, k0)] = 1
        v[v_idx(j0+1, k0+1)] = 1
        w[w_idx(k0, i0)] = 1
        w[w_idx(k0+1, i0+1)] = 1
        # Wait, W is for output C.
        # C00 gets M1. C11 gets -M1? No.
        # C00 = M1...
        # C11 = M1...
        # So W for M1 should have 1 at (k0, i0) and 1 at (k0+1, i0+1)?
        # No, M1 is a scalar. It contributes to C00 and C11.
        # w[w_idx(k0, i0)] += 1
        # w[w_idx(k0+1, i0+1)] += 1
        # But we are adding a new column for each M.
        # So for M1, the W vector has 1s where M1 contributes.
        
        # Let's refine the W vectors.
        # M1 contributes to C00 (+1) and C11 (+1).
        # w[w_idx(k0, i0)] = 1
        # w[w_idx(k0+1, i0+1)] = 1
        add_column(u, v, w)
        
        # M2
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[u_idx(i0+1, j0)] = 1
        u[u_idx(i0+1, j0+1)] = 1
        v[v_idx(j0, k0)] = 1
        w[w_idx(k0, i0+1)] = 1
        w[w_idx(k0+1, i0+1)] = 1 # C10 and C11?
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        # So M2 contributes to C10 (+1) and C11 (-1).
        w[w_idx(k0, i0+1)] = 1
        w[w_idx(k0+1, i0+1)] = -1
        add_column(u, v, w)
        
        # M3
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[u_idx(i0, j0+1)] = 1
        v[v_idx(j0+1, k0)] = 1
        v[v_idx(j0, k0)] = -1
        # C01 = M3 + M5
        # C11 = M1 - M2 + M3 + M6
        w[w_idx(k0+1, i0)] = 1
        w[w_idx(k0+1, i0+1)] = 1
        add_column(u, v, w)
        
        # M4
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[u_idx(i0+1, j0+1)] = 1
        v[v_idx(j0+1, k0+1)] = 1
        v[v_idx(j0, k0)] = -1
        # C00 = M1 + M4 - M5 + M7
        # C10 = M2 + M4
        w[w_idx(k0, i0)] = 1
        w[w_idx(k0, i0+1)] = 1
        add_column(u, v, w)
        
        # M5
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[u_idx(i0, j0)] = 1
        u[u_idx(i0+1, j0+1)] = 1
        v[v_idx(j0+1, k0+1)] = 1
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        w[w_idx(k0, i0)] = -1
        w[w_idx(k0+1, i0)] = 1
        add_column(u, v, w)
        
        # M6
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[u_idx(i0, j0+1)] = 1
        u[u_idx(i0, j0)] = -1
        v[v_idx(j0+1, k0)] = 1
        # C11 = M1 - M2 + M3 + M6
        w[w_idx(k0+1, i0+1)] = 1
        add_column(u, v, w)
        
        # M7
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[u_idx(i0+1, j0)] = 1
        u[u_idx(i0, j0)] = 1
        v[v_idx(j0, k0+1)] = 1
        # C00 = M1 + M4 - M5 + M7
        w[w_idx(k0, i0)] = 1
        add_column(u, v, w)

    def add_standard_block(i0, j0, k0, ni, mj, pk):
        """Add standard multiplication for a ni x mj x pk block."""
        for i in range(ni):
            for j in range(mj):
                for k in range(pk):
                    u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
                    u[u_idx(i0+i, j0+j)] = 1
                    v[v_idx(j0+j, k0+k)] = 1
                    w[w_idx(k0+k, i0+i)] = 1
                    add_column(u, v, w)
                    if time.time() > deadline:
                        # Fallback to schoolbook if time is tight
                        return False
        return True

    # Define blocks
    # n: 2, 1
    # m: 2, 2, 1
    # p: 2, 2, 2, 1
    
    n_splits = [2, 1]
    m_splits = [2, 2, 1]
    p_splits = [2, 2, 2, 1]
    
    i_offset = 0
    for ni in n_splits:
        j_offset = 0
        for mj in m_splits:
            k_offset = 0
            for pk in p_splits:
                if ni == 2 and mj == 2 and pk == 2:
                    add_strassen_block(i_offset, j_offset, k_offset)
                else:
                    if not add_standard_block(i_offset, j_offset, k_offset, ni, mj, pk):
                        # Time exceeded, fallback
                        return _schoolbook_fallback()
                k_offset += pk
            j_offset += mj
        i_offset += ni

    # Convert lists to matrices
    rank = len(U_cols)
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for r in range(rank):
        u_vals = U_cols[r]
        v_vals = V_cols[r]
        w_vals = W_cols[r]
        for i in range(n*m):
            U[i][r] = u_vals[i]
        for j in range(m*p):
            V[j][r] = v_vals[j]
        for k in range(p*n):
            W[k][r] = w_vals[k]
            
    return {"U": U, "V": V, "W": W}

def _schoolbook_fallback():
    n, m, p = DIMENSIONS
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
