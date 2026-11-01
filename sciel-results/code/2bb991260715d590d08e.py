"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """Construct rank-28 decomposition for 2x3x5 matrix multiplication.
    
    Uses block structure: 2x3x5 split as:
    - 2x2x3 (rank 11: Strassen 2x2x2 + schoolbook 2x2x1)
    - 2x2x2 (rank 7: Strassen)
    - 2x1x3 (rank 6: schoolbook)
    - 2x1x2 (rank 4: schoolbook)
    Total rank = 28
    """
    deadline = time.time() + 130
    n, m, p = DIMENSIONS  # 2, 3, 5
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vec, v_vec, w_vec):
        U_cols.append(list(u_vec))
        V_cols.append(list(v_vec))
        W_cols.append(list(w_vec))
    
    def zeros_l(length):
        return [0] * length
    
    # Strassen 7 terms for 2x2x2 (local indices)
    strassen_terms = [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),   # M1
        ([0,0,1,1], [1,0,0,0], [0,0,1,0]),   # M2
        ([1,0,0,0], [0,1,-1,0], [0,1,0,1]),  # M3
        ([0,0,0,1], [0,1,-1,0], [1,0,1,0]),  # M4
        ([1,1,0,0], [0,0,0,1], [0,1,-1,0]),  # M5
        ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),  # M6
        ([0,1,-1,0], [0,0,1,1], [1,0,0,0]),  # M7
    ]
    
    def add_strassen_block(j_offset, k_offset):
        """Add Strassen 2x2x2 block for j in {j_offset, j_offset+1}, k in {k_offset, k_offset+1}."""
        for u_loc, v_loc, w_loc in strassen_terms:
            u_g = zeros_l(6)
            v_g = zeros_l(15)
            w_g = zeros_l(10)
            for li in range(4):
                li_i, li_j = li // 2, li % 2
                gi = li_i * 3 + (li_j + j_offset)
                u_g[gi] += u_loc[li]
            for li in range(4):
                li_j, li_k = li // 2, li % 2
                gi = (li_j + j_offset) * 5 + (li_k + k_offset)
                v_g[gi] += v_loc[li]
            for li in range(4):
                li_k, li_i = li // 2, li % 2
                gi = (li_k + k_offset) * 2 + li_i
                w_g[gi] += w_loc[li]
            add_term(u_g, v_g, w_g)
    
    # Block 1a: Strassen 2x2x2 for j={0,1}, k={0,1}
    add_strassen_block(0, 0)
    
    # Block 1b: Schoolbook 2x2x1 for j={0,1}, k=2
    for i in range(2):
        for j in range(2):
            u_g = zeros_l(6)
            v_g = zeros_l(15)
            w_g = zeros_l(10)
            u_g[i * 3 + j] = 1
            v_g[j * 5 + 2] = 1
            w_g[2 * 2 + i] = 1
            add_term(u_g, v_g, w_g)
    
    # Block 2: Strassen 2x2x2 for j={0,1}, k={3,4}
    add_strassen_block(0, 3)
    
    # Block 3: Schoolbook 2x1x3 for j=2, k={0,1,2}
    for i in range(2):
        for k in range(3):
            u_g = zeros_l(6)
            v_g = zeros_l(15)
            w_g = zeros_l(10)
            u_g[i * 3 + 2] = 1
            v_g[2 * 5 + k] = 1
            w_g[k * 2 + i] = 1
            add_term(u_g, v_g, w_g)
    
    # Block 4: Schoolbook 2x1x2 for j=2, k={3,4}
    for i in range(2):
        for k in range(3, 5):
            u_g = zeros_l(6)
            v_g = zeros_l(15)
            w_g = zeros_l(10)
            u_g[i * 3 + 2] = 1
            v_g[2 * 5 + k] = 1
            w_g[k * 2 + i] = 1
            add_term(u_g, v_g, w_g)
    
    rank = len(U_cols)
    
    # Verify
    def verify(U, V, W, n, m, p, tol=1e-10):
        rank = len(U)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(rank):
                        total += U[r][i*m+j] * V[r][j*p+k] * W[r][k*n+i]
                    if abs(total - 1) > tol:
                        return False
        return True
    
    # Transpose to column-major for verification
    U_t = [[U_cols[r][i] for r in range(rank)] for i in range(6)]
    V_t = [[V_cols[r][i] for r in range(rank)] for i in range(15)]
    W_t = [[W_cols[r][i] for r in range(rank)] for i in range(10)]
    
    if not verify(U_t, V_t, W_t, n, m, p):
        # Fallback: schoolbook rank 30
        U_fb = []
        V_fb = []
        W_fb = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_g = zeros_l(6)
                    v_g = zeros_l(15)
                    w_g = zeros_l(10)
                    u_g[i * 3 + j] = 1
                    v_g[j * 5 + k] = 1
                    w_g[k * 2 + i] = 1
                    U_fb.append(u_g)
                    V_fb.append(v_g)
                    W_fb.append(w_g)
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U_cols, "V": V_cols, "W": W_cols}
# EVOLVE-BLOCK-END
