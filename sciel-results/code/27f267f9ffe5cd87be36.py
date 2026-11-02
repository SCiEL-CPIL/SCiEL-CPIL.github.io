"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 4
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
    
    def make_u():
        return [0] * (n * m)
    
    def make_v():
        return [0] * (m * p)
    
    def make_w():
        return [0] * (p * n)
    
    def add_u(vec, i, j, val):
        vec[i * m + j] += val
    
    def add_v(vec, j, k, val):
        vec[j * p + k] += val
    
    def add_w(vec, i, k, val):
        vec[k * n + i] += val
    
    def strassen_block(ia, ja, ka):
        """Strassen for A[ia:ia+2, ja:ja+2] x B[ja:ja+2, ka:ka+2] -> C[ia:ia+2, ka:ka+2]."""
        # M1 = (P00+P11)*(Q00+Q11)
        u, v, w = make_u(), make_v(), make_w()
        add_u(u, ia, ja, 1); add_u(u, ia+1, ja+1, 1)
        add_v(v, ja, ka, 1); add_v(v, ja+1, ka+1, 1)
        add_w(w, ia, ka, 1); add_w(w, ia+1, ka+1, 1)
        add_term(u, v, w)
        
        # M2 = (P10+P11)*Q00
        u, v, w = make_u(), make_v(), make_w()
        add_u(u, ia+1, ja, 1); add_u(u, ia+1, ja+1, 1)
        add_v(v, ja, ka, 1)
        add_w(w, ia+1, ka, 1)
        add_term(u, v, w)
        
        # M3 = P00*(Q01-Q11)
        u, v, w = make_u(), make_v(), make_w()
        add_u(u, ia, ja, 1)
        add_v(v, ja, ka+1, 1); add_v(v, ja+1, ka+1, -1)
        add_w(w, ia, ka+1, 1)
        add_term(u, v, w)
        
        # M4 = P11*(Q10-Q00)
        u, v, w = make_u(), make_v(), make_w()
        add_u(u, ia+1, ja+1, 1)
        add_v(v, ja+1, ka, 1); add_v(v, ja, ka, -1)
        add_w(w, ia, ka, 1); add_w(w, ia+1, ka, 1)
        add_term(u, v, w)
        
        # M5 = (P00+P01)*Q11
        u, v, w = make_u(), make_v(), make_w()
        add_u(u, ia, ja, 1); add_u(u, ia, ja+1, 1)
        add_v(v, ja+1, ka+1, 1)
        add_w(w, ia, ka, -1); add_w(w, ia, ka+1, 1)
        add_term(u, v, w)
        
        # M6 = (P10-P00)*(Q00+Q01)
        u, v, w = make_u(), make_v(), make_w()
        add_u(u, ia+1, ja, 1); add_u(u, ia, ja, -1)
        add_v(v, ja, ka, 1); add_v(v, ja, ka+1, 1)
        add_w(w, ia+1, ka+1, 1)
        add_term(u, v, w)
        
        # M7 = (P01-P11)*(Q10+Q11)
        u, v, w = make_u(), make_v(), make_w()
        add_u(u, ia, ja+1, 1); add_u(u, ia+1, ja+1, -1)
        add_v(v, ja+1, ka, 1); add_v(v, ja+1, ka+1, 1)
        add_w(w, ia, ka, 1); add_w(w, ia+1, ka+1, 1)
        add_term(u, v, w)
    
    def schoolbook_1x2x2(i, ja, ka):
        """Schoolbook for A[i, ja:ja+2] x B[ja:ja+2, ka:ka+2] -> C[i, ka:ka+2]."""
        # P0*Q00 -> R0
        u, v, w = make_u(), make_v(), make_w()
        add_u(u, i, ja, 1)
        add_v(v, ja, ka, 1)
        add_w(w, i, ka, 1)
        add_term(u, v, w)
        
        # P1*Q10 -> R0
        u, v, w = make_u(), make_v(), make_w()
        add_u(u, i, ja+1, 1)
        add_v(v, ja+1, ka, 1)
        add_w(w, i, ka, 1)
        add_term(u, v, w)
        
        # P0*Q01 -> R1
        u, v, w = make_u(), make_v(), make_w()
        add_u(u, i, ja, 1)
        add_v(v, ja, ka+1, 1)
        add_w(w, i, ka+1, 1)
        add_term(u, v, w)
        
        # P1*Q11 -> R1
        u, v, w = make_u(), make_v(), make_w()
        add_u(u, i, ja+1, 1)
        add_v(v, ja+1, ka+1, 1)
        add_w(w, i, ka+1, 1)
        add_term(u, v, w)
    
    # Top 2 rows with Strassen (4 blocks of rank 7 each = 28)
    strassen_block(0, 0, 0)
    strassen_block(0, 2, 0)
    strassen_block(0, 0, 2)
    strassen_block(0, 2, 2)
    
    # Bottom row with schoolbook (4 blocks of rank 4 each = 16)
    schoolbook_1x2x2(2, 0, 0)
    schoolbook_1x2x2(2, 2, 0)
    schoolbook_1x2x2(2, 0, 2)
    schoolbook_1x2x2(2, 2, 2)
    
    rank = len(U_cols)
    
    U = [[U_cols[r][row] for r in range(rank)] for row in range(n * m)]
    V = [[V_cols[r][row] for r in range(rank)] for row in range(m * p)]
    W = [[W_cols[r][row] for r in range(rank)] for row in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
