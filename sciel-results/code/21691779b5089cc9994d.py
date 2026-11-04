"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    
    columns = []  # Each entry is (u_dict, v_dict, w_dict) as sparse dicts
    
    def add_term(u_dict, v_dict, w_dict):
        columns.append((u_dict, v_dict, w_dict))
    
    def add_schoolblock(n_start, m_start, p_start, ni, mi, pi):
        """Add schoolbook terms for a sub-block."""
        for i in range(ni):
            for j in range(mi):
                for k in range(pi):
                    u_idx = (n_start + i) * m + (m_start + j)
                    v_idx = (m_start + j) * p + (p_start + k)
                    w_idx = (p_start + k) * n + (n_start + i)
                    add_term({u_idx: 1}, {v_idx: 1}, {w_idx: 1})
    
    # Full schoolbook: split n into [0] and [1,2], m into [0,1,2] and [3], p into [0,1,2,3] and [4,5,6,7]
    
    # n=0, m=0..2, p=0..3
    add_schoolblock(0, 0, 0, 1, 3, 4)
    # n=0, m=3, p=0..3
    add_schoolblock(0, 3, 0, 1, 1, 4)
    # n=0, m=0..2, p=4..7
    add_schoolblock(0, 0, 4, 1, 3, 4)
    # n=0, m=3, p=4..7
    add_schoolblock(0, 3, 4, 1, 1, 4)
    # n=1..2, m=0..2, p=0..3
    add_schoolblock(1, 0, 0, 2, 3, 4)
    # n=1..2, m=3, p=0..3
    add_schoolblock(1, 3, 0, 2, 1, 4)
    # n=1..2, m=0..2, p=4..7
    add_schoolblock(1, 0, 4, 2, 3, 4)
    # n=1..2, m=3, p=4..7
    add_schoolblock(1, 3, 4, 2, 1, 4)
    
    rank = len(columns)
    
    # Build U, V, W matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col_idx, (u_dict, v_dict, w_dict) in enumerate(columns):
        for idx, val in u_dict.items():
            U[idx][col_idx] = val
        for idx, val in v_dict.items():
            V[idx][col_idx] = val
        for idx, val in w_dict.items():
            W[idx][col_idx] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
