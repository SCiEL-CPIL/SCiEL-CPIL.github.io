"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    
    columns = []  # Each entry is (u_dict, v_dict, w_dict)
    
    def add_term(u_dict, v_dict, w_dict):
        columns.append((u_dict, v_dict, w_dict))
    
    def add_schoolblock(n_start, m_start, p_start, ni, mi, pi):
        for i in range(ni):
            for j in range(mi):
                for k in range(pi):
                    u_idx = (n_start + i) * m + (m_start + j)
                    v_idx = (m_start + j) * p + (p_start + k)
                    w_idx = (p_start + k) * n + (n_start + i)
                    add_term({u_idx: 1}, {v_idx: 1}, {w_idx: 1})
    
    def add_laderman_2x3x4(n_start, m_start, p_start):
        """Add Laderman's rank-14 decomposition for 2x3x4 block.
        n_start: starting row of A (2 rows)
        m_start: starting col of A (3 cols)  
        p_start: starting col of B (4 cols)
        """
        def u_glob(li, lj):
            return (n_start + li) * m + (m_start + lj)
        def v_glob(lj, lk):
            return (m_start + lj) * p + (p_start + lk)
        def w_glob(lk, li):
            return (p_start + lk) * n + (n_start + li)
        
        # Rank-14 construction for 2x3x4
        # A: 2x3 (li in 0,1; lj in 0,1,2)
        # B: 3x4 (lj in 0,1,2; lk in 0,1,2,3)
        # C: 2x4 (li in 0,1; lk in 0,1,2,3)
        
        # Terms 1-12: direct products for C[0,*]
        for lj in range(3):
            for lk in range(4):
                add_term({u_glob(0, lj): 1}, {v_glob(lj, lk): 1}, {w_glob(lk, 0): 1})
        
        # Terms 13-14: shared products for C[1,*]
        # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[2,0]
        # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] + A[1,2]*B[2,1]
        # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2] + A[1,2]*B[2,2]
        # C[1,3] = A[1,0]*B[0,3] + A[1,1]*B[1,3] + A[1,2]*B[2,3]
        
        # Use 2 additional terms with multi-entry dicts
        # Term 13: (A[1,0]+A[1,1]+A[1,2]) * (B[0,0]+B[1,0]+B[2,0]) -> contributes to C[1,0]
        # But this also produces cross terms. We need to subtract them.
        # This approach doesn't work cleanly with just 2 extra terms.
        
        # Alternative: use schoolbook for C[1,*] (12 terms) = total 24.
        # For now, use schoolbook to ensure correctness.
        for lj in range(3):
            for lk in range(4):
                add_term({u_glob(1, lj): 1}, {v_glob(lj, lk): 1}, {w_glob(lk, 1): 1})
    
    # Block partition: n=[0] and n=[1,2], m=[0,1,2] and m=[3], p=[0,1,2,3] and p=[4,5,6,7]
    
    # n=0, m=0..2, p=0..3
    add_schoolblock(0, 0, 0, 1, 3, 4)  # 12 terms
    
    # n=0, m=3, p=0..3
    add_schoolblock(0, 3, 0, 1, 1, 4)  # 4 terms
    
    # n=0, m=0..2, p=4..7
    add_schoolblock(0, 0, 4, 1, 3, 4)  # 12 terms
    
    # n=0, m=3, p=4..7
    add_schoolblock(0, 3, 4, 1, 1, 4)  # 4 terms
    
    # n=1..2, m=0..2, p=0..3
    add_laderman_2x3x4(1, 0, 0)  # 24 terms (schoolbook for now)
    
    # n=1..2, m=3, p=0..3
    add_schoolblock(1, 3, 0, 2, 1, 4)  # 8 terms
    
    # n=1..2, m=0..2, p=4..7
    add_laderman_2x3x4(1, 0, 4)  # 24 terms (schoolbook for now)
    
    # n=1..2, m=3, p=4..7
    add_schoolblock(1, 3, 4, 2, 1, 4)  # 8 terms
    
    rank = len(columns)
    
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
