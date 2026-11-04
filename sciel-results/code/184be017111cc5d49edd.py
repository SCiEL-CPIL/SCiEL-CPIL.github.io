import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    
    U = [[0] * (n * m * p) for _ in range(n * m)] # Pre-allocate max possible, will trim
    V = [[0] * (n * m * p) for _ in range(m * p)]
    W = [[0] * (n * m * p) for _ in range(p * n)]
    
    rank_counter = [0]
    
    def add_term_direct(u_entries, v_entries, w_entries, i_off, j_off, k_off):
        r = rank_counter[0]
        rank_counter[0] += 1
        for (li, lj, c) in u_entries:
            gi = i_off + li
            gj = j_off + lj
            U[gi * m + gj][r] = c
        for (lj, lk, c) in v_entries:
            gj = j_off + lj
            gk = k_off + lk
            V[gj * p + gk][r] = c
        for (lk, li, c) in w_entries:
            gk = k_off + lk
            gi = i_off + li
            W[gk * n + gi][r] = c

    def decompose_direct(n_sub, m_sub, p_sub, i_off, j_off, k_off):
        # Base case: Rank 1 if any dimension is 1
        if n_sub == 1 or m_sub == 1 or p_sub == 1:
            # Construct the single term
            u_entries = []
            for i in range(n_sub):
                for j in range(m_sub):
                    u_entries.append((i, j, 1))
            
            v_entries = []
            for j in range(m_sub):
                for k in range(p_sub):
                    v_entries.append((j, k, 1))
                    
            w_entries = []
            for k in range(p_sub):
                for i in range(n_sub):
                    w_entries.append((k, i, 1))
            
            add_term_direct(u_entries, v_entries, w_entries, i_off, j_off, k_off)
            return

        if n_sub == 2 and m_sub == 2 and p_sub == 2:
            # Strassen
            add_term_direct([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], i_off, j_off, k_off)
            add_term_direct([(1,0,1), (1,1,1)], [(0,0,1)], [(0,1,1), (1,1,-1)], i_off, j_off, k_off)
            add_term_direct([(0,0,1)], [(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], i_off, j_off, k_off)
            add_term_direct([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], i_off, j_off, k_off)
            add_term_direct([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (1,0,1)], i_off, j_off, k_off)
            add_term_direct([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)], i_off, j_off, k_off)
            add_term_direct([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)], i_off, j_off, k_off)
            return

        # Heuristic: Split the dimension that is largest or odd to expose 1s?
        # Priority: n > m > p
        if n_sub > 2:
            n1 = 2
            n2 = n_sub - 2
            decompose_direct(n1, m_sub, p_sub, i_off, j_off, k_off)
            decompose_direct(n2, m_sub, p_sub, i_off + n1, j_off, k_off)
        elif m_sub > 2:
            # Split into 2 and rest, or half if even?
            # Splitting into 2 and rest is consistent.
            m1 = 2
            m2 = m_sub - 2
            decompose_direct(n_sub, m1, p_sub, i_off, j_off, k_off)
            decompose_direct(n_sub, m2, p_sub, i_off, j_off + m1, k_off)
        elif p_sub > 2:
            p1 = 2
            p2 = p_sub - 2
            decompose_direct(n_sub, m_sub, p1, i_off, j_off, k_off)
            decompose_direct(n_sub, m_sub, p2, i_off, j_off, k_off + p1)
        else:
            # Schoolbook fallback (should be rare)
            for i in range(n_sub):
                for j in range(m_sub):
                    for k in range(p_sub):
                        add_term_direct([(i, j, 1)], [(j, k, 1)], [(k, i, 1)], i_off, j_off, k_off)

    decompose_direct(n, m, p, 0, 0, 0)
    
    rank = rank_counter[0]
    
    # Trim matrices to actual rank
    U_final = [row[:rank] for row in U]
    V_final = [row[:rank] for row in V]
    W_final = [row[:rank] for row in W]
    
    return {"U": U_final, "V": V_final, "W": W_final}
