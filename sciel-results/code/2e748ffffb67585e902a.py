def _build_decomposition_32(n, m, p):
    terms = []
    
    # Helper to add 2x2x2 Strassen terms with offsets
    def add_strassen_2x2x2(j_off, k_off):
        for u, v, w in _strassen_2x2x2_terms():
            u_off = {}
            for (ii, jj), val in u.items():
                u_off[(ii, jj + j_off)] = val
            v_off = {}
            for (jj, kk), val in v.items():
                v_off[(jj + j_off, kk + k_off)] = val
            w_off = {}
            for (kk, ii), val in w.items():
                w_off[(kk + k_off, ii)] = val
            terms.append((u_off, v_off, w_off))

    # Helper to add schoolbook terms for 2 x m_block x p_block
    def add_schoolbook_2(j_start, j_end, k_start, k_end):
        for i in range(2):
            for j in range(j_start, j_end):
                for k in range(k_start, k_end):
                    terms.append((
                        {(i, j): 1},
                        {(j, k): 1},
                        {(k, i): 1}
                    ))

    # Block 1: j in 0..2, k in 0..2 (2x3x3)
    # Split j: 0..1, 2. k: 0..1, 2.
    add_strassen_2x2x2(0, 0) # j 0..1, k 0..1
    add_schoolbook_2(0, 2, 2, 3) # j 0..1, k 2..2
    add_schoolbook_2(2, 3, 0, 2) # j 2..2, k 0..1
    add_schoolbook_2(2, 3, 2, 3) # j 2..2, k 2..2

    # Block 2: j in 0..2, k in 3..4 (2x3x2)
    # Split j: 0..1, 2. k: 3..4 (size 2)
    add_strassen_2x2x2(0, 3) # j 0..1, k 3..4
    add_schoolbook_2(2, 3, 3, 5) # j 2..2, k 3..4

    # Block 3: j in 3..4, k in 0..2 (2x2x3)
    # Split k: 0..1, 2. j: 3..4 (size 2)
    add_strassen_2x2x2(3, 0) # j 3..4, k 0..1
    add_schoolbook_2(3, 5, 2, 3) # j 3..4, k 2..2

    # Block 4: j in 3..4, k in 3..4 (2x2x2)
    add_strassen_2x2x2(3, 3)

    return terms
