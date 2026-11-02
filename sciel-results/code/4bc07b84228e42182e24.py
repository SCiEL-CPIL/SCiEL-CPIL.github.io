    laderman_terms = [
        # u, v, w
        ([1,0,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,0,0]), # a00*b00
        ([0,1,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0,0]), # a01*b10 -> c00? No, a01*b10 is c00?
        # c_ik = sum_j a_ij b_jk.
        # c_00 = a_00 b_00 + a_01 b_10 + a_02 b_20.
        # So a_01 b_10 contributes to c_00. w index for c_00 is k*n+i = 0*3+0=0.
        # So w should be [1,0,0,0,0,0,0,0,0].
        # My previous guess was wrong.
    ]
    