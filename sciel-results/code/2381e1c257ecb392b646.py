<<<<<<< SEARCH
    # --- Part 2: A1*B1b (2x2 * 2x3), rank 5 ---
    # A1 indices in full A: (0,0)=0, (0,1)=1, (1,0)=3, (1,1)=4
    # B1b indices in full B: (0,2)=2, (0,3)=3, (0,4)=4, (1,2)=7, (1,3)=8, (1,4)=9
    # C indices: (0,2)=2, (0,3)=3, (0,4)=4, (1,2)=7, (1,3)=8, (1,4)=9
    
    U_223, V_223, W_223 = _decomp_2x2x3()
    
    # Map 2x2*2x3 indices to full problem
    # 2x2*2x3 A: (0,0)->0, (0,1)->1, (1,0)->3, (1,1)->4
    # 2x2*2x3 B: (0,0)->2, (0,1)->3, (0,2)->4, (1,0)->7, (1,1)->8, (1,2)->9
    # 2x2*2x3 C: (0,0)->2, (0,1)->3, (0,2)->4, (1,0)->7, (1,1)->8, (1,2)->9
    
    a_map2 = [0, 1, 3, 4]
    b_map2 = [2, 3, 4, 7, 8, 9]
    c_map2 = [2, 3, 4, 7, 8, 9]
    
    for r in range(5):
        for s_idx, f_idx in enumerate(a_map2):
            U[f_idx, 7 + r] = U_223[s_idx, r]
        for s_idx, f_idx in enumerate(b_map2):
            V[f_idx, 7 + r] = V_223[s_idx, r]
        for s_idx, f_idx in enumerate(c_map2):
            W[f_idx, 7 + r] = W_223[s_idx, r]
=======
    # --- Part 2: A1*B1b (2x2 * 2x3), rank 11 ---
    # A1 indices in full A: (0,0)=0, (0,1)=1, (1,0)=3, (1,1)=4
    # B1b indices in full B: (0,2)=2, (0,3)=3, (0,4)=4, (1,2)=7, (1,3)=8, (1,4)=9
    # C indices: (0,2)=2, (0,3)=3, (0,4)=4, (1,2)=7, (1,3)=8, (1,4)=9
    
    U_223, V_223, W_223 = _decomp_2x2x3()
    
    # Map 2x2*2x3 indices to full problem
    a_map2 = [0, 1, 3, 4]
    b_map2 = [2, 3, 4, 7, 8, 9]
    c_map2 = [2, 3, 4, 7, 8, 9]
    
    for r in range(11):
        for s_idx, f_idx in enumerate(a_map2):
            U[f_idx, 7 + r] = U_223[s_idx, r]
        for s_idx, f_idx in enumerate(b_map2):
            V[f_idx, 7 + r] = V_223[s_idx, r]
        for s_idx, f_idx in enumerate(c_map2):
            W[f_idx, 7 + r] = W_223[s_idx, r]
>>>>>>> REPLACE

Now update the total rank and the Part 3 offset:

