<<<<<<< SEARCH
    # Part 5: Schoolbook for A[0:2, 2] * B[2, 0:7]
    for k in range(p):
        add_term({u_idx(0,2): 1}, {v_idx(2,k): 1}, {w_idx(k,0): 1})
        add_term({u_idx(1,2): 1}, {v_idx(2,k): 1}, {w_idx(k,1): 1})
        
=======
    # Part 5: Rank 1 decomposition for A[0:2, 2] * B[2, 0:7]
    # This part corresponds to j=2.
    # T[i, 2, k] = 1 for all i, k.
    # Factorized as U[i, 2] * V[2, k] * W[k, i]
    # U[i, 2] = 1 for i in {0, 1}
    # V[2, k] = 1 for k in {0..6}
    # W[k, i] = 1 for all k, i
    u_part5 = {u_idx(0,2): 1, u_idx(1,2): 1}
    v_part5 = {v_idx(2,k): 1 for k in range(p)}
    w_part5 = {w_idx(k,i): 1 for k in range(p) for i in range(n)}
    add_term(u_part5, v_part5, w_part5)
>>>>>>> REPLACE
