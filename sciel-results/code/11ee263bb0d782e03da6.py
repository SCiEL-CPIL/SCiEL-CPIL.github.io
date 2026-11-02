<<<<<<< SEARCH
# Part 5: A_right @ B_bottom (2x1 * 1x7) - Rank 1 term
    # T[i, 2, k] = 1 for all i, k
    # U[i, 2] = 1, V[2, k] = 1, W[k, i] = 1
    u_dict = {u_idx(0, 2): 1, u_idx(1, 2): 1}
    v_dict = {v_idx(2, k): 1 for k in range(7)}
    w_dict = {w_idx(k, i): 1 for k in range(7) for i in range(2)}
    add_term(u_dict, v_dict, w_dict)
=======
# Part 5: A_right @ B_bottom (2x1 * 1x7) - Rank 1 term
# T[i, 2, k] = 1 for all i, k
# U[i, 2] = 1, V[2, k] = 1, W[k, i] = 1
u_dict = {u_idx(0, 2): 1, u_idx(1, 2): 1}
v_dict = {v_idx(2, k): 1 for k in range(7)}
w_dict = {w_idx(k, i): 1 for k in range(7) for i in range(2)}
add_term(u_dict, v_dict, w_dict)
>>>>>>> REPLACE
