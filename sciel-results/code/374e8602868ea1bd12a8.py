w_idx = k * n + i
total = np.dot(U_arr[u_idx, :], V_arr[v_idx, :] * W_arr[w_idx, :])
if total != 1:
