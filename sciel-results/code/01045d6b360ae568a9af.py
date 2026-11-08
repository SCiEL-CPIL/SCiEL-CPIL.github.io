  for r in range(rank):
      u_val = np.dot(U_arr[:, r], A_flat)
      v_val = np.dot(V_arr[:, r], B_flat)
      for k in range(p):
          for i in range(n):
              C_computed[i, k] += W_arr[k * n + i, r] * u_val * v_val
  