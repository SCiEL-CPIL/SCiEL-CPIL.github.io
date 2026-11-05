     for i in range(5):
         for j in range(5):
             for k in range(5):
                 if (i, j, k) not in covered:
                     if j == k:
                         u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
                         u_vec[i*5 + j] = 1
                         v_vec[j*5 + k] = 1
                         w_vec[k*5 + i] = 1
                         add_col(u_vec, v_vec, w_vec)
     