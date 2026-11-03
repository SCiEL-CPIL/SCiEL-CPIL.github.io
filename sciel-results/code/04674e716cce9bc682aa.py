         def construct():
             n, m, p = 6, 6, 6
             # Strassen coefficients for 2x2
             # r=1..7
             # u[r][I][J], v[r][J][K], w[r][K][I]
             # ...
             
             U = [[0 for _ in range(189)] for _ in range(36)] # Guess rank
             # Better: build dynamically
             U_rows = [[] for _ in range(36)]
             V_rows = [[] for _ in range(36)]
             W_rows = [[] for _ in range(36)]
             
             for r in range(7):
                 for I in range(2):
                     for J in range(2):
                         for K in range(2):
                             u_val = u[r][I][J]
                             v_val = v[r][J][K]
                             w_val = w[r][K][I]
                             if u_val == 0 or v_val == 0 or w_val == 0:
                                 continue
                             
                             for ip in range(3):
                                 for jp in range(3):
                                     for kp in range(3):
                                         i = I * 3 + ip
                                         j = J * 3 + jp
                                         k = K * 3 + kp
                                         
                                         U_rows[i * 6 + j].append(u_val)
                                         V_rows[j * 6 + k].append(v_val)
                                         W_rows[k * 6 + i].append(w_val)
             
             # Check lengths
             rank = len(U_rows[0])
             # Verify all rows have same length
             # Return
         