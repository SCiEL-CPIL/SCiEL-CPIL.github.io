  for c in range(28):
      for i in range(4):
          for j in range(4):
              U_full[i*4+j][c] = U_str[i*4+j][c]
          for k in range(4):
              V_full[j*7+k][c] = V_str[j*4+k][c] # Fixed
          for k in range(4):
              W_full[k*4+i][c] = W_str[i*4+k][c]
  