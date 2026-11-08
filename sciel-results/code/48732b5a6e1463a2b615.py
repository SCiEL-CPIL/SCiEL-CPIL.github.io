  def build_4x4x2():
      S_U = [ ... ] # Same as above
      S_V = [ ... ]
      S_W = [ ... ]
      
      U = [[0]*28 for _ in range(16)]
      V = [[0]*28 for _ in range(8)]
      W = [[0]*28 for _ in range(8)]
      
      # 4 products
      # P1: A00 * B_top
      # P2: A01 * B_bot
      # P3: A10 * B_top
      # P4: A11 * B_bot
      
      # B_top is B[0:2, :], B_bot is B[2:4, :]
      # B is 4x2.
      # B_top rows 0,1. B_bot rows 2,3.
      
      # U is 4x4.
      # A00: rows 0,1 cols 0,1
      # A01: rows 0,1 cols 2,3
      # A10: rows 2,3 cols 0,1
      # A11: rows 2,3 cols 2,3
      
      # V is 4x2.
      # B_top: rows 0,1 cols 0,1 (of B)
      # B_bot: rows 2,3 cols 0,1 (of B)
      
      # W is 4x2.
      # C_top: rows 0,1 cols 0,1
      # C_bot: rows 2,3 cols 0,1
      
      # C_top = P1 + P2
      # C_bot = P3 + P4
      
      prods = [
          {'A_block': 0, 'B_part': 'top'},
          {'A_block': 1, 'B_part': 'bot'},
          {'A_block': 2, 'B_part': 'top'},
          {'A_block': 3, 'B_part': 'bot'},
      ]
      
      c_combos = [
          {0: 1, 1: 1}, # C_top
          {2: 1, 3: 1}, # C_bot
      ]
      
      for k in range(4):
          prod = prods[k]
          base_col = k * 7
          
          a_block = prod['A_block']
          b_part = prod['B_part']
          
          for c in range(7):
              col = base_col + c
              
              # U
              for lr in range(2):
                  for lc in range(2):
                      local_idx = lr * 2 + lc
                      coeff_su = S_U[local_idx][c]
                      if coeff_su == 0: continue
                      
                      # Global A index for block a_block, local lr,lc
                      br = (a_block // 2) * 2 + lr
                      bc = (a_block % 2) * 2 + lc
                      u_idx = br * 4 + bc
                      U[u_idx][col] += coeff_su
                      
              # V
              for lr in range(2):
                  for lc in range(2):
                      local_idx = lr * 2 + lc
                      coeff_sv = S_V[local_idx][c]
                      if coeff_sv == 0: continue
                      
                      # Global B index
                      # B is 4x2.
                      # If b_part is 'top', rows 0,1.
                      # If 'bot', rows 2,3.
                      if b_part == 'top':
                          br = lr
                      else:
                          br = 2 + lr
                      bc = lc
                      v_idx = br * 2 + bc
                      V[v_idx][col] += coeff_sv
          
          # W
          # W is 4x2.
          # C_top (rows 0,1) gets P1, P2.
          # C_bot (rows 2,3) gets P3, P4.
          
          for gk in range(4):
              for gi in range(2):
                  w_row = gk * 4 + gi
                  
                  # Determine if this is C_top or C_bot
                  if gk < 2:
                      # C_top
                      # lr = gk, lc = gi
                      lr = gk
                      lc = gi
                      local_idx = lr * 2 + lc
                      # Contributing prods: 0 (P1), 1 (P2)
                      for m in [0, 1]:
                          coeff_m = c_combos[0].get(m, 0)
                          if coeff_m == 0: continue
                          for c in range(7):
                              col = m * 7 + c
                              coeff_sw = S_W[local_idx][c]
                              if coeff_sw == 0: continue
                              W[w_row][col] += coeff_m * coeff_sw
                  else:
                      # C_bot
                      # gk in 2,3. lr = gk - 2. lc = gi.
                      lr = gk - 2
                      lc = gi
                      local_idx = lr * 2 + lc
                      # Contributing prods: 2 (P3), 3 (P4)
                      for m in [2, 3]:
                          coeff_m = c_combos[1].get(m, 0)
                          if coeff_m == 0: continue
                          for c in range(7):
                              col = m * 7 + c
                              coeff_sw = S_W[local_idx][c]
                              if coeff_sw == 0: continue
                              W[w_row][col] += coeff_m * coeff_sw
                              
      return U, V, W
  