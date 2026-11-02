  def add_strassen(U, V, W, i_offset, l1, l2, k_offset, start_col):
      col = start_col
      # M1
      U[i_offset*5 + l1][col] += 1
      U[(i_offset+1)*5 + l2][col] += 1
      V[l1*6 + k_offset][col] += 1
      V[l2*6 + k_offset+1][col] += 1
      W[k_offset*2 + i_offset][col] += 1
      W[(k_offset+1)*2 + i_offset+1][col] += 1
      col += 1
      
      # M2
      U[i_offset*5 + l1][col] += 1
      U[(i_offset+1)*5 + l2][col] += 1
      V[l2*6 + k_offset][col] += 1
      W[(k_offset+1)*2 + i_offset][col] += 1
      col += 1
      
      # M3
      U[i_offset*5 + l1][col] += 1
      V[l1*6 + k_offset+1][col] += 1
      V[l2*6 + k_offset+1][col] -= 1
      W[k_offset*2 + i_offset+1][col] += 1
      W[(k_offset+1)*2 + i_offset+1][col] += 1
      col += 1
      
      # M4
      U[(i_offset+1)*5 + l2][col] += 1
      V[l2*6 + k_offset][col] += 1
      V[l1*6 + k_offset][col] -= 1
      W[k_offset*2 + i_offset][col] += 1
      W[(k_offset+1)*2 + i_offset][col] += 1
      col += 1
      
      # M5
      U[i_offset*5 + l2][col] += 1
      U[(i_offset+1)*5 + l2][col] += 1
      V[l1*6 + k_offset][col] += 1
      W[k_offset*2 + i_offset][col] -= 1
      W[k_offset*2 + i_offset+1][col] += 1
      col += 1
      
      # M6
      U[i_offset*5 + l1][col] += 1
      U[(i_offset+1)*5 + l1][col] -= 1
      V[l1*6 + k_offset+1][col] += 1
      V[l2*6 + k_offset+1][col] += 1
      W[(k_offset+1)*2 + i_offset+1][col] += 1
      col += 1
      
      # M7
      U[(i_offset+1)*5 + l1][col] += 1
      U[i_offset*5 + l1][col] -= 1
      V[l2*6 + k_offset][col] += 1
      V[l2*6 + k_offset+1][col] -= 1
      W[k_offset*2 + i_offset][col] += 1
      col += 1
      
      return col
  