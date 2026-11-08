  import numpy as np

  def construct():
      n, m, p = 4, 4, 4
      rank = 56
      U = np.zeros((n*m, rank), dtype=float)
      V = np.zeros((m*p, rank), dtype=float)
      W = np.zeros((p*n, rank), dtype=float)
      
      col = 0
      # Blocks: (i_off, j_off, k_off)
      # C_00 = A_00 B_00 + A_01 B_10
      # A_00: i=0,1 j=0,1 -> i_off=0, j_off=0
      # B_00: j=0,1 k=0,1 -> j_off=0, k_off=0
      # C_00: i=0,1 k=0,1 -> i_off=0, k_off=0
      
      blocks = [
          (0, 0, 0), # A00 B00 -> C00
          (0, 2, 0), # A01 B10 -> C00
          (0, 0, 2), # A00 B01 -> C01
          (0, 2, 2), # A01 B11 -> C01
          (2, 0, 0), # A10 B00 -> C10
          (2, 2, 0), # A11 B10 -> C10
          (2, 0, 2), # A10 B01 -> C11
          (2, 2, 2), # A11 B11 -> C11
      ]
      
      for i_off, j_off, k_off in blocks:
          # Add Strassen 2x2 block
          # P1
          U[i_off*4 + j_off, col] = 1
          U[(i_off+1)*4 + (j_off+1), col] = 1
          V[j_off*4 + k_off, col] = 1
          V[(j_off+1)*4 + (k_off+1), col] = 1
          W[k_off*4 + i_off, col] = 1
          W[(k_off+1)*4 + (i_off+1), col] = 1
          col += 1
          
          # ... 6 more terms
          
      # Convert to lists
      return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
  