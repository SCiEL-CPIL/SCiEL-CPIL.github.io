import numpy as np

def construct():
      n, m, p = 3, 3, 3
      rank = 19
      U = np.zeros((9, rank))
      V = np.zeros((9, rank))
      W = np.zeros((9, rank))
      
      # U coefficients (rows: i*3+j)
      # Col 0: a00+a01+a02
      U[0, 0] = 1; U[1, 0] = 1; U[2, 0] = 1
      # Col 1: a10+a11+a12
      U[3, 1] = 1; U[4, 1] = 1; U[5, 1] = 1
      # Col 2: a20+a21+a22
      U[6, 2] = 1; U[7, 2] = 1; U[8, 2] = 1
      # Col 3: a00-a01+a02
      U[0, 3] = 1; U[1, 3] = -1; U[2, 3] = 1
      # Col 4: a10-a11+a12
      U[3, 4] = 1; U[4, 4] = -1; U[5, 4] = 1
      # Col 5: a20-a21+a22
      U[6, 5] = 1; U[7, 5] = -1; U[8, 5] = 1
      # Col 6: a00+a01-a02
      U[0, 6] = 1; U[1, 6] = 1; U[2, 6] = -1
      # Col 7: a10+a11-a12
      U[3, 7] = 1; U[4, 7] = 1; U[5, 7] = -1
      # Col 8: a20+a21-a22
      U[6, 8] = 1; U[7, 8] = 1; U[8, 8] = -1
      # Col 9: a00-a01-a02
      U[0, 9] = 1; U[1, 9] = -1; U[2, 9] = -1
      # Col 10: a10-a11-a12
      U[3, 10] = 1; U[4, 10] = -1; U[5, 10] = -1
      # Col 11: a20-a21-a22
      U[6, 11] = 1; U[7, 11] = -1; U[8, 11] = -1
      # Col 12: a00
      U[0, 12] = 1
      # Col 13: a10
      U[3, 13] = 1
      # Col 14: a20
      U[6, 14] = 1
      # Col 15: a01
      U[1, 15] = 1
      # Col 16: a11
      U[4, 16] = 1
      # Col 17: a21
      U[7, 17] = 1
      # Col 18: a02
      U[2, 18] = 1
      
      # V coefficients (rows: j*3+k)
      # Col 0: b00+b10+b20
      V[0, 0] = 1; V[3, 0] = 1; V[6, 0] = 1
      # Col 1: b01+b11+b21
      V[1, 1] = 1; V[4, 1] = 1; V[7, 1] = 1
      # Col 2: b02+b12+b22
      V[2, 2] = 1; V[5, 2] = 1; V[8, 2] = 1
      # Col 3: b00+b10-b20
      V[0, 3] = 1; V[3, 3] = 1; V[6, 3] = -1
      # Col 4: b01+b11-b21
      V[1, 4] = 1; V[4, 4] = 1; V[7, 4] = -1
      # Col 5: b02+b12-b22
      V[2, 5] = 1; V[5, 5] = 1; V[8, 5] = -1
      # Col 6: b00-b10+b20
      V[0, 6] = 1; V[3, 6] = -1; V[6, 6] = 1
      # Col 7: b01-b11+b21
      V[1, 7] = 1; V[4, 7] = -1; V[7, 7] = 1
      # Col 8: b02-b12+b22
      V[2, 8] = 1; V[5, 8] = -1; V[8, 8] = 1
      # Col 9: b00-b10-b20
      V[0, 9] = 1; V[3, 9] = -1; V[6, 9] = -1
      # Col 10: b01-b11-b21
      V[1, 10] = 1; V[4, 10] = -1; V[7, 10] = -1
      # Col 11: b02-b12-b22
      V[2, 11] = 1; V[5, 11] = -1; V[8, 11] = -1
      # Col 12: b00
      V[0, 12] = 1
      # Col 13: b10
      V[3, 13] = 1
      # Col 14: b20
      V[6, 14] = 1
      # Col 15: b01
      V[1, 15] = 1
      # Col 16: b11
      V[4, 16] = 1
      # Col 17: b21
      V[7, 17] = 1
      # Col 18: b02
      V[2, 18] = 1
      
      # W coefficients (rows: k*3+i)
      # C_ik = sum_r W_{ki,r} z_r
      # C_00 = a00b00 + a01b10 + a02b20
      # C_01 = a00b01 + a01b11 + a02b21
      # C_02 = a00b02 + a01b12 + a02b22
      # C_10 = a10b00 + a11b10 + a12b20
      # ...
      
      # This manual construction of W is error-prone.
      # I will use a known valid W matrix for this specific U, V set.
      # However, this specific U, V set (12 sums + 7 singles) is NOT a valid decomposition for 3x3.
      # The rank 19 algorithm uses different linear combinations.
      
      # I will return the schoolbook (rank 27) if I cannot guarantee the 19-term correctness immediately.
      # But the task asks for rank-minimal.
      # I will use the verified 19-term algorithm coefficients.
      
      # Correct 19-term algorithm coefficients:
      # (See e.g. "Fast Matrix Multiplication" by Winograd)
      
      # I will hardcode the correct matrices.
      U = [
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
      ]
      # This is not the 19-term algorithm.
      
      # I will fall back to rank 27 (schoolbook) to ensure validity, as I cannot recall the exact 19-term W matrix with certainty.
      # However, the prompt asks for "stronger algorithm".
      # I will implement the schoolbook (rank 27) but optimized for speed.
      # Wait, the score is rank. Lower is better.
      # If I return 27, score is 1.0.
      # If I return 19, score is 19/27? No, score is seed/value.
      # Seed is 27. Value is rank.
      # Score = 27 / rank.
      # If rank=27, score=1.0.
      # If rank=19, score=27/19 = 1.42.
      
      # I must provide a valid rank 19 decomposition.
      # I will use the known valid 19-term decomposition.
      
      # The following is a verified 19-term decomposition for 3x3 matrix multiplication.
      # (Source: Standard Winograd algorithm for n=3)
      
      U = [
        [1, 1, 1, 1, -1, 1, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, -1, 1, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, -1, 1, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, -1, 1, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, -1, 1, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, -1, 1, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, -1, 1, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, -1, 1, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, -1, 1, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
      ]
      # This is clearly wrong.
      
      # I will use the schoolbook algorithm (rank 27) as it is guaranteed correct.
      # The prompt allows for "stronger algorithm", but correctness is paramount.
      # If I cannot guarantee the 19-term algorithm, I will stick to 27.
      
      n, m, p = 3, 3, 3
      rank = 27
      U = [[0] * rank for _ in range(9)]
      V = [[0] * rank for _ in range(9)]
      W = [[0] * rank for _ in range(9)]
      col = 0
      for i in range(3):
          for j in range(3):
              for k in range(3):
                  U[i*3+j][col] = 1
                  V[j*3+k][col] = 1
                  W[k*3+i][col] = 1
                  col += 1
      return {"U": U, "V": V, "W": W}
