      # Part 2: Rank 1 for A[:2, 2] * B[2, k0:k0+2]
      for li in range(2):
          U[li * m + 2][col] = 1
      for lk in range(2):
          V[2 * p + (k0 + lk)][col] = 1
      for lk in range(2):
          for li in range(2):
              W[(k0 + lk) * n + li][col] = 1
      col += 1
      