import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
      n, m, p = DIMENSIONS
      target_rank = 24
      # Try to find lower rank
      # ... search logic ...
      
      # Fallback: Schoolbook
      rank = n * m * p
      U = np.zeros((n*m, rank), dtype=int)
      V = np.zeros((m*p, rank), dtype=int)
      W = np.zeros((p*n, rank), dtype=int)
      
      col = 0
      for i in range(n):
          for j in range(m):
              for k in range(p):
                  U[i*m+j, col] = 1
                  V[j*p+k, col] = 1
                  W[k*n+i, col] = 1
                  col += 1
                  
      return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
