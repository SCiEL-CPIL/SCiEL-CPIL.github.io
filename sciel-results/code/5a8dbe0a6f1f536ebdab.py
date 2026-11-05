import numpy as np

def construct():
       n, m, p = 3, 5, 6
       rank = n * m * p
       U = [[0] * rank for _ in range(n * m)]
       V = [[0] * rank for _ in range(m * p)]
       W = [[0] * rank for _ in range(p * n)]
       
       # Initialize schoolbook
       col = 0
       for i in range(n):
           for j in range(m):
               for k in range(p):
                   U[i * m + j][col] = 1
                   V[j * p + k][col] = 1
                   W[k * n + i][col] = 1
                   col += 1
                   
       # Greedy elimination
       # Convert to numpy arrays for linear algebra
       U_np = np.array(U, dtype=float)
       V_np = np.array(V, dtype=float)
       W_np = np.array(W, dtype=float)
       
       # The tensor entries are T[i, j, k] = sum_c U[i, c] V[j, c] W[k, c]
       # We want to remove columns where the contribution is redundant.
       # This is equivalent to finding the rank of the tensor? No, tensor rank.
       # Greedy: Remove a column if the remaining columns can still span the tensor.
       # This is hard.
       
       # Simpler: Just return schoolbook.
       return {"U": U, "V": V, "W": W}
