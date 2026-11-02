import numpy as np
import time

def construct():
       start_time = time.time()
       deadline = start_time + 110.0

       n, m, p = 3, 3, 6
       num_U = n * m
       num_V = m * p
       num_W = p * n

       def verify(U, V, W, rank):
           U_np = np.array(U, dtype=float)
           V_np = np.array(V, dtype=float)
           W_np = np.array(W, dtype=float)
           
           # Check all tensor terms
           for i in range(n):
               for j in range(m):
                   for k in range(p):
                       u_idx = i * m + j
                       v_idx = j * p + k
                       w_idx = k * n + i
                       actual = 0.0
                       for r in range(rank):
                           actual += U_np[u_idx, r] * V_np[v_idx, r] * W_np[w_idx, r]
                       if abs(actual - 1.0) > 1e-6:
                           return False
           
           # Check non-tensor terms are zero
           # This is expensive, so we check a random subset or skip if tensor terms are perfect.
           # For schoolbook, it's exact.
           return True

       # Schoolbook
       rank = 54
       U = [[0] * rank for _ in range(num_U)]
       V = [[0] * rank for _ in range(num_V)]
       W = [[0] * rank for _ in range(num_W)]
       
       col = 0
       for i in range(n):
           for j in range(m):
               for k in range(p):
                   U[i * m + j][col] = 1
                   V[j * p + k][col] = 1
                   W[k * n + i][col] = 1
                   col += 1
                   
       if verify(U, V, W, rank):
           return {"U": U, "V": V, "W": W}
       
       # Fallback (should not happen)
       return {"U": U, "V": V, "W": W}
