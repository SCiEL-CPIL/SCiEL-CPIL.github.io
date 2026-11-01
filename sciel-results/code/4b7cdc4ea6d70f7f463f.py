import numpy as np
import time
import random

DIMENSIONS = [2, 3, 4]

def construct():
       n, m, p = DIMENSIONS
       target_rank = 22
       
       # Deadline guard
       start_time = time.time()
       deadline = start_time + 110 # 10s margin
       
       # Fallback: Schoolbook
       def get_schoolbook():
           rank = n * m * p
           U = [[0] * rank for _ in range(n * m)]
           V = [[0] * rank for _ in range(m * p)]
           W = [[0] * rank for _ in range(p * n)]
           col = 0
           for i in range(n):
               for j in range(m):
                   for k in range(p):
                       U[i * m + j][col] = 1
                       V[j * p + k][col] = 1
                       W[k * n + i][col] = 1
                       col += 1
           return {"U": U, "V": V, "W": W}
       
       best_result = get_schoolbook()
       
       # Search for rank 22
       # We fix V and W, solve for U.
       # U_{i,j,r} is solved from: sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} = 1 for all k.
       # This is a linear system A U = b for each (i,j).
       # A is (p x R), b is (p x 1).
       # A_{k, r} = V_{j, k, r} W_{k, i, r}. Note: V index is j*p+k, W index is k*n+i.
       
       for attempt in range(5000):
           if time.time() > deadline:
               break
           
           # Random V, W with small half-integers
           vals = [-1, -0.5, 0, 0.5, 1]
           V = np.array([[random.choice(vals) for _ in range(target_rank)] for _ in range(m * p)])
           W = np.array([[random.choice(vals) for _ in range(target_rank)] for _ in range(p * n)])
           
           U = np.zeros((n * m, target_rank))
           valid = True
           
           for i in range(n):
               for j in range(m):
                   # Construct A (p x R)
                   A = np.zeros((p, target_rank))
                   for k in range(p):
                       for r in range(target_rank):
                           # V index: j * p + k
                           # W index: k * n + i
                           A[k, r] = V[j * p + k, r] * W[k * n + i, r]
                   
                   # b is ones (p x 1)
                   b = np.ones(p)
                   
                   # Solve A U_col = b
                   # U_col is (R x 1)
                   try:
                       # Use least squares if not square
                       U_col, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                   except:
                       valid = False
                       break
                       
                   # Check if U_col is close to half-integers
                   # Round to nearest 0.5
                   U_rounded = np.round(U_col * 2) / 2
                   
                   # Check error
                   if np.linalg.norm(A @ U_rounded - b) > 1e-9:
                       valid = False
                       break
                       
                   U[i * m + j, :] = U_rounded
               
           if not valid:
               continue
           
           # Verify full decomposition
           # Check sum_r U[i,j,r] V[j,k,r] W[k,i,r] == 1 for all i,j,k
           is_exact = True
           for i in range(n):
               for j in range(m):
                   for k in range(p):
                       s = 0
                       for r in range(target_rank):
                           s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                       if abs(s - 1) > 1e-9:
                           is_exact = False
                           break
                   if not is_exact:
                       break
               if not is_exact:
                   break
           
           if is_exact:
               # Convert to lists of floats/ints
               U_list = [[float(x) for x in row] for row in U]
               V_list = [[float(x) for x in row] for row in V]
               W_list = [[float(x) for x in row] for row in W]
               return {"U": U_list, "V": V_list, "W": W_list}
       
       return best_result
