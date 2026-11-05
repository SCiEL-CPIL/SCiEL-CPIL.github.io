import time
import numpy as np

DIMENSIONS = [4, 5, 6]

def construct():
       start_time = time.time()
       deadline = start_time + 110  # 10s margin
       
       n, m, p = DIMENSIONS
       rank_schoolbook = n * m * p
       
       # Schoolbook decomposition
       def get_schoolbook(rank):
           U = np.zeros((n * m, rank), dtype=np.float64)
           V = np.zeros((m * p, rank), dtype=np.float64)
           W = np.zeros((p * n, rank), dtype=np.float64)
           col = 0
           for i in range(n):
               for j in range(m):
                   for k in range(p):
                       if col < rank:
                           U[i * m + j, col] = 1
                           V[j * p + k, col] = 1
                           W[k * n + i, col] = 1
                           col += 1
           return U, V, W

       U_school, V_school, W_school = get_schoolbook(rank_schoolbook)
       
       best_rank = rank_schoolbook
       best_U, best_V, best_W = U_school, V_school, W_school

       # Search for lower ranks
       # Start from 119 down to 100
       # Limit total time
       
       def check_solution(U, V, W):
           # U, V, W are flattened or reshaped? 
           # Let's assume they are reshaped to (n,m,R), (m,p,R), (p,n,R)
           valid = True
           # Vectorized check?
           # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
           # We want T[i,j,k] == 1 for all i,j,k
           
           # Use einsum for speed
           # U: (n, m, R)
           # V: (m, p, R)
           # W: (p, n, R)
           # Result: (n, m, p)
           # np.einsum('ijr,jkr,kin->ijk', U, V, W)
           try:
               T = np.einsum('ijr,jkr,kin->ijk', U, V, W)
               if not np.allclose(T, 1.0, atol=1e-9):
                   return False
           except Exception:
               return False
           return True

       def run_als(n, m, p, rank, max_iter, seed):
           np.random.seed(seed)
           U = np.random.uniform(-1, 1, size=(n, m, rank))
           V = np.random.uniform(-1, 1, size=(m, p, rank))
           W = np.random.uniform(-1, 1, size=(p, n, rank))
           
           for it in range(max_iter):
               if time.time() > deadline - 5:
                   return None
                   
               # Update U
               for i in range(n):
                   for j in range(m):
                       # A[k, r] = V[j, k, r] * W[k, i, r]
                       A = V[j] * W[:, i] # (p, rank)
                       b = np.ones(p)
                       try:
                           u_sol, residuals, rank_m, sv = np.linalg.lstsq(A, b, rcond=None)
                           U[i, j] = u_sol
                       except np.linalg.LinAlgError:
                           pass
                           
               # Update V
               for j in range(m):
                   for k in range(p):
                       # A[i, r] = U[i, j, r] * W[k, i, r]
                       A = U[:, j] * W[k] # (n, rank)
                       b = np.ones(n)
                       try:
                           v_sol, residuals, rank_m, sv = np.linalg.lstsq(A, b, rcond=None)
                           V[j, k] = v_sol
                       except np.linalg.LinAlgError:
                           pass
                           
               # Update W
               for k in range(p):
                   for i in range(n):
                       # A[j, r] = U[i, j, r] * V[j, k, r]
                       A = U[i] * V[:, k] # (m, rank)
                       b = np.ones(m)
                       try:
                           w_sol, residuals, rank_m, sv = np.linalg.lstsq(A, b, rcond=None)
                           W[k, i] = w_sol
                       except np.linalg.LinAlgError:
                           pass
           
           # Round to nearest 0.5
           U_rounded = np.round(U * 2) / 2
           V_rounded = np.round(V * 2) / 2
           W_rounded = np.round(W * 2) / 2
           
           if check_solution(U_rounded, V_rounded, W_rounded):
               return U_rounded.flatten(), V_rounded.flatten(), W_rounded.flatten()
           
           return None

       # Try ranks
       for rank in range(rank_schoolbook - 1, 99, -1):
           if time.time() > deadline - 10:
               break
           
           # Try a few seeds
           for seed in range(3):
               if time.time() > deadline - 5:
                   break
               
               res = run_als(n, m, p, rank, max_iter=20, seed=seed)
               if res is not None:
                   U_flat, V_flat, W_flat = res
                   return {
                       "U": U_flat.tolist(),
                       "V": V_flat.tolist(),
                       "W": W_flat.tolist()
                   }

       # If no better rank found, return schoolbook
       return {
           "U": U_school.tolist(),
           "V": V_school.tolist(),
           "W": W_school.tolist()
       }
